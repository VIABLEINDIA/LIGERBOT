"""Reconciliation and session-recording tests (DESIGN.md 2.6, Phase 4).

The property these pin: **attribution, not just divergence.** A reconciliation reporting
only "paper made 12,000 less" invites the wrong conclusion. Each cause has a different fix,
so each must be measured separately — and the residual must stay near zero, because a large
unattributed remainder means the diagnosis itself cannot be trusted.
"""
from __future__ import annotations

import datetime as dt

import pytest

from src.reconciliation import ReconciliationTolerance, reconcile
from src.session_recorder import RecordedTrade, SessionRecord, SessionStore

DAY = "2026-07-23"


def trade(instrument="nse_cm:1", entry_hour=10, entry_price=100.0, exit_price=102.0,
          net_pnl=180.0, costs=20.0, r_multiple=0.9, exit_reason="signal",
          quantity=100, slippage=5.0) -> RecordedTrade:
    return RecordedTrade(
        instrument_id=instrument, direction="LONG", quantity=quantity,
        entry_at=f"2026-07-23T{entry_hour:02d}:00:00+05:30",
        entry_price=entry_price,
        exit_at=f"2026-07-23T{entry_hour + 1:02d}:00:00+05:30",
        exit_price=exit_price, exit_reason=exit_reason,
        gross_pnl=net_pnl + costs, costs=costs, slippage=slippage,
        net_pnl=net_pnl, risk_amount=200.0, r_multiple=r_multiple,
    )


def record(source: str, trades, *, day: str = DAY, halted: bool = False,
           halt_reason: str = "") -> SessionRecord:
    return SessionRecord(
        day=day, source=source, starting_equity=500_000.0,
        ending_equity=500_000.0 + sum(t.net_pnl for t in trades),
        trades=list(trades), halted=halted, halt_reason=halt_reason,
    )


@pytest.fixture
def store(tmp_path) -> SessionStore:
    return SessionStore(tmp_path)


class TestSessionStore:
    def test_round_trip(self, store):
        original = record("paper", [trade(), trade(instrument="nse_cm:2")])
        store.save(original)
        loaded = store.load("paper", DAY)
        assert loaded is not None
        assert loaded.trade_count == 2
        assert loaded.net_pnl == pytest.approx(original.net_pnl)

    def test_missing_record_returns_none(self, store):
        assert store.load("paper", "2020-01-01") is None

    def test_sources_are_separate(self, store):
        store.save(record("paper", [trade()]))
        store.save(record("backtest", [trade(), trade()]))
        assert len(store.load("paper", DAY).trades) == 1
        assert len(store.load("backtest", DAY).trades) == 2

    def test_common_days_intersects(self, store):
        store.save(record("paper", [trade()], day="2026-07-23"))
        store.save(record("paper", [trade()], day="2026-07-24"))
        store.save(record("backtest", [trade()], day="2026-07-24"))
        assert store.common_days("paper", "backtest") == ["2026-07-24"]

    def test_corrupt_record_returns_none(self, store):
        path = store.path_for("paper", DAY)
        path.write_text("{not json", encoding="utf-8")
        assert store.load("paper", DAY) is None

    def test_derived_metrics(self):
        session = record("paper", [trade(net_pnl=100.0, r_multiple=0.5),
                                   trade(net_pnl=-50.0, r_multiple=-0.25)])
        assert session.trade_count == 2
        assert session.win_count == 1
        assert session.win_rate == 0.5
        assert session.expectancy_r == pytest.approx(0.125)

    def test_halted_session_is_not_comparable(self):
        assert not record("paper", [], halted=True, halt_reason="drawdown").comparable
        assert record("paper", []).comparable


class TestPerfectAgreement:
    def test_identical_sessions_reconcile_clean(self, store):
        trades = [trade(), trade(instrument="nse_cm:2", entry_hour=11)]
        store.save(record("paper", trades))
        store.save(record("backtest", trades))

        result = reconcile(store)
        assert result.days_compared == 1
        assert result.match_rate == 1.0
        assert result.pnl_divergence == pytest.approx(0.0)
        assert result.passed, result.failures


class TestAttribution:
    def test_missing_trade_is_attributed(self, store):
        """The backtest traded and paper did not — usually a feed gap or a block."""
        store.save(record("paper", [trade()]))
        store.save(record("backtest", [trade(), trade(instrument="nse_cm:2",
                                                      entry_hour=13)]))

        result = reconcile(store)
        assert len(result.missing_in_paper) == 1
        assert result.attribution.missing_trades == pytest.approx(-180.0)
        assert result.unexplained == pytest.approx(0.0, abs=0.01)

    def test_extra_trade_is_attributed(self, store):
        store.save(record("paper", [trade(), trade(instrument="nse_cm:2",
                                                   entry_hour=13)]))
        store.save(record("backtest", [trade()]))

        result = reconcile(store)
        assert len(result.extra_in_paper) == 1
        assert result.attribution.extra_trades == pytest.approx(180.0)

    def test_fill_price_divergence_is_attributed(self, store):
        """Same trade, worse fill — the slippage model or execution speed is wrong."""
        store.save(record("paper", [trade(entry_price=100.5, net_pnl=130.0)]))
        store.save(record("backtest", [trade(entry_price=100.0, net_pnl=180.0)]))

        result = reconcile(store)
        assert len(result.pairs) == 1
        assert result.attribution.fill_prices == pytest.approx(-50.0)
        assert result.pairs[0].entry_divergence_bps == pytest.approx(50.0)

    def test_cost_divergence_is_attributed(self, store):
        store.save(record("paper", [trade(costs=35.0, net_pnl=165.0)]))
        store.save(record("backtest", [trade(costs=20.0, net_pnl=180.0)]))

        result = reconcile(store)
        assert result.attribution.costs == pytest.approx(-15.0)

    def test_attribution_sums_to_the_total_divergence(self, store):
        """The residual must stay near zero, or the diagnosis cannot be trusted."""
        store.save(record("paper", [
            trade(entry_price=100.4, net_pnl=140.0, costs=25.0),
            trade(instrument="nse_cm:3", entry_hour=14, net_pnl=90.0),
        ]))
        store.save(record("backtest", [
            trade(entry_price=100.0, net_pnl=180.0, costs=20.0),
            trade(instrument="nse_cm:2", entry_hour=12, net_pnl=200.0),
        ]))

        result = reconcile(store)
        assert result.unexplained == pytest.approx(0.0, abs=0.01)
        assert result.attribution.total == pytest.approx(result.pnl_divergence, abs=0.01)


class TestMatching:
    def test_matches_within_the_time_window(self, store):
        store.save(record("paper", [trade(entry_hour=10)]))
        store.save(record("backtest", [trade(entry_hour=10)]))
        assert len(reconcile(store).pairs) == 1

    def test_does_not_match_outside_the_window(self, store):
        store.save(record("paper", [trade(entry_hour=14)]))
        store.save(record("backtest", [trade(entry_hour=10)]))
        result = reconcile(store)
        assert result.pairs == []
        assert len(result.missing_in_paper) == 1
        assert len(result.extra_in_paper) == 1

    def test_does_not_match_across_instruments(self, store):
        store.save(record("paper", [trade(instrument="nse_cm:2")]))
        store.save(record("backtest", [trade(instrument="nse_cm:1")]))
        assert reconcile(store).pairs == []

    def test_exit_reason_agreement_is_measured(self, store):
        store.save(record("paper", [trade(exit_reason="stop_loss")]))
        store.save(record("backtest", [trade(exit_reason="signal")]))
        result = reconcile(store)
        assert result.exit_reason_agreement == 0.0
        assert not result.pairs[0].same_exit_reason


class TestTolerance:
    def test_low_match_rate_blocks(self, store):
        store.save(record("paper", [trade()]))
        store.save(record("backtest", [
            trade(), trade(instrument="nse_cm:2", entry_hour=11),
            trade(instrument="nse_cm:3", entry_hour=12),
        ]))
        result = reconcile(store)
        assert not result.passed
        assert any("appeared in paper" in f for f in result.failures)

    def test_large_fill_divergence_blocks(self, store):
        store.save(record("paper", [trade(entry_price=105.0)]))
        store.save(record("backtest", [trade(entry_price=100.0)]))
        result = reconcile(store)
        assert not result.passed
        assert any("slippage model" in f for f in result.failures)

    def test_expectancy_gap_blocks(self, store):
        store.save(record("paper", [trade(r_multiple=0.2)]))
        store.save(record("backtest", [trade(r_multiple=0.9)]))
        result = reconcile(store)
        assert not result.passed
        assert any("expectancy differs" in f for f in result.failures)

    def test_tolerance_is_configurable(self, store):
        store.save(record("paper", [trade(entry_price=100.2)]))
        store.save(record("backtest", [trade(entry_price=100.0)]))
        strict = reconcile(store, tolerance=ReconciliationTolerance(
            max_entry_divergence_bps=1.0))
        assert not strict.passed

    def test_no_comparable_days_does_not_pass(self, store):
        result = reconcile(store)
        assert result.days_compared == 0
        assert not result.passed
        assert "No comparable sessions" in result.report()


class TestHaltedSessions:
    def test_halted_paper_day_is_skipped(self, store):
        """A day that halted at 11:00 cannot be compared with one that ran to 15:10."""
        store.save(record("paper", [trade()], halted=True, halt_reason="drawdown"))
        store.save(record("backtest", [trade(), trade(entry_hour=13)]))

        result = reconcile(store)
        assert result.days_compared == 0
        assert len(result.days_skipped) == 1
        assert "drawdown" in result.days_skipped[0]


class TestReport:
    def test_report_shows_attribution(self, store):
        store.save(record("paper", [trade(entry_price=100.5, net_pnl=130.0)]))
        store.save(record("backtest", [trade()]))
        text = reconcile(store).report()
        assert "Attribution" in text
        assert "fill-price divergence" in text
        assert "unexplained residual" in text

    def test_failing_report_does_not_claim_live_clearance(self, store):
        store.save(record("paper", [trade(entry_price=110.0)]))
        store.save(record("backtest", [trade()]))
        assert "BLOCKED" in reconcile(store).report()

    def test_passing_report_still_defers_live(self, store):
        trades = [trade()]
        store.save(record("paper", trades))
        store.save(record("backtest", trades))
        text = reconcile(store).report()
        assert "does NOT by itself clear" in text
