"""Paper broker and briefing tests (Phase 4).

The central property: **paper must fill exactly as the backtester does.** Phase 4's whole
purpose is comparing the two, so any divergence in fill semantics would be measured as a
strategy result rather than as the modelling artefact it is.
"""
from __future__ import annotations

import datetime as dt

import fakeredis
import pytest

import config
from src import event_bus
from src import market_calendar as cal
from src.backtest.costs import CostModel, SlippageModel
from src.backtest.sim_broker import FillReason
from src.bars import Bar

DAY = dt.date(2026, 7, 23)


@pytest.fixture
def client(monkeypatch):
    server = fakeredis.FakeServer()

    def factory(*_a, **_k):
        return fakeredis.FakeStrictRedis(server=server, decode_responses=True)

    monkeypatch.setattr(event_bus, "get_client", factory)
    monkeypatch.setattr(event_bus, "ping", lambda *a, **k: True)
    return factory()


@pytest.fixture
def broker(client):
    from src.paper_broker import PaperBroker

    return PaperBroker(CostModel(), SlippageModel(slippage_bps=2.5, half_spread_bps=1.0))


def bar_event(minute: int, o: float, h: float, l: float, c: float,
              volume: float = 100_000.0, instrument="nse_cm:1") -> dict:
    start = cal.at(DAY, dt.time(11, minute))
    return Bar(instrument, start, start + dt.timedelta(minutes=1),
               o, h, l, c, volume=volume, vwap=(h + l + c) / 3,
               tick_count=200).to_event()


def order_event(quantity=100, stop=99.0, instrument="nse_cm:1", side="BUY",
                intent="OPEN_LONG", price=100.0) -> dict:
    return {
        "client_order_id": "lbtest001", "instrument_id": instrument,
        "side": side, "intent": intent, "quantity": quantity,
        "price": price, "stop_loss": stop, "strategy_name": "test",
    }


def fills(client) -> list:
    return [f for _, f in client.xrange(config.STREAM_FILLED_ORDERS)]


class TestNextBarExecution:
    """The anti-look-ahead rule, carried into paper trading."""

    def test_order_does_not_fill_until_a_bar_arrives(self, broker, client):
        broker._handle_order(order_event())
        assert fills(client) == []

    def test_fill_uses_the_next_bars_open_not_the_signal_price(self, broker, client):
        broker._handle_order(order_event(price=100.0))
        broker._handle_bar(bar_event(1, 102.0, 103.0, 101.0, 102.5))

        recorded = fills(client)
        assert len(recorded) == 1
        # Filled near 102 (the bar's open), NOT at the 100.0 signal price. The old
        # DRY_RUN path filled at the signal price, which flattered every result.
        assert float(recorded[0]["average_fill_price"]) == pytest.approx(102.0, abs=0.1)

    def test_fill_is_worse_than_the_open_for_a_buy(self, broker, client):
        broker._handle_order(order_event())
        broker._handle_bar(bar_event(1, 100.0, 101.0, 99.5, 100.5))
        recorded = fills(client)[0]
        assert float(recorded["average_fill_price"]) > 100.0
        assert float(recorded["slippage_per_share"]) > 0


class TestFillsMatchTheBacktester:
    def test_stop_is_respected_within_the_bar(self, broker, client):
        broker._handle_order(order_event(stop=99.0))
        broker._handle_bar(bar_event(1, 100.0, 100.5, 98.0, 98.5))
        reasons = [f["fill_reason"] for f in fills(client)]
        assert FillReason.STOP_LOSS.value in reasons

    def test_costs_are_charged_on_every_fill(self, broker, client):
        broker._handle_order(order_event())
        broker._handle_bar(bar_event(1, 100.0, 101.0, 99.5, 100.5))
        assert float(fills(client)[0]["costs"]) > 0

    def test_liquidity_cap_trims_the_order(self, broker, client):
        broker._handle_order(order_event(quantity=100_000))
        broker._handle_bar(bar_event(1, 100.0, 101.0, 99.5, 100.5, volume=1000.0))
        assert int(fills(client)[0]["filled_quantity"]) == 100  # 10% of bar volume

    def test_nothing_fills_on_a_synthetic_bar(self, broker, client):
        broker._handle_order(order_event())
        synthetic = Bar("nse_cm:1", cal.at(DAY, dt.time(11, 1)),
                        cal.at(DAY, dt.time(11, 2)), 100.0, 100.0, 100.0, 100.0,
                        volume=0.0, vwap=100.0, tick_count=0, synthetic=True).to_event()
        broker._handle_bar(synthetic)
        assert fills(client) == []


class TestSessionControl:
    def test_position_is_flattened_at_the_square_off(self, broker, client):
        broker._handle_order(order_event(stop=90.0))
        broker._handle_bar(bar_event(1, 100.0, 100.5, 99.5, 100.0))
        assert broker.broker.has_position("nse_cm:1")

        late = cal.at(DAY, dt.time(15, 15))
        forced = Bar("nse_cm:1", late, late + dt.timedelta(minutes=1),
                     100.0, 100.5, 99.5, 100.0, volume=50_000.0,
                     vwap=100.0, tick_count=100).to_event()
        broker._handle_bar(forced)

        assert not broker.broker.has_position("nse_cm:1")
        assert FillReason.SQUARE_OFF.value in [f["fill_reason"] for f in fills(client)]


class TestPublishedShape:
    def test_fill_looks_like_a_live_fill(self, broker, client):
        broker._handle_order(order_event())
        broker._handle_bar(bar_event(1, 100.0, 101.0, 99.5, 100.5))
        recorded = fills(client)[0]
        for key in ("client_order_id", "instrument_id", "side", "status",
                    "filled_quantity", "average_fill_price", "costs"):
            assert key in recorded
        assert recorded["status"] == "FILLED"
        assert recorded["mode"] == "paper"

    def test_client_order_id_is_carried_through(self, broker, client):
        broker._handle_order(order_event())
        broker._handle_bar(bar_event(1, 100.0, 101.0, 99.5, 100.5))
        assert fills(client)[0]["client_order_id"] == "lbtest001"

    def test_zero_quantity_order_raises(self, broker):
        with pytest.raises(ValueError, match="quantity"):
            broker._handle_order(order_event(quantity=0))

    def test_missing_instrument_raises(self, broker):
        with pytest.raises(ValueError, match="without an instrument"):
            broker._handle_order({"side": "BUY", "quantity": 10})


class TestTradingMode:
    def test_three_modes_are_recognised(self):
        assert config.TRADING_MODE in ("dry_run", "paper", "live")

    def test_dry_run_default_maps_to_dry_run_mode(self, monkeypatch):
        # DRY_RUN=true historically meant "do not send"; that must keep working.
        assert config.TRADING_MODE == "dry_run" or not config.DRY_RUN


class TestMorningBriefing:
    def _build(self, **kwargs):
        from src.briefing import build_morning

        defaults = dict(day=DAY, equity=500_000.0, open_positions=0,
                        strategy="trend_pullback v1")
        defaults.update(kwargs)
        return build_morning(**defaults)

    def test_healthy_state_is_ready(self):
        assert self._build().ready

    def test_unresolved_equity_blocks(self):
        briefing = self._build(equity=None)
        assert not briefing.ready
        assert any("mis-sized" in c.detail for c in briefing.blockers)

    def test_equity_below_the_floor_blocks(self):
        briefing = self._build(equity=50_000.0)
        assert not briefing.ready

    def test_overnight_position_blocks(self):
        """Intraday means intraday — a carried position needs investigating."""
        briefing = self._build(open_positions=2)
        assert not briefing.ready
        assert any("overnight" in c.detail for c in briefing.blockers)

    def test_non_trading_day_blocks(self):
        assert not self._build(day=dt.date(2026, 7, 25)).ready   # Saturday

    def test_halt_blocks(self, client):
        from src.kill_switch import KillSwitch

        KillSwitch(client).halt("investigating")
        briefing = self._build(client=client)
        assert not briefing.ready
        assert any("investigating" in c.detail for c in briefing.blockers)

    def test_report_leads_with_blockers(self):
        text = self._build(equity=None).render()
        head = text.split("Pre-flight")[0]
        assert "NOT READY" in head

    def test_report_shows_phase_progress(self, tmp_path):
        from src.session_recorder import SessionStore

        text = self._build(store=SessionStore(tmp_path)).render()
        assert "Paper sessions" in text


class TestEveningBriefing:
    def test_reports_the_session(self, tmp_path):
        from src.briefing import build_evening
        from src.session_recorder import RecordedTrade, SessionRecord, SessionStore

        store = SessionStore(tmp_path)
        store.save(SessionRecord(
            day=DAY.isoformat(), source="paper",
            starting_equity=500_000.0, ending_equity=500_450.0,
            trades=[RecordedTrade(
                instrument_id="nse_cm:1", direction="LONG", quantity=100,
                entry_at="2026-07-23T10:00:00+05:30", entry_price=100.0,
                exit_at="2026-07-23T11:00:00+05:30", exit_price=105.0,
                exit_reason="signal", gross_pnl=500.0, costs=50.0, slippage=10.0,
                net_pnl=450.0, risk_amount=500.0, r_multiple=0.9)],
            signals_generated=5, signals_rejected=4,
            rejection_reasons={"max open positions reached": 4},
        ))
        text = build_evening(DAY, store=store).render()
        assert "+450.00" in text
        assert "Signal flow" in text
        assert "max open positions reached" in text

    def test_missing_session_is_flagged(self, tmp_path):
        from src.briefing import build_evening
        from src.session_recorder import SessionStore

        text = build_evening(DAY, store=SessionStore(tmp_path)).render()
        assert "No session recorded" in text

    def test_halted_session_notes_exclusion(self, tmp_path):
        from src.briefing import build_evening
        from src.session_recorder import SessionRecord, SessionStore

        store = SessionStore(tmp_path)
        store.save(SessionRecord(
            day=DAY.isoformat(), source="paper", starting_equity=500_000.0,
            ending_equity=490_000.0, halted=True, halt_reason="daily drawdown"))
        text = build_evening(DAY, store=store).render()
        assert "HALTED" in text
        assert "excluded from reconciliation" in text
