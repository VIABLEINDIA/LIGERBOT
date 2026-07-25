"""Live-trading guard and scaling-ladder tests (DESIGN.md Phase 5).

The property that matters most: **``TRADING_MODE=live`` must not be sufficient.** An
environment variable is one typo, one copied ``.env``, one careless export away from
committing real capital. Every one of these tests exists to keep that true.
"""
from __future__ import annotations

import datetime as dt

import pytest

import config
from src.live_guard import (
    LiveAuthorisation, LiveTradingBlocked, evaluate, read_authorisation,
    require_live_clearance, write_authorisation,
)
from src.live_scaling import DEFAULT_LADDER, ScalingLadder

DAY = dt.date(2026, 7, 23)


@pytest.fixture
def auth_path(tmp_path, monkeypatch):
    path = tmp_path / "live_authorisation.json"
    monkeypatch.setattr(config, "LIVE_AUTH_PATH", str(path))
    return path


def cleared_kwargs(**overrides):
    """Everything passing, for one-at-a-time negation."""
    base = dict(
        day=DAY,
        backtest_gates_passed=True,
        phase4_gates_passed=True,
        paper_sessions=25,
        probe_completed=True,
        equity=400_000.0,
        instrument_master_loaded=True,
    )
    base.update(overrides)
    return base


def blocker_names(report) -> set:
    return {c.name for c in report.blockers}


class TestNothingClearsByDefault:
    def test_unknown_prerequisites_block(self, auth_path):
        """Unknown must never read as passed — the same rule the §2.6 gates follow."""
        report = evaluate(day=DAY)
        assert not report.cleared
        assert len(report.blockers) >= 4

    def test_env_var_alone_is_not_enough(self, auth_path, monkeypatch):
        monkeypatch.setattr(config, "TRADING_MODE", "live")
        assert not evaluate(day=DAY).cleared

    def test_full_prerequisites_plus_authorisation_clears(self, auth_path):
        write_authorisation(400_000.0, "tester", day=DAY)
        report = evaluate(**cleared_kwargs())
        assert report.cleared, blocker_names(report)


class TestIndividualBlockers:
    @pytest.fixture(autouse=True)
    def _authorised(self, auth_path):
        write_authorisation(400_000.0, "tester", day=DAY)

    def test_failing_backtest_gates_block(self):
        report = evaluate(**cleared_kwargs(backtest_gates_passed=False))
        assert not report.cleared
        assert any("Backtest gates" in n for n in blocker_names(report))

    def test_failing_paper_gate_blocks(self):
        report = evaluate(**cleared_kwargs(phase4_gates_passed=False))
        assert not report.cleared
        assert any("Paper trading gate" in n for n in blocker_names(report))

    def test_unrun_probe_blocks(self):
        """The equity field names are guesses until the probe verifies them."""
        report = evaluate(**cleared_kwargs(probe_completed=False))
        assert not report.cleared
        blocker = next(c for c in report.blockers if "probe" in c.name)
        assert "mis-sizes every trade" in blocker.remedy

    def test_missing_instrument_master_blocks(self):
        report = evaluate(**cleared_kwargs(instrument_master_loaded=False))
        assert not report.cleared
        assert any("Instrument master" in n for n in blocker_names(report))

    def test_unresolved_equity_blocks(self):
        report = evaluate(**cleared_kwargs(equity=None))
        assert not report.cleared

    def test_equity_below_the_floor_blocks(self):
        report = evaluate(**cleared_kwargs(equity=50_000.0))
        assert not report.cleared
        assert any("viability floor" in n for n in blocker_names(report))


class TestAuthorisation:
    def test_missing_authorisation_blocks(self, auth_path):
        report = evaluate(**cleared_kwargs())
        assert not report.cleared
        assert any("authorisation" in n for n in blocker_names(report))

    def test_authorisation_round_trips(self, auth_path):
        write_authorisation(300_000.0, "operator", strategy="trend_pullback", day=DAY)
        loaded = read_authorisation()
        assert loaded is not None
        assert loaded.authorised_by == "operator"
        assert loaded.capital == 300_000.0

    def test_stale_authorisation_blocks(self, auth_path):
        """A decision made weeks ago is not today's decision."""
        write_authorisation(400_000.0, "tester", day=DAY - dt.timedelta(days=30))
        report = evaluate(**cleared_kwargs())
        assert not report.cleared
        blocker = next(c for c in report.blockers if "authorisation" in c.name)
        assert "days old" in blocker.detail

    def test_future_dated_authorisation_blocks(self, auth_path):
        write_authorisation(400_000.0, "tester", day=DAY + dt.timedelta(days=5))
        assert not evaluate(**cleared_kwargs()).cleared

    def test_capital_mismatch_blocks(self, auth_path):
        """Authorising for a small account and running against a large one."""
        write_authorisation(50_000.0, "tester", day=DAY)
        report = evaluate(**cleared_kwargs(equity=500_000.0))
        assert not report.cleared
        assert any("authorised capital" in n for n in blocker_names(report))

    def test_equity_below_authorised_is_fine(self, auth_path):
        # Trading less than authorised is not a risk; trading more is.
        write_authorisation(500_000.0, "tester", day=DAY)
        assert evaluate(**cleared_kwargs(equity=400_000.0)).cleared

    def test_corrupt_authorisation_blocks(self, auth_path):
        auth_path.parent.mkdir(parents=True, exist_ok=True)
        auth_path.write_text("{not json", encoding="utf-8")
        assert read_authorisation() is None
        assert not evaluate(**cleared_kwargs()).cleared


class TestRequireClearance:
    def test_raises_when_blocked(self, auth_path):
        with pytest.raises(LiveTradingBlocked) as exc:
            require_live_clearance(day=DAY)
        assert "blocked" in str(exc.value).lower()

    def test_passes_when_cleared(self, auth_path):
        write_authorisation(400_000.0, "tester", day=DAY)
        assert require_live_clearance(**cleared_kwargs()).cleared

    def test_no_bypass_parameter_exists(self):
        """A bypass becomes the thing someone reaches for at 09:14.

        Inspects the actual signatures rather than the source text, so the module can
        still *describe* having no override without tripping its own check.
        """
        import inspect

        from src import live_guard

        banned = {"force", "override", "skip_checks", "ignore_blockers", "bypass"}
        for name, func in inspect.getmembers(live_guard, inspect.isfunction):
            params = set(inspect.signature(func).parameters)
            leaked = params & banned
            assert not leaked, f"{name}() exposes a bypass parameter: {leaked}"


class TestReport:
    def test_blocked_report_lists_remedies(self, auth_path):
        text = evaluate(day=DAY).render()
        assert "BLOCKED" in text
        assert "->" in text          # remedies are shown
        assert "no override" in text

    def test_cleared_report_advises_minimum_size(self, auth_path):
        write_authorisation(400_000.0, "tester", day=DAY)
        text = evaluate(**cleared_kwargs()).render()
        assert "CLEARED" in text
        assert "smallest tradable size" in text


# ---------------------------------------------------------------------------
class TestScalingLadder:
    @pytest.fixture
    def ladder(self, tmp_path):
        return ScalingLadder(state_path=tmp_path / "scaling.json")

    def test_starts_at_the_smallest_size(self, ladder):
        assert ladder.rung.name == "minimum"
        assert ladder.size_multiplier == 0.10

    def test_scales_the_equity_base_not_the_rules(self, ladder):
        """Keeps every proportional guarantee from D2 intact through the ramp."""
        assert ladder.scaled_equity(1_000_000.0) == pytest.approx(100_000.0)

    def test_promotion_requires_sessions_and_trades(self, ladder):
        rung = ladder.rung
        # Enough sessions but not enough trades — five quiet days prove nothing.
        for i in range(rung.min_sessions):
            ladder.record_session(DAY, trades=1, net_pnl=100.0, expectancy_r_sum=0.3)
        assert ladder.rung.name == "minimum"

    def test_promotion_happens_on_sufficient_evidence(self, ladder):
        rung = ladder.rung
        for _ in range(rung.min_sessions):
            ladder.record_session(DAY, trades=rung.min_trades, net_pnl=500.0,
                                  expectancy_r_sum=0.3 * rung.min_trades)
        assert ladder.rung.name == "quarter"

    def test_promotion_blocked_by_poor_expectancy(self, ladder):
        rung = ladder.rung
        for _ in range(rung.min_sessions):
            ladder.record_session(DAY, trades=rung.min_trades, net_pnl=-50.0,
                                  expectancy_r_sum=-0.5 * rung.min_trades)
        assert ladder.rung.name == "minimum"

    def test_demotion_skips_straight_to_the_floor(self, ladder, monkeypatch):
        """Promoting too slowly costs upside; too quickly costs capital."""
        ladder.state.rung_index = 3      # three-quarter size
        monkeypatch.setattr(config, "LIVE_MAX_LOSING_SESSIONS", 3)
        for _ in range(3):
            ladder.record_session(DAY, trades=2, net_pnl=-500.0, expectancy_r_sum=-1.0)
        assert ladder.rung.name == "minimum"
        assert ladder.state.demotions == 1

    def test_halted_session_demotes(self, ladder):
        ladder.state.rung_index = 2
        note = ladder.record_session(DAY, trades=1, net_pnl=-100.0,
                                     expectancy_r_sum=-0.5, halted=True)
        assert "DEMOTED" in note
        assert ladder.rung.name == "minimum"

    def test_poor_expectancy_demotes(self, ladder):
        ladder.state.rung_index = 2
        ladder.record_session(DAY, trades=15, net_pnl=-200.0, expectancy_r_sum=-3.0)
        assert ladder.rung.name == "minimum"

    def test_winning_session_resets_the_losing_streak(self, ladder):
        ladder.record_session(DAY, trades=2, net_pnl=-100.0, expectancy_r_sum=-0.2)
        ladder.record_session(DAY, trades=2, net_pnl=+100.0, expectancy_r_sum=+0.2)
        assert ladder.state.consecutive_losing_sessions == 0

    def test_state_persists_across_restarts(self, tmp_path):
        path = tmp_path / "scaling.json"
        first = ScalingLadder(state_path=path)
        first.state.rung_index = 2
        first.save()
        assert ScalingLadder(state_path=path).rung.name == "half"

    def test_unreadable_state_falls_to_the_floor(self, tmp_path):
        """If we cannot read how much size was earned, assume none."""
        path = tmp_path / "scaling.json"
        path.write_text("{corrupt", encoding="utf-8")
        assert ScalingLadder(state_path=path).rung.name == "minimum"

    def test_force_floor(self, ladder):
        ladder.state.rung_index = 3
        ladder.force_floor("incident")
        assert ladder.rung.name == "minimum"

    def test_full_size_does_not_promote_further(self, ladder):
        ladder.state.rung_index = len(DEFAULT_LADDER) - 1
        assert ladder.at_full_size
        for _ in range(50):
            ladder.record_session(DAY, trades=10, net_pnl=1000.0, expectancy_r_sum=5.0)
        assert ladder.rung.name == "full"

    def test_ladder_never_skips_a_rung_upward(self, ladder):
        seen = [ladder.state.rung_index]
        for _ in range(200):
            rung = ladder.rung
            ladder.record_session(DAY, trades=max(1, rung.min_trades),
                                  net_pnl=500.0,
                                  expectancy_r_sum=0.3 * max(1, rung.min_trades))
            if ladder.state.rung_index != seen[-1]:
                seen.append(ladder.state.rung_index)
        assert seen == sorted(seen)
        assert all(b - a == 1 for a, b in zip(seen, seen[1:]))


class TestAbsoluteBackstops:
    """Percentage caps scale with a wrong equity figure; these do not."""

    def _engine(self, **limit_overrides):
        from src.risk_engine import RiskEngine, RiskLimits

        engine = RiskEngine(RiskLimits(**limit_overrides))
        engine.start_session(DAY, 1_000_000.0)
        return engine

    def _signal(self, instrument="nse_cm:1"):
        from src.risk_engine import Intent, Signal

        return Signal(instrument_id=instrument, intent=Intent.OPEN_LONG,
                      ref_price=100.0, stop_loss=99.0,
                      bar_time=dt.datetime(2026, 7, 23, 10, 30))

    def test_order_cap_blocks_further_entries(self):
        engine = self._engine(max_orders_per_session=2, max_open_positions=10,
                              max_daily_drawdown=0.10, max_open_risk=0.05,
                              risk_per_trade=0.005)
        for i in range(2):
            decision = engine.evaluate(self._signal(f"nse_cm:{i}"),
                                       allows_entry=True, allows_exit=True)
            assert decision.approved
            engine.on_open_fill(f"nse_cm:{i}", decision.order.quantity, 100.0, 99.0)

        blocked = engine.evaluate(self._signal("nse_cm:9"),
                                  allows_entry=True, allows_exit=True)
        assert not blocked.approved
        assert "order cap" in blocked.reason

    def test_absolute_loss_cap_blocks_entries(self):
        engine = self._engine(max_daily_loss_absolute=5_000.0)
        engine.realized_pnl_today = -6_000.0   # under the 2% percentage limit
        decision = engine.evaluate(self._signal(), allows_entry=True, allows_exit=True)
        assert not decision.approved
        assert "absolute daily loss cap" in decision.reason

    def test_backstops_are_disabled_by_default(self):
        engine = self._engine()
        assert engine.limits.max_daily_loss_absolute == 0.0
        assert engine.limits.max_orders_per_session == 0
        assert engine.evaluate(self._signal(), allows_entry=True,
                               allows_exit=True).approved

    def test_order_count_resets_each_session(self):
        engine = self._engine(max_orders_per_session=1)
        engine.evaluate(self._signal(), allows_entry=True, allows_exit=True)
        assert engine.orders_this_session == 1
        engine.start_session(dt.date(2026, 7, 24), 1_000_000.0)
        assert engine.orders_this_session == 0

    def test_backstops_only_apply_in_live_mode(self, monkeypatch):
        """In backtest and paper they would cap activity for non-strategy reasons."""
        monkeypatch.setattr(config, "TRADING_MODE", "paper")
        assert config.risk_limits().max_orders_per_session == 0
        monkeypatch.setattr(config, "TRADING_MODE", "live")
        assert config.risk_limits().max_orders_per_session == config.LIVE_MAX_ORDERS_PER_DAY
