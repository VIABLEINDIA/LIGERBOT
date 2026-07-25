"""Risk engine tests.

The centrepiece is :class:`TestOpenRiskInvariant`, which is Phase 0's exit criterion:
the engine must be *provably* incapable of carrying more than the configured total open
risk, across arbitrary sequences of signals — not merely observed to behave on a few
hand-picked cases.
"""
from __future__ import annotations

import datetime as dt
import random

import pytest

from src.risk_engine import (
    Intent, OrderRequest, Position, RiskDecision, RiskEngine, RiskLimits, Side, Signal,
)

DAY = dt.date(2026, 7, 23)
BAR_TIME = dt.datetime(2026, 7, 23, 10, 0)
EQUITY = 1_000_000.0


def make_engine(**overrides) -> RiskEngine:
    engine = RiskEngine(RiskLimits(**overrides))
    engine.start_session(DAY, EQUITY)
    return engine


def long_signal(instrument="nse_cm:1", price=100.0, stop=99.0, **kw) -> Signal:
    return Signal(
        instrument_id=instrument, intent=Intent.OPEN_LONG,
        ref_price=price, stop_loss=stop, bar_time=BAR_TIME, **kw
    )


def approve(engine: RiskEngine, signal: Signal) -> RiskDecision:
    return engine.evaluate(signal, allows_entry=True, allows_exit=True)


# ---------------------------------------------------------------------------
# Limits validation
# ---------------------------------------------------------------------------
class TestRiskLimits:
    def test_defaults_are_the_d2_set(self):
        limits = RiskLimits()
        assert limits.max_daily_drawdown == 0.02
        assert limits.max_open_risk == 0.015
        assert limits.risk_per_trade == 0.005
        assert limits.max_open_positions == 3
        assert limits.allow_short is False

    def test_open_risk_must_sit_below_daily_limit(self):
        # The whole point of the cap: if open risk could equal the daily limit, a
        # simultaneous stop-out breaches the day before the breaker can act.
        with pytest.raises(ValueError, match="must be strictly below"):
            RiskLimits(max_open_risk=0.02, max_daily_drawdown=0.02)

    def test_per_trade_times_positions_cannot_exceed_open_risk(self):
        with pytest.raises(ValueError, match="contradict"):
            RiskLimits(risk_per_trade=0.01, max_open_positions=3, max_open_risk=0.015)

    def test_the_old_inconsistent_config_is_now_rejected(self):
        # The pre-existing config (1% per trade, 3 positions, 2% daily) allowed 3% of
        # concurrent risk against a 2% daily limit. It must no longer be constructible.
        with pytest.raises(ValueError):
            RiskLimits(risk_per_trade=0.01, max_open_positions=3,
                       max_open_risk=0.03, max_daily_drawdown=0.02)


# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------
class TestSizing:
    def test_quantity_comes_from_the_risk_budget(self):
        engine = make_engine()
        # 0.5% of 1,000,000 = 5,000 risk budget; stop is 1.00 away -> 5,000 shares.
        decision = approve(engine, long_signal(price=100.0, stop=99.0))
        assert decision.approved
        assert decision.order.quantity == 5_000
        assert decision.order.risk_amount == pytest.approx(5_000.0)

    def test_rounds_down_never_up(self):
        engine = make_engine()
        # 5,000 / 3.0 = 1666.67 -> 1666, so realised risk is under budget.
        decision = approve(engine, long_signal(price=100.0, stop=97.0))
        assert decision.order.quantity == 1_666
        assert decision.order.risk_amount <= engine.risk_budget_per_trade

    def test_tight_stop_is_clamped_by_the_exposure_guard(self):
        engine = make_engine()
        # A 0.2% stop would demand 250% of equity in notional for a full 0.5% risk.
        decision = approve(engine, long_signal(price=100.0, stop=99.8))
        assert decision.approved
        assert "exposure" in decision.reason
        # Clamped to 75% of equity -> 7,500 shares at 100.
        assert decision.order.quantity == 7_500
        # And the realised risk is therefore *below* budget, never above.
        assert decision.order.risk_amount < engine.risk_budget_per_trade

    def test_lot_size_rounds_down(self):
        engine = make_engine()
        decision = engine.evaluate(
            long_signal(price=100.0, stop=97.0), allows_entry=True, allows_exit=True,
            lot_size=100,
        )
        assert decision.order.quantity == 1_600  # 1666 -> 1600

    def test_equity_too_small_is_rejected_not_silently_zero(self):
        engine = RiskEngine()
        engine.start_session(DAY, 10_000.0)  # 0.5% = Rs 50 budget
        decision = approve(engine, long_signal(price=3_000.0, stop=2_900.0))
        assert not decision.approved
        assert "sized to zero" in decision.reason


# ---------------------------------------------------------------------------
# Gatekeeping
# ---------------------------------------------------------------------------
class TestGatekeeping:
    def test_open_without_stop_loss_is_rejected(self):
        # Fixes B7: previously a missing stop silently fell back to notional sizing,
        # so the documented per-trade risk cap never actually applied.
        engine = make_engine()
        signal = Signal(instrument_id="nse_cm:1", intent=Intent.OPEN_LONG,
                        ref_price=100.0, stop_loss=None, bar_time=BAR_TIME)
        decision = approve(engine, signal)
        assert not decision.approved
        assert "stop_loss" in decision.reason

    def test_stop_on_the_wrong_side_is_rejected(self):
        engine = make_engine()
        decision = approve(engine, long_signal(price=100.0, stop=101.0))
        assert not decision.approved
        assert "must be below entry" in decision.reason

    def test_stop_below_the_distance_floor_is_rejected(self):
        engine = make_engine()
        decision = approve(engine, long_signal(price=100.0, stop=99.95))  # 0.05%
        assert not decision.approved
        assert "floor" in decision.reason

    def test_shorting_disabled_by_default(self):
        engine = make_engine()
        signal = Signal(instrument_id="nse_cm:1", intent=Intent.OPEN_SHORT,
                        ref_price=100.0, stop_loss=101.0, bar_time=BAR_TIME)
        decision = approve(engine, signal)
        assert not decision.approved
        assert "long-only" in decision.reason

    def test_shorting_works_when_enabled(self):
        engine = make_engine(allow_short=True)
        signal = Signal(instrument_id="nse_cm:1", intent=Intent.OPEN_SHORT,
                        ref_price=100.0, stop_loss=101.0, bar_time=BAR_TIME)
        decision = approve(engine, signal)
        assert decision.approved
        assert decision.order.side is Side.SELL

    def test_max_open_positions_enforced(self):
        engine = make_engine()
        for i in range(3):
            decision = approve(engine, long_signal(f"nse_cm:{i}", price=100.0, stop=97.0))
            assert decision.approved
            engine.on_open_fill(f"nse_cm:{i}", decision.order.quantity, 100.0, 97.0)
        decision = approve(engine, long_signal("nse_cm:99", price=100.0, stop=97.0))
        assert not decision.approved
        assert "max open positions" in decision.reason

    def test_pyramiding_rejected(self):
        engine = make_engine()
        decision = approve(engine, long_signal(price=100.0, stop=97.0))
        engine.on_open_fill("nse_cm:1", decision.order.quantity, 100.0, 97.0)
        again = approve(engine, long_signal(price=100.0, stop=97.0))
        assert not again.approved
        assert "already holding" in again.reason

    def test_entry_blocked_outside_the_entry_phase(self):
        engine = make_engine()
        decision = engine.evaluate(long_signal(), allows_entry=False, allows_exit=True)
        assert not decision.approved
        assert "phase" in decision.reason

    def test_exit_permitted_when_entry_is_not(self):
        # The asymmetry that matters: the bot must always be able to reduce risk.
        engine = make_engine()
        opened = approve(engine, long_signal(price=100.0, stop=97.0))
        engine.on_open_fill("nse_cm:1", opened.order.quantity, 100.0, 97.0)

        close = Signal(instrument_id="nse_cm:1", intent=Intent.CLOSE_LONG,
                       ref_price=98.0, bar_time=BAR_TIME)
        decision = engine.evaluate(close, allows_entry=False, allows_exit=True)
        assert decision.approved
        assert decision.order.side is Side.SELL
        assert decision.order.quantity == opened.order.quantity

    def test_close_without_position_rejected(self):
        engine = make_engine()
        close = Signal(instrument_id="nse_cm:1", intent=Intent.CLOSE_LONG,
                       ref_price=98.0, bar_time=BAR_TIME)
        assert not engine.evaluate(close, allows_entry=True, allows_exit=True).approved

    def test_close_short_against_a_long_is_rejected(self):
        # Fixes B8: CLOSE_LONG and OPEN_SHORT were previously indistinguishable.
        engine = make_engine()
        opened = approve(engine, long_signal(price=100.0, stop=97.0))
        engine.on_open_fill("nse_cm:1", opened.order.quantity, 100.0, 97.0)
        close = Signal(instrument_id="nse_cm:1", intent=Intent.CLOSE_SHORT,
                       ref_price=98.0, bar_time=BAR_TIME)
        decision = engine.evaluate(close, allows_entry=True, allows_exit=True)
        assert not decision.approved
        assert "does not match" in decision.reason

    def test_no_session_equity_rejects_everything(self):
        engine = RiskEngine()  # start_session never called
        assert not approve(engine, long_signal()).approved


# ---------------------------------------------------------------------------
# The daily drawdown breaker (fixes B2 — previously dead code)
# ---------------------------------------------------------------------------
class TestDailyDrawdown:
    def test_breaker_trips_at_the_limit(self):
        engine = make_engine()
        assert not engine.halted
        engine.realized_pnl_today = -engine.daily_loss_cap
        assert engine.check_daily_drawdown()
        assert "daily drawdown breached" in engine.halt_reason

    def test_breaker_blocks_new_entries_but_not_exits(self):
        engine = make_engine()
        opened = approve(engine, long_signal(price=100.0, stop=97.0))
        engine.on_open_fill("nse_cm:1", opened.order.quantity, 100.0, 97.0)
        engine.halt("test halt")

        assert not approve(engine, long_signal("nse_cm:2", price=100.0, stop=97.0)).approved
        close = Signal(instrument_id="nse_cm:1", intent=Intent.CLOSE_LONG,
                       ref_price=98.0, bar_time=BAR_TIME)
        assert engine.evaluate(close, allows_entry=True, allows_exit=True).approved

    def test_losses_accumulate_through_close_fills(self):
        # The loop that was open before: nothing ever updated realized P&L.
        engine = make_engine()
        opened = approve(engine, long_signal(price=100.0, stop=97.0))
        qty = opened.order.quantity
        engine.on_open_fill("nse_cm:1", qty, 100.0, 97.0)
        pnl = engine.on_close_fill("nse_cm:1", 97.0)
        assert pnl == pytest.approx(-3.0 * qty)
        assert engine.realized_pnl_today == pytest.approx(-3.0 * qty)

    def test_four_full_risk_losses_trip_the_breaker(self):
        # D2's arithmetic: 4 x 0.5% = 2.0% = the daily limit.
        engine = make_engine()
        for i in range(4):
            assert not engine.halted, f"halted early after {i} losses"
            decision = approve(engine, long_signal(f"nse_cm:{i}", price=100.0, stop=99.0))
            assert decision.approved
            engine.on_open_fill(f"nse_cm:{i}", decision.order.quantity, 100.0, 99.0)
            engine.on_close_fill(f"nse_cm:{i}", 99.0)  # stopped out
        assert engine.halted

    def test_costs_count_against_the_limit(self):
        engine = make_engine()
        engine.apply_costs(engine.daily_loss_cap)
        assert engine.halted


# ---------------------------------------------------------------------------
# The invariant — Phase 0's exit criterion
# ---------------------------------------------------------------------------
class TestOpenRiskInvariant:
    """Total open risk must never exceed the configured cap, for any signal sequence."""

    def test_third_position_capped_by_open_risk_not_position_count(self):
        engine = make_engine()
        # Two positions at full 0.5% risk consume 1.0% of the 1.5% cap.
        for i in range(2):
            decision = approve(engine, long_signal(f"nse_cm:{i}", price=100.0, stop=99.0))
            engine.on_open_fill(f"nse_cm:{i}", decision.order.quantity, 100.0, 99.0)
        assert engine.total_open_risk_pct() == pytest.approx(0.010)

        # A third full-risk position fits exactly, reaching the cap.
        third = approve(engine, long_signal("nse_cm:3", price=100.0, stop=99.0))
        assert third.approved
        engine.on_open_fill("nse_cm:3", third.order.quantity, 100.0, 99.0)
        assert engine.total_open_risk_pct() == pytest.approx(0.015)

    def test_open_risk_cap_binds_before_the_position_count_does(self):
        engine = make_engine(max_open_positions=3, risk_per_trade=0.005)
        # Two positions opened at full risk.
        for i in range(2):
            d = approve(engine, long_signal(f"nse_cm:{i}", price=100.0, stop=99.0))
            engine.on_open_fill(f"nse_cm:{i}", d.order.quantity, 100.0, 99.0)
        # Now inflate one position's risk by widening its stop after the fact, as a
        # trailing-stop adjustment in the wrong direction would.
        engine.positions["nse_cm:0"].stop_loss = 90.0
        assert engine.total_open_risk_pct() > 0.015

        # A third position must now be refused even though only 2 of 3 slots are used.
        third = approve(engine, long_signal("nse_cm:3", price=100.0, stop=99.0))
        assert not third.approved
        assert "total open risk" in third.reason

    @pytest.mark.parametrize("seed", range(25))
    def test_invariant_holds_over_random_sequences(self, seed):
        """Fuzz: random signals, prices, stops and closes. The cap must always hold."""
        rng = random.Random(seed)
        engine = make_engine()
        cap = engine.limits.max_open_risk

        for _ in range(200):
            instrument = f"nse_cm:{rng.randint(0, 6)}"
            if instrument in engine.positions and rng.random() < 0.4:
                close = Signal(instrument_id=instrument, intent=Intent.CLOSE_LONG,
                               ref_price=rng.uniform(50, 200), bar_time=BAR_TIME)
                decision = engine.evaluate(close, allows_entry=True, allows_exit=True)
                if decision.approved:
                    engine.on_close_fill(instrument, close.ref_price)
            else:
                price = rng.uniform(50, 4_000)
                stop = price * (1 - rng.uniform(0.002, 0.05))
                decision = approve(engine, long_signal(instrument, price=price, stop=stop))
                if decision.approved:
                    # Fill exactly at the reference price: no slippage.
                    engine.on_open_fill(
                        instrument, decision.order.quantity, price, stop
                    )

            assert engine.total_open_risk_pct() <= cap + 1e-9, (
                f"open risk {engine.total_open_risk_pct():.5%} exceeded cap {cap:.5%} "
                f"with {len(engine.positions)} positions"
            )
            assert len(engine.positions) <= engine.limits.max_open_positions

    @pytest.mark.parametrize("seed", range(10))
    def test_entry_slippage_overshoot_stays_within_the_daily_limit(self, seed):
        """Slippage on entry can push realised risk above the cap — bounded, by design.

        The engine sizes on the reference price but recomputes risk from the actual fill,
        so a bad entry fill carries more risk than budgeted. This is why the open-risk cap
        (1.5%) sits below the daily loss limit (2.0%) rather than at it: the gap absorbs
        exactly this overshoot. The test pins that the gap is sufficient for realistic
        slippage.
        """
        rng = random.Random(1000 + seed)
        engine = make_engine()

        for _ in range(150):
            instrument = f"nse_cm:{rng.randint(0, 4)}"
            if instrument in engine.positions:
                engine.on_close_fill(instrument, rng.uniform(50, 200))
                continue
            price = rng.uniform(100, 2_000)
            stop = price * (1 - rng.uniform(0.005, 0.03))
            decision = approve(engine, long_signal(instrument, price=price, stop=stop))
            if decision.approved:
                # Adverse fill: up to 20 bps worse than the reference price.
                fill = price * (1 + rng.uniform(0, 0.002))
                engine.on_open_fill(instrument, decision.order.quantity, fill, stop)

            assert engine.total_open_risk_pct() <= engine.limits.max_daily_drawdown, (
                f"slippage pushed open risk to {engine.total_open_risk_pct():.4%}, "
                f"past the {engine.limits.max_daily_drawdown:.2%} daily limit"
            )


class TestWorstCaseBound:
    """The daily limit must bound the worst case, not just what has been realised."""

    def test_projected_loss_counts_open_risk(self):
        engine = make_engine()
        decision = approve(engine, long_signal(price=100.0, stop=99.0))
        engine.on_open_fill("nse_cm:1", decision.order.quantity, 100.0, 99.0)
        assert engine.projected_loss() == pytest.approx(-5_000.0)  # 0.5% of 1m

    def test_entry_refused_when_worst_case_would_breach_the_daily_limit(self):
        engine = make_engine()
        # Bank 1.8% of losses, leaving only 0.2% of headroom to the 2% limit.
        engine.realized_pnl_today = -0.018 * EQUITY
        decision = approve(engine, long_signal(price=100.0, stop=99.0))
        assert not decision.approved
        assert "worst case" in decision.reason

    def test_realised_losses_plus_open_risk_cannot_exceed_the_limit(self):
        engine = make_engine()
        engine.realized_pnl_today = -0.01 * EQUITY  # 1% banked
        # Only 1% of headroom remains, so at most two 0.5% positions may be opened.
        opened = 0
        for i in range(5):
            decision = approve(engine, long_signal(f"nse_cm:{i}", price=100.0, stop=99.0))
            if not decision.approved:
                break
            engine.on_open_fill(f"nse_cm:{i}", decision.order.quantity, 100.0, 99.0)
            opened += 1
        assert opened == 2
        worst_case = engine.projected_loss()
        assert worst_case >= -engine.daily_loss_cap - 1e-6

    @pytest.mark.parametrize("seed", range(15))
    def test_worst_case_never_exceeds_the_daily_limit(self, seed):
        """Fuzz the property that actually protects the account."""
        rng = random.Random(5000 + seed)
        engine = make_engine()

        for _ in range(200):
            instrument = f"nse_cm:{rng.randint(0, 5)}"
            if instrument in engine.positions and rng.random() < 0.5:
                # Stop out, the worst realistic outcome.
                position = engine.positions[instrument]
                engine.on_close_fill(instrument, position.stop_loss)
            else:
                price = rng.uniform(100, 2_000)
                stop = price * (1 - rng.uniform(0.005, 0.04))
                decision = approve(engine, long_signal(instrument, price=price, stop=stop))
                if decision.approved:
                    engine.on_open_fill(instrument, decision.order.quantity, price, stop)

            assert engine.projected_loss() >= -engine.daily_loss_cap - 1e-6, (
                f"worst case {engine.projected_loss() / EQUITY:.4%} exceeds the "
                f"{engine.limits.max_daily_drawdown:.2%} daily limit"
            )


class TestSnapshot:
    def test_snapshot_reports_the_key_figures(self):
        engine = make_engine()
        decision = approve(engine, long_signal(price=100.0, stop=99.0))
        engine.on_open_fill("nse_cm:1", decision.order.quantity, 100.0, 99.0)
        snap = engine.snapshot()
        assert snap["open_positions"] == 1
        assert snap["total_open_risk_pct"] == pytest.approx(0.005)
        assert snap["halted"] is False

    def test_start_session_rejects_non_positive_equity(self):
        # Fail closed (D1): a bad broker response must not become a placeholder.
        engine = RiskEngine()
        with pytest.raises(ValueError, match="must be positive"):
            engine.start_session(DAY, 0.0)
