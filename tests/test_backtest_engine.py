"""Backtest engine tests.

Two of these matter more than the rest:

* :class:`TestNoLookAhead` — the property test from DESIGN.md 3.10. A strategy's decisions
  over bars ``0..t`` must not change when bars ``t+1..n`` are appended to the source. If
  they do, the engine is leaking the future and every result it produced is void.
* :class:`TestNegativeControl` — the Phase 1 exit criterion. The SMA reference must lose
  money after costs on a driftless random walk. If it profits, the harness is broken.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from src import market_calendar as cal
from src.backtest.bar_source import InMemoryBarSource
from src.backtest.costs import CostModel, SlippageModel
from src.backtest.engine import BacktestConfig, BacktestEngine
from src.backtest.synthetic import generate_history
from src.risk_engine import Intent, RiskLimits
from src.strategies.sma_crossover import SmaCrossover
from src.strategy_base import Strategy, StrategyContext

START, END = dt.date(2026, 3, 2), dt.date(2026, 4, 30)
INSTRUMENTS = ["nse_cm:2885", "nse_cm:1333"]


@pytest.fixture(scope="module")
def history():
    return generate_history(
        INSTRUMENTS, START, END,
        start_prices={"nse_cm:2885": 1_300.0, "nse_cm:1333": 1_650.0},
        seed=11,
    )


@pytest.fixture
def source(history):
    return InMemoryBarSource(history)


def make_config(**overrides) -> BacktestConfig:
    defaults = dict(
        starting_equity=1_000_000.0,
        risk_limits=RiskLimits(),
        skip_quality_gate=True,
    )
    defaults.update(overrides)
    return BacktestConfig(**defaults)


class RecordingStrategy(Strategy):
    """Records every bar it is shown, for the look-ahead property test."""

    name = "recording"

    def __init__(self, **params):
        super().__init__(**params)
        self.seen: list[tuple[str, dt.datetime, float]] = []

    def on_bar(self, bar, ctx: StrategyContext):
        self.seen.append((bar.instrument_id, bar.bar_start, bar.close))
        return []


class BuyOnceStrategy(Strategy):
    """Opens one long on the Nth bar of each session, then holds."""

    name = "buy_once"

    def __init__(self, trigger_bar: int = 30, stop_pct: float = 0.01, **params):
        super().__init__(trigger_bar=trigger_bar, stop_pct=stop_pct, **params)
        self.trigger_bar = trigger_bar
        self.stop_pct = stop_pct
        self._count: dict[str, int] = {}

    def on_session_start(self, day):
        self._count.clear()

    def on_bar(self, bar, ctx: StrategyContext):
        seen = self._count.get(bar.instrument_id, 0) + 1
        self._count[bar.instrument_id] = seen
        if seen == self.trigger_bar and not ctx.in_position and ctx.allows_entry:
            return [self._signal(
                bar, Intent.OPEN_LONG,
                stop_loss=round(bar.close * (1 - self.stop_pct), 2),
                reason="test entry",
            )]
        return []


class TestNoLookAhead:
    """The invariant that makes every other result meaningful."""

    def test_decisions_are_unchanged_by_future_bars(self, history):
        cut = dt.date(2026, 3, 31)

        truncated = {
            i: f[f["bar_start"].dt.date <= cut].reset_index(drop=True)
            for i, f in history.items()
        }

        short_run = RecordingStrategy()
        BacktestEngine(short_run, make_config()).run(
            InMemoryBarSource(truncated), INSTRUMENTS, START, cut)

        long_run = RecordingStrategy()
        BacktestEngine(long_run, make_config()).run(
            InMemoryBarSource(history), INSTRUMENTS, START, END)

        # Everything the short run saw must be a prefix of what the long run saw.
        assert len(long_run.seen) > len(short_run.seen)
        assert long_run.seen[:len(short_run.seen)] == short_run.seen

    def test_results_over_a_window_do_not_depend_on_later_data(self, history):
        cut = dt.date(2026, 3, 31)
        truncated = {
            i: f[f["bar_start"].dt.date <= cut].reset_index(drop=True)
            for i, f in history.items()
        }

        def trades_up_to_cut(source, end):
            engine = BacktestEngine(BuyOnceStrategy(), make_config())
            result = engine.run(source, INSTRUMENTS, START, end)
            return [
                (t.instrument_id, t.entry_at, round(t.entry_price, 2), round(t.exit_price, 2))
                for t in result.portfolio.trades if t.entry_at.date() <= cut
            ]

        assert (trades_up_to_cut(InMemoryBarSource(truncated), cut)
                == trades_up_to_cut(InMemoryBarSource(history), END))


class TestDeterminism:
    def test_identical_runs_produce_identical_results(self, source):
        def run():
            engine = BacktestEngine(SmaCrossover(short_period=5, long_period=20),
                                    make_config())
            return engine.run(source, INSTRUMENTS, START, END)

        first, second = run(), run()
        assert first.metrics.net_pnl == second.metrics.net_pnl
        assert first.metrics.trade_count == second.metrics.trade_count


class TestSessionControl:
    def test_no_position_survives_past_the_square_off(self, source):
        # Bar 20 is ~09:34, inside the entry window. Anything before bar 15 lands in the
        # opening range, where entries are correctly refused.
        engine = BacktestEngine(BuyOnceStrategy(trigger_bar=20), make_config())
        result = engine.run(source, INSTRUMENTS, START, END)
        assert result.portfolio.trades
        for trade in result.portfolio.trades:
            assert trade.exit_at.time() <= cal.SESSION_CLOSE
            # Entry and exit must fall on the same session — intraday means intraday.
            assert trade.entry_at.date() == trade.exit_at.date()

    def test_entries_only_inside_the_entry_window(self, source):
        engine = BacktestEngine(SmaCrossover(short_period=5, long_period=20),
                                make_config())
        result = engine.run(source, INSTRUMENTS, START, END)
        assert result.portfolio.trades
        for trade in result.portfolio.trades:
            assert cal.ENTRY_START <= trade.entry_at.time() < cal.SQUARE_OFF

    def test_square_off_exits_are_recorded_as_such(self, source):
        engine = BacktestEngine(BuyOnceStrategy(trigger_bar=300, stop_pct=0.20),
                                make_config())
        result = engine.run(source, INSTRUMENTS, START, END)
        reasons = {t.exit_reason.value for t in result.portfolio.trades}
        assert "square_off" in reasons


class TestRiskIntegration:
    def test_open_risk_cap_holds_through_a_full_backtest(self, source):
        """The Phase 0 invariant must survive contact with the real engine."""
        engine = BacktestEngine(SmaCrossover(short_period=5, long_period=20),
                                make_config())
        engine.run(source, INSTRUMENTS, START, END)
        assert engine.risk.total_open_risk_pct() <= engine.risk.limits.max_open_risk + 1e-9

    def test_signals_without_stops_are_rejected(self, source):
        class NoStop(Strategy):
            name = "no_stop"

            def on_bar(self, bar, ctx):
                if not ctx.in_position and ctx.allows_entry:
                    return [self._signal(bar, Intent.OPEN_LONG, reason="no stop")]
                return []

        engine = BacktestEngine(NoStop(), make_config())
        result = engine.run(source, INSTRUMENTS, START, END)
        assert result.metrics.trade_count == 0
        assert any("stop_loss" in key for key in result.rejections)

    def test_shorts_rejected_while_long_only(self, source):
        class ShortOnly(Strategy):
            name = "short_only"

            def on_bar(self, bar, ctx):
                if not ctx.in_position and ctx.allows_entry:
                    return [self._signal(bar, Intent.OPEN_SHORT,
                                         stop_loss=round(bar.close * 1.01, 2))]
                return []

        result = BacktestEngine(ShortOnly(), make_config()).run(
            source, INSTRUMENTS, START, END)
        assert result.metrics.trade_count == 0


class TestAccounting:
    def test_net_equals_gross_minus_costs(self, source):
        engine = BacktestEngine(SmaCrossover(short_period=5, long_period=20),
                                make_config())
        result = engine.run(source, INSTRUMENTS, START, END)
        metrics = result.metrics
        assert metrics.net_pnl == pytest.approx(
            metrics.gross_pnl - metrics.total_costs, abs=0.01)

    def test_friction_splits_into_slippage_plus_charges(self, source):
        result = BacktestEngine(SmaCrossover(short_period=5, long_period=20),
                                make_config()).run(source, INSTRUMENTS, START, END)
        metrics = result.metrics
        assert metrics.total_friction == pytest.approx(
            metrics.total_slippage + metrics.total_costs, abs=0.01)
        # Frictionless P&L is gross before slippage was deducted.
        assert metrics.frictionless_pnl == pytest.approx(
            metrics.gross_pnl + metrics.total_slippage, abs=0.01)

    def test_ending_equity_tracks_the_trade_ledger(self, source):
        result = BacktestEngine(SmaCrossover(short_period=5, long_period=20),
                                make_config()).run(source, INSTRUMENTS, START, END)
        expected = result.portfolio.starting_equity + result.metrics.net_pnl
        assert result.portfolio.equity == pytest.approx(expected, abs=0.01)

    def test_zero_slippage_makes_gross_and_frictionless_equal(self, source):
        config = make_config(slippage=SlippageModel(slippage_bps=0.0, half_spread_bps=0.0))
        result = BacktestEngine(SmaCrossover(short_period=5, long_period=20),
                                config).run(source, INSTRUMENTS, START, END)
        assert result.metrics.total_slippage == pytest.approx(0.0, abs=0.01)


class TestNegativeControl:
    """Phase 1 exit criterion (DESIGN.md 2.5 rule 5)."""

    def test_sma_crossover_loses_money_on_a_random_walk(self, source):
        result = BacktestEngine(
            SmaCrossover(short_period=10, long_period=50, stop_pct=0.01),
            make_config(),
        ).run(source, INSTRUMENTS, START, END)

        assert result.metrics.trade_count > 20, "too few trades to conclude anything"
        assert result.metrics.net_pnl < 0, (
            "The negative control turned a profit on a driftless random walk. The "
            "harness is wrong — check the cost model, the fill model and the "
            "look-ahead guard before believing anything else it reports."
        )

    def test_the_signal_itself_has_no_edge(self, source):
        """Frictionless expectancy should sit near zero — a random walk has no pattern.

        This separates the two possible reasons the control loses. It must lose because
        friction exceeds a *zero* edge, not because the engine mis-prices trades.
        """
        result = BacktestEngine(
            SmaCrossover(short_period=10, long_period=50, stop_pct=0.01),
            make_config(),
        ).run(source, INSTRUMENTS, START, END)
        assert abs(result.metrics.frictionless_expectancy_r) < 0.10

    def test_friction_is_the_cause_of_the_loss(self, source):
        result = BacktestEngine(
            SmaCrossover(short_period=10, long_period=50, stop_pct=0.01),
            make_config(),
        ).run(source, INSTRUMENTS, START, END)
        metrics = result.metrics
        # net = frictionless - friction, per trade.
        assert metrics.expectancy_r == pytest.approx(
            metrics.frictionless_expectancy_r - metrics.friction_drag_r, abs=0.005)
        assert metrics.friction_drag_r > 0.03

    def test_removing_all_friction_removes_most_of_the_loss(self, source):
        """Sanity check on the attribution: no costs, no slippage -> roughly break-even."""
        free = make_config(
            cost_model=CostModel(brokerage_flat=0.0, brokerage_pct=0.0, stt_sell=0.0,
                                 exchange_txn=0.0, sebi=0.0, stamp_duty_buy=0.0, gst=0.0),
            slippage=SlippageModel(slippage_bps=0.0, half_spread_bps=0.0),
        )
        result = BacktestEngine(
            SmaCrossover(short_period=10, long_period=50, stop_pct=0.01), free,
        ).run(source, INSTRUMENTS, START, END)
        assert abs(result.metrics.expectancy_r) < 0.10


class TestReporting:
    def test_report_includes_provenance(self, source):
        result = BacktestEngine(SmaCrossover(short_period=5, long_period=20),
                                make_config()).run(source, INSTRUMENTS, START, END)
        text = result.report()
        assert "Provenance" in text
        assert "sma_crossover" in text
        assert "brokerage" in text

    def test_session_direction_breakdown_is_produced(self, source):
        result = BacktestEngine(SmaCrossover(short_period=5, long_period=20),
                                make_config()).run(source, INSTRUMENTS, START, END)
        # D3: a long-only strategy's dependence on market direction must be visible.
        assert not result.metrics.by_session_direction.empty
        assert "session_direction" in result.metrics.by_session_direction.columns

    def test_empty_result_reports_cleanly(self, source):
        class Silent(Strategy):
            name = "silent"

            def on_bar(self, bar, ctx):
                return []

        result = BacktestEngine(Silent(), make_config()).run(
            source, INSTRUMENTS, START, END)
        assert result.metrics.trade_count == 0
        assert "No trades" in result.report()
