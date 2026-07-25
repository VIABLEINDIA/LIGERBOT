"""Walk-forward and anti-overfitting protocol tests."""
from __future__ import annotations

import datetime as dt

import pytest

from src import market_calendar as cal
from src.backtest.bar_source import InMemoryBarSource
from src.backtest.engine import BacktestConfig
from src.backtest.synthetic import generate_history
from src.backtest.walk_forward import (
    parameter_grid, parameter_surface, rolling_windows, run_walk_forward, split_holdout,
)
from src.strategies.sma_crossover import SmaCrossover

START, END = dt.date(2026, 1, 1), dt.date(2026, 6, 30)
INSTRUMENTS = ["nse_cm:2885"]


@pytest.fixture(scope="module")
def source():
    return InMemoryBarSource(generate_history(
        INSTRUMENTS, START, END, start_prices={"nse_cm:2885": 1_300.0}, seed=5))


def config() -> BacktestConfig:
    return BacktestConfig(starting_equity=1_000_000.0, skip_quality_gate=True)


class TestHoldout:
    def test_holdout_is_the_most_recent_slice(self):
        (dev_start, dev_end), (hold_start, hold_end) = split_holdout(START, END, 0.2)
        assert dev_start < dev_end < hold_start < hold_end
        assert hold_end == max(cal.trading_days_between(START, END))

    def test_holdout_fraction_is_respected(self):
        total = len(cal.trading_days_between(START, END))
        _, (hold_start, hold_end) = split_holdout(START, END, 0.2)
        held = len(cal.trading_days_between(hold_start, hold_end))
        assert held == pytest.approx(total * 0.2, abs=2)

    def test_windows_do_not_overlap(self):
        (_, dev_end), (hold_start, _) = split_holdout(START, END, 0.2)
        assert dev_end < hold_start

    def test_too_little_data_raises(self):
        with pytest.raises(ValueError, match="too few"):
            split_holdout(dt.date(2026, 3, 2), dt.date(2026, 3, 6))


class TestRollingWindows:
    def test_train_precedes_test_and_they_do_not_overlap(self):
        windows = rolling_windows(START, END, train_days=40, test_days=20)
        assert windows
        for window in windows:
            assert window.train_start < window.train_end < window.test_start
            assert window.test_start <= window.test_end

    def test_windows_step_forward(self):
        windows = rolling_windows(START, END, train_days=40, test_days=20)
        for earlier, later in zip(windows, windows[1:]):
            assert later.train_start > earlier.train_start
            assert later.test_start > earlier.test_start

    def test_no_windows_when_the_range_is_too_short(self):
        assert rolling_windows(START, END, train_days=500, test_days=100) == []

    def test_windows_use_trading_days_not_calendar_days(self):
        windows = rolling_windows(START, END, train_days=20, test_days=10)
        first = windows[0]
        assert len(cal.trading_days_between(first.train_start, first.train_end)) == 20


class TestParameterGrid:
    def test_expands_every_combination(self):
        combos = parameter_grid({"a": [1, 2], "b": [3, 4, 5]})
        assert len(combos) == 6
        assert {"a": 1, "b": 3} in combos

    def test_empty_grid_yields_one_empty_config(self):
        assert parameter_grid({}) == [{}]


class TestParameterSurface:
    def test_returns_a_row_per_configuration(self, source):
        surface = parameter_surface(
            SmaCrossover, {"short_period": [5, 10], "long_period": [30]},
            config(), source, INSTRUMENTS, START, dt.date(2026, 2, 28),
        )
        assert len(surface) == 2
        assert {"short_period", "long_period", "expectancy_r", "trades"} <= set(surface.columns)

    def test_invalid_combinations_are_skipped_not_fatal(self, source):
        # short >= long raises in the strategy constructor; the sweep must survive it.
        surface = parameter_surface(
            SmaCrossover, {"short_period": [10, 50], "long_period": [30]},
            config(), source, INSTRUMENTS, START, dt.date(2026, 2, 28),
        )
        assert len(surface) == 1


class TestWalkForward:
    @pytest.fixture(scope="class")
    def result(self, source):
        return run_walk_forward(
            SmaCrossover,
            {"short_period": [5, 10], "long_period": [30, 50]},
            config(), source, INSTRUMENTS, START, END,
            train_days=40, test_days=20, min_trades_in_sample=3,
        )

    def test_produces_folds(self, result):
        assert result.folds
        for fold in result.folds:
            assert fold.best_params
            assert fold.window.train_end < fold.window.test_start

    def test_counts_every_trial(self, result):
        """DESIGN.md 2.5 rule 4 — the trial count is not a footnote."""
        assert result.total_trials == sum(f.trials for f in result.folds)
        assert result.total_trials >= len(result.folds)

    def test_reports_degradation(self, result):
        # In-sample minus out-of-sample: the signature of curve fitting.
        for fold in result.folds:
            assert fold.degradation == pytest.approx(
                fold.in_sample.expectancy_r - fold.out_of_sample.expectancy_r)

    def test_parameter_stability_table(self, result):
        stability = result.parameter_stability()
        assert not stability.empty
        assert "short_period" in stability.columns

    def test_report_warns_about_thin_samples(self, result):
        text = result.report()
        assert "Walk-forward" in text
        # The synthetic run is far under the 200-trade gate; the report must say so.
        if result.oos_trade_count < 200:
            assert "200-trade" in text

    def test_negative_control_does_not_survive_walk_forward(self, result):
        """Optimising a no-edge strategy must not manufacture out-of-sample profit.

        In-sample, a grid search will always find *something*. The point of walk-forward
        is that it does not carry over — and on a random walk it must not.
        """
        assert result.oos_expectancy_r < 0.05

    def test_raises_when_no_window_fits(self, source):
        with pytest.raises(ValueError, match="No walk-forward windows"):
            run_walk_forward(
                SmaCrossover, {"short_period": [5]}, config(), source, INSTRUMENTS,
                START, END, train_days=1000, test_days=100,
            )
