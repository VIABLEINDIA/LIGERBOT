"""Walk-forward analysis and the anti-overfitting protocol (DESIGN.md 2.5).

A single backtest over a single period is worthless. Given enough parameter combinations,
a good-looking result is guaranteed by noise alone — which is why this module counts and
reports the trials rather than quietly presenting the winner.

What it enforces:

* **Rolling optimise-then-test.** Parameters are chosen on a training window and evaluated
  on the *next*, unseen window. Only the stitched out-of-sample results are reported.
  In-sample numbers are recorded for comparison but never quoted as evidence.
* **Trial counting.** Every configuration evaluated is counted. Testing 500 and reporting
  the best is not a discovery, and the reported Sharpe must be discounted accordingly.
* **Robustness over optimality.** :func:`parameter_surface` exposes the whole grid so a
  broad plateau can be told apart from a lone spike. A spike is an overfit.
* **A locked holdout.** :func:`split_holdout` carves off the most recent slice, to be
  touched exactly once. Examining it and then changing parameters burns it, and the code
  says so rather than leaving it to discipline.
"""
from __future__ import annotations

import datetime as dt
import itertools
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from src import market_calendar as cal
from src.backtest.bar_source import BarSource
from src.backtest.engine import BacktestConfig, BacktestEngine, BacktestResult
from src.backtest.metrics import Metrics
from src.strategy_base import Strategy

log = logging.getLogger("ligerbot.walk_forward")

StrategyFactory = Callable[..., Strategy]


@dataclass
class Window:
    train_start: dt.date
    train_end: dt.date
    test_start: dt.date
    test_end: dt.date

    def __str__(self) -> str:
        return (f"train {self.train_start}..{self.train_end} -> "
                f"test {self.test_start}..{self.test_end}")


@dataclass
class FoldResult:
    window: Window
    best_params: Dict[str, Any]
    in_sample: Metrics
    out_of_sample: Metrics
    trials: int

    @property
    def degradation(self) -> float:
        """In-sample minus out-of-sample expectancy.

        Large positive degradation is the signature of curve fitting: the parameters
        described the training noise rather than anything that persisted.
        """
        return self.in_sample.expectancy_r - self.out_of_sample.expectancy_r


@dataclass
class WalkForwardResult:
    folds: List[FoldResult] = field(default_factory=list)
    total_trials: int = 0
    objective_name: str = "expectancy_r"

    @property
    def oos_trade_count(self) -> int:
        return sum(f.out_of_sample.trade_count for f in self.folds)

    @property
    def oos_net_pnl(self) -> float:
        return sum(f.out_of_sample.net_pnl for f in self.folds)

    @property
    def oos_expectancy_r(self) -> float:
        """Trade-weighted out-of-sample expectancy across all folds."""
        total = self.oos_trade_count
        if not total:
            return 0.0
        return sum(f.out_of_sample.expectancy_r * f.out_of_sample.trade_count
                   for f in self.folds) / total

    @property
    def mean_degradation(self) -> float:
        return (sum(f.degradation for f in self.folds) / len(self.folds)
                if self.folds else 0.0)

    @property
    def profitable_folds(self) -> int:
        return sum(1 for f in self.folds if f.out_of_sample.net_pnl > 0)

    def parameter_stability(self) -> pd.DataFrame:
        """Chosen parameters per fold.

        Parameters that jump around between folds mean the optimiser is fitting noise —
        a more damning signal than any single fold's numbers.
        """
        if not self.folds:
            return pd.DataFrame()
        rows = []
        for fold in self.folds:
            row = {"test_start": fold.window.test_start, **fold.best_params}
            row["oos_expectancy_r"] = round(fold.out_of_sample.expectancy_r, 4)
            row["oos_trades"] = fold.out_of_sample.trade_count
            rows.append(row)
        return pd.DataFrame(rows)

    def folds_frame(self) -> pd.DataFrame:
        if not self.folds:
            return pd.DataFrame()
        return pd.DataFrame([{
            "test_start": f.window.test_start,
            "test_end": f.window.test_end,
            "is_expectancy_r": round(f.in_sample.expectancy_r, 4),
            "oos_expectancy_r": round(f.out_of_sample.expectancy_r, 4),
            "degradation": round(f.degradation, 4),
            "oos_trades": f.out_of_sample.trade_count,
            "oos_net_pnl": round(f.out_of_sample.net_pnl, 2),
            "oos_win_rate": round(f.out_of_sample.win_rate, 4),
            "trials": f.trials,
        } for f in self.folds])

    def report(self) -> str:
        rule = "=" * 78
        lines = [rule, "Walk-forward analysis (out-of-sample only)", rule]
        if not self.folds:
            return "\n".join(lines + ["No folds completed.", rule])

        lines += [
            f"  Folds                {len(self.folds):>10}   "
            f"({self.profitable_folds} profitable)",
            f"  OOS trades           {self.oos_trade_count:>10,}",
            f"  OOS net P&L          {self.oos_net_pnl:>+10,.2f}",
            f"  OOS expectancy       {self.oos_expectancy_r:>+10.4f}R",
            f"  Mean degradation     {self.mean_degradation:>+10.4f}R   "
            f"(in-sample minus out-of-sample)",
            f"  Configurations tried {self.total_trials:>10,}",
            "",
        ]

        # DESIGN.md 2.5 rule 4: the trial count is not a footnote.
        if self.total_trials > 20:
            lines += [
                f"  ! {self.total_trials} configurations were evaluated. With that many",
                "    trials the best result is substantially selection bias — discount the",
                "    headline figures accordingly and weight the fold-to-fold consistency",
                "    above the aggregate.",
                "",
            ]
        if self.oos_trade_count < 200:
            lines += [
                f"  ! Only {self.oos_trade_count} out-of-sample trades, below the 200-trade",
                "    gate in DESIGN.md 2.6. Not statistically meaningful yet.",
                "",
            ]
        if self.mean_degradation > 0.05:
            lines += [
                f"  ! Mean degradation {self.mean_degradation:+.4f}R suggests the optimiser",
                "    is fitting training noise rather than a persistent effect.",
                "",
            ]

        lines += ["Per fold", self.folds_frame().to_string(index=False), ""]
        stability = self.parameter_stability()
        if not stability.empty:
            lines += ["Parameter stability (jumpy parameters = fitting noise)",
                      stability.to_string(index=False), ""]
        lines.append(rule)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
def split_holdout(
    start: dt.date, end: dt.date, holdout_fraction: float = 0.2
) -> Tuple[Tuple[dt.date, dt.date], Tuple[dt.date, dt.date]]:
    """Split into (development, holdout), holdout being the most recent slice.

    The holdout is touched **once**, at the very end. If it is examined and parameters
    then change, it is burned and a new one must be carved from fresh data — there is no
    way to un-see a result.
    """
    days = cal.trading_days_between(start, end)
    if len(days) < 10:
        raise ValueError(f"Only {len(days)} trading days — too few to split meaningfully.")
    cut = int(len(days) * (1 - holdout_fraction))
    return (days[0], days[cut - 1]), (days[cut], days[-1])


def rolling_windows(
    start: dt.date,
    end: dt.date,
    *,
    train_days: int = 120,
    test_days: int = 40,
    step_days: Optional[int] = None,
) -> List[Window]:
    """Rolling train/test windows measured in **trading** days, not calendar days."""
    days = cal.trading_days_between(start, end)
    step = step_days or test_days
    windows: List[Window] = []
    index = 0
    while index + train_days + test_days <= len(days):
        train = days[index:index + train_days]
        test = days[index + train_days:index + train_days + test_days]
        windows.append(Window(train[0], train[-1], test[0], test[-1]))
        index += step
    return windows


def parameter_grid(grid: Dict[str, Sequence[Any]]) -> List[Dict[str, Any]]:
    """Every combination in the grid."""
    if not grid:
        return [{}]
    keys = list(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*(grid[k] for k in keys))]


def _run_one(
    factory: StrategyFactory,
    params: Dict[str, Any],
    config: BacktestConfig,
    source: BarSource,
    instrument_ids: Sequence[str],
    start: dt.date,
    end: dt.date,
) -> Optional[BacktestResult]:
    try:
        engine = BacktestEngine(factory(**params), config)
        return engine.run(source, instrument_ids, start, end)
    except Exception as exc:  # a bad parameter combination must not abort the sweep
        log.warning("Backtest failed for %s: %s", params, exc)
        return None


def parameter_surface(
    factory: StrategyFactory,
    grid: Dict[str, Sequence[Any]],
    config: BacktestConfig,
    source: BarSource,
    instrument_ids: Sequence[str],
    start: dt.date,
    end: dt.date,
    *,
    objective: str = "expectancy_r",
) -> pd.DataFrame:
    """Evaluate the whole grid and return every result.

    Publish this, not just the maximum. A broad plateau of decent performance is a
    finding; a single sharp spike surrounded by poor results is an artefact.
    """
    rows = []
    for params in parameter_grid(grid):
        result = _run_one(factory, params, config, source, instrument_ids, start, end)
        if result is None:
            continue
        metrics = result.metrics
        rows.append({
            **params,
            "objective": getattr(metrics, objective, 0.0),
            "expectancy_r": round(metrics.expectancy_r, 4),
            "net_pnl": round(metrics.net_pnl, 2),
            "trades": metrics.trade_count,
            "win_rate": round(metrics.win_rate, 4),
            "profit_factor": round(metrics.profit_factor, 3),
            "max_dd_pct": round(metrics.max_drawdown_pct, 4),
        })
    return pd.DataFrame(rows)


def run_walk_forward(
    factory: StrategyFactory,
    grid: Dict[str, Sequence[Any]],
    config: BacktestConfig,
    source: BarSource,
    instrument_ids: Sequence[str],
    start: dt.date,
    end: dt.date,
    *,
    train_days: int = 120,
    test_days: int = 40,
    objective: str = "expectancy_r",
    min_trades_in_sample: int = 20,
) -> WalkForwardResult:
    """Optimise on each training window, evaluate on the next unseen window."""
    windows = rolling_windows(start, end, train_days=train_days, test_days=test_days)
    if not windows:
        raise ValueError(
            f"No walk-forward windows fit in {start}..{end} with train={train_days} "
            f"test={test_days} trading days. Widen the range or shrink the windows."
        )

    combinations = parameter_grid(grid)
    outcome = WalkForwardResult(objective_name=objective)
    log.info("Walk-forward: %d folds x %d configurations", len(windows), len(combinations))

    for window in windows:
        best_params, best_metrics, best_score, trials = None, None, float("-inf"), 0

        for params in combinations:
            trials += 1
            result = _run_one(factory, params, config, source, instrument_ids,
                              window.train_start, window.train_end)
            if result is None:
                continue
            # A configuration that barely traded in-sample tells us nothing about
            # whether it works; picking it would be selecting on sample size.
            if result.metrics.trade_count < min_trades_in_sample:
                continue
            score = getattr(result.metrics, objective, float("-inf"))
            if score > best_score:
                best_params, best_metrics, best_score = params, result.metrics, score

        outcome.total_trials += trials
        if best_params is None:
            log.warning("%s: no configuration met the in-sample minimum — fold skipped.",
                        window)
            continue

        test = _run_one(factory, best_params, config, source, instrument_ids,
                        window.test_start, window.test_end)
        if test is None:
            continue

        outcome.folds.append(FoldResult(
            window=window, best_params=best_params,
            in_sample=best_metrics, out_of_sample=test.metrics, trials=trials,
        ))
        log.info("%s | best=%s | IS %.4fR -> OOS %.4fR (%d trades)",
                 window, best_params, best_metrics.expectancy_r,
                 test.metrics.expectancy_r, test.metrics.trade_count)

    return outcome
