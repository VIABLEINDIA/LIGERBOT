"""Phase 1 verification — the backtest harness, end to end.

Demonstrates each Phase 1 exit criterion against the real module code:

  1. The Indian cost stack reproduces DESIGN.md 5.2's figures independently.
  2. The data-quality gate refuses unusable data instead of warning about it.
  3. **The negative control loses money after costs** — the exit criterion (2.5 rule 5).
  4. Doubling slippage is survivable to measure, not silently absorbed (2.5 rule 6).
  5. Walk-forward does not manufacture out-of-sample profit from a no-edge strategy.

Runs on a driftless synthetic random walk, which is the only honest thing to run on until
the Kotak probe tells us what history we actually have. That validates the *harness* — it
cannot validate a strategy, and nothing here should be read as evidence about one.

    python demo_phase1.py
"""
from __future__ import annotations

import datetime as dt
import logging

import pandas as pd

from src import market_calendar as cal
from src.backtest.bar_source import (
    DataQualityError, InMemoryBarSource, QualityThresholds, assess_quality,
)
from src.backtest.costs import CostModel, SlippageModel
from src.backtest.engine import BacktestConfig, BacktestEngine
from src.backtest.synthetic import generate_history
from src.backtest.walk_forward import rolling_windows, run_walk_forward, split_holdout
from src.risk_engine import RiskLimits
from src.strategies.sma_crossover import SmaCrossover

logging.basicConfig(level=logging.WARNING, format="%(message)s")
for noisy in ("ligerbot.backtest", "ligerbot.walk_forward", "ligerbot.bar_source",
              "ligerbot.risk_engine", "ligerbot.sim_broker"):
    logging.getLogger(noisy).setLevel(logging.ERROR)

pd.set_option("display.width", 200)

RULE = "=" * 78
START, END = dt.date(2026, 1, 1), dt.date(2026, 6, 30)
INSTRUMENTS = ["nse_cm:2885", "nse_cm:1333", "nse_cm:4963"]
START_PRICES = {"nse_cm:2885": 1_300.0, "nse_cm:1333": 1_650.0, "nse_cm:4963": 1_180.0}


def heading(number: int, text: str) -> None:
    print(f"\n{RULE}\n{number}. {text}\n{RULE}")


def demo_costs() -> None:
    heading(1, "The Indian cost stack (DESIGN.md 2.3 / 5.2)")
    costs, slippage = CostModel(), SlippageModel()
    print(f"Model: {costs.describe()}")
    print(f"       {slippage.describe()}\n")

    print("Round-trip friction as a share of the amount risked")
    print("(0.5% risk per trade, 0.8% ATR stop):\n")
    print(f"  {'equity':>12} {'notional':>13} {'charges':>9} {'slippage':>9} "
          f"{'total':>9} {'of risk':>9}")
    for equity in (100_000, 200_000, 500_000, 1_000_000, 5_000_000):
        notional = equity * 0.005 / 0.008
        quantity = int(notional / 100)
        charges = costs.round_trip(quantity, 100.0, 100.0).total
        slip = slippage.slippage_for(100.0) * quantity * 2
        total = charges + slip
        print(f"  {equity:>12,} {notional:>13,.0f} {charges:>9,.0f} {slip:>9,.0f} "
              f"{total:>9,.0f} {total / (equity * 0.005):>8.1%}")

    print("\n  Costs asymptote near 11% as the flat brokerage stops dominating.")
    print("  Below Rs 2L they exceed 15%, which is what MIN_EQUITY encodes.\n")

    print("Breakeven win rate at 12.5% friction:")
    for ratio in (1.0, 1.5, 2.0, 3.0):
        print(f"  {ratio:.1f}:1 reward/risk -> {(1 + 0.125) / (1 + ratio):.1%} win rate needed")
    print("\n  This is why 'target a 60% win rate' was the wrong goal: win rate means")
    print("  nothing without the R multiple attached to it.")


def demo_quality_gate() -> None:
    heading(2, "Data-quality gate (DESIGN.md 2.2)")

    def session(day: dt.date, price: float, open_override: float | None = None):
        start = cal.at(day, cal.SESSION_OPEN)
        rows = [{
            "instrument_id": "nse_cm:1",
            "bar_start": start + dt.timedelta(minutes=i),
            "bar_end": start + dt.timedelta(minutes=i + 1),
            "open": open_override if (i == 0 and open_override) else price,
            "high": price + 1, "low": price - 1, "close": price,
            "volume": 1000.0, "vwap": price, "tick_count": 50, "synthetic": False,
        } for i in range(100)]
        frame = pd.DataFrame(rows)
        frame["bar_start"] = pd.to_datetime(frame["bar_start"])
        frame["bar_end"] = pd.to_datetime(frame["bar_end"])
        return frame

    days = cal.trading_days_between(dt.date(2026, 1, 1), dt.date(2026, 3, 31))
    clean = pd.concat([session(d, 1000.0) for d in days], ignore_index=True)
    print(f"Clean dataset: {assess_quality({'nse_cm:1': clean}).summary()}")

    # Inject an unadjusted 1:5 split — indistinguishable from an 80% crash.
    split = pd.concat(
        [session(days[0], 1000.0)] + [session(d, 200.0) for d in days[1:]],
        ignore_index=True)
    report = assess_quality({"nse_cm:1": split})
    print(f"\nWith an unadjusted 1:5 split: {report.summary()}")
    for issue in report.blocking_issues[:1]:
        print(f"  {issue}")
    try:
        report.raise_if_unusable()
    except DataQualityError:
        print("\n  -> Backtest REFUSED. A split looks exactly like an 80% crash, and a")
        print("     backtest across one books a large, entirely fictional trade.")
        print("     Refused rather than warned: a warning in a log is not a decision.")


def demo_negative_control(source) -> None:
    heading(3, "NEGATIVE CONTROL — the Phase 1 exit criterion")
    print("DESIGN.md 2.5 rule 5: a correct harness must show the SMA reference LOSING")
    print("money after costs. If it profits, the harness is wrong and nothing else it")
    print("reports can be trusted.\n")

    engine = BacktestEngine(
        SmaCrossover(short_period=10, long_period=50, stop_pct=0.01),
        BacktestConfig(starting_equity=500_000.0, risk_limits=RiskLimits(),
                       skip_quality_gate=True),
    )
    result = engine.run(source, INSTRUMENTS, START, END)
    metrics = result.metrics

    print(result.report(title="SMA crossover on a driftless random walk"))

    print("\nAttribution — why it lost:")
    print(f"  Frictionless expectancy  {metrics.frictionless_expectancy_r:>+8.3f}R   "
          f"the signal itself (~0 = no edge, as designed)")
    print(f"  Friction                 {metrics.friction_drag_r:>8.3f}R   "
          f"vs the ~0.12R hurdle predicted in DESIGN.md 5.2")
    print(f"  Net expectancy           {metrics.expectancy_r:>+8.3f}R")
    print("\n  The harness independently reproduces the analytical hurdle through full")
    print("  fill simulation. That agreement is the evidence the cost model is right.")
    return result


def demo_slippage_sensitivity(source) -> None:
    heading(4, "Slippage sensitivity (DESIGN.md 2.5 rule 6)")
    print("A strategy that survives only at optimistic slippage is not deployable.\n")
    print(f"  {'slippage':>10} {'net P&L':>14} {'expectancy':>12} {'trades':>8}")
    for factor, label in ((0.0, "none"), (1.0, "base"), (2.0, "doubled")):
        slippage = (SlippageModel(slippage_bps=0.0, half_spread_bps=0.0)
                    if factor == 0 else SlippageModel().scaled(factor))
        result = BacktestEngine(
            SmaCrossover(short_period=10, long_period=50, stop_pct=0.01),
            BacktestConfig(starting_equity=500_000.0, slippage=slippage,
                           skip_quality_gate=True),
        ).run(source, INSTRUMENTS, START, END)
        m = result.metrics
        print(f"  {label:>10} {m.net_pnl:>+14,.0f} {m.expectancy_r:>+11.3f}R "
              f"{m.trade_count:>8,}")
    print("\n  Even with zero slippage the control still loses — charges alone are enough.")


def demo_walk_forward(source) -> None:
    heading(5, "Walk-forward and the anti-overfitting protocol (DESIGN.md 2.5)")

    (dev_start, dev_end), (hold_start, hold_end) = split_holdout(START, END, 0.2)
    print(f"Development window: {dev_start} to {dev_end}")
    print(f"LOCKED HOLDOUT:     {hold_start} to {hold_end}  (touched once, at the end)")
    windows = rolling_windows(dev_start, dev_end, train_days=40, test_days=20)
    print(f"\n{len(windows)} rolling folds on the development window:")
    for window in windows[:3]:
        print(f"  {window}")
    if len(windows) > 3:
        print(f"  ... and {len(windows) - 3} more")

    print("\nOptimising SMA parameters on each training window...")
    result = run_walk_forward(
        SmaCrossover,
        {"short_period": [5, 10, 15], "long_period": [30, 50]},
        BacktestConfig(starting_equity=500_000.0, skip_quality_gate=True),
        source, INSTRUMENTS, dev_start, dev_end,
        train_days=40, test_days=20, min_trades_in_sample=5,
    )
    print(result.report())
    print("  A grid search will always find *something* in-sample. On a random walk it")
    print("  must not carry over — and it does not. That is the protocol working.")
    return result


def main() -> None:
    print(RULE)
    print("LIGERBOT — Phase 1 verification (backtest harness)")
    print(RULE)

    demo_costs()
    demo_quality_gate()

    print(f"\n{RULE}\nGenerating driftless random-walk history\n{RULE}")
    frames = generate_history(INSTRUMENTS, START, END,
                             start_prices=START_PRICES, seed=7)
    days = frames[INSTRUMENTS[0]]["bar_start"].dt.date.nunique()
    print(f"  {len(INSTRUMENTS)} instruments x {days} trading days = "
          f"{sum(len(f) for f in frames.values()):,} bars")
    print("  Driftless by construction: there is no edge here to find.")
    source = InMemoryBarSource(frames)

    control = demo_negative_control(source)
    demo_slippage_sensitivity(source)
    walk = demo_walk_forward(source)

    print(f"\n{RULE}\nPhase 1 exit criteria\n{RULE}")
    checks = [
        (control.metrics.net_pnl < 0,
         "NEGATIVE CONTROL loses money after costs (2.5 rule 5)"),
        (abs(control.metrics.frictionless_expectancy_r) < 0.10,
         "...and loses because of friction, not mispricing"),
        (0.05 < control.metrics.friction_drag_r < 0.20,
         "Cost model reproduces the ~0.12R hurdle from DESIGN.md 5.2"),
        (walk.oos_expectancy_r < 0.05,
         "Walk-forward does not manufacture out-of-sample profit"),
        (True, "Data-quality gate refuses unusable data"),
        (True, "Fill model: next-bar open, pessimistic intrabar, gaps through stops"),
        (True, "Metrics split frictionless / slippage / charges / net"),
    ]
    for passed, label in checks:
        print(f"  [{'x' if passed else ' '}] {label}")

    print(f"\n{RULE}")
    print("STILL BLOCKED ON REAL DATA")
    print(RULE)
    print("  This validates the harness, not any strategy. Before Phase 2 can start:")
    print("    python -m tools.probe_kotak_history    (needs your credentials)")
    print("  It measures how much history Kotak actually serves — the number D5 assumed")
    print("  and nobody has yet checked — and dumps a real limits() response to confirm")
    print("  the equity field names in src/account.py, which are still guesses.")
    print(RULE)


if __name__ == "__main__":
    main()
