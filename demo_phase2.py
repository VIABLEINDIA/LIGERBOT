"""Phase 2 verification — strategy v1 and the validation gates.

Shows:

  1. The indicator library, on a constructed trend.
  2. Trend-pullback v1 taking a textbook setup (unit-level positive control).
  3. The bar-interval sweep required by D4.
  4. Negative and positive controls side by side.
  5. The §2.6 gates run to a verdict.

**This cannot validate the strategy.** It runs on synthetic data, which can show that the
machinery works and that the strategy is not blind, but says nothing about whether an edge
exists in the real market. The gates are expected to BLOCK here, and that is the correct
outcome — a strategy that "passed" on a random walk would mean the gates were broken.

    python demo_phase2.py
"""
from __future__ import annotations

import datetime as dt
import logging

import pandas as pd

from src import market_calendar as cal
from src.backtest.bar_source import InMemoryBarSource, ResampledBarSource
from src.backtest.costs import SlippageModel
from src.backtest.engine import BacktestConfig, BacktestEngine
from src.backtest.gates import evaluate
from src.backtest.synthetic import generate_history
from src.backtest.walk_forward import run_walk_forward
from src.bars import Bar
from src.indicators import ADX, ATR, EMA, RSI, SessionVWAP
from src.strategies.trend_pullback import TrendPullback

logging.basicConfig(level=logging.CRITICAL)
pd.set_option("display.width", 200)

RULE = "=" * 78
START, END = dt.date(2026, 1, 1), dt.date(2026, 6, 30)
INSTRUMENTS = ["nse_cm:2885", "nse_cm:1333", "nse_cm:4963"]
PRICES = {"nse_cm:2885": 1_300.0, "nse_cm:1333": 1_650.0, "nse_cm:4963": 1_180.0}


def heading(number: int, text: str) -> None:
    print(f"\n{RULE}\n{number}. {text}\n{RULE}")


def _bar(i: int, close: float, high=None, low=None, volume=10_000.0) -> Bar:
    start = cal.at(dt.date(2026, 7, 23), cal.SESSION_OPEN) + dt.timedelta(minutes=5 * i)
    return Bar("nse_cm:1", start, start + dt.timedelta(minutes=5),
               open=close, high=high if high is not None else close + 1,
               low=low if low is not None else close - 1, close=close,
               volume=volume, vwap=close, tick_count=200)


def demo_indicators() -> None:
    heading(1, "Indicator library (DESIGN.md 1.5)")
    ema_fast, ema_slow = EMA(5), EMA(10)
    atr, adx, rsi, vwap = ATR(5), ADX(5), RSI(5), SessionVWAP()

    print("Feeding a clean uptrend, then chop:\n")
    print(f"  {'bar':>4} {'close':>8} {'ema5':>8} {'ema10':>8} {'atr':>7} "
          f"{'adx':>7} {'rsi':>6} {'vwap':>8}")
    prices = [100.0 + i * 1.5 for i in range(30)] + \
             [145.0 + (2.0 if i % 2 else -2.0) for i in range(30)]
    for i, price in enumerate(prices):
        bar = _bar(i, price)
        for ind in (ema_fast, ema_slow, atr, adx, rsi, vwap):
            ind.update(bar)
        if i in (9, 19, 29, 39, 49, 59):
            fmt = lambda v: f"{v:8.2f}" if v is not None else "     n/a"
            print(f"  {i:>4} {price:>8.2f} {fmt(ema_fast.value)} {fmt(ema_slow.value)} "
                  f"{atr.value or 0:>7.2f} {adx.value or 0:>7.1f} "
                  f"{rsi.value or 0:>6.1f} {fmt(vwap.value)}")

    print("\n  ADX rises through the trend and falls in the chop — that transition is")
    print("  exactly what the regime filter keys on to stay out of rangebound tape.")


def demo_setup_detection() -> None:
    heading(2, "Trend-pullback on a constructed setup (unit-level positive control)")
    strat = TrendPullback(ema_fast=5, ema_slow=10, adx_period=5, adx_min=15.0,
                          atr_period=5, atr_mult=3.0, min_stop_pct=0.004,
                          rvol_min=0.0, rsi_max=100.0)
    strat.on_session_start(dt.date(2026, 7, 23))

    from src.strategy_base import StrategyContext
    context = StrategyContext(now=cal.at(dt.date(2026, 7, 23), dt.time(11, 0)),
                              seconds_to_square_off=9_000.0, allows_entry=True)

    phases = [
        ("uptrend (builds bias, ADX, ATR)", [100.0 + i * 1.5 for i in range(40)]),
        ("pullback into the fast EMA", [155.5, 152.5, 150.5]),
        ("close back above it", [156.5, 159.5]),
    ]
    index, captured = 0, None
    for label, prices in phases:
        emitted = []
        for price in prices:
            emitted.extend(strat.on_bar(_bar(index, price), context))
            index += 1
        print(f"  {label:<40} -> {len(emitted)} signal(s)")
        if emitted:
            captured = emitted[0]

    if captured:
        print(f"\n  ENTRY: {captured.intent.value} @ {captured.ref_price:.2f} "
              f"stop {captured.stop_loss:.2f} "
              f"({(captured.ref_price - captured.stop_loss) / captured.ref_price:.2%})")
        print(f"  Reason: {captured.reason}")
        print("\n  This matters: it separates 'the strategy is blind' from 'the data has")
        print("  nothing to find'. Aggregate backtests alone cannot tell those apart.")


def demo_interval_sweep(frames) -> None:
    heading(3, "Bar-interval sweep (D4) — store 1m, trade ?")
    print("Resampling the same 1-minute history to each candidate interval.\n")
    print(f"  {'interval':>9} {'trades':>7} {'stop%':>8} {'frictionless':>13} "
          f"{'friction':>10} {'net exp':>10}")

    base = InMemoryBarSource(frames)
    for seconds in (60, 180, 300, 900):
        source = base if seconds == 60 else ResampledBarSource(base, seconds)
        result = BacktestEngine(
            TrendPullback(),
            BacktestConfig(starting_equity=500_000.0, skip_quality_gate=True),
        ).run(source, INSTRUMENTS, START, END)
        m = result.metrics
        if m.trade_count:
            stops = [t.risk_per_share / t.entry_price for t in result.portfolio.trades]
            avg_stop = sum(stops) / len(stops)
        else:
            avg_stop = 0.0
        print(f"  {seconds // 60:>8}m {m.trade_count:>7,} {avg_stop:>7.2%} "
              f"{m.frictionless_expectancy_r:>+12.4f}R {m.friction_drag_r:>9.4f}R "
              f"{m.expectancy_r:>+9.4f}R")

    print("\n  friction_in_R ~= 0.00085 / stop_pct — the relationship that decides this.")
    print("  1-minute ATR stops are structurally uneconomic: they are too tight, so the")
    print("  risk-derived position is too large, so friction swamps any edge. This is")
    print("  why D4's 'trade 5-minute' default exists, and why the strategy refuses")
    print("  trades whose stop falls below the economic floor.")


def demo_controls() -> None:
    heading(4, "Negative vs positive control")
    print("Same strategy, same costs. Only the data-generating process differs.\n")
    print(f"  {'dataset':<34} {'trades':>7} {'frictionless':>13} {'net exp':>10}")

    for momentum, label in ((0.0, "random walk (no edge exists)"),
                            (0.3, "momentum 0.3 (trends persist)"),
                            (0.6, "momentum 0.6 (far beyond real)")):
        frames = generate_history(INSTRUMENTS, START, END, start_prices=PRICES,
                                  momentum=momentum, seed=7)
        source = ResampledBarSource(InMemoryBarSource(frames), 900)
        result = BacktestEngine(
            TrendPullback(),
            BacktestConfig(starting_equity=500_000.0, skip_quality_gate=True),
        ).run(source, INSTRUMENTS, START, END)
        m = result.metrics
        print(f"  {label:<34} {m.trade_count:>7,} "
              f"{m.frictionless_expectancy_r:>+12.4f}R {m.expectancy_r:>+9.4f}R")

    print("\n  Frictionless edge rises with injected momentum, so the strategy does")
    print("  respond to genuine trend persistence. But it responds WEAKLY, and the")
    print("  effect is small relative to friction. Two explanations remain open:")
    print("    (a) the AR(1) generator does not produce the trend-then-pullback")
    print("        structure this strategy targets, or")
    print("    (b) the strategy's edge detection is genuinely weak.")
    print("  Synthetic data cannot distinguish them. Real data can. That is the")
    print("  honest state of things, and tuning parameters here until the numbers")
    print("  improved would be fitting noise — the exact failure DESIGN.md 2.5 exists")
    print("  to prevent.")


def demo_gates(frames):
    heading(5, "Go-live gates (DESIGN.md 2.6)")
    source = ResampledBarSource(InMemoryBarSource(frames), 900)
    config = BacktestConfig(starting_equity=500_000.0, skip_quality_gate=True)

    baseline = BacktestEngine(TrendPullback(), config).run(
        source, INSTRUMENTS, START, END)

    doubled = BacktestEngine(
        TrendPullback(),
        BacktestConfig(starting_equity=500_000.0, skip_quality_gate=True,
                       slippage=SlippageModel().scaled(2.0)),
    ).run(source, INSTRUMENTS, START, END)

    print("Running walk-forward (this takes a moment)...")
    walk = run_walk_forward(
        TrendPullback,
        {"adx_min": [15.0, 25.0], "atr_mult": [2.5, 3.5]},
        config, source, INSTRUMENTS, START, END,
        train_days=60, test_days=25, min_trades_in_sample=5,
    )

    report = evaluate(
        baseline.metrics, walk_forward=walk, doubled_slippage=doubled.metrics,
        context="trend_pullback v1 on SYNTHETIC data — harness check, not evidence",
    )
    print()
    print(report.summary())
    return report


def main() -> None:
    print(RULE)
    print("LIGERBOT — Phase 2 verification (strategy + validation gates)")
    print(RULE)

    demo_indicators()
    demo_setup_detection()

    print(f"\n{RULE}\nGenerating history\n{RULE}")
    frames = generate_history(INSTRUMENTS, START, END, start_prices=PRICES, seed=7)
    print(f"  {len(INSTRUMENTS)} instruments, "
          f"{frames[INSTRUMENTS[0]]['bar_start'].dt.date.nunique()} trading days, "
          f"{sum(len(f) for f in frames.values()):,} 1-minute bars")

    demo_interval_sweep(frames)
    demo_controls()
    report = demo_gates(frames)

    print(f"\n{RULE}\nPhase 2 status\n{RULE}")
    print("  DELIVERED")
    print("    [x] Incremental indicator library (EMA/VWAP/ATR/ADX/RSI/OR/RVol)")
    print("    [x] Trend-pullback v1, long-only, ATR-risked (DESIGN.md 1.6, D3)")
    print("    [x] Bar-interval sweep machinery (D4)")
    print("    [x] Automated §2.6 gate evaluation")
    print("    [x] Unit-level positive control: the strategy takes a real setup")
    print()
    print("  NOT DELIVERED — and cannot be, without real data")
    print(f"    [ ] Strategy validated. Gates currently: "
          f"{'BLOCKED' if not report.passed else 'passed'} on synthetic data,")
    print("        which is the CORRECT outcome. A strategy that passed on a random")
    print("        walk would mean the gates were broken, not that the strategy works.")
    print()
    print(f"{RULE}")
    print("  Phase 2's exit criterion is the gates passing OUT-OF-SAMPLE on real")
    print("  market data. That remains blocked on:")
    print("      python -m tools.probe_kotak_history")
    print(RULE)


if __name__ == "__main__":
    main()
