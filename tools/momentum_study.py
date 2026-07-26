"""Does momentum ranking predict forward returns on NSE?

A **cross-sectional** study, and it exists to decompose a failure. The intraday backtest
(§5.12.1) attributed the loss to the entry — but "entry" bundles two separable decisions:

1. **Selection** — which stocks are worth watching at all
2. **Timing** — when to enter one of them

If selection has predictive power and timing does not, the fix is a different intraday
trigger on the same watchlist. If selection has none either, the momentum premise itself is
wrong on this universe and no trigger will rescue it. Those are very different projects, and
until now nothing distinguished them.

## Method

At each rebalance date, rank the universe by momentum using **only bars strictly before that
date**, then measure realised forward return over the following `horizon` sessions. Repeat
across the sample and compare the top quintile against the bottom and against the universe
mean.

Three things are load-bearing:

**No look-ahead.** The ranking window ends before the forward window begins, with a
one-session gap so the ranking date's own close is never both an input and part of the
measured return.

**The universe is momentum-neutral.** Ranking within a universe pre-filtered on momentum
would answer "do stocks that rose keep rising, given that they rose", which is circular.
The universe is liquidity-filtered only.

**Excess, not absolute.** The top quintile's raw return mostly measures the market. What
matters is the *spread* over the universe mean — that is the part attributable to ranking
rather than to being long in a rising market (D3's concern, applied cross-sectionally).

A positive spread is not a strategy. It is evidence that the selection layer carries signal,
which is the precondition for a timing layer being worth building.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from src.momentum_screen import MomentumCriteria, score


def load_daily(path: Path) -> Dict[str, List[Tuple[dt.date, float]]]:
    """symbol -> [(date, close)], ascending."""
    series: Dict[str, List[Tuple[dt.date, float]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                # TradingView stamps daily bars at the session; only the date matters.
                day = dt.datetime.fromtimestamp(int(row["time"])).date()
                close = float(row["close"])
            except (KeyError, TypeError, ValueError):
                continue
            if close > 0:
                series[row["symbol"]].append((day, close))
    for symbol in series:
        series[symbol].sort(key=lambda pair: pair[0])
    return dict(series)


def study(series: Dict[str, List[Tuple[dt.date, float]]], *, lookback: int = 60,
          horizon: int = 10, gap: int = 1, quantiles: int = 5,
          step: int = 5) -> dict:
    """Roll through the sample ranking then measuring. Returns aggregate statistics."""
    calendar = sorted({day for rows in series.values() for day, _ in rows})
    closes = {sym: {day: px for day, px in rows} for sym, rows in series.items()}

    first = lookback + gap
    last = len(calendar) - horizon
    rebalances = range(first, last, step)

    top_ex: List[float] = []
    bottom_ex: List[float] = []
    spreads: List[float] = []
    hit = 0
    used = 0

    for i in rebalances:
        rank_end = calendar[i - gap]
        entry_day = calendar[i]
        exit_day = calendar[i + horizon]

        scored: List[Tuple[float, str]] = []
        for symbol, rows in series.items():
            history = [px for day, px in rows if day <= rank_end]
            if len(history) < lookback:
                continue
            s = score(symbol, history[-lookback:],
                      criteria=MomentumCriteria(skip_bars=0, lookback_bars=lookback))
            if s.usable:
                scored.append((s.risk_adjusted, symbol))

        forward = {}
        for _, symbol in scored:
            a = closes[symbol].get(entry_day)
            b = closes[symbol].get(exit_day)
            if a and b and a > 0:
                forward[symbol] = (b - a) / a
        eligible = [(k, sym) for k, sym in scored if sym in forward]
        if len(eligible) < quantiles * 4:
            continue

        eligible.sort(reverse=True)
        size = max(1, len(eligible) // quantiles)
        top = [forward[sym] for _, sym in eligible[:size]]
        bottom = [forward[sym] for _, sym in eligible[-size:]]
        universe = list(forward.values())
        mean = sum(universe) / len(universe)

        t_ex = sum(top) / len(top) - mean
        b_ex = sum(bottom) / len(bottom) - mean
        top_ex.append(t_ex)
        bottom_ex.append(b_ex)
        spreads.append(t_ex - b_ex)
        hit += 1 if t_ex > 0 else 0
        used += 1

    if not used:
        return {"rebalances": 0}

    sd = statistics.stdev(spreads) if len(spreads) > 1 else 0.0
    mean_spread = sum(spreads) / used
    t_stat = (mean_spread / (sd / math.sqrt(used))) if sd > 0 else 0.0
    return {
        "rebalances": used,
        "lookback": lookback, "horizon": horizon,
        "top_excess": sum(top_ex) / used,
        "bottom_excess": sum(bottom_ex) / used,
        "spread": mean_spread,
        "spread_sd": sd,
        "t_stat": t_stat,
        "hit_rate": hit / used,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Cross-sectional momentum study")
    parser.add_argument("csv", type=Path)
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--horizons", default="5,10,20")
    args = parser.parse_args()

    if not args.csv.exists():
        print(f"  no such file: {args.csv}")
        return 1

    series = load_daily(args.csv)
    sessions = len({day for rows in series.values() for day, _ in rows})
    print(f"  {len(series)} symbols, {sessions} sessions\n")
    print(f"  {'horizon':>8} {'rebal':>6} {'top ex':>9} {'bottom ex':>10} "
          f"{'spread':>9} {'t':>6} {'hit':>6}")
    print("  " + "-" * 60)

    for horizon in [int(h) for h in args.horizons.split(",")]:
        r = study(series, lookback=args.lookback, horizon=horizon)
        if not r["rebalances"]:
            print(f"  {horizon:>8}  insufficient history")
            continue
        print(f"  {horizon:>8} {r['rebalances']:>6} {r['top_excess']:>+8.2%} "
              f"{r['bottom_excess']:>+9.2%} {r['spread']:>+8.2%} "
              f"{r['t_stat']:>6.2f} {r['hit_rate']:>5.0%}")

    print()
    print("  spread = top quintile minus bottom quintile, both in excess of the")
    print("  universe mean. |t| > 2 is the conventional bar for 'probably not noise',")
    print("  and with this few independent rebalances it is a weak bar.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
