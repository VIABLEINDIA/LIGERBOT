"""Import TradingView CSV bars into the Parquet store.

A **bootstrap**, not a data source. The store starts empty and fills at one session per
day (D5 mitigation 2); this puts real NSE prices in it today so the backtest harness can be
exercised on something other than a synthetic generator.

Provenance is stamped on every row and the source is recorded in the filename, because a
dataset whose origin cannot be established is one nobody should trust a backtest to. The
imported bars are **not** broker data: they carry TradingView's adjustments, its session
boundaries and its idea of volume, none of which are guaranteed to match what Kotak
reports. That is fine for validating a pipeline and misleading for validating a strategy,
so the two uses are kept clearly apart.

    python -m tools.import_tv_bars ~/Downloads/ligerbot_bars.csv --interval 900
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from src import market_calendar as cal
from src.bar_store import ParquetBarStore
from src.bars import Bar


def load_csv(path: Path, interval_seconds: int) -> Dict[str, List[Bar]]:
    """Read the export into Bars, keyed by canonical-ish instrument id.

    The symbol is kept as ``tv:NSE:RELIANCE`` rather than guessed into ``nse_cm:<token>``:
    resolving to a canonical id needs the instrument master, and a guessed token is an
    order on the wrong instrument (B4). Backtesting does not need the real token, and
    pretending to have it would let this data leak into a live path.
    """
    by_instrument: Dict[str, List[Bar]] = defaultdict(list)
    skipped = 0

    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                # Keep the timezone. Stripping it looks harmless and is not: the store
                # writes UTC and converts back to IST on read, so a naive IST wall time
                # is read as UTC and comes back +5:30 — putting every 09:15 bar at 14:45
                # and every afternoon bar past the close, where the engine's phase check
                # discards it. The backtest then reports zero trades and looks like a
                # market observation rather than a timezone bug.
                #
                # `cal.at()` returns tz-aware datetimes for exactly this reason, and the
                # bot's own bar builder round-trips correctly because of it.
                start = dt.datetime.fromtimestamp(int(row["time"]), tz=cal.IST)
                bar = Bar(
                    instrument_id=f"tv:{row['symbol']}",
                    bar_start=start,
                    bar_end=start + dt.timedelta(seconds=interval_seconds),
                    open=float(row["open"]), high=float(row["high"]),
                    low=float(row["low"]), close=float(row["close"]),
                    volume=float(row["volume"] or 0.0),
                    vwap=(float(row["high"]) + float(row["low"])
                          + float(row["close"])) / 3.0,
                    tick_count=0,
                )
            except (KeyError, TypeError, ValueError):
                skipped += 1
                continue

            # Bars outside the NSE session are TradingView's, not the exchange's. Keeping
            # them would put trades in windows the live bot can never trade in.
            if not cal.is_trading_day(bar.bar_start.date()):
                skipped += 1
                continue
            if not (cal.SESSION_OPEN <= bar.bar_start.time() < cal.SESSION_CLOSE):
                skipped += 1
                continue
            by_instrument[bar.instrument_id].append(bar)

    if skipped:
        print(f"  skipped {skipped} row(s): unparseable, non-trading-day, or "
              f"outside 09:15-15:30")
    return by_instrument


def main() -> int:
    parser = argparse.ArgumentParser(description="Import TradingView bars")
    parser.add_argument("csv", type=Path)
    parser.add_argument("--interval", type=int, default=900,
                        help="bar interval in seconds (900 = 15 minutes)")
    parser.add_argument("--root", default="bar_data/tv_import")
    args = parser.parse_args()

    if not args.csv.exists():
        print(f"  no such file: {args.csv}")
        return 1

    label = (f"{args.interval // 60}m" if args.interval < 3600
             else f"{args.interval // 3600}h")
    store = ParquetBarStore(args.root, label)
    by_instrument = load_csv(args.csv, args.interval)

    total = 0
    for instrument_id, bars in sorted(by_instrument.items()):
        bars.sort(key=lambda b: b.bar_start)
        total += store.write(bars)
        days = sorted({b.bar_start.date() for b in bars})
        print(f"  {instrument_id:<22} {len(bars):>5} bars  "
              f"{days[0]} -> {days[-1]}  ({len(days)} sessions)")

    print(f"\n  {total} bars written to {args.root} ({label})")
    print("  NOTE: TradingView data — adjustments, session boundaries and volume are")
    print("  its own, not the broker's. Good for exercising the harness; not a")
    print("  substitute for broker data when validating a strategy.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
