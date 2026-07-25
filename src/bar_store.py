"""Parquet bar store — the historical dataset, written from day one.

This lands in Phase 0 rather than Phase 1, deliberately (DESIGN.md 4). Kotak is our only
history source (D5) and its depth is expected to be thin, so the dataset we record
ourselves is the slowest-maturing asset in the project. It has to start accumulating
before anything that depends on it is built.

Self-recorded bars are also the *highest-fidelity* data available: they are literally what
the live system saw, including its gaps and its latency, rather than a vendor's cleaned-up
reconstruction.

Layout — partitioned so a backtest can read one instrument-day without touching the rest::

    bar_data/
      1m/
        nse_cm_11536/
          2026-07-23.parquet

Writes are idempotent on ``(instrument_id, bar_start)``: re-running a day merges rather
than duplicating, so a crashed session can simply be replayed.
"""
from __future__ import annotations

import datetime as dt
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd

from src.bars import Bar
from src.market_calendar import IST as IST_TZ

log = logging.getLogger("ligerbot.bar_store")

BAR_COLUMNS = [
    "instrument_id", "bar_start", "bar_end",
    "open", "high", "low", "close",
    "volume", "vwap", "tick_count", "synthetic",
]

# instrument_id looks like "nse_cm:11536"; ':' is illegal in Windows paths.
_UNSAFE_PATH_CHARS = re.compile(r"[^A-Za-z0-9._-]")


def _safe_name(instrument_id: str) -> str:
    return _UNSAFE_PATH_CHARS.sub("_", instrument_id)


def bars_to_frame(bars: Sequence[Bar]) -> pd.DataFrame:
    """Convert bars to a DataFrame with a stable column order and dtypes."""
    if not bars:
        return pd.DataFrame(columns=BAR_COLUMNS)
    frame = pd.DataFrame([{
        "instrument_id": b.instrument_id,
        "bar_start": b.bar_start,
        "bar_end": b.bar_end,
        "open": b.open, "high": b.high, "low": b.low, "close": b.close,
        "volume": b.volume, "vwap": b.vwap,
        "tick_count": b.tick_count, "synthetic": b.synthetic,
    } for b in bars])
    frame["bar_start"] = pd.to_datetime(frame["bar_start"], utc=True)
    frame["bar_end"] = pd.to_datetime(frame["bar_end"], utc=True)
    return frame[BAR_COLUMNS]


class ParquetBarStore:
    """Append-only, partitioned store of completed bars.

    Not thread-safe by design — a single writer owns the store. Concurrent writers to the
    same partition would interleave read-modify-write cycles and silently lose bars.
    """

    def __init__(self, root: str | Path, interval_label: str = "1m") -> None:
        self.root = Path(root) / interval_label
        self.interval_label = interval_label
        self.root.mkdir(parents=True, exist_ok=True)
        self._buffer: List[Bar] = []

    # -- paths -------------------------------------------------------------
    def partition_path(self, instrument_id: str, day: dt.date) -> Path:
        return self.root / _safe_name(instrument_id) / f"{day.isoformat()}.parquet"

    # -- writing -----------------------------------------------------------
    def buffer(self, bars: Iterable[Bar]) -> int:
        """Queue bars for a later :meth:`flush`. Returns the buffered count.

        Buffering exists so the live path never blocks on disk. Archiving must never
        apply backpressure to trading (DESIGN.md 3.9).
        """
        before = len(self._buffer)
        self._buffer.extend(bars)
        return len(self._buffer) - before

    def flush(self) -> int:
        """Write all buffered bars to disk, grouped by partition. Returns bars written."""
        if not self._buffer:
            return 0
        pending, self._buffer = self._buffer, []
        return self.write(pending)

    def write(self, bars: Sequence[Bar]) -> int:
        """Merge ``bars`` into their partitions immediately."""
        if not bars:
            return 0

        groups: Dict[tuple, List[Bar]] = defaultdict(list)
        for bar in bars:
            # Partition by the bar's own session date, not today's — replaying an old
            # session must land in that session's file.
            groups[(bar.instrument_id, bar.bar_start.date())].append(bar)

        written = 0
        for (instrument_id, day), group in groups.items():
            written += self._write_partition(instrument_id, day, group)
        return written

    def _write_partition(self, instrument_id: str, day: dt.date, bars: List[Bar]) -> int:
        path = self.partition_path(instrument_id, day)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame = bars_to_frame(bars)

        if path.exists():
            try:
                existing = pd.read_parquet(path)
                frame = pd.concat([existing, frame], ignore_index=True)
            except (OSError, ValueError) as exc:
                # A corrupt partition must not take down the writer, but it also must
                # not be silently overwritten — the old file is kept for inspection.
                quarantine = path.with_suffix(".parquet.corrupt")
                log.error("Unreadable partition %s (%s) — moved to %s and rewriting.",
                          path, exc, quarantine)
                path.replace(quarantine)

        # Last write wins per (instrument, bar_start): a replayed session supersedes a
        # partial one, which is what makes re-running a crashed day safe.
        frame = (
            frame.drop_duplicates(subset=["instrument_id", "bar_start"], keep="last")
                 .sort_values("bar_start")
                 .reset_index(drop=True)
        )
        frame.to_parquet(path, index=False, compression="snappy")
        return len(bars)

    # -- reading -----------------------------------------------------------
    @staticmethod
    def _to_ist(frame: pd.DataFrame) -> pd.DataFrame:
        """Present timestamps in IST.

        Storage is UTC (unambiguous, portable); everything that reasons about sessions
        works in IST. Converting on the way out means a caller never has to remember
        which convention a given frame is in — and a 09:15 bar reads as 09:15, not 03:45.
        """
        for column in ("bar_start", "bar_end"):
            if column in frame.columns and len(frame):
                frame[column] = pd.to_datetime(frame[column], utc=True).dt.tz_convert(IST_TZ)
        return frame

    def read_day(self, instrument_id: str, day: dt.date) -> pd.DataFrame:
        path = self.partition_path(instrument_id, day)
        if not path.exists():
            return pd.DataFrame(columns=BAR_COLUMNS)
        return self._to_ist(pd.read_parquet(path))

    def read_range(
        self,
        instrument_id: str,
        start: dt.date,
        end: dt.date,
        *,
        drop_synthetic: bool = False,
    ) -> pd.DataFrame:
        """Read ``[start, end]`` inclusive for one instrument.

        ``drop_synthetic`` removes gap-filler bars. Useful for measuring genuine data
        coverage, but generally leave them in for backtesting — removing them silently
        re-introduces the time-misalignment the filler exists to prevent.
        """
        frames = []
        for day in self.available_days(instrument_id):
            if start <= day <= end:
                frames.append(self.read_day(instrument_id, day))
        if not frames:
            return pd.DataFrame(columns=BAR_COLUMNS)
        out = pd.concat(frames, ignore_index=True).sort_values("bar_start")
        if drop_synthetic:
            out = out[~out["synthetic"].astype(bool)]
        return self._to_ist(out.reset_index(drop=True))

    def available_days(self, instrument_id: str) -> List[dt.date]:
        folder = self.root / _safe_name(instrument_id)
        if not folder.exists():
            return []
        days = []
        for file in folder.glob("*.parquet"):
            try:
                days.append(dt.date.fromisoformat(file.stem))
            except ValueError:
                continue
        return sorted(days)

    def instruments(self) -> List[str]:
        return sorted(p.name for p in self.root.iterdir() if p.is_dir())

    # -- coverage ----------------------------------------------------------
    def coverage(self) -> pd.DataFrame:
        """Per-instrument summary: day count, span, bar count, synthetic fraction.

        The synthetic fraction is the number that matters. A high ratio means the feed
        was mostly silent, so the "data" is largely our own gap filler — bars that exist
        but carry no information. Backtesting on that produces confident nonsense.
        """
        rows = []
        for folder_name in self.instruments():
            days = [
                dt.date.fromisoformat(f.stem)
                for f in (self.root / folder_name).glob("*.parquet")
                if _is_date(f.stem)
            ]
            if not days:
                continue
            total = synthetic = 0
            for day in days:
                frame = pd.read_parquet(self.root / folder_name / f"{day.isoformat()}.parquet")
                total += len(frame)
                synthetic += int(frame["synthetic"].astype(bool).sum())
            rows.append({
                "instrument": folder_name,
                "days": len(days),
                "first_day": min(days),
                "last_day": max(days),
                "bars": total,
                "synthetic_pct": round(100.0 * synthetic / total, 2) if total else 0.0,
            })
        return pd.DataFrame(rows)


def _is_date(text: str) -> bool:
    try:
        dt.date.fromisoformat(text)
        return True
    except ValueError:
        return False
