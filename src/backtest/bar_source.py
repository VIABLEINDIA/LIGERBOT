"""Bar sources and the data-quality gate.

``BarSource`` is the seam that lets the *same* strategy code run against historical
Parquet, a broker backfill, or the live stream (DESIGN.md 2.2). Only the transport differs;
the strategy cannot tell which it is attached to, which is the property that makes a
backtest predictive of anything.

The quality gate exists because bad data produces confident nonsense. Three failure modes
matter enough to block a run:

  * **Missing bars** — a session with holes is a session where the strategy silently saw a
    different market than the one that traded.
  * **Unadjusted corporate actions** — a 1:5 split looks exactly like an 80% overnight
    crash. A strategy backtested across one will show a spectacular, entirely fictional
    trade. We cannot adjust without reference data, so we detect and refuse.
  * **Excess synthetic bars** — gap fillers are the *absence* of information. Past a
    threshold the "data" is mostly our own padding, and any statistics computed on it
    describe the filler rather than the market.

A run against ungated data is refused rather than warned about, because a warning in a log
is not a decision anyone actually makes.
"""
from __future__ import annotations

import datetime as dt
import heapq
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Optional, Sequence

import pandas as pd

from src import market_calendar as cal
from src.bar_store import ParquetBarStore
from src.bars import Bar

log = logging.getLogger("ligerbot.bar_source")


class DataQualityError(RuntimeError):
    """Raised when a dataset is not fit to backtest on."""


@dataclass
class QualityIssue:
    instrument_id: str
    day: Optional[dt.date]
    kind: str
    detail: str
    blocking: bool = True

    def __str__(self) -> str:
        where = f"{self.instrument_id}" + (f" {self.day}" if self.day else "")
        flag = "BLOCK" if self.blocking else "warn "
        return f"[{flag}] {where}: {self.kind} — {self.detail}"


@dataclass
class QualityReport:
    """Verdict on a dataset. Stamped into every backtest report for provenance."""

    instrument_ids: List[str] = field(default_factory=list)
    start: Optional[dt.date] = None
    end: Optional[dt.date] = None
    total_bars: int = 0
    synthetic_bars: int = 0
    trading_days: int = 0
    issues: List[QualityIssue] = field(default_factory=list)

    @property
    def synthetic_pct(self) -> float:
        return (100.0 * self.synthetic_bars / self.total_bars) if self.total_bars else 0.0

    @property
    def blocking_issues(self) -> List[QualityIssue]:
        return [i for i in self.issues if i.blocking]

    @property
    def is_usable(self) -> bool:
        return self.total_bars > 0 and not self.blocking_issues

    def raise_if_unusable(self) -> None:
        if self.total_bars == 0:
            raise DataQualityError(
                "Dataset is empty. Backfill has not run, or the requested window "
                "contains no trading days."
            )
        if self.blocking_issues:
            detail = "\n  ".join(str(i) for i in self.blocking_issues[:20])
            extra = ("" if len(self.blocking_issues) <= 20
                     else f"\n  ... and {len(self.blocking_issues) - 20} more")
            raise DataQualityError(
                f"Dataset failed {len(self.blocking_issues)} quality check(s) — refusing "
                f"to backtest on it:\n  {detail}{extra}"
            )

    def summary(self) -> str:
        span = f"{self.start} to {self.end}" if self.start else "empty"
        return (
            f"{len(self.instrument_ids)} instrument(s), {span}, "
            f"{self.trading_days} trading day(s), {self.total_bars:,} bars "
            f"({self.synthetic_pct:.1f}% synthetic), "
            f"{len(self.blocking_issues)} blocking / "
            f"{len(self.issues) - len(self.blocking_issues)} advisory issue(s)"
        )


@dataclass
class QualityThresholds:
    """What counts as unusable.

    Defaults are deliberately strict. It is far cheaper to loosen one knowingly than to
    discover months later that a result rested on padding.
    """

    max_synthetic_pct: float = 40.0
    # An overnight move this large is far more likely an unadjusted split/bonus than a
    # real price change. NSE has a 20% circuit limit on most scrips, so a genuine
    # overnight move beyond that is rare enough to warrant a human look.
    corporate_action_gap_pct: float = 25.0
    # Intrabar spikes: high/low this far from the close suggests a bad print.
    max_bar_range_pct: float = 20.0
    min_bars_per_day: int = 30
    min_trading_days: int = 20
    # Advisory only — a low count is a power problem, not a correctness problem.
    recommended_trading_days: int = 250


def assess_quality(
    frames: Dict[str, pd.DataFrame],
    thresholds: Optional[QualityThresholds] = None,
) -> QualityReport:
    """Inspect loaded bars and decide whether they can be backtested on."""
    thresholds = thresholds or QualityThresholds()
    report = QualityReport(instrument_ids=sorted(frames))

    all_days: set[dt.date] = set()
    for instrument_id, frame in frames.items():
        if frame.empty:
            report.issues.append(QualityIssue(
                instrument_id, None, "no-data", "no bars in the requested window"))
            continue

        frame = frame.sort_values("bar_start").reset_index(drop=True)
        report.total_bars += len(frame)
        report.synthetic_bars += int(frame["synthetic"].astype(bool).sum())

        days = sorted({ts.date() for ts in frame["bar_start"]})
        all_days.update(days)

        _check_synthetic_ratio(instrument_id, frame, thresholds, report)
        _check_daily_bar_counts(instrument_id, frame, days, thresholds, report)
        _check_corporate_actions(instrument_id, frame, days, thresholds, report)
        _check_price_sanity(instrument_id, frame, thresholds, report)
        _check_ohlc_consistency(instrument_id, frame, report)

    if all_days:
        report.start, report.end = min(all_days), max(all_days)
        report.trading_days = len(all_days)

        if report.trading_days < thresholds.min_trading_days:
            report.issues.append(QualityIssue(
                "*", None, "insufficient-history",
                f"{report.trading_days} trading days is below the "
                f"{thresholds.min_trading_days}-day minimum",
                blocking=True))
        elif report.trading_days < thresholds.recommended_trading_days:
            # Advisory: this is the D5 risk made concrete. Not wrong, just underpowered.
            report.issues.append(QualityIssue(
                "*", None, "thin-history",
                f"{report.trading_days} trading days is below the recommended "
                f"{thresholds.recommended_trading_days}. Results will be underpowered "
                f"and likely span a single market regime — extend the paper period "
                f"rather than trusting these numbers (DESIGN.md D5)",
                blocking=False))
    return report


def _check_synthetic_ratio(instrument_id, frame, thresholds, report) -> None:
    synthetic_pct = 100.0 * frame["synthetic"].astype(bool).mean()
    if synthetic_pct > thresholds.max_synthetic_pct:
        report.issues.append(QualityIssue(
            instrument_id, None, "excess-synthetic",
            f"{synthetic_pct:.1f}% of bars are gap fillers, over the "
            f"{thresholds.max_synthetic_pct:.0f}% limit — the feed was mostly silent, "
            f"so these bars carry no information"))


def _check_daily_bar_counts(instrument_id, frame, days, thresholds, report) -> None:
    counts = frame.groupby(frame["bar_start"].dt.date).size()
    for day, count in counts.items():
        if count < thresholds.min_bars_per_day:
            report.issues.append(QualityIssue(
                instrument_id, day, "sparse-session",
                f"only {count} bars, below the {thresholds.min_bars_per_day} minimum"))


def _check_corporate_actions(instrument_id, frame, days, thresholds, report) -> None:
    """Flag overnight gaps that look like unadjusted splits or bonuses."""
    closes = frame.groupby(frame["bar_start"].dt.date)["close"].last()
    opens = frame.groupby(frame["bar_start"].dt.date)["open"].first()
    for previous_day, current_day in zip(days, days[1:]):
        previous_close = closes.get(previous_day)
        current_open = opens.get(current_day)
        if not previous_close or not current_open or previous_close <= 0:
            continue
        gap_pct = abs(current_open - previous_close) / previous_close * 100.0
        if gap_pct > thresholds.corporate_action_gap_pct:
            ratio = current_open / previous_close
            report.issues.append(QualityIssue(
                instrument_id, current_day, "suspected-corporate-action",
                f"{gap_pct:.1f}% overnight gap ({previous_close:.2f} -> "
                f"{current_open:.2f}, ratio {ratio:.3f}). Almost certainly an unadjusted "
                f"split/bonus rather than a real move — a backtest across this will book "
                f"a large fictional trade"))


def _check_price_sanity(instrument_id, frame, thresholds, report) -> None:
    real = frame[~frame["synthetic"].astype(bool)]
    if real.empty:
        return
    if (real["close"] <= 0).any():
        report.issues.append(QualityIssue(
            instrument_id, None, "non-positive-price",
            f"{int((real['close'] <= 0).sum())} bars have a close <= 0"))
    span = (real["high"] - real["low"]) / real["close"].replace(0, pd.NA) * 100.0
    extreme = real[span > thresholds.max_bar_range_pct]
    if not extreme.empty:
        worst = span.max()
        report.issues.append(QualityIssue(
            instrument_id, None, "price-spike",
            f"{len(extreme)} bar(s) span more than "
            f"{thresholds.max_bar_range_pct:.0f}% of price (worst {worst:.1f}%) — "
            f"likely bad prints"))


def _check_ohlc_consistency(instrument_id, frame, report) -> None:
    """high >= max(open, close) and low <= min(open, close), or the bar is corrupt."""
    bad_high = frame["high"] < frame[["open", "close"]].max(axis=1) - 1e-9
    bad_low = frame["low"] > frame[["open", "close"]].min(axis=1) + 1e-9
    broken = int((bad_high | bad_low).sum())
    if broken:
        report.issues.append(QualityIssue(
            instrument_id, None, "inconsistent-ohlc",
            f"{broken} bar(s) violate high >= max(o,c) or low <= min(o,c)"))


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------
class BarSource(ABC):
    """Where bars come from. The seam between backtest, paper and live."""

    @abstractmethod
    def load(self, instrument_id: str, start: dt.date, end: dt.date) -> pd.DataFrame:
        """All bars for one instrument in ``[start, end]`` inclusive, time-ordered."""

    @abstractmethod
    def available_days(self, instrument_id: str) -> List[dt.date]:
        ...

    def load_all(
        self, instrument_ids: Sequence[str], start: dt.date, end: dt.date
    ) -> Dict[str, pd.DataFrame]:
        return {i: self.load(i, start, end) for i in instrument_ids}

    def assess(
        self,
        instrument_ids: Sequence[str],
        start: dt.date,
        end: dt.date,
        thresholds: Optional[QualityThresholds] = None,
    ) -> QualityReport:
        return assess_quality(self.load_all(instrument_ids, start, end), thresholds)

    def stream(
        self, instrument_ids: Sequence[str], start: dt.date, end: dt.date
    ) -> Iterator[Bar]:
        """Merged, strictly chronological bar stream across instruments.

        Chronological order across the whole universe is not a convenience — it is what
        stops the engine from seeing instrument B's 14:00 bar before instrument A's
        10:00 bar, which would let a strategy act on information it could not have had.
        """
        frames = self.load_all(instrument_ids, start, end)
        iterators = []
        for instrument_id, frame in frames.items():
            if not frame.empty:
                iterators.append(_frame_to_bars(instrument_id, frame))
        # heapq.merge keeps the whole thing lazy — no need to materialise every bar.
        yield from heapq.merge(*iterators, key=lambda b: (b.bar_start, b.instrument_id))


def _frame_to_bars(instrument_id: str, frame: pd.DataFrame) -> Iterator[Bar]:
    for row in frame.sort_values("bar_start").itertuples(index=False):
        yield Bar(
            instrument_id=instrument_id,
            bar_start=row.bar_start.to_pydatetime(),
            bar_end=row.bar_end.to_pydatetime(),
            open=float(row.open), high=float(row.high),
            low=float(row.low), close=float(row.close),
            volume=float(row.volume), vwap=float(row.vwap),
            tick_count=int(row.tick_count), synthetic=bool(row.synthetic),
        )


class ParquetBarSource(BarSource):
    """Reads the self-recorded Parquet store — the primary backtest source."""

    def __init__(self, store: ParquetBarStore) -> None:
        self.store = store

    def load(self, instrument_id: str, start: dt.date, end: dt.date) -> pd.DataFrame:
        return self.store.read_range(instrument_id, start, end)

    def available_days(self, instrument_id: str) -> List[dt.date]:
        return self.store.available_days(instrument_id)

    def coverage_gaps(
        self, instrument_id: str, start: dt.date, end: dt.date
    ) -> List[dt.date]:
        """Trading days in the window with no data at all.

        The honest measure of how much history we really hold, as opposed to how wide a
        date range the files happen to span.
        """
        have = set(self.available_days(instrument_id))
        return [d for d in cal.trading_days_between(start, end) if d not in have]


def resample_frame(frame: pd.DataFrame, interval_seconds: int) -> pd.DataFrame:
    """Aggregate finer bars into coarser ones, anchored to each session's open.

    The mechanism behind D4: the store holds 1-minute bars, and a strategy consumes
    whatever interval it was configured for. Anchoring per session matters — resampling
    against a midnight-anchored grid would make the first bar of the day a partial one,
    and every indicator would inherit that distortion.
    """
    if frame.empty:
        return frame
    frame = frame.sort_values("bar_start").reset_index(drop=True)
    out: List[dict] = []

    for day, session in frame.groupby(frame["bar_start"].dt.date):
        window = cal.session_window(day)
        if window is None:
            continue
        session_open = pd.Timestamp(window[0])
        elapsed = (session["bar_start"] - session_open).dt.total_seconds()
        bucket = (elapsed // interval_seconds).astype(int)

        for bucket_index, group in session.groupby(bucket):
            start = session_open + pd.Timedelta(seconds=int(bucket_index) * interval_seconds)
            volume = float(group["volume"].sum())
            # Volume-weighted where volume exists, else a plain mean — matching how the
            # bar builder computes it, so resampled and native bars agree.
            if volume > 0:
                vwap = float((group["vwap"] * group["volume"]).sum() / volume)
            else:
                vwap = float(group["vwap"].mean())
            out.append({
                "instrument_id": group["instrument_id"].iloc[0],
                "bar_start": start,
                "bar_end": start + pd.Timedelta(seconds=interval_seconds),
                "open": float(group["open"].iloc[0]),
                "high": float(group["high"].max()),
                "low": float(group["low"].min()),
                "close": float(group["close"].iloc[-1]),
                "volume": volume,
                "vwap": vwap,
                "tick_count": int(group["tick_count"].sum()),
                # Only fully-synthetic buckets stay synthetic: one real trade in the
                # window means something genuinely happened in it.
                "synthetic": bool(group["synthetic"].astype(bool).all()),
            })

    result = pd.DataFrame(out)
    if not result.empty:
        result["bar_start"] = pd.to_datetime(result["bar_start"])
        result["bar_end"] = pd.to_datetime(result["bar_end"])
    return result


class ResampledBarSource(BarSource):
    """Wraps a source, serving coarser bars aggregated from its finer ones."""

    def __init__(self, source: BarSource, interval_seconds: int) -> None:
        self.source = source
        self.interval_seconds = interval_seconds
        self._cache: Dict[tuple, pd.DataFrame] = {}

    def load(self, instrument_id: str, start: dt.date, end: dt.date) -> pd.DataFrame:
        key = (instrument_id, start, end)
        if key not in self._cache:
            self._cache[key] = resample_frame(
                self.source.load(instrument_id, start, end), self.interval_seconds)
        return self._cache[key]

    def available_days(self, instrument_id: str) -> List[dt.date]:
        return self.source.available_days(instrument_id)


class InMemoryBarSource(BarSource):
    """Bars held in memory. For tests and for replaying a generated series."""

    def __init__(self, frames: Dict[str, pd.DataFrame]) -> None:
        self.frames = frames

    def load(self, instrument_id: str, start: dt.date, end: dt.date) -> pd.DataFrame:
        frame = self.frames.get(instrument_id)
        if frame is None or frame.empty:
            return pd.DataFrame(columns=[
                "instrument_id", "bar_start", "bar_end", "open", "high", "low",
                "close", "volume", "vwap", "tick_count", "synthetic",
            ])
        days = frame["bar_start"].dt.date
        return frame[(days >= start) & (days <= end)].reset_index(drop=True)

    def available_days(self, instrument_id: str) -> List[dt.date]:
        frame = self.frames.get(instrument_id)
        if frame is None or frame.empty:
            return []
        return sorted({ts.date() for ts in frame["bar_start"]})
