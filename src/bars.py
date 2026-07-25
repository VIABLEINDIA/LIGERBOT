"""Time-bar aggregation — pure logic, no I/O.

Fixes **B1** (DESIGN.md 0.1). The old strategy engine computed SMA(10)/SMA(50) over raw
*ticks*, so "50 periods" might be two seconds or two hours depending on how busy the tape
was. A strategy whose horizon changes with liquidity has no defined behaviour at all.

Everything here is deliberately free of Redis, files, and wall-clock reads, for one
reason: the backtester and the live pipeline must produce **byte-identical bars** from the
same ticks. Any divergence between the two makes every backtest result a fiction. Keeping
the aggregation pure means both paths physically share this code.

Two subtleties that are easy to get wrong and expensive to discover later:

  * **Bars must close on time, not on the next tick.** If an instrument goes quiet, a
    tick-driven aggregator emits nothing and the strategy silently stops receiving data.
    :meth:`BarAggregator.flush_until` closes bars against the clock instead.
  * **Broker volume is cumulative for the day.** Publishing it raw would make every bar's
    "volume" the day's running total. We difference it, and handle the counter resetting.
"""
from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterator, List, Optional

from src import market_calendar as cal

log = logging.getLogger("ligerbot.bars")


class VolumeMode(Enum):
    """How to interpret the ``volume`` field on an incoming tick."""

    CUMULATIVE = "cumulative"    # day-running total (Kotak Neo's default)
    INCREMENTAL = "incremental"  # quantity traded since the previous tick
    NONE = "none"                # feed carries no volume at all


@dataclass(frozen=True)
class Bar:
    """One completed OHLCV bar.

    ``synthetic`` marks a bar built from no trades at all — a gap filler. Strategies
    should generally distrust these: a flat bar is the *absence* of information, not
    evidence of a stable price, and indicators that treat it as real will understate
    volatility.
    """

    instrument_id: str
    bar_start: dt.datetime
    bar_end: dt.datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    vwap: float = 0.0
    tick_count: int = 0
    synthetic: bool = False

    @property
    def typical_price(self) -> float:
        return (self.high + self.low + self.close) / 3.0

    @property
    def range(self) -> float:
        return self.high - self.low

    def to_event(self) -> Dict[str, object]:
        """Flatten for the event bus. Datetimes become ISO-8601 strings."""
        return {
            "instrument_id": self.instrument_id,
            "bar_start": self.bar_start.isoformat(),
            "bar_end": self.bar_end.isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "vwap": self.vwap,
            "tick_count": self.tick_count,
            "synthetic": self.synthetic,
            "closed": True,  # we never publish a forming bar; see module docstring
        }

    @classmethod
    def from_event(cls, event: Dict[str, object]) -> "Bar":
        """Rebuild from an event-bus payload."""
        return cls(
            instrument_id=str(event["instrument_id"]),
            bar_start=dt.datetime.fromisoformat(str(event["bar_start"])),
            bar_end=dt.datetime.fromisoformat(str(event["bar_end"])),
            open=float(event["open"]),
            high=float(event["high"]),
            low=float(event["low"]),
            close=float(event["close"]),
            volume=float(event.get("volume", 0.0)),
            vwap=float(event.get("vwap", 0.0)),
            tick_count=int(event.get("tick_count", 0)),
            synthetic=bool(event.get("synthetic", False)),
        )


def align_to_interval(
    moment: dt.datetime, interval_seconds: int, session_open: dt.datetime
) -> dt.datetime:
    """Floor ``moment`` to its bar boundary, anchored at the session open.

    Anchoring to the session open rather than to midnight matters for intervals that
    don't divide the trading day evenly: 09:15 must always start a bar, otherwise the
    first bar of the day is a partial one and every indicator inherits the distortion.
    """
    elapsed = (moment - session_open).total_seconds()
    if elapsed < 0:
        return session_open
    return session_open + dt.timedelta(seconds=(int(elapsed) // interval_seconds) * interval_seconds)


class BarAggregator:
    """Builds bars for a single instrument from a stream of ticks.

    Usage is strictly ordered: feed ticks in non-decreasing timestamp order, collect any
    bars that :meth:`add_tick` returns, and call :meth:`flush_until` on a timer so quiet
    instruments still produce bars.
    """

    def __init__(
        self,
        instrument_id: str,
        interval_seconds: int,
        session_open: dt.datetime,
        session_close: dt.datetime,
        *,
        volume_mode: VolumeMode = VolumeMode.CUMULATIVE,
        fill_gaps: bool = True,
    ) -> None:
        self.instrument_id = instrument_id
        self.interval_seconds = interval_seconds
        self.session_open = session_open
        self.session_close = session_close
        self.volume_mode = volume_mode
        self.fill_gaps = fill_gaps

        self._bar_start: Optional[dt.datetime] = None
        self._open = self._high = self._low = self._close = 0.0
        self._tick_count = 0
        self._notional = 0.0   # sum(price * qty), for a volume-weighted VWAP
        self._qty = 0.0
        self._bar_volume = 0.0

        self._last_cum_volume: Optional[float] = None
        self._last_close: Optional[float] = None  # carries gap fills across quiet periods
        self._last_tick_ts: Optional[dt.datetime] = None
        # Start of the most recently emitted bar, real or synthetic. This is the anchor
        # gap filling resumes from, so an instrument that goes quiet for an hour still
        # produces a continuous run of bars rather than one giant hole.
        self._last_emitted_start: Optional[dt.datetime] = None

    # -- volume ------------------------------------------------------------
    def _delta_volume(self, raw: Optional[float]) -> float:
        """Convert the tick's volume field into quantity traded since the last tick."""
        if raw is None or self.volume_mode is VolumeMode.NONE:
            return 0.0
        if self.volume_mode is VolumeMode.INCREMENTAL:
            return max(0.0, raw)

        # CUMULATIVE: difference against the previous reading.
        previous = self._last_cum_volume
        self._last_cum_volume = raw
        if previous is None:
            # First tick seen — we have no baseline, so we cannot attribute any of the
            # day's accumulated volume to this bar. Counting it all here would put the
            # entire session's volume into one bar.
            return 0.0
        if raw < previous:
            # Counter went backwards: a new session, or a feed reconnect replaying from
            # zero. Treat the new reading as the baseline rather than emitting a
            # nonsensical negative delta.
            log.debug("%s cumulative volume reset (%.0f -> %.0f).",
                      self.instrument_id, previous, raw)
            return 0.0
        return raw - previous

    # -- bar lifecycle -----------------------------------------------------
    def _start_bar(self, bar_start: dt.datetime, price: float) -> None:
        self._bar_start = bar_start
        self._open = self._high = self._low = self._close = price
        self._tick_count = 0
        self._notional = 0.0
        self._qty = 0.0
        self._bar_volume = 0.0

    def _finish_bar(self) -> Optional[Bar]:
        if self._bar_start is None:
            return None
        # Volume-weighted where we have quantities; otherwise fall back to a simple
        # tick average, which is the best available estimate on a feed without volume.
        vwap = (self._notional / self._qty) if self._qty > 0 else self._close
        bar = Bar(
            instrument_id=self.instrument_id,
            bar_start=self._bar_start,
            bar_end=self._bar_start + dt.timedelta(seconds=self.interval_seconds),
            open=self._open, high=self._high, low=self._low, close=self._close,
            volume=self._bar_volume,
            vwap=round(vwap, 4),
            tick_count=self._tick_count,
            synthetic=False,
        )
        self._last_close = self._close
        self._bar_start = None
        return bar

    def _synthetic_bar(self, bar_start: dt.datetime) -> Bar:
        """A flat placeholder for an interval in which nothing traded."""
        price = self._last_close if self._last_close is not None else 0.0
        return Bar(
            instrument_id=self.instrument_id,
            bar_start=bar_start,
            bar_end=bar_start + dt.timedelta(seconds=self.interval_seconds),
            open=price, high=price, low=price, close=price,
            volume=0.0, vwap=price, tick_count=0, synthetic=True,
        )

    def _gap_bars(self, from_start: dt.datetime, to_start: dt.datetime) -> Iterator[Bar]:
        """Synthetic bars for every empty interval in ``(from_start, to_start)``.

        Without these, indicator windows drift out of time alignment: an EMA(20) would
        span 20 minutes on a busy instrument and two hours on a quiet one.
        """
        if not self.fill_gaps or self._last_close is None:
            return
        step = dt.timedelta(seconds=self.interval_seconds)
        cursor = from_start + step
        while cursor < to_start and cursor < self.session_close:
            yield self._synthetic_bar(cursor)
            cursor += step

    @property
    def last_tick_time(self) -> Optional[dt.datetime]:
        """Timestamp of the most recent accepted tick, or None if none yet.

        Used to bound end-of-run flushing. Padding to the session close instead would
        manufacture synthetic bars for periods the bot simply wasn't running, and
        overstate how much real data the store holds.
        """
        return self._last_tick_ts

    def _emit(self, bars: List[Bar]) -> List[Bar]:
        """Record the emission watermark. Every bar leaves the aggregator through here."""
        if bars:
            self._last_emitted_start = bars[-1].bar_start
        return bars

    def _elapsed_boundary(self, moment: dt.datetime) -> dt.datetime:
        """Start of the bar forming at ``moment``; everything before it has elapsed."""
        return align_to_interval(
            min(moment, self.session_close), self.interval_seconds, self.session_open
        )

    # -- public API --------------------------------------------------------
    def add_tick(
        self,
        timestamp: dt.datetime,
        price: float,
        volume: Optional[float] = None,
    ) -> List[Bar]:
        """Feed one tick. Returns any bars this tick caused to close (usually none)."""
        timestamp = cal.to_ist(timestamp)

        # Ignore anything outside the session — pre-open auction prints and any
        # after-hours noise must not contaminate the session's bars.
        if not (self.session_open <= timestamp < self.session_close):
            return []
        if price <= 0:
            return []

        # Guard against out-of-order delivery. Reordering a live feed is not worth the
        # latency, but silently folding a stale tick into the wrong bar is a real error,
        # so it is dropped and counted.
        if self._last_tick_ts is not None and timestamp < self._last_tick_ts:
            log.debug("%s out-of-order tick (%s < %s) dropped.",
                      self.instrument_id, timestamp, self._last_tick_ts)
            return []
        self._last_tick_ts = timestamp

        qty = self._delta_volume(volume)
        target_start = align_to_interval(timestamp, self.interval_seconds, self.session_open)
        closed: List[Bar] = []

        if self._bar_start is None:
            # Resuming after an idle stretch: backfill the intervals we sat out, so the
            # returning tick doesn't appear adjacent to a bar from an hour ago.
            if self._last_emitted_start is not None:
                closed.extend(self._gap_bars(self._last_emitted_start, target_start))
            self._start_bar(target_start, price)
        elif target_start > self._bar_start:
            previous_start = self._bar_start
            finished = self._finish_bar()
            if finished is not None:
                closed.append(finished)
            closed.extend(self._gap_bars(previous_start, target_start))
            self._start_bar(target_start, price)

        # Accumulate into the (now current) bar.
        self._high = max(self._high, price)
        self._low = min(self._low, price)
        self._close = price
        self._tick_count += 1
        if qty > 0:
            self._notional += price * qty
            self._qty += qty
            self._bar_volume += qty
        return self._emit(closed)

    def flush_until(self, moment: dt.datetime) -> List[Bar]:
        """Close every bar that has fully elapsed as of ``moment``.

        This is what makes bar emission time-driven rather than tick-driven. Call it on
        a timer; without it, an instrument that stops trading also stops producing bars,
        and the strategy has no way to distinguish "quiet" from "feed is dead".

        Idempotent: calling it repeatedly for the same moment emits nothing the second
        time, so it is safe to drive from a fast loop.
        """
        target = self._elapsed_boundary(cal.to_ist(moment))
        closed: List[Bar] = []

        if self._bar_start is not None and target > self._bar_start:
            previous_start = self._bar_start
            finished = self._finish_bar()
            if finished is not None:
                closed.append(finished)
            closed.extend(self._gap_bars(previous_start, target))
        elif self._bar_start is None and self._last_emitted_start is not None:
            closed.extend(self._gap_bars(self._last_emitted_start, target))

        return self._emit(closed)


class MultiInstrumentAggregator:
    """One :class:`BarAggregator` per instrument, sharing a session window."""

    def __init__(
        self,
        interval_seconds: int,
        session_day: dt.date,
        *,
        volume_mode: VolumeMode = VolumeMode.CUMULATIVE,
        fill_gaps: bool = True,
    ) -> None:
        window = cal.session_window(session_day)
        if window is None:
            raise ValueError(f"{session_day} is not an NSE trading day — cannot build bars.")
        self.session_open, self.session_close = window
        self.session_day = session_day
        self.interval_seconds = interval_seconds
        self.volume_mode = volume_mode
        self.fill_gaps = fill_gaps
        self._aggregators: Dict[str, BarAggregator] = {}

    def _for(self, instrument_id: str) -> BarAggregator:
        if instrument_id not in self._aggregators:
            self._aggregators[instrument_id] = BarAggregator(
                instrument_id,
                self.interval_seconds,
                self.session_open,
                self.session_close,
                volume_mode=self.volume_mode,
                fill_gaps=self.fill_gaps,
            )
        return self._aggregators[instrument_id]

    def add_tick(
        self,
        instrument_id: str,
        timestamp: dt.datetime,
        price: float,
        volume: Optional[float] = None,
    ) -> List[Bar]:
        return self._for(instrument_id).add_tick(timestamp, price, volume)

    def flush_until(self, moment: dt.datetime) -> List[Bar]:
        """Flush every instrument, returning all closed bars in time order."""
        bars: List[Bar] = []
        for aggregator in self._aggregators.values():
            bars.extend(aggregator.flush_until(moment))
        bars.sort(key=lambda b: (b.bar_start, b.instrument_id))
        return bars

    @property
    def instrument_ids(self) -> List[str]:
        return list(self._aggregators)

    @property
    def last_tick_time(self) -> Optional[dt.datetime]:
        """Latest tick timestamp across all instruments, or None if no ticks yet."""
        seen = [
            a.last_tick_time for a in self._aggregators.values()
            if a.last_tick_time is not None
        ]
        return max(seen) if seen else None
