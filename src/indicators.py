"""Incremental indicators (DESIGN.md 1.5).

Three requirements shape every class here:

1. **O(1) per update.** The original ``strategy_engine.py`` recomputed its whole window on
   every tick — O(N) per instrument per update. At a hundred instruments and a busy tape
   that is the difference between keeping up and falling behind.

2. **Replay-deterministic.** No wall-clock reads, no randomness, no hidden global state.
   The same bar sequence must produce the same values in the backtester and in production,
   or the two systems are quietly running different strategies.

3. **Explicit warmup.** Every indicator returns ``None`` until it has enough data. Emitting
   a half-warmed average as though it were a real one is how a strategy ends up trading on
   an artefact of its own initialisation.

The set is chosen for intraday NSE equities specifically. Session VWAP is here because it
is the single most-watched intraday reference on the exchange — institutional flow anchors
to it, so price above or below it is the cleanest bias filter available. ATR is here
because volatility-normalised stops are what make risk-based sizing comparable across
instruments trading at different prices.
"""
from __future__ import annotations

import datetime as dt
from abc import ABC, abstractmethod
from collections import deque
from typing import Deque, Optional

from src.bars import Bar


class Indicator(ABC):
    """Base class. ``update`` returns the new value, or None while warming up."""

    @abstractmethod
    def update(self, bar: Bar) -> Optional[float]:
        ...

    @property
    @abstractmethod
    def value(self) -> Optional[float]:
        """Most recent value without advancing state."""

    @property
    def ready(self) -> bool:
        return self.value is not None

    def reset(self) -> None:
        """Clear state. Called at each session start for session-anchored indicators."""
        self.__init__(**self._params)  # type: ignore[attr-defined]


class SMA(Indicator):
    """Simple moving average, maintained with a running sum."""

    def __init__(self, period: int) -> None:
        if period < 1:
            raise ValueError("period must be >= 1")
        self.period = period
        self._window: Deque[float] = deque()
        self._sum = 0.0

    def update_value(self, price: float) -> Optional[float]:
        self._window.append(price)
        self._sum += price
        if len(self._window) > self.period:
            self._sum -= self._window.popleft()
        return self.value

    def update(self, bar: Bar) -> Optional[float]:
        return self.update_value(bar.close)

    @property
    def value(self) -> Optional[float]:
        if len(self._window) < self.period:
            return None
        return self._sum / self.period

    def reset(self) -> None:
        self._window.clear()
        self._sum = 0.0


class EMA(Indicator):
    """Exponential moving average.

    Seeded with an SMA of the first ``period`` values rather than with the first value
    alone. Seeding from a single price makes the early readings depend heavily on one
    arbitrary bar, and that distortion persists for several multiples of the period.
    """

    def __init__(self, period: int) -> None:
        if period < 1:
            raise ValueError("period must be >= 1")
        self.period = period
        self.alpha = 2.0 / (period + 1.0)
        self._value: Optional[float] = None
        self._seed: Deque[float] = deque()

    def update_value(self, price: float) -> Optional[float]:
        if self._value is None:
            self._seed.append(price)
            if len(self._seed) < self.period:
                return None
            self._value = sum(self._seed) / len(self._seed)
            return self._value
        self._value += self.alpha * (price - self._value)
        return self._value

    def update(self, bar: Bar) -> Optional[float]:
        return self.update_value(bar.close)

    @property
    def value(self) -> Optional[float]:
        return self._value

    def reset(self) -> None:
        self._value = None
        self._seed.clear()


class SessionVWAP(Indicator):
    """Volume-weighted average price, anchored to the session open.

    The reference intraday traders actually watch. Uses each bar's own VWAP where volume
    is available and falls back to the typical price otherwise, so a feed without volume
    still produces something meaningful rather than silently returning None all day.
    """

    def __init__(self) -> None:
        self._notional = 0.0
        self._volume = 0.0
        self._fallback_sum = 0.0
        self._fallback_count = 0

    def update(self, bar: Bar) -> Optional[float]:
        # Synthetic bars represent no trading at all; folding them in would drag VWAP
        # toward a price at which nothing changed hands.
        if bar.synthetic:
            return self.value
        price = bar.vwap if bar.vwap > 0 else bar.typical_price
        if bar.volume > 0:
            self._notional += price * bar.volume
            self._volume += bar.volume
        else:
            self._fallback_sum += price
            self._fallback_count += 1
        return self.value

    @property
    def value(self) -> Optional[float]:
        if self._volume > 0:
            return self._notional / self._volume
        if self._fallback_count > 0:
            return self._fallback_sum / self._fallback_count
        return None

    def reset(self) -> None:
        self._notional = self._volume = self._fallback_sum = 0.0
        self._fallback_count = 0


class ATR(Indicator):
    """Average True Range with Wilder smoothing.

    Wilder's original formulation (``prev * (n-1)/n + tr/n``), not a simple average —
    it is what every charting package and every published ATR figure uses, so matching it
    keeps our numbers comparable to anything a human looks at.
    """

    def __init__(self, period: int = 14) -> None:
        if period < 1:
            raise ValueError("period must be >= 1")
        self.period = period
        self._value: Optional[float] = None
        self._prev_close: Optional[float] = None
        self._seed: Deque[float] = deque()

    @staticmethod
    def true_range(bar: Bar, prev_close: Optional[float]) -> float:
        if prev_close is None:
            return bar.high - bar.low
        return max(
            bar.high - bar.low,
            abs(bar.high - prev_close),
            abs(bar.low - prev_close),
        )

    def update(self, bar: Bar) -> Optional[float]:
        tr = self.true_range(bar, self._prev_close)
        self._prev_close = bar.close

        if self._value is None:
            self._seed.append(tr)
            if len(self._seed) < self.period:
                return None
            self._value = sum(self._seed) / self.period
            return self._value

        self._value = (self._value * (self.period - 1) + tr) / self.period
        return self._value

    @property
    def value(self) -> Optional[float]:
        return self._value

    def reset(self) -> None:
        self._value = None
        self._prev_close = None
        self._seed.clear()


class RSI(Indicator):
    """Relative Strength Index, Wilder-smoothed."""

    def __init__(self, period: int = 14) -> None:
        self.period = period
        self._prev_close: Optional[float] = None
        self._avg_gain: Optional[float] = None
        self._avg_loss: Optional[float] = None
        self._gains: Deque[float] = deque()
        self._losses: Deque[float] = deque()

    def update(self, bar: Bar) -> Optional[float]:
        if self._prev_close is None:
            self._prev_close = bar.close
            return None

        change = bar.close - self._prev_close
        self._prev_close = bar.close
        gain, loss = max(change, 0.0), max(-change, 0.0)

        if self._avg_gain is None:
            self._gains.append(gain)
            self._losses.append(loss)
            if len(self._gains) < self.period:
                return None
            self._avg_gain = sum(self._gains) / self.period
            self._avg_loss = sum(self._losses) / self.period
        else:
            self._avg_gain = (self._avg_gain * (self.period - 1) + gain) / self.period
            self._avg_loss = (self._avg_loss * (self.period - 1) + loss) / self.period
        return self.value

    @property
    def value(self) -> Optional[float]:
        if self._avg_gain is None or self._avg_loss is None:
            return None
        if self._avg_loss == 0:
            return 100.0
        rs = self._avg_gain / self._avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def reset(self) -> None:
        self._prev_close = self._avg_gain = self._avg_loss = None
        self._gains.clear()
        self._losses.clear()


class ADX(Indicator):
    """Average Directional Index — the regime filter (DESIGN.md 1.6).

    Measures trend *strength* regardless of direction. This is the gate that keeps the
    strategy out of chop: a pullback entry in a rangebound market is just buying a random
    dip, and paying the full round trip for the privilege.

    Standard Wilder construction: directional movement -> smoothed DI -> DX -> smoothed ADX.
    Needs roughly ``2 * period`` bars before it reads anything.
    """

    def __init__(self, period: int = 14) -> None:
        self.period = period
        self._prev_high: Optional[float] = None
        self._prev_low: Optional[float] = None
        self._prev_close: Optional[float] = None

        self._smooth_tr: Optional[float] = None
        self._smooth_plus: Optional[float] = None
        self._smooth_minus: Optional[float] = None
        self._seed_tr: Deque[float] = deque()
        self._seed_plus: Deque[float] = deque()
        self._seed_minus: Deque[float] = deque()

        self._adx: Optional[float] = None
        self._seed_dx: Deque[float] = deque()

    def update(self, bar: Bar) -> Optional[float]:
        if self._prev_high is None:
            self._prev_high, self._prev_low, self._prev_close = bar.high, bar.low, bar.close
            return None

        up_move = bar.high - self._prev_high
        down_move = self._prev_low - bar.low
        # Only the larger move counts, and only if it is positive — a bar that expands in
        # both directions carries no directional information.
        plus_dm = up_move if (up_move > down_move and up_move > 0) else 0.0
        minus_dm = down_move if (down_move > up_move and down_move > 0) else 0.0
        tr = ATR.true_range(bar, self._prev_close)

        self._prev_high, self._prev_low, self._prev_close = bar.high, bar.low, bar.close

        if self._smooth_tr is None:
            self._seed_tr.append(tr)
            self._seed_plus.append(plus_dm)
            self._seed_minus.append(minus_dm)
            if len(self._seed_tr) < self.period:
                return None
            self._smooth_tr = sum(self._seed_tr)
            self._smooth_plus = sum(self._seed_plus)
            self._smooth_minus = sum(self._seed_minus)
        else:
            # Wilder's running total: drop 1/n of the accumulated value, add the new one.
            self._smooth_tr = self._smooth_tr - self._smooth_tr / self.period + tr
            self._smooth_plus = self._smooth_plus - self._smooth_plus / self.period + plus_dm
            self._smooth_minus = self._smooth_minus - self._smooth_minus / self.period + minus_dm

        dx = self._dx()
        if dx is None:
            return None

        if self._adx is None:
            self._seed_dx.append(dx)
            if len(self._seed_dx) < self.period:
                return None
            self._adx = sum(self._seed_dx) / self.period
        else:
            self._adx = (self._adx * (self.period - 1) + dx) / self.period
        return self._adx

    def _dx(self) -> Optional[float]:
        if not self._smooth_tr:
            return None
        plus_di = 100.0 * (self._smooth_plus or 0.0) / self._smooth_tr
        minus_di = 100.0 * (self._smooth_minus or 0.0) / self._smooth_tr
        total = plus_di + minus_di
        if total == 0:
            return 0.0
        return 100.0 * abs(plus_di - minus_di) / total

    @property
    def plus_di(self) -> Optional[float]:
        if not self._smooth_tr:
            return None
        return 100.0 * (self._smooth_plus or 0.0) / self._smooth_tr

    @property
    def minus_di(self) -> Optional[float]:
        if not self._smooth_tr:
            return None
        return 100.0 * (self._smooth_minus or 0.0) / self._smooth_tr

    @property
    def value(self) -> Optional[float]:
        return self._adx

    def reset(self) -> None:
        self.__init__(self.period)


class OpeningRange(Indicator):
    """High and low of the first N minutes — a classic intraday reference level.

    Session-anchored: it locks once the window closes and stays fixed for the day.
    """

    def __init__(self, minutes: int = 15) -> None:
        self.minutes = minutes
        self._high: Optional[float] = None
        self._low: Optional[float] = None
        self._start: Optional[dt.datetime] = None
        self._locked = False

    def update(self, bar: Bar) -> Optional[float]:
        if self._locked or bar.synthetic:
            return self.value
        if self._start is None:
            self._start = bar.bar_start
        elapsed = (bar.bar_end - self._start).total_seconds() / 60.0
        if elapsed > self.minutes:
            self._locked = True
            return self.value
        self._high = bar.high if self._high is None else max(self._high, bar.high)
        self._low = bar.low if self._low is None else min(self._low, bar.low)
        return self.value

    @property
    def high(self) -> Optional[float]:
        return self._high

    @property
    def low(self) -> Optional[float]:
        return self._low

    @property
    def locked(self) -> bool:
        return self._locked

    @property
    def value(self) -> Optional[float]:
        """Midpoint of the range, or None until it has formed."""
        if self._high is None or self._low is None:
            return None
        return (self._high + self._low) / 2.0

    def reset(self) -> None:
        self._high = self._low = self._start = None
        self._locked = False


class RelativeVolume(Indicator):
    """Current bar volume against the session's running median.

    Guards against trading drift on no participation. A breakout on a tenth of normal
    volume is usually noise, and it will be expensive to exit.
    """

    def __init__(self, min_samples: int = 10) -> None:
        self.min_samples = min_samples
        self._volumes: list[float] = []
        self._value: Optional[float] = None

    def update(self, bar: Bar) -> Optional[float]:
        if bar.synthetic or bar.volume <= 0:
            return self._value
        if len(self._volumes) >= self.min_samples:
            ordered = sorted(self._volumes)
            median = ordered[len(ordered) // 2]
            self._value = bar.volume / median if median > 0 else None
        self._volumes.append(bar.volume)
        return self._value

    @property
    def value(self) -> Optional[float]:
        return self._value

    def reset(self) -> None:
        self._volumes.clear()
        self._value = None
