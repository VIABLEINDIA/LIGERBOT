"""SMA crossover — the backtester's **negative control**.

This is not a strategy to trade. It exists for one purpose: DESIGN.md 2.5 rule 5 requires
that a correct backtester show it **losing money after costs**. If the harness ever reports
this as profitable, the harness is wrong — the cost model, the fill model, or the
look-ahead guard — and nothing else it says can be believed until that is fixed.

Why it is expected to lose: a raw moving-average cross on short intraday bars is a
chop-maximising machine. It buys every false break and pays the full round trip on each.
With costs at ~11-15% of the amount risked (DESIGN.md 5.2), a strategy with no real edge
does not merely fail to profit — it bleeds at a predictable rate.

Unlike the original ``strategy_engine.py``, this version operates on **time bars** rather
than raw ticks (fixing B1) and emits a mandatory stop-loss (fixing B7), so it can actually
travel through the risk engine.
"""
from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Optional

from src.bars import Bar
from src.risk_engine import Intent, Signal
from src.strategy_base import Strategy, StrategyContext, register


@register
class SmaCrossover(Strategy):
    """Long-only SMA crossover with a fixed percentage stop."""

    name = "sma_crossover"
    version = "1"

    def __init__(
        self,
        short_period: int = 10,
        long_period: int = 50,
        stop_pct: float = 0.01,
        **extra,
    ) -> None:
        if short_period >= long_period:
            raise ValueError(
                f"short_period ({short_period}) must be below long_period ({long_period})"
            )
        super().__init__(short_period=short_period, long_period=long_period,
                         stop_pct=stop_pct, **extra)
        self.short_period = short_period
        self.long_period = long_period
        self.stop_pct = stop_pct
        # Running sums keep each update O(1). The original recomputed the whole window
        # on every tick, which is O(N) per instrument per update.
        self._windows: Dict[str, Deque[float]] = {}
        self._short_sum: Dict[str, float] = {}
        self._long_sum: Dict[str, float] = {}
        self._above: Dict[str, Optional[bool]] = {}

    @property
    def warmup_bars(self) -> int:
        return self.long_period

    def on_session_start(self, day) -> None:
        """Reset every session.

        Carrying an average across the overnight gap would blend two different regimes
        and produce a crossover that no intraday trader could have acted on.
        """
        self._windows.clear()
        self._short_sum.clear()
        self._long_sum.clear()
        self._above.clear()

    def _update(self, instrument_id: str, close: float) -> Optional[tuple[float, float]]:
        """Push a close and return (short_ma, long_ma) once both are available."""
        window = self._windows.setdefault(instrument_id, deque())
        self._short_sum.setdefault(instrument_id, 0.0)
        self._long_sum.setdefault(instrument_id, 0.0)
        self._above.setdefault(instrument_id, None)

        window.append(close)
        self._short_sum[instrument_id] += close
        self._long_sum[instrument_id] += close

        if len(window) > self.short_period:
            self._short_sum[instrument_id] -= window[-(self.short_period + 1)]
        if len(window) > self.long_period:
            self._long_sum[instrument_id] -= window.popleft()

        if len(window) < self.long_period:
            return None
        return (self._short_sum[instrument_id] / self.short_period,
                self._long_sum[instrument_id] / self.long_period)

    def on_bar(self, bar: Bar, ctx: StrategyContext) -> List[Signal]:
        averages = self._update(bar.instrument_id, bar.close)
        if averages is None:
            return []
        short_ma, long_ma = averages

        was_above = self._above[bar.instrument_id]
        is_above = short_ma > long_ma
        self._above[bar.instrument_id] = is_above

        # First reading establishes state only. Acting on it would be a level check,
        # not a crossover — the bug the original carried at startup.
        if was_above is None:
            return []
        if is_above == was_above:
            return []

        reason = (f"sma{self.short_period}={short_ma:.2f} "
                  f"{'crossed above' if is_above else 'crossed below'} "
                  f"sma{self.long_period}={long_ma:.2f}")

        if is_above and not ctx.in_position and ctx.allows_entry:
            return [self._signal(
                bar, Intent.OPEN_LONG,
                stop_loss=round(bar.close * (1.0 - self.stop_pct), 2),
                reason=reason,
            )]
        if not is_above and ctx.is_long:
            return [self._signal(bar, Intent.CLOSE_LONG, reason=reason)]
        return []
