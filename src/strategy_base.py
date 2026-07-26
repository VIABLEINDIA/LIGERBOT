"""The ``Strategy`` interface.

The core rule of DESIGN.md 1.1: **the same strategy object runs unchanged in backtest,
paper and live.** Only what feeds it bars differs. A strategy reimplemented for
backtesting is a strategy that will drift from the live one, and every result it produced
becomes a fiction.

Two boundaries are enforced by the interface itself:

  * **Strategies decide direction and levels, never size.** ``StrategyContext`` exposes no
    account equity and no broker. Sizing is the risk engine's job and stays there, so a
    strategy cannot accidentally take more risk than the caps allow.
  * **Strategies see only closed bars.** The engine never passes a forming bar, so a
    strategy physically cannot act on information that wasn't available at decision time.

A minimal registry lives here too, so the backtester, paper runner and live engine all
select strategies the same way and every signal records which exact configuration
produced it.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type

from src.bars import Bar
from src.risk_engine import Intent, Position, Signal


@dataclass(frozen=True)
class StrategyContext:
    """What a strategy is allowed to know when deciding.

    Deliberately narrow. No equity, no margin, no broker handle — a strategy that could
    see those would eventually be written to size itself, which is precisely the coupling
    the risk engine exists to prevent.
    """

    now: dt.datetime
    position: Optional[Position] = None
    seconds_to_square_off: float = 0.0
    allows_entry: bool = True
    session_day: Optional[dt.date] = None

    @property
    def in_position(self) -> bool:
        return self.position is not None

    @property
    def is_long(self) -> bool:
        return self.position is not None and self.position.quantity > 0


class Strategy(ABC):
    """Base class for every strategy."""

    name: str = "unnamed"
    version: str = "0"

    def __init__(self, **params: Any) -> None:
        self.params: Dict[str, Any] = dict(params)

    # -- identity ----------------------------------------------------------
    @property
    def params_hash(self) -> str:
        """Stable digest of the parameter set.

        Stamped onto every signal so any trade in the archive is traceable to the exact
        configuration that produced it (DESIGN.md 1.4).
        """
        blob = json.dumps(self.params, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:10]

    @property
    def warmup_bars(self) -> int:
        """Bars required before the strategy's output should be trusted."""
        return 0

    # -- interval feasibility ----------------------------------------------
    def bars_per_session(self, bar_seconds: int) -> int:
        """How many bars of this size an NSE session contains."""
        from src import market_calendar as cal

        span = (dt.datetime.combine(dt.date(2000, 1, 1), cal.SESSION_CLOSE)
                - dt.datetime.combine(dt.date(2000, 1, 1), cal.SESSION_OPEN))
        return int(span.total_seconds() // max(1, bar_seconds))

    def usable_bars_per_session(self, bar_seconds: int) -> int:
        """Bars left after warmup, per session. Negative means never."""
        return self.bars_per_session(bar_seconds) - self.warmup_bars

    def check_interval(self, bar_seconds: int) -> tuple[bool, str]:
        """Can this strategy trade at all on bars of this size?

        A real defect this exists to prevent, found by a backtest on live data reporting
        zero trades. **Session-anchored indicators reset in `on_session_start`**, so
        warmup must fit *inside a single session* — it does not accumulate across days.
        At 15-minute bars an NSE session holds 25 bars; `trend_pullback` needs 28 and
        `sma_crossover` needs 50. Both are then **structurally incapable of ever
        trading**, and nothing said so: the bot runs all day, logs normally, passes every
        health check, and takes no trades.

        That is the same silent-failure shape as the feed that reconnects forever and the
        consumer that falls behind — healthy-looking and useless. It gets a check.
        """
        available = self.bars_per_session(bar_seconds)
        needed = self.warmup_bars
        if needed <= 0:
            return True, ""
        if available <= needed:
            return False, (
                f"{self.name} needs {needed} bars of warmup but a session holds only "
                f"{available} at {bar_seconds}s bars, and indicators reset each session "
                f"— it can NEVER trade at this interval. Use a finer interval "
                f"(<= {self._max_interval_seconds()}s) or reduce the warmup.")
        # A strategy that spends most of the day warming up is technically able to trade
        # and practically crippled, which is worth saying out loud rather than leaving to
        # be inferred from a thin trade count.
        if available - needed < available * 0.25:
            return True, (
                f"{self.name} spends {needed} of {available} bars warming up at "
                f"{bar_seconds}s — only {available - needed} usable bars per session.")
        return True, ""

    def _max_interval_seconds(self) -> int:
        """Coarsest interval at which warmup still leaves room to trade."""
        from src import market_calendar as cal

        span = (dt.datetime.combine(dt.date(2000, 1, 1), cal.SESSION_CLOSE)
                - dt.datetime.combine(dt.date(2000, 1, 1), cal.SESSION_OPEN))
        if self.warmup_bars <= 0:
            return int(span.total_seconds())
        return int(span.total_seconds() // (self.warmup_bars + 1))

    def describe(self) -> str:
        items = " ".join(f"{k}={v}" for k, v in sorted(self.params.items()))
        return f"{self.name} v{self.version} [{items}] ({self.params_hash})"

    # -- lifecycle ---------------------------------------------------------
    def on_session_start(self, day: dt.date) -> None:
        """Reset per-session state. Session-anchored indicators rebuild from here."""

    def on_session_end(self, day: dt.date, ctx: StrategyContext) -> List[Signal]:
        """Last chance to emit exits before the forced flat.

        The engine flattens anything still open regardless; this hook exists so a
        strategy can exit on its own terms first.
        """
        return []

    @abstractmethod
    def on_bar(self, bar: Bar, ctx: StrategyContext) -> List[Signal]:
        """React to one **closed** bar. Return zero or more signals."""

    # -- helpers -----------------------------------------------------------
    def _signal(
        self,
        bar: Bar,
        intent: Intent,
        *,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        reason: str = "",
    ) -> Signal:
        """Build a signal stamped with this strategy's provenance."""
        return Signal(
            instrument_id=bar.instrument_id,
            intent=intent,
            ref_price=bar.close,
            bar_time=bar.bar_end,
            stop_loss=stop_loss,
            take_profit=take_profit,
            strategy_name=self.name,
            strategy_version=self.version,
            params_hash=self.params_hash,
            reason=reason,
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
_REGISTRY: Dict[str, Type[Strategy]] = {}


def register(cls: Type[Strategy]) -> Type[Strategy]:
    """Class decorator that makes a strategy selectable by name."""
    _REGISTRY[cls.name] = cls
    return cls


def create(name: str, **params: Any) -> Strategy:
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown strategy {name!r}. Registered: {sorted(_REGISTRY) or '(none)'}"
        )
    return _REGISTRY[name](**params)


def available() -> List[str]:
    return sorted(_REGISTRY)
