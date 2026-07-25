"""Trend-pullback v1 — the reference strategy (DESIGN.md 1.6).

Long-only (D3). The thesis: establish a directional bias, wait for price to pull back
against it, and enter only when the pullback *fails* — i.e. price closes back in the
direction of the trend. Entering on pullbacks rather than on breakouts is the point;
a breakout entry pays the spread at every false break, which is exactly how the SMA
negative control bleeds.

Structure::

    bias      close > session VWAP  AND  EMA(fast) > EMA(slow)
    regime    ADX > adx_min         AND  atr_floor < ATR% < atr_ceiling
    entry     price pulled back to EMA(fast), then closed back above it
    exits     ATR stop (resting) + trail after +1R + time stop

**Session anchoring.** VWAP, the opening range and relative volume reset every session —
they are defined relative to the day's open. EMA, ATR and ADX deliberately *carry across*
sessions. Two reasons: an overnight gap is genuine volatility information that ATR should
see, and on 5-minute bars a daily-reset ADX(14) needs ~2h20m to warm up, which would
consume most of the tradable session before the strategy could act at all.

**On the exits.** Only the initial ATR stop is a resting order the broker holds. The trail
and time stops are evaluated at bar close, so they exit at the close rather than at the
trail level. That is looser than an exchange-resting trail — and deliberately so, since it
errs toward worse fills rather than better ones.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.bars import Bar
from src.indicators import ADX, ATR, EMA, RSI, RelativeVolume, SessionVWAP
from src.risk_engine import Intent, Signal
from src.strategy_base import Strategy, StrategyContext, register


@dataclass
class InstrumentState:
    """Per-instrument indicator set and pullback state machine."""

    ema_fast: EMA
    ema_slow: EMA
    atr: ATR
    adx: ADX
    rsi: RSI
    vwap: SessionVWAP = field(default_factory=SessionVWAP)
    rvol: RelativeVolume = field(default_factory=RelativeVolume)

    # Pullback state machine.
    pulled_back: bool = False
    # Position tracking, for the trail and time stops.
    bars_in_trade: int = 0
    peak_since_entry: Optional[float] = None
    trailing_armed: bool = False

    def reset_session(self) -> None:
        """Reset only what is genuinely session-scoped."""
        self.vwap.reset()
        self.rvol.reset()
        self.pulled_back = False

    def reset_trade(self) -> None:
        self.bars_in_trade = 0
        self.peak_since_entry = None
        self.trailing_armed = False


@register
class TrendPullback(Strategy):
    """Long-only intraday trend-pullback with ATR-based risk."""

    name = "trend_pullback"
    version = "1"

    def __init__(
        self,
        ema_fast: int = 9,
        ema_slow: int = 21,
        adx_period: int = 14,
        adx_min: float = 20.0,
        atr_period: int = 14,
        atr_mult: float = 3.0,
        min_stop_pct: float = 0.007,
        atr_ceiling_pct: float = 0.05,
        rvol_min: float = 0.8,
        rsi_max: float = 75.0,
        pullback_atr: float = 0.5,
        trail_after_r: float = 1.0,
        trail_atr_mult: float = 1.5,
        time_stop_bars: int = 20,
        time_stop_min_r: float = 0.5,
        **extra,
    ) -> None:
        if ema_fast >= ema_slow:
            raise ValueError(f"ema_fast ({ema_fast}) must be below ema_slow ({ema_slow})")
        if min_stop_pct >= atr_ceiling_pct:
            raise ValueError("min_stop_pct must be below atr_ceiling_pct")

        super().__init__(
            ema_fast=ema_fast, ema_slow=ema_slow, adx_period=adx_period,
            adx_min=adx_min, atr_period=atr_period, atr_mult=atr_mult,
            min_stop_pct=min_stop_pct, atr_ceiling_pct=atr_ceiling_pct,
            rvol_min=rvol_min, rsi_max=rsi_max, pullback_atr=pullback_atr,
            trail_after_r=trail_after_r, trail_atr_mult=trail_atr_mult,
            time_stop_bars=time_stop_bars, time_stop_min_r=time_stop_min_r, **extra,
        )
        self.ema_fast_period = ema_fast
        self.ema_slow_period = ema_slow
        self.adx_period = adx_period
        self.adx_min = adx_min
        self.atr_period = atr_period
        self.atr_mult = atr_mult
        self.min_stop_pct = min_stop_pct
        self.atr_ceiling_pct = atr_ceiling_pct
        self.rvol_min = rvol_min
        self.rsi_max = rsi_max
        self.pullback_atr = pullback_atr
        self.trail_after_r = trail_after_r
        self.trail_atr_mult = trail_atr_mult
        self.time_stop_bars = time_stop_bars
        self.time_stop_min_r = time_stop_min_r

        self._state: Dict[str, InstrumentState] = {}

    @property
    def warmup_bars(self) -> int:
        # ADX needs roughly two periods before it reads anything at all.
        return max(self.ema_slow_period, self.atr_period, self.adx_period * 2)

    def _state_for(self, instrument_id: str) -> InstrumentState:
        if instrument_id not in self._state:
            self._state[instrument_id] = InstrumentState(
                ema_fast=EMA(self.ema_fast_period),
                ema_slow=EMA(self.ema_slow_period),
                atr=ATR(self.atr_period),
                adx=ADX(self.adx_period),
                rsi=RSI(14),
            )
        return self._state[instrument_id]

    def on_session_start(self, day: dt.date) -> None:
        for state in self._state.values():
            state.reset_session()
            state.reset_trade()

    # -- filters -----------------------------------------------------------
    def _regime_ok(self, state: InstrumentState, price: float) -> tuple[bool, str]:
        """Is there a trend worth trading, at a volatility that can be sized economically?

        The ``min_stop_pct`` check is the non-obvious one, and it is an economic
        constraint rather than a technical one. Risk-based sizing sets
        ``notional = risk_budget / stop_distance``, so a *tighter* stop means a *larger*
        position — while friction scales with notional and the risk budget does not.

        The relationship falls straight out of the cost structure (DESIGN.md 5.2, where
        round-trip friction is ~0.085% of notional)::

            friction_in_R  ~=  0.00085 / stop_pct

        A 0.4% stop therefore costs 0.21R per trade; a 0.8% stop costs 0.11R. To stay
        under the ~0.12R hurdle the stop must be at least **~0.7% of price**, and that is
        where the default comes from — derived, not chosen.

        This has a sharp consequence for bar interval (D4): 5-minute ATR on a liquid
        large-cap runs around 0.2% of price, so a 1.5x ATR stop lands near 0.3% and is
        structurally uneconomic. Either the multiple widens or the bars must be coarser.
        """
        adx = state.adx.value
        atr = state.atr.value
        if adx is None or atr is None or price <= 0:
            return False, "warming up"
        if adx < self.adx_min:
            return False, f"adx {adx:.1f} < {self.adx_min:.0f} (chop)"

        stop_pct = (self.atr_mult * atr) / price
        if stop_pct < self.min_stop_pct:
            return False, (f"stop {stop_pct:.3%} below the {self.min_stop_pct:.2%} "
                           f"economic floor — friction would dominate")
        if atr / price > self.atr_ceiling_pct:
            return False, f"atr {atr / price:.3%} above ceiling (too wild)"
        return True, f"adx {adx:.1f} stop {stop_pct:.2%}"

    def _bias_is_long(self, state: InstrumentState, bar: Bar) -> bool:
        fast, slow, vwap = state.ema_fast.value, state.ema_slow.value, state.vwap.value
        if fast is None or slow is None or vwap is None:
            return False
        return fast > slow and bar.close > vwap

    # -- exits -------------------------------------------------------------
    def _exit_signal(
        self, bar: Bar, ctx: StrategyContext, state: InstrumentState
    ) -> Optional[Signal]:
        position = ctx.position
        if position is None:
            return None

        risk_per_share = abs(position.entry_price - position.stop_loss)
        if risk_per_share <= 0:
            return None
        r_multiple = (bar.close - position.entry_price) / risk_per_share

        # 1. Bias has broken down — the reason for the trade is gone.
        if state.ema_fast.value and state.ema_slow.value:
            if state.ema_fast.value < state.ema_slow.value:
                return self._signal(bar, Intent.CLOSE_LONG,
                                    reason=f"bias lost (ema cross down) at {r_multiple:+.2f}R")

        # 2. Trailing stop, armed once the trade has paid for itself.
        if r_multiple >= self.trail_after_r:
            state.trailing_armed = True
        if state.trailing_armed:
            state.peak_since_entry = max(state.peak_since_entry or bar.high, bar.high)
            atr = state.atr.value or 0.0
            trail_level = state.peak_since_entry - self.trail_atr_mult * atr
            if bar.close <= trail_level:
                return self._signal(
                    bar, Intent.CLOSE_LONG,
                    reason=f"trailing stop at {r_multiple:+.2f}R "
                           f"(peak {state.peak_since_entry:.2f})")

        # 3. Time stop. Intraday capital is finite; a position that has not moved is
        #    occupying a slot and a share of the open-risk budget for nothing.
        if (state.bars_in_trade >= self.time_stop_bars
                and r_multiple < self.time_stop_min_r):
            return self._signal(
                bar, Intent.CLOSE_LONG,
                reason=f"time stop after {state.bars_in_trade} bars at {r_multiple:+.2f}R")
        return None

    # -- main --------------------------------------------------------------
    def on_bar(self, bar: Bar, ctx: StrategyContext) -> List[Signal]:
        state = self._state_for(bar.instrument_id)

        # Synthetic bars carry no information — updating indicators with a fabricated
        # flat price would understate volatility and fake a calm market.
        if bar.synthetic:
            return []

        state.ema_fast.update(bar)
        state.ema_slow.update(bar)
        state.atr.update(bar)
        state.adx.update(bar)
        state.rsi.update(bar)
        state.vwap.update(bar)
        state.rvol.update(bar)

        if ctx.in_position:
            state.bars_in_trade += 1
            exit_signal = self._exit_signal(bar, ctx, state)
            if exit_signal is not None:
                state.reset_trade()
                state.pulled_back = False
                return [exit_signal]
            return []

        state.reset_trade()

        if not ctx.allows_entry:
            return []

        bias_long = self._bias_is_long(state, bar)
        if not bias_long:
            state.pulled_back = False
            return []

        fast = state.ema_fast.value
        atr = state.atr.value
        if fast is None or atr is None:
            return []

        # Pullback detection: price must first trade down into the fast EMA...
        if bar.low <= fast + self.pullback_atr * atr:
            state.pulled_back = True

        # ...and then close back above it. A pullback that keeps going is not a setup.
        if not (state.pulled_back and bar.close > fast):
            return []

        regime_ok, regime_note = self._regime_ok(state, bar.close)
        if not regime_ok:
            return []

        rvol = state.rvol.value
        if rvol is not None and rvol < self.rvol_min:
            return []

        rsi = state.rsi.value
        if rsi is not None and rsi > self.rsi_max:
            return []  # already extended; a pullback entry here is chasing

        stop = round(bar.close - self.atr_mult * atr, 2)
        if stop >= bar.close:
            return []

        state.pulled_back = False
        return [self._signal(
            bar, Intent.OPEN_LONG,
            stop_loss=stop,
            reason=f"pullback held above ema{self.ema_fast_period} | {regime_note}",
        )]
