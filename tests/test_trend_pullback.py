"""Trend-pullback strategy tests.

These are *behavioural* tests: given a hand-built sequence, does the strategy fire when it
should and stay out when it should not. They deliberately do not test profitability —
synthetic data cannot establish that, and a test that asserted it would only be pinning
the generator's quirks.

:class:`TestEntryFiresOnAConstructedSetup` is the unit-level positive control. It answers a
question the aggregate backtests cannot: when a textbook trend-pullback is placed directly
in front of the strategy, does it take it? Without that, a weak result on generated data is
ambiguous between "the strategy is blind" and "the data has nothing to find".
"""
from __future__ import annotations

import datetime as dt

import pytest

from src import market_calendar as cal
from src.bars import Bar
from src.risk_engine import Intent, Position
from src.strategies.trend_pullback import TrendPullback
from src.strategy_base import StrategyContext

DAY = dt.date(2026, 7, 23)


def bar(index: int, close: float, *, high=None, low=None, open_=None,
        volume: float = 10_000.0) -> Bar:
    """One 5-minute bar, indexed from the session open."""
    start = cal.at(DAY, cal.SESSION_OPEN) + dt.timedelta(minutes=5 * index)
    return Bar(
        "nse_cm:1", start, start + dt.timedelta(minutes=5),
        open=open_ if open_ is not None else close,
        high=high if high is not None else close + 1.0,
        low=low if low is not None else close - 1.0,
        close=close, volume=volume, vwap=close, tick_count=200,
    )


def ctx(*, position: Position | None = None, allows_entry: bool = True) -> StrategyContext:
    return StrategyContext(
        now=cal.at(DAY, dt.time(11, 0)), position=position,
        seconds_to_square_off=10_000.0, allows_entry=allows_entry, session_day=DAY,
    )


def strategy(**overrides) -> TrendPullback:
    defaults = dict(ema_fast=5, ema_slow=10, adx_period=5, adx_min=15.0,
                    atr_period=5, atr_mult=3.0, min_stop_pct=0.004,
                    rvol_min=0.0, rsi_max=100.0, pullback_atr=0.5)
    defaults.update(overrides)
    strat = TrendPullback(**defaults)
    strat.on_session_start(DAY)
    return strat


def drive(strat: TrendPullback, prices, *, context=None, start=0):
    """Feed closes and return every signal emitted, with its bar index."""
    signals = []
    for offset, price in enumerate(prices):
        emitted = strat.on_bar(bar(start + offset, price),
                               context or ctx())
        for signal in emitted:
            signals.append((start + offset, signal))
    return signals


class TestEntryFiresOnAConstructedSetup:
    """Unit-level positive control — the strategy must take a textbook setup."""

    def test_uptrend_then_pullback_then_recovery_triggers_a_long(self):
        strat = strategy()
        # 1. A clean, sustained uptrend: builds bias, ADX and ATR.
        uptrend = [100.0 + i * 1.5 for i in range(40)]
        signals = drive(strat, uptrend)
        assert not signals, "should not enter mid-trend without a pullback"

        # 2. A pullback into the fast EMA.
        peak = uptrend[-1]
        pullback = [peak - 3.0, peak - 6.0, peak - 8.0]
        signals = drive(strat, pullback, start=40)
        assert not signals, "should not enter while price is still falling"

        # 3. Price closes back above the fast EMA — the pullback failed.
        recovery = [peak - 2.0, peak + 1.0, peak + 3.0]
        signals = drive(strat, recovery, start=43)

        opens = [s for _, s in signals if s.intent is Intent.OPEN_LONG]
        assert opens, "constructed trend-pullback setup produced no entry"
        signal = opens[0]
        assert signal.stop_loss is not None and signal.stop_loss < signal.ref_price
        assert "pullback held" in signal.reason

    def test_entry_stop_is_atr_based_and_below_entry(self):
        strat = strategy()
        drive(strat, [100.0 + i * 1.5 for i in range(40)])
        peak = 100.0 + 39 * 1.5
        drive(strat, [peak - 3.0, peak - 6.0, peak - 8.0], start=40)
        signals = drive(strat, [peak - 2.0, peak + 1.0], start=43)
        opens = [s for _, s in signals if s.intent is Intent.OPEN_LONG]
        assert opens
        distance = opens[0].ref_price - opens[0].stop_loss
        assert distance > 0
        # Stop distance should be a meaningful fraction of price, not a token amount.
        assert distance / opens[0].ref_price >= strat.min_stop_pct * 0.9


class TestFiltersBlockEntries:
    def test_no_entry_in_chop(self):
        """The regime filter is the whole reason this is not a breakout system."""
        strat = strategy(adx_min=30.0)
        prices = [100.0 + (2.0 if i % 2 else -2.0) for i in range(80)]
        assert not [s for _, s in drive(strat, prices) if s.intent is Intent.OPEN_LONG]

    def test_no_entry_when_the_stop_would_be_uneconomic(self):
        """A stop too tight to be economic is refused, however good the setup looks.

        friction_in_R ~= 0.00085 / stop_pct, so a very tight stop guarantees the trade
        loses to costs regardless of whether the signal was right.
        """
        # A floor no ATR stop in this fixture can reach (its stops land near 7%).
        strat = strategy(min_stop_pct=0.15, atr_ceiling_pct=0.5)
        drive(strat, [100.0 + i * 1.5 for i in range(40)])
        peak = 100.0 + 39 * 1.5
        drive(strat, [peak - 3.0, peak - 6.0, peak - 8.0], start=40)
        signals = drive(strat, [peak - 2.0, peak + 1.0], start=43)
        assert not [s for _, s in signals if s.intent is Intent.OPEN_LONG]

    def test_no_entry_below_session_vwap(self):
        strat = strategy()
        # A sustained downtrend: price stays below VWAP throughout.
        assert not [s for _, s in drive(strat, [200.0 - i * 1.5 for i in range(60)])
                    if s.intent is Intent.OPEN_LONG]

    def test_no_entry_when_the_phase_forbids_it(self):
        strat = strategy()
        blocked = ctx(allows_entry=False)
        prices = [100.0 + i * 1.5 for i in range(40)] + [140.0, 130.0, 145.0]
        signals = drive(strat, prices, context=blocked)
        assert not [s for _, s in signals if s.intent is Intent.OPEN_LONG]

    def test_low_relative_volume_blocks_entry(self):
        strat = strategy(rvol_min=5.0)  # demand 5x median volume
        drive(strat, [100.0 + i * 1.5 for i in range(40)])
        peak = 100.0 + 39 * 1.5
        drive(strat, [peak - 3.0, peak - 6.0], start=40)
        signals = drive(strat, [peak + 1.0], start=42)
        assert not [s for _, s in signals if s.intent is Intent.OPEN_LONG]

    def test_overbought_rsi_blocks_chasing(self):
        strat = strategy(rsi_max=10.0)  # anything counts as overbought
        drive(strat, [100.0 + i * 1.5 for i in range(40)])
        peak = 100.0 + 39 * 1.5
        drive(strat, [peak - 3.0, peak - 6.0], start=40)
        signals = drive(strat, [peak + 1.0], start=42)
        assert not [s for _, s in signals if s.intent is Intent.OPEN_LONG]

    def test_synthetic_bars_are_ignored_entirely(self):
        """Updating indicators from fabricated flat prices would fake a calm market."""
        strat = strategy()
        synthetic = Bar("nse_cm:1", cal.at(DAY, dt.time(11, 0)),
                        cal.at(DAY, dt.time(11, 5)), 100.0, 100.0, 100.0, 100.0,
                        volume=0.0, vwap=100.0, tick_count=0, synthetic=True)
        assert strat.on_bar(synthetic, ctx()) == []
        assert strat._state_for("nse_cm:1").atr.value is None


class TestExits:
    def _in_trade(self, strat: TrendPullback, entry: float, stop: float) -> Position:
        return Position(instrument_id="nse_cm:1", quantity=100,
                        entry_price=entry, stop_loss=stop)

    def test_bias_loss_closes_the_position(self):
        strat = strategy()
        drive(strat, [100.0 + i * 1.5 for i in range(40)])
        position = self._in_trade(strat, 158.5, 150.0)
        # A sharp reversal flips the EMAs.
        signals = drive(strat, [140.0 - i * 3.0 for i in range(20)],
                        context=ctx(position=position), start=40)
        closes = [s for _, s in signals if s.intent is Intent.CLOSE_LONG]
        assert closes
        assert "bias lost" in closes[0].reason

    def test_time_stop_fires_on_a_stalled_trade(self):
        strat = strategy(time_stop_bars=5, time_stop_min_r=0.5)
        drive(strat, [100.0 + i * 1.5 for i in range(40)])
        # Entry at 158.5 with a 8.5-point risk; price goes nowhere.
        position = self._in_trade(strat, 158.5, 150.0)
        signals = drive(strat, [158.6] * 10, context=ctx(position=position), start=40)
        closes = [s for _, s in signals if s.intent is Intent.CLOSE_LONG]
        assert closes
        assert "time stop" in closes[0].reason

    def test_time_stop_does_not_fire_on_a_working_trade(self):
        strat = strategy(time_stop_bars=5, time_stop_min_r=0.5)
        drive(strat, [100.0 + i * 1.5 for i in range(40)])
        position = self._in_trade(strat, 158.5, 150.0)
        # Up 2R — well past the time stop's profit threshold.
        signals = drive(strat, [175.5 + i for i in range(10)],
                        context=ctx(position=position), start=40)
        assert not [s for _, s in signals if "time stop" in s.reason]

    def test_trailing_stop_arms_only_after_the_trade_pays_for_itself(self):
        strat = strategy(trail_after_r=1.0, trail_atr_mult=1.0, time_stop_bars=999)
        drive(strat, [100.0 + i * 1.5 for i in range(40)])
        position = self._in_trade(strat, 158.5, 150.0)  # 8.5 risk per share

        # Rally past +1R to arm the trail, then give it all back.
        drive(strat, [170.0, 175.0, 180.0], context=ctx(position=position), start=40)
        assert strat._state_for("nse_cm:1").trailing_armed

        signals = drive(strat, [160.0], context=ctx(position=position), start=43)
        closes = [s for _, s in signals if s.intent is Intent.CLOSE_LONG]
        assert closes
        assert "trailing stop" in closes[0].reason

    def test_no_exit_signal_when_flat(self):
        strat = strategy()
        assert not drive(strat, [100.0 + i * 1.5 for i in range(40)])


class TestSessionHandling:
    def test_vwap_resets_but_trend_indicators_carry_over(self):
        """Session anchoring is per-indicator and deliberate.

        A daily-reset ADX(14) on 5-minute bars would need ~2h20m to warm up, consuming
        most of the tradable session before the strategy could act at all.
        """
        strat = strategy()
        drive(strat, [100.0 + i * 1.5 for i in range(40)])
        state = strat._state_for("nse_cm:1")
        assert state.vwap.value is not None
        atr_before = state.atr.value

        strat.on_session_start(dt.date(2026, 7, 24))
        assert state.vwap.value is None          # session-scoped: reset
        assert state.atr.value == atr_before     # multi-day: carried

    def test_trade_state_resets_between_sessions(self):
        strat = strategy()
        state = strat._state_for("nse_cm:1")
        state.bars_in_trade = 10
        state.trailing_armed = True
        strat.on_session_start(dt.date(2026, 7, 24))
        assert state.bars_in_trade == 0
        assert not state.trailing_armed


class TestConfiguration:
    def test_rejects_inverted_emas(self):
        with pytest.raises(ValueError, match="ema_fast"):
            TrendPullback(ema_fast=21, ema_slow=9)

    def test_rejects_inverted_volatility_bounds(self):
        with pytest.raises(ValueError, match="min_stop_pct"):
            TrendPullback(min_stop_pct=0.5, atr_ceiling_pct=0.01)

    def test_warmup_accounts_for_adx(self):
        # ADX needs roughly two periods before it reads anything at all.
        assert TrendPullback(adx_period=14, ema_slow=21).warmup_bars >= 28

    def test_params_hash_is_stable_and_sensitive(self):
        assert TrendPullback().params_hash == TrendPullback().params_hash
        assert TrendPullback().params_hash != TrendPullback(adx_min=25.0).params_hash

    def test_is_registered(self):
        from src.strategy_base import available, create
        assert "trend_pullback" in available()
        assert isinstance(create("trend_pullback"), TrendPullback)
