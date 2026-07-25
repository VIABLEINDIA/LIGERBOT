"""Indicator tests against hand-computed reference values.

Required by DESIGN.md 3.10. An indicator that is subtly wrong produces a strategy that
looks plausible and is not, which is far worse than one that crashes — so every formula
here is pinned to a value that can be checked by hand, not merely to its own past output.
"""
from __future__ import annotations

import datetime as dt

import pytest

from src import market_calendar as cal
from src.bars import Bar
from src.indicators import (
    ADX, ATR, EMA, RSI, SMA, OpeningRange, RelativeVolume, SessionVWAP,
)

DAY = dt.date(2026, 7, 23)


def bar(minute: int, close: float, *, high=None, low=None, open_=None,
        volume: float = 1000.0, vwap: float = 0.0, synthetic: bool = False) -> Bar:
    start = cal.at(DAY, cal.SESSION_OPEN) + dt.timedelta(minutes=minute)
    return Bar(
        "nse_cm:1", start, start + dt.timedelta(minutes=1),
        open=open_ if open_ is not None else close,
        high=high if high is not None else close,
        low=low if low is not None else close,
        close=close, volume=volume, vwap=vwap or close,
        tick_count=10, synthetic=synthetic,
    )


def feed(indicator, closes, **kw):
    out = []
    for i, close in enumerate(closes):
        out.append(indicator.update(bar(i, close, **kw)))
    return out


class TestSMA:
    def test_returns_none_until_warm(self):
        sma = SMA(3)
        assert feed(sma, [1.0, 2.0])[-1] is None
        assert sma.update(bar(2, 3.0)) == pytest.approx(2.0)

    def test_hand_computed(self):
        sma = SMA(3)
        values = feed(sma, [1.0, 2.0, 3.0, 4.0, 5.0])
        assert values[2] == pytest.approx(2.0)   # (1+2+3)/3
        assert values[3] == pytest.approx(3.0)   # (2+3+4)/3
        assert values[4] == pytest.approx(4.0)   # (3+4+5)/3

    def test_window_drops_oldest(self):
        sma = SMA(2)
        feed(sma, [100.0, 100.0, 1.0])
        assert sma.value == pytest.approx(50.5)

    def test_rejects_bad_period(self):
        with pytest.raises(ValueError):
            SMA(0)


class TestEMA:
    def test_seeded_with_sma_then_recursive(self):
        """alpha = 2/(3+1) = 0.5; seed = mean(1,2,3) = 2."""
        ema = EMA(3)
        values = feed(ema, [1.0, 2.0, 3.0, 4.0, 5.0])
        assert values[0] is None
        assert values[1] is None
        assert values[2] == pytest.approx(2.0)            # seed
        assert values[3] == pytest.approx(3.0)            # 2 + 0.5*(4-2)
        assert values[4] == pytest.approx(4.0)            # 3 + 0.5*(5-3)

    def test_alpha_matches_the_standard_formula(self):
        assert EMA(9).alpha == pytest.approx(0.2)

    def test_converges_toward_a_constant(self):
        ema = EMA(5)
        feed(ema, [100.0] * 50)
        assert ema.value == pytest.approx(100.0)

    def test_seeding_from_sma_avoids_first_value_distortion(self):
        """A single-value seed would leave the first bar dominating for many periods."""
        ema = EMA(4)
        feed(ema, [1000.0, 10.0, 10.0, 10.0])
        # Seed is the mean of the four, not anchored to the 1000.
        assert ema.value == pytest.approx(257.5)


class TestATR:
    def test_constant_range_gives_that_range(self):
        atr = ATR(3)
        bars = [bar(i, 100.0, high=105.0, low=95.0) for i in range(5)]
        for b in bars:
            atr.update(b)
        assert atr.value == pytest.approx(10.0)

    def test_wilder_smoothing_is_hand_computable(self):
        """Seed = mean of first 3 TRs, then prev*(n-1)/n + tr/n."""
        atr = ATR(3)
        for i in range(3):
            atr.update(bar(i, 100.0, high=105.0, low=95.0))   # TR = 10 each
        assert atr.value == pytest.approx(10.0)
        # A bar with TR = 20 (range 90..110, prev close 100).
        atr.update(bar(3, 100.0, high=110.0, low=90.0))
        assert atr.value == pytest.approx((10.0 * 2 + 20.0) / 3)

    def test_true_range_uses_the_previous_close_on_a_gap(self):
        # Gap up: prev close 100, this bar 120..125. TR = |125 - 100| = 25.
        assert ATR.true_range(bar(1, 122.0, high=125.0, low=120.0), 100.0) == pytest.approx(25.0)

    def test_true_range_without_a_previous_close_is_the_bar_range(self):
        assert ATR.true_range(bar(0, 100.0, high=105.0, low=95.0), None) == pytest.approx(10.0)

    def test_none_until_warm(self):
        atr = ATR(5)
        for i in range(4):
            assert atr.update(bar(i, 100.0, high=101.0, low=99.0)) is None
        assert atr.update(bar(4, 100.0, high=101.0, low=99.0)) is not None


class TestRSI:
    def test_monotonic_rise_gives_100(self):
        rsi = RSI(3)
        feed(rsi, [10.0, 11.0, 12.0, 13.0, 14.0])
        assert rsi.value == pytest.approx(100.0)

    def test_monotonic_fall_gives_0(self):
        rsi = RSI(3)
        feed(rsi, [14.0, 13.0, 12.0, 11.0, 10.0])
        assert rsi.value == pytest.approx(0.0)

    def test_balanced_moves_sit_near_50(self):
        rsi = RSI(14)
        prices = [10.0 + (1.0 if i % 2 else 0.0) for i in range(60)]
        feed(rsi, prices)
        assert 40.0 < rsi.value < 60.0

    def test_bounded_between_0_and_100(self):
        rsi = RSI(14)
        values = [v for v in feed(rsi, [10.0, 12.0, 11.0, 15.0, 9.0, 20.0, 8.0] * 6)
                  if v is not None]
        assert values
        assert all(0.0 <= v <= 100.0 for v in values)


class TestADX:
    def test_strong_uptrend_reads_high(self):
        """The regime gate must actually detect a trend."""
        adx = ADX(5)
        for i in range(40):
            price = 100.0 + i * 2.0
            adx.update(bar(i, price, high=price + 0.5, low=price - 0.5))
        assert adx.value is not None
        assert adx.value > 40.0

    def test_strong_downtrend_also_reads_high(self):
        """ADX measures strength, not direction."""
        adx = ADX(5)
        for i in range(40):
            price = 200.0 - i * 2.0
            adx.update(bar(i, price, high=price + 0.5, low=price - 0.5))
        assert adx.value > 40.0

    def test_chop_reads_low(self):
        # The case the filter exists to suppress: a market going nowhere.
        adx = ADX(5)
        for i in range(60):
            price = 100.0 + (1.0 if i % 2 else -1.0)
            adx.update(bar(i, price, high=price + 0.5, low=price - 0.5))
        assert adx.value is not None
        assert adx.value < 25.0

    def test_directional_indicators_point_the_right_way(self):
        rising = ADX(5)
        for i in range(40):
            price = 100.0 + i * 2.0
            rising.update(bar(i, price, high=price + 0.5, low=price - 0.5))
        assert rising.plus_di > rising.minus_di

        falling = ADX(5)
        for i in range(40):
            price = 200.0 - i * 2.0
            falling.update(bar(i, price, high=price + 0.5, low=price - 0.5))
        assert falling.minus_di > falling.plus_di

    def test_needs_roughly_two_periods_to_warm(self):
        adx = ADX(14)
        for i in range(20):
            assert adx.update(bar(i, 100.0 + i, high=100.5 + i, low=99.5 + i)) is None

    def test_bounded_between_0_and_100(self):
        adx = ADX(7)
        values = []
        for i in range(200):
            price = 100.0 + (i % 13) - 6
            v = adx.update(bar(i, price, high=price + 1, low=price - 1))
            if v is not None:
                values.append(v)
        assert values
        assert all(0.0 <= v <= 100.0 for v in values)


class TestSessionVWAP:
    def test_volume_weighted_hand_computed(self):
        vwap = SessionVWAP()
        vwap.update(bar(0, 100.0, vwap=100.0, volume=100.0))
        vwap.update(bar(1, 200.0, vwap=200.0, volume=300.0))
        assert vwap.value == pytest.approx((100 * 100 + 200 * 300) / 400)

    def test_synthetic_bars_are_excluded(self):
        """A gap-filler represents no trading; folding it in drags VWAP to a dead price."""
        vwap = SessionVWAP()
        vwap.update(bar(0, 100.0, vwap=100.0, volume=100.0))
        vwap.update(bar(1, 999.0, vwap=999.0, volume=1000.0, synthetic=True))
        assert vwap.value == pytest.approx(100.0)

    def test_falls_back_to_typical_price_without_volume(self):
        vwap = SessionVWAP()
        vwap.update(bar(0, 100.0, high=110.0, low=90.0, volume=0.0, vwap=0.0))
        assert vwap.value == pytest.approx(100.0)

    def test_reset_clears_the_session(self):
        vwap = SessionVWAP()
        vwap.update(bar(0, 100.0, volume=100.0))
        vwap.reset()
        assert vwap.value is None


class TestOpeningRange:
    def test_captures_the_first_window_only(self):
        opening = OpeningRange(minutes=15)
        for i in range(15):
            opening.update(bar(i, 100.0, high=100.0 + i, low=100.0 - i))
        assert opening.high == pytest.approx(114.0)
        assert opening.low == pytest.approx(86.0)

        # A later bar outside the window must not extend it.
        opening.update(bar(30, 500.0, high=500.0, low=50.0))
        assert opening.high == pytest.approx(114.0)
        assert opening.low == pytest.approx(86.0)
        assert opening.locked

    def test_midpoint_is_the_value(self):
        opening = OpeningRange(minutes=5)
        opening.update(bar(0, 100.0, high=110.0, low=90.0))
        assert opening.value == pytest.approx(100.0)

    def test_none_before_any_bar(self):
        assert OpeningRange().value is None


class TestRelativeVolume:
    def test_ratio_against_the_running_median(self):
        rvol = RelativeVolume(min_samples=3)
        for i in range(3):
            rvol.update(bar(i, 100.0, volume=1000.0))
        rvol.update(bar(3, 100.0, volume=3000.0))
        assert rvol.value == pytest.approx(3.0)

    def test_none_before_enough_samples(self):
        rvol = RelativeVolume(min_samples=5)
        for i in range(3):
            assert rvol.update(bar(i, 100.0, volume=1000.0)) is None

    def test_synthetic_bars_ignored(self):
        rvol = RelativeVolume(min_samples=2)
        for i in range(3):
            rvol.update(bar(i, 100.0, volume=1000.0))
        before = rvol.value
        rvol.update(bar(3, 100.0, volume=99999.0, synthetic=True))
        assert rvol.value == before


class TestDeterminism:
    def test_same_bars_produce_same_values(self):
        """Replay determinism — the property that keeps backtest and live in agreement."""
        closes = [100.0 + (i % 17) - 8 for i in range(200)]

        def run():
            indicators = [SMA(10), EMA(10), ATR(14), RSI(14), ADX(14), SessionVWAP()]
            out = []
            for i, close in enumerate(closes):
                b = bar(i, close, high=close + 1, low=close - 1, volume=1000.0 + i)
                out.append(tuple(ind.update(b) for ind in indicators))
            return out

        assert run() == run()

    def test_incremental_matches_recomputation(self):
        """An incremental SMA must equal the naive full-window computation."""
        closes = [100.0 + (i % 23) for i in range(120)]
        sma = SMA(20)
        for i, close in enumerate(closes):
            sma.update(bar(i, close))
            if i >= 19:
                expected = sum(closes[i - 19:i + 1]) / 20
                assert sma.value == pytest.approx(expected)
