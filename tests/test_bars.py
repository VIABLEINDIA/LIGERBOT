"""Bar aggregation tests — the fix for B1.

The determinism test at the end is the important one: it pins the property that makes
backtesting meaningful, namely that the same ticks always produce the same bars.
"""
from __future__ import annotations

import datetime as dt

import pytest

from src import market_calendar as cal
from src.bars import Bar, BarAggregator, MultiInstrumentAggregator, VolumeMode, align_to_interval

DAY = dt.date(2026, 7, 23)
SESSION_OPEN, SESSION_CLOSE = cal.session_window(DAY)


def t(hour: int, minute: int, second: int = 0) -> dt.datetime:
    return dt.datetime.combine(DAY, dt.time(hour, minute, second), tzinfo=cal.IST)


def agg(interval: int = 60, **kw) -> BarAggregator:
    return BarAggregator("nse_cm:1", interval, SESSION_OPEN, SESSION_CLOSE, **kw)


class TestAlignment:
    def test_floors_to_the_interval(self):
        assert align_to_interval(t(9, 15, 30), 60, SESSION_OPEN) == t(9, 15)
        assert align_to_interval(t(9, 17, 59), 60, SESSION_OPEN) == t(9, 17)

    def test_five_minute_bars_anchor_at_the_session_open(self):
        # 09:15 must always start a bar; otherwise the first bar of the day is partial
        # and every indicator inherits the distortion.
        assert align_to_interval(t(9, 19), 300, SESSION_OPEN) == t(9, 15)
        assert align_to_interval(t(9, 20), 300, SESSION_OPEN) == t(9, 20)

    def test_odd_interval_still_anchors_at_the_open(self):
        # 7-minute boundaries run 09:15, 09:22, 09:29 — anchored at the open, which a
        # midnight-anchored scheme would not reproduce.
        assert align_to_interval(t(9, 21), 420, SESSION_OPEN) == t(9, 15)
        assert align_to_interval(t(9, 22), 420, SESSION_OPEN) == t(9, 22)
        assert align_to_interval(t(9, 23), 420, SESSION_OPEN) == t(9, 22)


class TestOHLC:
    def test_builds_correct_ohlc(self):
        a = agg(volume_mode=VolumeMode.NONE)
        a.add_tick(t(9, 15, 1), 100.0)
        a.add_tick(t(9, 15, 20), 102.0)
        a.add_tick(t(9, 15, 40), 98.0)
        a.add_tick(t(9, 15, 55), 101.0)
        bars = a.flush_until(t(9, 16, 30))
        assert len(bars) == 1
        bar = bars[0]
        assert (bar.open, bar.high, bar.low, bar.close) == (100.0, 102.0, 98.0, 101.0)
        assert bar.tick_count == 4
        assert bar.synthetic is False

    def test_bar_end_is_exclusive_of_the_next_bar(self):
        a = agg(volume_mode=VolumeMode.NONE)
        a.add_tick(t(9, 15, 1), 100.0)
        bars = a.flush_until(t(9, 17))
        assert bars[0].bar_start == t(9, 15)
        assert bars[0].bar_end == t(9, 16)

    def test_ticks_outside_the_session_are_ignored(self):
        a = agg()
        assert a.add_tick(t(9, 0), 100.0) == []      # pre-open auction
        assert a.add_tick(t(15, 45), 100.0) == []    # after close
        assert a.add_tick(t(16, 30), 100.0) == []

    def test_non_positive_prices_ignored(self):
        a = agg()
        assert a.add_tick(t(9, 20), 0.0) == []
        assert a.add_tick(t(9, 20), -5.0) == []

    def test_out_of_order_ticks_are_dropped(self):
        a = agg(volume_mode=VolumeMode.NONE)
        a.add_tick(t(9, 20, 30), 100.0)
        a.add_tick(t(9, 20, 10), 999.0)  # stale, must not affect the bar
        bars = a.flush_until(t(9, 22))
        assert bars[0].high == 100.0

    def test_no_bar_extends_past_the_session_close(self):
        a = agg(volume_mode=VolumeMode.NONE)
        a.add_tick(t(15, 28), 100.0)
        bars = a.flush_until(t(16, 30))
        assert all(b.bar_end <= SESSION_CLOSE for b in bars)


class TestVolume:
    def test_cumulative_volume_is_differenced(self):
        # The broker reports a day-running total; publishing it raw would make every
        # bar's volume the whole day's.
        a = agg(volume_mode=VolumeMode.CUMULATIVE)
        a.add_tick(t(9, 15, 1), 100.0, 1000)   # baseline only
        a.add_tick(t(9, 15, 30), 102.0, 1200)  # +200
        a.add_tick(t(9, 15, 50), 101.0, 1300)  # +100
        bars = a.flush_until(t(9, 17))
        assert bars[0].volume == 300.0

    def test_first_tick_contributes_no_volume(self):
        # Without a baseline we cannot attribute the day's accumulated volume to this
        # bar; counting it would dump the whole session into the first bar.
        a = agg(volume_mode=VolumeMode.CUMULATIVE)
        a.add_tick(t(9, 15, 1), 100.0, 50_000)
        bars = a.flush_until(t(9, 17))
        assert bars[0].volume == 0.0

    def test_counter_reset_does_not_produce_negative_volume(self):
        a = agg(volume_mode=VolumeMode.CUMULATIVE)
        a.add_tick(t(9, 15, 1), 100.0, 5000)
        a.add_tick(t(9, 15, 2), 100.0, 5500)  # +500
        a.add_tick(t(9, 15, 3), 100.0, 100)   # reconnect/replay: reset
        a.add_tick(t(9, 15, 4), 100.0, 300)   # +200
        bars = a.flush_until(t(9, 17))
        assert bars[0].volume == 700.0

    def test_incremental_mode_sums_quantities(self):
        a = agg(volume_mode=VolumeMode.INCREMENTAL)
        a.add_tick(t(9, 15, 1), 100.0, 10)
        a.add_tick(t(9, 15, 2), 100.0, 25)
        bars = a.flush_until(t(9, 17))
        assert bars[0].volume == 35.0

    def test_vwap_is_volume_weighted(self):
        a = agg(volume_mode=VolumeMode.INCREMENTAL)
        a.add_tick(t(9, 15, 1), 100.0, 100)
        a.add_tick(t(9, 15, 2), 200.0, 300)
        bars = a.flush_until(t(9, 17))
        assert bars[0].vwap == pytest.approx((100 * 100 + 200 * 300) / 400)

    def test_vwap_falls_back_to_close_without_volume(self):
        a = agg(volume_mode=VolumeMode.NONE)
        a.add_tick(t(9, 15, 1), 100.0)
        a.add_tick(t(9, 15, 2), 200.0)
        bars = a.flush_until(t(9, 17))
        assert bars[0].volume == 0.0
        assert bars[0].vwap == 200.0


class TestGapFilling:
    def test_gaps_are_filled_with_synthetic_bars(self):
        a = agg(volume_mode=VolumeMode.NONE)
        a.add_tick(t(9, 15, 1), 100.0)
        bars = a.add_tick(t(9, 20, 10), 110.0)
        # Real 09:15, then synthetic 09:16-09:19.
        assert len(bars) == 5
        assert bars[0].synthetic is False
        assert all(b.synthetic for b in bars[1:])
        assert all(b.open == b.high == b.low == b.close == 100.0 for b in bars[1:])
        assert all(b.volume == 0.0 and b.tick_count == 0 for b in bars[1:])

    def test_synthetic_bars_carry_the_previous_close(self):
        a = agg(volume_mode=VolumeMode.NONE)
        a.add_tick(t(9, 15, 1), 100.0)
        a.add_tick(t(9, 15, 50), 105.0)
        bars = a.add_tick(t(9, 18, 10), 110.0)
        assert [b.close for b in bars] == [105.0, 105.0, 105.0]

    def test_gap_filling_can_be_disabled(self):
        a = agg(volume_mode=VolumeMode.NONE, fill_gaps=False)
        a.add_tick(t(9, 15, 1), 100.0)
        bars = a.add_tick(t(9, 20, 10), 110.0)
        assert len(bars) == 1
        assert bars[0].synthetic is False

    def test_no_synthetic_bars_before_the_first_real_tick(self):
        a = agg(volume_mode=VolumeMode.NONE)
        bars = a.add_tick(t(10, 0), 100.0)
        assert bars == []

    def test_bars_form_a_contiguous_series(self):
        # The reason gap filling exists: indicator windows must stay time-aligned.
        a = agg(volume_mode=VolumeMode.NONE)
        a.add_tick(t(9, 15, 1), 100.0)
        collected = a.add_tick(t(9, 25, 10), 110.0)
        collected += a.flush_until(t(9, 30))
        starts = [b.bar_start for b in collected]
        assert starts == sorted(starts)
        for earlier, later in zip(collected, collected[1:]):
            assert later.bar_start == earlier.bar_end


class TestFlush:
    def test_flush_closes_the_forming_bar(self):
        a = agg(volume_mode=VolumeMode.NONE)
        a.add_tick(t(9, 15, 30), 100.0)
        assert a.flush_until(t(9, 16, 1)) != []

    def test_flush_is_idempotent(self):
        a = agg(volume_mode=VolumeMode.NONE)
        a.add_tick(t(9, 15, 30), 100.0)
        first = a.flush_until(t(9, 17))
        assert first
        assert a.flush_until(t(9, 17)) == []

    def test_flush_does_not_close_a_bar_still_forming(self):
        a = agg(volume_mode=VolumeMode.NONE)
        a.add_tick(t(9, 15, 30), 100.0)
        assert a.flush_until(t(9, 15, 59)) == []

    def test_idle_instrument_keeps_producing_bars(self):
        # Without time-driven closure, a quiet instrument silently stops emitting and
        # downstream cannot tell "quiet" from "feed is dead".
        a = agg(volume_mode=VolumeMode.NONE)
        a.add_tick(t(9, 15, 30), 100.0)
        a.flush_until(t(9, 16, 30))
        later = a.flush_until(t(9, 20))
        # 09:16 through 09:19 have all fully elapsed by 09:20; 09:20 is still forming.
        assert [b.bar_start for b in later] == [t(9, 16), t(9, 17), t(9, 18), t(9, 19)]
        assert all(b.synthetic for b in later)


class TestMultiInstrument:
    def test_instruments_are_independent(self):
        multi = MultiInstrumentAggregator(60, DAY, volume_mode=VolumeMode.NONE)
        multi.add_tick("nse_cm:1", t(9, 15, 10), 100.0)
        multi.add_tick("nse_cm:2", t(9, 15, 10), 500.0)
        bars = multi.flush_until(t(9, 17))
        by_id = {b.instrument_id: b for b in bars}
        assert by_id["nse_cm:1"].close == 100.0
        assert by_id["nse_cm:2"].close == 500.0

    def test_flush_returns_bars_in_time_order(self):
        multi = MultiInstrumentAggregator(60, DAY, volume_mode=VolumeMode.NONE)
        multi.add_tick("nse_cm:1", t(9, 15, 10), 100.0)
        multi.add_tick("nse_cm:2", t(9, 16, 10), 500.0)
        bars = multi.flush_until(t(9, 20))
        assert [b.bar_start for b in bars] == sorted(b.bar_start for b in bars)

    def test_rejects_a_non_trading_day(self):
        with pytest.raises(ValueError, match="not an NSE trading day"):
            MultiInstrumentAggregator(60, dt.date(2026, 1, 26))


class TestSerialization:
    def test_round_trips_through_the_event_schema(self):
        bar = Bar("nse_cm:1", t(9, 15), t(9, 16), 100.0, 102.0, 99.0, 101.0,
                  volume=500.0, vwap=100.5, tick_count=12)
        restored = Bar.from_event(bar.to_event())
        assert restored == bar

    def test_events_are_always_marked_closed(self):
        # A forming bar is never published; that would be look-ahead.
        bar = Bar("nse_cm:1", t(9, 15), t(9, 16), 100.0, 100.0, 100.0, 100.0)
        assert bar.to_event()["closed"] is True


class TestDeterminism:
    def test_same_ticks_produce_identical_bars(self):
        """The property that makes backtesting meaningful.

        If the live and backtest paths could produce different bars from the same ticks,
        every backtest result would be a fiction. Both paths share this code precisely
        so that cannot happen.
        """
        ticks = [(t(9, 15, s % 60 + (s // 60) * 0), 100.0 + (s % 7)) for s in range(0, 50)]
        ticks = [(t(9, 15 + i // 20, (i * 3) % 60), 100.0 + (i % 11)) for i in range(60)]

        def build():
            a = agg(volume_mode=VolumeMode.NONE)
            out = []
            for timestamp, price in ticks:
                out.extend(a.add_tick(timestamp, price))
            out.extend(a.flush_until(t(10, 0)))
            return out

        assert build() == build()
