"""Bar resampling tests — the mechanism behind D4 (store 1m, trade 5m)."""
from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from src import market_calendar as cal
from src.backtest.bar_source import InMemoryBarSource, ResampledBarSource, resample_frame

DAY = dt.date(2026, 7, 23)


def minute_frame(count: int, *, start_price: float = 100.0,
                 instrument="nse_cm:1", synthetic_from: int | None = None) -> pd.DataFrame:
    session_open = cal.at(DAY, cal.SESSION_OPEN)
    rows = []
    for i in range(count):
        start = session_open + dt.timedelta(minutes=i)
        price = start_price + i
        rows.append({
            "instrument_id": instrument,
            "bar_start": start, "bar_end": start + dt.timedelta(minutes=1),
            "open": price, "high": price + 2.0, "low": price - 2.0, "close": price + 0.5,
            "volume": 100.0 * (i + 1), "vwap": price,
            "tick_count": 10, "synthetic": synthetic_from is not None and i >= synthetic_from,
        })
    frame = pd.DataFrame(rows)
    frame["bar_start"] = pd.to_datetime(frame["bar_start"])
    frame["bar_end"] = pd.to_datetime(frame["bar_end"])
    return frame


class TestResampleFrame:
    def test_bar_count_divides_correctly(self):
        assert len(resample_frame(minute_frame(60), 300)) == 12

    def test_ohlc_aggregates_correctly(self):
        resampled = resample_frame(minute_frame(10), 300)
        source = minute_frame(10)
        first = resampled.iloc[0]
        window = source.iloc[:5]
        assert first["open"] == window["open"].iloc[0]
        assert first["high"] == window["high"].max()
        assert first["low"] == window["low"].min()
        assert first["close"] == window["close"].iloc[-1]

    def test_volume_sums(self):
        resampled = resample_frame(minute_frame(10), 300)
        # First five bars: 100 + 200 + 300 + 400 + 500.
        assert resampled.iloc[0]["volume"] == pytest.approx(1500.0)

    def test_vwap_is_volume_weighted(self):
        source = minute_frame(5)
        resampled = resample_frame(source, 300)
        expected = (source["vwap"] * source["volume"]).sum() / source["volume"].sum()
        assert resampled.iloc[0]["vwap"] == pytest.approx(expected)

    def test_anchored_to_the_session_open(self):
        """A midnight-anchored grid would make the day's first bar a partial one."""
        resampled = resample_frame(minute_frame(30), 300)
        assert resampled.iloc[0]["bar_start"] == pd.Timestamp(
            cal.at(DAY, cal.SESSION_OPEN))

    def test_partially_synthetic_bucket_is_not_synthetic(self):
        """One real trade in the window means something genuinely happened in it."""
        resampled = resample_frame(minute_frame(10, synthetic_from=3), 300)
        assert not bool(resampled.iloc[0]["synthetic"])   # bars 0-2 real
        assert bool(resampled.iloc[1]["synthetic"])       # bars 5-9 all synthetic

    def test_empty_frame_passes_through(self):
        assert resample_frame(pd.DataFrame(), 300).empty

    def test_intervals_are_consistent(self):
        for seconds in (180, 300, 900):
            resampled = resample_frame(minute_frame(90), seconds)
            spans = (resampled["bar_end"] - resampled["bar_start"]).dt.total_seconds()
            assert (spans == seconds).all()

    def test_no_information_is_lost_in_the_extremes(self):
        """The coarse high/low must bracket every underlying bar's."""
        source = minute_frame(60)
        resampled = resample_frame(source, 300)
        assert resampled["high"].max() == pytest.approx(source["high"].max())
        assert resampled["low"].min() == pytest.approx(source["low"].min())


class TestResampledBarSource:
    def test_serves_coarser_bars(self):
        base = InMemoryBarSource({"nse_cm:1": minute_frame(60)})
        coarse = ResampledBarSource(base, 300)
        assert len(coarse.load("nse_cm:1", DAY, DAY)) == 12
        assert len(base.load("nse_cm:1", DAY, DAY)) == 60

    def test_available_days_delegates(self):
        base = InMemoryBarSource({"nse_cm:1": minute_frame(60)})
        assert ResampledBarSource(base, 300).available_days("nse_cm:1") == [DAY]

    def test_stream_yields_resampled_bars(self):
        base = InMemoryBarSource({"nse_cm:1": minute_frame(60)})
        bars = list(ResampledBarSource(base, 300).stream(["nse_cm:1"], DAY, DAY))
        assert len(bars) == 12
        assert all((b.bar_end - b.bar_start).total_seconds() == 300 for b in bars)

    def test_results_are_cached(self):
        base = InMemoryBarSource({"nse_cm:1": minute_frame(60)})
        coarse = ResampledBarSource(base, 300)
        assert coarse.load("nse_cm:1", DAY, DAY) is coarse.load("nse_cm:1", DAY, DAY)
