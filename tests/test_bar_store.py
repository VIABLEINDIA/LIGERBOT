"""Parquet bar store tests."""
from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from src import market_calendar as cal
from src.bar_store import ParquetBarStore, _safe_name, bars_to_frame
from src.bars import Bar

DAY = dt.date(2026, 7, 23)


def t(hour: int, minute: int) -> dt.datetime:
    return dt.datetime.combine(DAY, dt.time(hour, minute), tzinfo=cal.IST)


def bar(minute: int, close: float = 100.0, instrument="nse_cm:11536",
        synthetic: bool = False) -> Bar:
    return Bar(
        instrument_id=instrument,
        bar_start=t(9, minute), bar_end=t(9, minute + 1),
        open=close, high=close + 1, low=close - 1, close=close,
        volume=1000.0, vwap=close, tick_count=10, synthetic=synthetic,
    )


@pytest.fixture
def store(tmp_path) -> ParquetBarStore:
    return ParquetBarStore(tmp_path, "1m")


class TestPathSafety:
    def test_colon_is_stripped_from_instrument_ids(self):
        # instrument_id is "nse_cm:11536"; ':' is illegal in Windows paths, so an
        # unsanitised name would fail to write at all on the target platform.
        assert _safe_name("nse_cm:11536") == "nse_cm_11536"
        assert ":" not in _safe_name("nse_cm:11536")

    def test_partition_path_layout(self, store):
        path = store.partition_path("nse_cm:11536", DAY)
        assert path.parent.name == "nse_cm_11536"
        assert path.name == "2026-07-23.parquet"


class TestWriting:
    def test_write_and_read_back(self, store):
        store.write([bar(15), bar(16, 101.0), bar(17, 102.0)])
        frame = store.read_day("nse_cm:11536", DAY)
        assert len(frame) == 3
        assert list(frame["close"]) == [100.0, 101.0, 102.0]

    def test_buffer_then_flush(self, store):
        assert store.buffer([bar(15), bar(16)]) == 2
        # Nothing on disk until flushed — archiving must not block the trading path.
        assert store.read_day("nse_cm:11536", DAY).empty
        assert store.flush() == 2
        assert len(store.read_day("nse_cm:11536", DAY)) == 2

    def test_flush_when_empty_is_a_no_op(self, store):
        assert store.flush() == 0

    def test_appending_merges_with_existing(self, store):
        store.write([bar(15), bar(16)])
        store.write([bar(17), bar(18)])
        assert len(store.read_day("nse_cm:11536", DAY)) == 4

    def test_rewriting_the_same_bar_deduplicates(self, store):
        # Re-running a crashed session must merge, not duplicate.
        store.write([bar(15, 100.0), bar(16, 101.0)])
        store.write([bar(15, 999.0)])  # corrected replay of the same bar
        frame = store.read_day("nse_cm:11536", DAY)
        assert len(frame) == 2
        assert frame.iloc[0]["close"] == 999.0  # last write wins

    def test_bars_are_sorted_by_time(self, store):
        store.write([bar(18), bar(15), bar(17), bar(16)])
        frame = store.read_day("nse_cm:11536", DAY)
        assert list(frame["bar_start"]) == sorted(frame["bar_start"])

    def test_partitions_by_instrument(self, store):
        store.write([bar(15, instrument="nse_cm:1"), bar(15, instrument="nse_cm:2")])
        assert len(store.read_day("nse_cm:1", DAY)) == 1
        assert len(store.read_day("nse_cm:2", DAY)) == 1
        assert sorted(store.instruments()) == ["nse_cm_1", "nse_cm_2"]

    def test_partitions_by_the_bars_own_date(self, store):
        # Replaying an old session must land in that session's file, not today's.
        old_day = dt.date(2026, 7, 22)
        old = Bar("nse_cm:1",
                  dt.datetime.combine(old_day, dt.time(9, 15), tzinfo=cal.IST),
                  dt.datetime.combine(old_day, dt.time(9, 16), tzinfo=cal.IST),
                  100.0, 100.0, 100.0, 100.0)
        store.write([old, bar(15, instrument="nse_cm:1")])
        assert len(store.read_day("nse_cm:1", old_day)) == 1
        assert len(store.read_day("nse_cm:1", DAY)) == 1

    def test_corrupt_partition_is_quarantined_not_overwritten(self, store):
        path = store.partition_path("nse_cm:1", DAY)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("this is not parquet", encoding="utf-8")
        store.write([bar(15, instrument="nse_cm:1")])
        assert path.with_suffix(".parquet.corrupt").exists()
        assert len(store.read_day("nse_cm:1", DAY)) == 1


class TestReading:
    def test_missing_partition_returns_empty_frame(self, store):
        assert store.read_day("nse_cm:nope", DAY).empty

    def test_available_days(self, store):
        store.write([bar(15)])
        assert store.available_days("nse_cm:11536") == [DAY]
        assert store.available_days("nse_cm:missing") == []

    def test_read_range_filters_by_date(self, store):
        store.write([bar(15)])
        assert len(store.read_range("nse_cm:11536", DAY, DAY)) == 1
        assert store.read_range(
            "nse_cm:11536", dt.date(2026, 1, 1), dt.date(2026, 1, 2)
        ).empty

    def test_drop_synthetic(self, store):
        store.write([bar(15), bar(16, synthetic=True), bar(17)])
        assert len(store.read_range("nse_cm:11536", DAY, DAY)) == 3
        assert len(store.read_range("nse_cm:11536", DAY, DAY, drop_synthetic=True)) == 2


class TestCoverage:
    def test_coverage_reports_the_synthetic_fraction(self, store):
        # The number that matters: a high synthetic ratio means the "data" is mostly
        # our own gap filler, carrying no information.
        store.write([bar(15), bar(16, synthetic=True),
                     bar(17, synthetic=True), bar(18)])
        report = store.coverage()
        assert len(report) == 1
        row = report.iloc[0]
        assert row["bars"] == 4
        assert row["synthetic_pct"] == pytest.approx(50.0)
        assert row["days"] == 1

    def test_coverage_is_empty_for_an_empty_store(self, store):
        assert store.coverage().empty


class TestFrameConversion:
    def test_empty_input_yields_typed_empty_frame(self):
        frame = bars_to_frame([])
        assert frame.empty
        assert "instrument_id" in frame.columns

    def test_timestamps_survive_the_round_trip(self, store):
        store.write([bar(15)])
        frame = store.read_day("nse_cm:11536", DAY)
        restored = pd.to_datetime(frame.iloc[0]["bar_start"], utc=True)
        assert restored == pd.Timestamp(t(9, 15)).tz_convert("UTC")
