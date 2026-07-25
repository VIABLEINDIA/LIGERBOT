"""Bar source and data-quality gate tests."""
from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from src import market_calendar as cal
from src.backtest.bar_source import (
    DataQualityError, InMemoryBarSource, ParquetBarSource, QualityThresholds,
    assess_quality,
)
from src.backtest.synthetic import generate_history
from src.bar_store import ParquetBarStore
from src.bars import Bar

DAY = dt.date(2026, 7, 23)
NEXT_DAY = dt.date(2026, 7, 24)


def frame_for(day: dt.date, *, bars: int = 100, price: float = 100.0,
              instrument="nse_cm:1", synthetic: bool = False,
              open_override: float | None = None) -> pd.DataFrame:
    session_open = cal.at(day, cal.SESSION_OPEN)
    rows = []
    for i in range(bars):
        start = session_open + dt.timedelta(minutes=i)
        o = open_override if (i == 0 and open_override) else price
        rows.append({
            "instrument_id": instrument,
            "bar_start": start, "bar_end": start + dt.timedelta(minutes=1),
            "open": o, "high": max(o, price) + 0.5, "low": min(o, price) - 0.5,
            "close": price, "volume": 1000.0, "vwap": price,
            "tick_count": 50, "synthetic": synthetic,
        })
    out = pd.DataFrame(rows)
    out["bar_start"] = pd.to_datetime(out["bar_start"])
    out["bar_end"] = pd.to_datetime(out["bar_end"])
    return out


def many_days(count: int, **kw) -> pd.DataFrame:
    days = cal.trading_days_between(dt.date(2026, 1, 1), dt.date(2026, 12, 31))[:count]
    return pd.concat([frame_for(d, **kw) for d in days], ignore_index=True)


class TestQualityGate:
    def test_clean_data_passes(self):
        report = assess_quality({"nse_cm:1": many_days(30)})
        assert report.is_usable
        report.raise_if_unusable()

    def test_empty_dataset_is_rejected(self):
        report = assess_quality({"nse_cm:1": pd.DataFrame()})
        assert not report.is_usable
        with pytest.raises(DataQualityError, match="empty"):
            report.raise_if_unusable()

    def test_too_few_trading_days_blocks(self):
        report = assess_quality({"nse_cm:1": many_days(5)})
        assert not report.is_usable
        with pytest.raises(DataQualityError, match="insufficient-history"):
            report.raise_if_unusable()

    def test_thin_history_warns_but_does_not_block(self):
        """The D5 risk made visible without stopping the run."""
        report = assess_quality({"nse_cm:1": many_days(40)})
        assert report.is_usable
        kinds = {i.kind for i in report.issues}
        assert "thin-history" in kinds
        assert all(not i.blocking for i in report.issues if i.kind == "thin-history")

    def test_excess_synthetic_blocks(self):
        # Gap fillers are the absence of information; past a threshold the "data" is
        # mostly our own padding.
        report = assess_quality({"nse_cm:1": many_days(30, synthetic=True)})
        assert not report.is_usable
        assert any(i.kind == "excess-synthetic" for i in report.issues)

    def test_suspected_corporate_action_blocks(self):
        """A 1:5 split looks exactly like an 80% crash — and would book a fake trade."""
        first = frame_for(DAY, price=1000.0)
        # Next session opens at a fifth of the price: an unadjusted 1:5 split.
        second = frame_for(NEXT_DAY, price=200.0, open_override=200.0)
        rest = pd.concat([frame_for(d, price=200.0)
                          for d in cal.trading_days_between(
                              dt.date(2026, 7, 27), dt.date(2026, 9, 15))],
                         ignore_index=True)
        combined = pd.concat([first, second, rest], ignore_index=True)

        report = assess_quality({"nse_cm:1": combined})
        assert any(i.kind == "suspected-corporate-action" for i in report.issues)
        with pytest.raises(DataQualityError, match="corporate-action"):
            report.raise_if_unusable()

    def test_sparse_session_blocks(self):
        report = assess_quality({"nse_cm:1": many_days(30, bars=5)})
        assert any(i.kind == "sparse-session" for i in report.issues)

    def test_inconsistent_ohlc_blocks(self):
        frame = many_days(30)
        frame.loc[0, "high"] = frame.loc[0, "close"] - 10  # high below close
        report = assess_quality({"nse_cm:1": frame})
        assert any(i.kind == "inconsistent-ohlc" for i in report.issues)

    def test_price_spike_blocks(self):
        frame = many_days(30)
        frame.loc[0, "high"] = frame.loc[0, "close"] * 2.0
        report = assess_quality({"nse_cm:1": frame})
        assert any(i.kind == "price-spike" for i in report.issues)

    def test_non_positive_price_blocks(self):
        frame = many_days(30)
        frame.loc[0, "close"] = 0.0
        report = assess_quality({"nse_cm:1": frame})
        assert any(i.kind == "non-positive-price" for i in report.issues)

    def test_thresholds_are_adjustable(self):
        loose = QualityThresholds(max_synthetic_pct=100.0, min_trading_days=1,
                                  recommended_trading_days=1)
        report = assess_quality({"nse_cm:1": many_days(30, synthetic=True)}, loose)
        assert report.is_usable

    def test_summary_reports_the_synthetic_fraction(self):
        report = assess_quality({"nse_cm:1": many_days(30)})
        assert "synthetic" in report.summary()
        assert report.synthetic_pct == 0.0


class TestInMemorySource:
    def test_load_filters_by_date(self):
        source = InMemoryBarSource({"nse_cm:1": many_days(10)})
        days = source.available_days("nse_cm:1")
        assert len(source.load("nse_cm:1", days[0], days[0])) == 100
        assert len(source.load("nse_cm:1", days[0], days[2])) == 300

    def test_unknown_instrument_returns_empty(self):
        source = InMemoryBarSource({})
        assert source.load("nse_cm:missing", DAY, DAY).empty
        assert source.available_days("nse_cm:missing") == []

    def test_stream_is_chronological_across_instruments(self):
        """Cross-instrument ordering stops a strategy seeing B's future before A's past."""
        source = InMemoryBarSource({
            "nse_cm:1": many_days(3, instrument="nse_cm:1"),
            "nse_cm:2": many_days(3, instrument="nse_cm:2"),
        })
        days = source.available_days("nse_cm:1")
        bars = list(source.stream(["nse_cm:1", "nse_cm:2"], days[0], days[-1]))
        starts = [b.bar_start for b in bars]
        assert starts == sorted(starts)
        assert len({b.instrument_id for b in bars}) == 2

    def test_stream_yields_bar_objects(self):
        source = InMemoryBarSource({"nse_cm:1": many_days(2)})
        days = source.available_days("nse_cm:1")
        first = next(iter(source.stream(["nse_cm:1"], days[0], days[-1])))
        assert isinstance(first, Bar)
        assert first.instrument_id == "nse_cm:1"


class TestParquetSource:
    def test_round_trips_through_the_store(self, tmp_path):
        store = ParquetBarStore(tmp_path, "1m")
        frames = generate_history(
            ["nse_cm:1"], dt.date(2026, 3, 2), dt.date(2026, 3, 13), seed=3)
        bars = [
            Bar("nse_cm:1", row.bar_start.to_pydatetime().replace(tzinfo=cal.IST),
                row.bar_end.to_pydatetime().replace(tzinfo=cal.IST),
                row.open, row.high, row.low, row.close,
                volume=row.volume, vwap=row.vwap, tick_count=row.tick_count)
            for row in frames["nse_cm:1"].itertuples(index=False)
        ]
        store.write(bars)

        source = ParquetBarSource(store)
        loaded = source.load("nse_cm:1", dt.date(2026, 3, 2), dt.date(2026, 3, 13))
        assert len(loaded) == len(bars)

    def test_coverage_gaps_lists_missing_trading_days(self, tmp_path):
        """The honest measure of held history, versus the span the files happen to cover."""
        store = ParquetBarStore(tmp_path, "1m")
        store.write([Bar("nse_cm:1", cal.at(DAY, cal.SESSION_OPEN),
                         cal.at(DAY, dt.time(9, 16)), 100.0, 100.0, 100.0, 100.0)])
        source = ParquetBarSource(store)
        gaps = source.coverage_gaps("nse_cm:1", dt.date(2026, 7, 20), dt.date(2026, 7, 24))
        assert DAY not in gaps
        assert dt.date(2026, 7, 20) in gaps
        assert len(gaps) == 4
