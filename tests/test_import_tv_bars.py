"""Importer tests — and a regression test for the bug that made a backtest lie.

The first run of this importer produced **zero trades on real NSE data**, from both the
strategy and the negative control. That reads as a market observation. It was a timezone
bug.

`ParquetBarStore` stores UTC and converts to IST on read, which is why `cal.at()` returns
**tz-aware** datetimes and why the bot's own bar builder round-trips correctly. The importer
stripped the timezone before writing. A naive 09:15 IST was then read back as 09:15 *UTC* →
**14:45 IST**, and every afternoon bar landed past the session close, where the engine's
phase check discards it silently.

That is the most dangerous shape a data bug can take: no exception, no warning, a plausible
result, and a conclusion ("this strategy takes no trades") that is entirely an artefact.
`TestTimestampsSurviveTheRoundTrip` is the guard.
"""
from __future__ import annotations

import datetime as dt

import pytest

from src import market_calendar as cal
from src.bar_store import ParquetBarStore
from tools.import_tv_bars import load_csv

DAY = dt.date(2026, 7, 21)


def epoch_for(day: dt.date, hour: int, minute: int) -> int:
    """The epoch TradingView would emit for an IST wall-clock time."""
    return int(cal.at(day, dt.time(hour, minute)).timestamp())


def write_csv(path, rows):
    lines = ["symbol,time,open,high,low,close,volume"]
    for symbol, ts, close in rows:
        lines.append(f"{symbol},{ts},{close},{close + 1},{close - 1},{close},10000")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


class TestTimestampsSurviveTheRoundTrip:
    """The regression test. Everything else here is ordinary parsing."""

    def test_a_0915_bar_reads_back_as_0915(self, tmp_path):
        csv = write_csv(tmp_path / "b.csv",
                        [("NSE:RELIANCE", epoch_for(DAY, 9, 15), 1300.0)])
        bars = load_csv(csv, 300)["tv:NSE:RELIANCE"]
        assert bars[0].bar_start.hour == 9
        assert bars[0].bar_start.minute == 15

    def test_the_bar_is_timezone_aware(self):
        """Stripping tzinfo is the bug. Naive IST is read back as UTC, +5:30."""
        csv_rows = [("NSE:X", epoch_for(DAY, 10, 0), 100.0)]
        import tempfile
        from pathlib import Path

        path = write_csv(Path(tempfile.mkdtemp()) / "b.csv", csv_rows)
        bar = load_csv(path, 300)["tv:NSE:X"][0]
        assert bar.bar_start.tzinfo is not None, (
            "naive timestamps are read back as UTC and shift the whole session")

    def test_a_full_session_survives_storage(self, tmp_path):
        """The end-to-end shape of the bug: write a session, read it back, and check
        every bar is still inside 09:15-15:30."""
        rows = []
        for i in range(75):                       # a full 5-minute session
            moment = cal.at(DAY, cal.SESSION_OPEN) + dt.timedelta(minutes=5 * i)
            rows.append(("NSE:RELIANCE", int(moment.timestamp()), 1300.0 + i))
        csv = write_csv(tmp_path / "b.csv", rows)

        store = ParquetBarStore(tmp_path / "store", "5m")
        store.write(load_csv(csv, 300)["tv:NSE:RELIANCE"])
        frame = store.read_day("tv:NSE:RELIANCE", DAY)

        assert len(frame) == 75
        times = frame["bar_start"].dt.time
        assert times.min() == cal.SESSION_OPEN
        assert times.max() < cal.SESSION_CLOSE

    def test_afternoon_bars_do_not_leak_past_the_close(self, tmp_path):
        """The specific failure: a 15:00 bar became 20:30 and the engine discarded it as
        'closed', so the strategy never saw the second half of any day."""
        csv = write_csv(tmp_path / "b.csv",
                        [("NSE:X", epoch_for(DAY, 15, 0), 100.0)])
        store = ParquetBarStore(tmp_path / "s", "5m")
        store.write(load_csv(csv, 300)["tv:NSE:X"])
        got = store.read_day("tv:NSE:X", DAY)["bar_start"].iloc[0]
        assert got.hour == 15 and got.minute == 0


class TestSessionFiltering:
    def test_bars_outside_the_session_are_dropped(self, tmp_path):
        """Pre-open and post-close bars are TradingView's, not the exchange's. Keeping
        them would put trades in windows the live bot can never trade in."""
        csv = write_csv(tmp_path / "b.csv", [
            ("NSE:X", epoch_for(DAY, 8, 0), 100.0),      # pre-open
            ("NSE:X", epoch_for(DAY, 10, 0), 100.0),     # in session
            ("NSE:X", epoch_for(DAY, 16, 0), 100.0),     # after close
        ])
        assert len(load_csv(csv, 300)["tv:NSE:X"]) == 1

    def test_non_trading_days_are_dropped(self, tmp_path):
        sunday = dt.date(2026, 7, 26)
        assert not cal.is_trading_day(sunday)
        csv = write_csv(tmp_path / "b.csv",
                        [("NSE:X", epoch_for(sunday, 10, 0), 100.0)])
        assert load_csv(csv, 300) == {}

    def test_a_malformed_row_does_not_cost_the_others(self, tmp_path):
        path = tmp_path / "b.csv"
        path.write_text(
            "symbol,time,open,high,low,close,volume\n"
            f"NSE:X,{epoch_for(DAY, 10, 0)},100,101,99,100,1000\n"
            "NSE:X,not-a-number,1,1,1,1,1\n"
            f"NSE:X,{epoch_for(DAY, 10, 5)},101,102,100,101,1000\n",
            encoding="utf-8")
        assert len(load_csv(path, 300)["tv:NSE:X"]) == 2


class TestProvenance:
    def test_instrument_ids_are_prefixed_not_guessed(self, tmp_path):
        """Resolving to nse_cm:<token> needs the instrument master, and a guessed token
        is an order on the wrong instrument (B4). The prefix also stops this data being
        mistaken for broker data in a live path."""
        csv = write_csv(tmp_path / "b.csv",
                        [("NSE:RELIANCE", epoch_for(DAY, 10, 0), 1300.0)])
        assert list(load_csv(csv, 300)) == ["tv:NSE:RELIANCE"]

    def test_the_bar_interval_sets_bar_end(self, tmp_path):
        csv = write_csv(tmp_path / "b.csv",
                        [("NSE:X", epoch_for(DAY, 10, 0), 100.0)])
        bar = load_csv(csv, 300)["tv:NSE:X"][0]
        assert (bar.bar_end - bar.bar_start).total_seconds() == 300

    def test_bars_are_not_marked_synthetic(self, tmp_path):
        """These are real prints. Marking them synthetic would make the strategy skip
        them, and the data-quality gate would report fabricated coverage."""
        csv = write_csv(tmp_path / "b.csv",
                        [("NSE:X", epoch_for(DAY, 10, 0), 100.0)])
        assert load_csv(csv, 300)["tv:NSE:X"][0].synthetic is False
