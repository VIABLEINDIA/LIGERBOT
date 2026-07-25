"""Market calendar and session-phase tests."""
from __future__ import annotations

import datetime as dt

import pytest

from src import market_calendar as cal

# 2026-07-23 is a Thursday and not in the holiday list.
TRADING_DAY = dt.date(2026, 7, 23)
SATURDAY = dt.date(2026, 7, 25)
SUNDAY = dt.date(2026, 7, 26)
REPUBLIC_DAY = dt.date(2026, 1, 26)


def at(hour: int, minute: int, day: dt.date = TRADING_DAY) -> dt.datetime:
    return dt.datetime.combine(day, dt.time(hour, minute), tzinfo=cal.IST)


class TestTradingDays:
    def test_weekday_is_a_trading_day(self):
        assert cal.is_trading_day(TRADING_DAY)

    def test_weekends_are_not(self):
        assert not cal.is_trading_day(SATURDAY)
        assert not cal.is_trading_day(SUNDAY)

    def test_listed_holiday_is_not(self):
        assert not cal.is_trading_day(REPUBLIC_DAY)

    def test_next_trading_day_skips_the_weekend(self):
        friday = dt.date(2026, 7, 24)
        assert cal.next_trading_day(friday) == dt.date(2026, 7, 27)  # Monday

    def test_trading_days_between_excludes_weekends(self):
        days = cal.trading_days_between(dt.date(2026, 7, 20), dt.date(2026, 7, 26))
        assert days == [dt.date(2026, 7, d) for d in (20, 21, 22, 23, 24)]

    def test_covers_year_flags_unverified_years(self):
        assert cal.covers_year(2026)
        # An uncovered year must not silently pass as fully-trading weekdays.
        assert not cal.covers_year(2031)


class TestVerifiedHolidays:
    """Dates cross-checked against two independent published NSE calendars.

    The list shipped in Phase 0 was written from recall and marked provisional. It turned
    out to be substantially wrong for 2026: three dates off by one day, two spurious, and
    six missing. These tests pin the corrected values so a future edit cannot quietly
    reintroduce the error — an omitted holiday makes the bar builder fabricate gap-fill
    bars for a day the market never opened.
    """

    @pytest.mark.parametrize("day,name", [
        ("2026-01-15", "Maharashtra municipal elections"),
        ("2026-01-26", "Republic Day"),
        ("2026-03-03", "Holi"),
        ("2026-03-26", "Shri Ram Navami"),
        ("2026-03-31", "Shri Mahavir Jayanti"),
        ("2026-04-03", "Good Friday"),
        ("2026-04-14", "Ambedkar Jayanti"),
        ("2026-05-01", "Maharashtra Day"),
        ("2026-05-28", "Bakri Id"),
        ("2026-06-26", "Muharram"),
        ("2026-09-14", "Ganesh Chaturthi"),
        ("2026-10-02", "Gandhi Jayanti"),
        ("2026-10-20", "Dussehra"),
        ("2026-11-10", "Diwali Balipratipada"),
        ("2026-11-24", "Guru Nanak Jayanti"),
        ("2026-12-25", "Christmas"),
    ])
    def test_2026_weekday_holidays_are_closed(self, day, name):
        assert not cal.is_trading_day(dt.date.fromisoformat(day)), name

    @pytest.mark.parametrize("day", [
        "2026-03-04",  # was in the old list; Holi is actually 03-03
        "2026-03-25",  # was in the old list; Ram Navami is actually 03-26
        "2026-04-01",  # was in the old list; Mahavir Jayanti is actually 03-31
        "2026-08-26",  # was in the old list; Ganesh Chaturthi is actually 09-14
        "2026-11-11",  # was in the old list; spurious
    ])
    def test_previously_wrong_dates_are_trading_days(self, day):
        """The old list closed the market on days it actually traded."""
        assert cal.is_trading_day(dt.date.fromisoformat(day))

    @pytest.mark.parametrize("day,name", [
        ("2025-02-26", "Mahashivratri"),
        ("2025-03-14", "Holi"),
        ("2025-03-31", "Id-Ul-Fitr"),
        ("2025-04-10", "Mahavir Jayanti"),
        ("2025-04-14", "Ambedkar Jayanti"),
        ("2025-04-18", "Good Friday"),
        ("2025-05-01", "Maharashtra Day"),
        ("2025-08-15", "Independence Day"),
        ("2025-08-27", "Ganesh Chaturthi"),
        ("2025-10-02", "Gandhi Jayanti"),
        ("2025-10-21", "Diwali Laxmi Pujan"),
        ("2025-10-22", "Balipratipada"),
        ("2025-11-05", "Guru Nanak Jayanti"),
        ("2025-12-25", "Christmas"),
    ])
    def test_2025_holidays_are_closed(self, day, name):
        assert not cal.is_trading_day(dt.date.fromisoformat(day)), name

    def test_2026_weekday_holiday_count(self):
        """16 weekday closures in 2026 — a count mismatch means the list drifted."""
        weekday_holidays = [
            d for d in cal.HOLIDAYS if d.year == 2026 and d.weekday() < 5
        ]
        assert len(weekday_holidays) == 16

    def test_weekend_holidays_are_recorded_for_completeness(self):
        """Harmless for trading, but keeps the list diffable against the circular."""
        for day in ("2026-02-15", "2026-03-21", "2026-08-15", "2026-11-08"):
            assert dt.date.fromisoformat(day) in cal.HOLIDAYS


class TestMuhurat:
    def test_muhurat_days_are_identified(self):
        assert cal.is_muhurat_session(dt.date(2026, 11, 8))
        assert cal.is_muhurat_session(dt.date(2025, 10, 21))
        assert not cal.is_muhurat_session(TRADING_DAY)

    def test_bot_does_not_trade_muhurat(self):
        """A deliberate skip, not an accident of the weekend check.

        The session runs about an hour at non-standard times, so every session constant
        in the module would misfire on it.
        """
        assert not cal.is_trading_day(dt.date(2026, 11, 8))   # Sunday
        assert not cal.is_trading_day(dt.date(2025, 10, 21))  # Tuesday, but a holiday


class TestPhases:
    @pytest.mark.parametrize("hour,minute,expected", [
        (8, 30, cal.Phase.CLOSED),
        (9, 5, cal.Phase.PRE_OPEN),
        (9, 15, cal.Phase.OPENING_RANGE),
        (9, 29, cal.Phase.OPENING_RANGE),
        (9, 30, cal.Phase.ENTRY),
        (14, 44, cal.Phase.ENTRY),
        (14, 45, cal.Phase.NO_NEW_ENTRY),
        (15, 9, cal.Phase.NO_NEW_ENTRY),
        (15, 10, cal.Phase.SQUARE_OFF),
        (15, 29, cal.Phase.SQUARE_OFF),
        (15, 30, cal.Phase.CLOSED),
        (16, 0, cal.Phase.CLOSED),
    ])
    def test_phase_boundaries(self, hour, minute, expected):
        assert cal.phase(at(hour, minute)) is expected

    def test_holiday_is_closed_all_day(self):
        assert cal.phase(at(11, 0, REPUBLIC_DAY)) is cal.Phase.CLOSED

    def test_only_the_entry_phase_permits_entries(self):
        assert cal.Phase.ENTRY.allows_entry
        for other in (cal.Phase.CLOSED, cal.Phase.PRE_OPEN, cal.Phase.OPENING_RANGE,
                      cal.Phase.NO_NEW_ENTRY, cal.Phase.SQUARE_OFF):
            assert not other.allows_entry

    def test_exits_permitted_whenever_the_market_is_open(self):
        # The asymmetry: risk can always be reduced, even when it cannot be taken on.
        for phase in (cal.Phase.OPENING_RANGE, cal.Phase.ENTRY,
                      cal.Phase.NO_NEW_ENTRY, cal.Phase.SQUARE_OFF):
            assert phase.allows_exit
        assert not cal.Phase.CLOSED.allows_exit
        assert not cal.Phase.PRE_OPEN.allows_exit

    def test_square_off_requires_flat(self):
        assert cal.Phase.SQUARE_OFF.requires_flat
        assert not cal.Phase.ENTRY.requires_flat

    def test_square_off_precedes_the_brokers_auto_squareoff(self):
        # We flatten at 15:10; the broker's MIS cutoff is ~15:20. The gap is the point.
        assert cal.SQUARE_OFF < dt.time(15, 20)


class TestSessionWindow:
    def test_window_on_a_trading_day(self):
        window = cal.session_window(TRADING_DAY)
        assert window is not None
        open_dt, close_dt = window
        assert open_dt.time() == dt.time(9, 15)
        assert close_dt.time() == dt.time(15, 30)
        assert open_dt.tzinfo is not None

    def test_no_window_on_a_holiday(self):
        assert cal.session_window(REPUBLIC_DAY) is None

    def test_is_market_open(self):
        assert cal.is_market_open(at(10, 0))
        assert not cal.is_market_open(at(8, 0))
        assert not cal.is_market_open(at(16, 0))


class TestClock:
    def test_naive_datetimes_are_treated_as_ist(self):
        naive = dt.datetime(2026, 7, 23, 10, 0)
        assert cal.to_ist(naive).utcoffset() == dt.timedelta(hours=5, minutes=30)

    def test_utc_input_is_converted_not_relabelled(self):
        utc = dt.datetime(2026, 7, 23, 4, 30, tzinfo=dt.timezone.utc)
        assert cal.to_ist(utc).time() == dt.time(10, 0)

    def test_seconds_to_square_off(self):
        assert cal.seconds_to_square_off(at(15, 0)) == pytest.approx(600.0)
        # Negative once past the deadline — strategies use the sign.
        assert cal.seconds_to_square_off(at(15, 20)) == pytest.approx(-600.0)

    def test_seconds_to_square_off_is_none_on_a_holiday(self):
        assert cal.seconds_to_square_off(at(11, 0, REPUBLIC_DAY)) is None
