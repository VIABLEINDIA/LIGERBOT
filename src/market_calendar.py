"""NSE market calendar and session clock.

Every time-of-day decision in the bot routes through here. Nothing else is allowed to
call ``datetime.now()`` and reason about market hours — scattering that logic is how you
end up with a backtester that disagrees with the live system about when a session ended.

Two things this module exists to prevent:

  * **Trading outside the session.** Orders placed before 09:15 or after 15:30 are
    rejected by the exchange; orders placed in the first 15 minutes are placed into the
    opening auction's price discovery, which is a different statistical regime from the
    rest of the day (see DESIGN.md 1.6).
  * **Letting the broker choose our exit.** MIS positions are force-squared-off by the
    broker around 15:20 at whatever the market offers. We flatten ourselves at 15:10 so
    the exit is ours, taken at a price we chose to accept.

All times are IST (``Asia/Kolkata``) and computed with an explicit timezone — never the
machine's local clock, so a laptop in the wrong timezone can't quietly shift the session.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
from enum import Enum
from pathlib import Path
from typing import Optional, Set
from zoneinfo import ZoneInfo

log = logging.getLogger("ligerbot.calendar")

IST = ZoneInfo("Asia/Kolkata")

# --------------------------------------------------------------------------
# Session structure (NSE equities, normal trading session)
# --------------------------------------------------------------------------
PRE_OPEN_START = dt.time(9, 0)
SESSION_OPEN = dt.time(9, 15)
# No entries during opening price discovery — different regime, wide spreads.
ENTRY_START = dt.time(9, 30)
# Late entries can't reach a profit target before the flat deadline.
ENTRY_CUTOFF = dt.time(14, 45)
# We flatten here. The broker's own MIS square-off is ~15:20; we never let it get there.
SQUARE_OFF = dt.time(15, 10)
SESSION_CLOSE = dt.time(15, 30)


class Phase(Enum):
    """Where we are in the trading day. Drives what the bot is permitted to do."""

    CLOSED = "closed"
    PRE_OPEN = "pre_open"          # feed live, bars building, no orders
    OPENING_RANGE = "opening"      # 09:15-09:30, bars building, no entries
    ENTRY = "entry"                # 09:30-14:45, entries permitted
    NO_NEW_ENTRY = "no_new_entry"  # 14:45-15:10, manage/exit only
    SQUARE_OFF = "square_off"      # 15:10-15:30, force flat

    @property
    def allows_entry(self) -> bool:
        return self is Phase.ENTRY

    @property
    def allows_exit(self) -> bool:
        """Exits are permitted whenever the market is open.

        This asymmetry is deliberate and load-bearing: the bot must always be able to
        reduce risk, even in states where it is forbidden from taking any on.
        """
        return self in (
            Phase.OPENING_RANGE,
            Phase.ENTRY,
            Phase.NO_NEW_ENTRY,
            Phase.SQUARE_OFF,
        )

    @property
    def requires_flat(self) -> bool:
        return self is Phase.SQUARE_OFF


# --------------------------------------------------------------------------
# Holidays
# --------------------------------------------------------------------------
# NSE equity-segment trading holidays, cross-checked against two independent published
# calendars (Zerodha and ClearTax) in July 2026.
#
# Still not the primary source: NSE issues the authoritative list by circular and amends
# it during the year — 2026-01-15 (Maharashtra municipal elections) is exactly that kind
# of late addition. Verify against the circular before trading, and override via
# NSE_HOLIDAYS_FILE (a JSON list of "YYYY-MM-DD") when it changes.
#
# Why this matters more than it looks: an omitted holiday makes the bar builder emit
# synthetic gap-fill bars for a day the market never opened, and every backtest spanning
# that day silently inherits fabricated data. `covers_year()` refuses to vouch for years
# absent here so the gap surfaces as a warning rather than as quiet corruption.
#
# Weekend-falling holidays are listed for completeness even though `is_trading_day`
# already excludes weekends — keeping the full circular makes diffing against next
# year's version straightforward.
_BUILTIN_HOLIDAYS: dict[int, tuple[str, ...]] = {
    2025: (
        "2025-02-26",  # Mahashivratri
        "2025-03-14",  # Holi
        "2025-03-31",  # Id-Ul-Fitr
        "2025-04-10",  # Shri Mahavir Jayanti
        "2025-04-14",  # Dr. Baba Saheb Ambedkar Jayanti
        "2025-04-18",  # Good Friday
        "2025-05-01",  # Maharashtra Day
        "2025-08-15",  # Independence Day
        "2025-08-27",  # Shri Ganesh Chaturthi
        "2025-10-02",  # Mahatma Gandhi Jayanti / Dussehra
        "2025-10-21",  # Diwali Laxmi Pujan
        "2025-10-22",  # Balipratipada
        "2025-11-05",  # Prakash Gurpurb Sri Guru Nanak Dev
        "2025-12-25",  # Christmas
    ),
    2026: (
        "2026-01-15",  # Maharashtra municipal elections
        "2026-01-26",  # Republic Day
        "2026-02-15",  # Mahashivratri            (Sunday)
        "2026-03-03",  # Holi
        "2026-03-21",  # Id-Ul-Fitr               (Saturday)
        "2026-03-26",  # Shri Ram Navami
        "2026-03-31",  # Shri Mahavir Jayanti
        "2026-04-03",  # Good Friday
        "2026-04-14",  # Dr. Baba Saheb Ambedkar Jayanti
        "2026-05-01",  # Maharashtra Day
        "2026-05-28",  # Bakri Id
        "2026-06-26",  # Muharram
        "2026-08-15",  # Independence Day         (Saturday)
        "2026-09-14",  # Shri Ganesh Chaturthi
        "2026-10-02",  # Mahatma Gandhi Jayanti
        "2026-10-20",  # Dussehra
        "2026-11-08",  # Diwali Laxmi Pujan       (Sunday — Muhurat session, see below)
        "2026-11-10",  # Diwali Balipratipada
        "2026-11-24",  # Prakash Gurpurb Sri Guru Nanak Dev
        "2026-12-25",  # Christmas
    ),
}

# Muhurat trading: a short ceremonial session held on Diwali Laxmi Pujan, often on a day
# the market is otherwise closed. Real trades happen, but the session is roughly an hour
# long at non-standard times, so every session constant in this module (09:15, 09:30,
# 14:45, 15:10, 15:30) is wrong for it.
#
# The bot deliberately does NOT trade Muhurat. Being closed on it is the correct outcome
# here, but it should be a decision rather than an accident of the weekend check — an
# intraday strategy has no edge in a one-hour ceremonial session with unusual liquidity,
# and the session-timing constants would misfire if it tried.
MUHURAT_SESSIONS: frozenset[str] = frozenset({
    "2025-10-21",
    "2026-11-08",
})


def _load_holidays() -> tuple[Set[dt.date], Set[int]]:
    """Return (holiday dates, years we have data for).

    An external file, when present, fully replaces the built-ins — a half-merged
    calendar is worse than either source alone.
    """
    path = os.getenv("NSE_HOLIDAYS_FILE", "").strip()
    if path:
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
            dates = {dt.date.fromisoformat(s) for s in raw}
            years = {d.year for d in dates}
            log.info("Loaded %d NSE holidays from %s (years: %s)",
                     len(dates), path, sorted(years))
            return dates, years
        except (OSError, ValueError, TypeError) as exc:
            # Fall back rather than crash, but make the degradation impossible to miss.
            log.error("Could not read NSE_HOLIDAYS_FILE=%s (%s) — using built-in "
                      "provisional list instead.", path, exc)

    dates = {
        dt.date.fromisoformat(s)
        for year_dates in _BUILTIN_HOLIDAYS.values()
        for s in year_dates
    }
    return dates, set(_BUILTIN_HOLIDAYS)


HOLIDAYS, _COVERED_YEARS = _load_holidays()


def covers_year(year: int) -> bool:
    """True if we hold a holiday list for ``year``.

    Callers that care about correctness (the backfill pipeline, the backtester) should
    check this and refuse to proceed on uncovered years rather than assume every
    weekday was a trading day.
    """
    return year in _COVERED_YEARS


def is_trading_day(day: dt.date) -> bool:
    """Weekday and not an NSE holiday.

    Muhurat sessions count as non-trading days here — see :data:`MUHURAT_SESSIONS`.
    They are genuine sessions, but at times none of this module's constants describe.
    """
    if day.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    return day not in HOLIDAYS


def is_muhurat_session(day: dt.date) -> bool:
    """True if ``day`` holds a ceremonial Muhurat session the bot deliberately skips."""
    return day.isoformat() in MUHURAT_SESSIONS


# --------------------------------------------------------------------------
# Clock
# --------------------------------------------------------------------------
def now_ist() -> dt.datetime:
    """Current time in IST. The single source of 'now' for the whole bot."""
    return dt.datetime.now(IST)


def to_ist(moment: dt.datetime) -> dt.datetime:
    """Coerce any datetime to IST, treating naive input as already-IST.

    Naive datetimes are ambiguous, but every naive timestamp in this system originates
    from Indian market data, so that assumption is safe here and nowhere else.
    """
    if moment.tzinfo is None:
        return moment.replace(tzinfo=IST)
    return moment.astimezone(IST)


def at(day: dt.date, clock: dt.time) -> dt.datetime:
    """Build an IST-aware datetime from a date and a session time constant."""
    return dt.datetime.combine(day, clock, tzinfo=IST)


def session_window(day: dt.date) -> Optional[tuple[dt.datetime, dt.datetime]]:
    """(open, close) for ``day``, or None if it isn't a trading day."""
    if not is_trading_day(day):
        return None
    return at(day, SESSION_OPEN), at(day, SESSION_CLOSE)


def phase(moment: Optional[dt.datetime] = None) -> Phase:
    """Which session phase ``moment`` (default: now) falls in."""
    moment = to_ist(moment or now_ist())
    if not is_trading_day(moment.date()):
        return Phase.CLOSED

    clock = moment.time()
    if clock < PRE_OPEN_START or clock >= SESSION_CLOSE:
        return Phase.CLOSED
    if clock < SESSION_OPEN:
        return Phase.PRE_OPEN
    if clock < ENTRY_START:
        return Phase.OPENING_RANGE
    if clock < ENTRY_CUTOFF:
        return Phase.ENTRY
    if clock < SQUARE_OFF:
        return Phase.NO_NEW_ENTRY
    return Phase.SQUARE_OFF


def is_market_open(moment: Optional[dt.datetime] = None) -> bool:
    """True between 09:15 and 15:30 on a trading day."""
    return phase(moment) in (
        Phase.OPENING_RANGE, Phase.ENTRY, Phase.NO_NEW_ENTRY, Phase.SQUARE_OFF
    )


def seconds_to_square_off(moment: Optional[dt.datetime] = None) -> Optional[float]:
    """Seconds until the 15:10 flat deadline, or None outside a trading day.

    Negative once the deadline has passed. Strategies use this for time-based exits —
    a position with 4 minutes of runway left is a different proposition from one with 4
    hours, and the strategy should be able to see the difference.
    """
    moment = to_ist(moment or now_ist())
    if not is_trading_day(moment.date()):
        return None
    return (at(moment.date(), SQUARE_OFF) - moment).total_seconds()


def next_trading_day(day: dt.date, *, max_lookahead: int = 30) -> Optional[dt.date]:
    """The next trading day strictly after ``day``.

    Bounded so a wrong holiday file can't spin forever; returns None if no trading day
    is found within the window, which is itself a signal the calendar is broken.
    """
    for offset in range(1, max_lookahead + 1):
        candidate = day + dt.timedelta(days=offset)
        if is_trading_day(candidate):
            return candidate
    log.error("No trading day found within %d days of %s — holiday data looks wrong.",
              max_lookahead, day)
    return None


def trading_days_between(start: dt.date, end: dt.date) -> list[dt.date]:
    """Every trading day in ``[start, end]`` inclusive. Used to find backfill gaps."""
    days, current = [], start
    while current <= end:
        if is_trading_day(current):
            days.append(current)
        current += dt.timedelta(days=1)
    return days
