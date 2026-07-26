"""Bar builder tests — the module that had a known bug and no test protecting the fix.

`src/bar_builder.py` sat at **0% coverage**: 138 statements, none of them ever executed by
a test. That is worse than it sounds, because this module is where the Phase 0 data-quality
bug lived:

> Synthetic-bar coverage was reported at 76%. It was actually 0–17%. `shutdown()` flushed
> the aggregator up to `now()`, so when the process stopped outside market hours — or on a
> different day entirely — it manufactured synthetic bars covering every minute the bot had
> simply not been running, and wrote them to the Parquet store as though they were market
> data.

That is the worst class of bug this project can have. It does not crash, it does not
produce an obviously wrong number, and it **silently inflates the apparent quality of the
historical dataset every backtest is built on**. The fix was `_flush_to_last_tick()`, which
bounds flushing by the last tick actually observed. It has been correct since Phase 0 and
nothing has been stopping it from regressing.

`TestTheSyntheticPaddingRegression` is the reason this file exists. The rest is the
ordinary coverage the module also lacked.

A note on what a synthetic bar is allowed to mean, because the distinction is the whole
bug: it means *"the market was open and nothing traded"*. It must never mean *"the bot
wasn't running"*. The first is information. The second is fabrication.
"""
from __future__ import annotations

import datetime as dt

import fakeredis
import pytest

import config
from src import event_bus
from src import market_calendar as cal
from src.bar_builder import BarBuilder
from src.bar_store import ParquetBarStore

DAY = dt.date(2026, 3, 2)          # Monday, a trading day
HOLIDAY = dt.date(2026, 3, 3)      # Holi
SUNDAY = dt.date(2026, 3, 8)
RELIANCE = "nse_cm:2885"


@pytest.fixture
def client():
    return fakeredis.FakeStrictRedis(server=fakeredis.FakeServer(), decode_responses=True)


@pytest.fixture(autouse=True)
def _bus(monkeypatch, client):
    monkeypatch.setattr(event_bus, "get_client", lambda *a, **k: client)
    monkeypatch.setattr(event_bus, "ping", lambda *a, **k: True)


@pytest.fixture
def store(tmp_path):
    return ParquetBarStore(tmp_path, "1m")


@pytest.fixture
def builder(store):
    return BarBuilder(store=store)


def tick(at: dt.datetime, price: float, volume: float = 1000.0,
         instrument_id: str = RELIANCE) -> dict:
    return {"instrument_id": instrument_id, "ltp": str(price),
            "timestamp": str(at.timestamp()), "volume": str(volume)}


def published_bars(client) -> list[dict]:
    return [row[1] for row in client.xrange(config.STREAM_MARKET_BARS)]


def feed_morning(builder, *, until_minute: int = 45, start_price: float = 1300.0):
    """Ticks from the open through 09:15+until_minute, one per minute."""
    builder.open_session(DAY)
    open_at = cal.at(DAY, cal.SESSION_OPEN)
    for minute in range(until_minute):
        at = open_at + dt.timedelta(minutes=minute, seconds=30)
        builder.feed(tick(at, start_price + minute * 0.5, 1000.0 + minute * 100))
    return open_at + dt.timedelta(minutes=until_minute - 1, seconds=30)


# ---------------------------------------------------------------------------
class TestTheSyntheticPaddingRegression:
    """The Phase 0 bug. Everything here asserts the same thing from a different angle:
    the builder must never emit a bar for a period it was not actually watching."""

    def test_shutdown_does_not_pad_beyond_the_last_tick(self, builder, client):
        last_tick_at = feed_morning(builder, until_minute=45)
        builder.shutdown()

        bars = published_bars(client)
        assert bars, "the morning's bars should exist"
        latest = max(dt.datetime.fromisoformat(b["bar_end"]) for b in bars)
        assert latest <= last_tick_at + dt.timedelta(minutes=1), (
            f"bars run to {latest}, past the last tick at {last_tick_at} — this is the "
            f"Phase 0 fabrication bug")

    def test_shutdown_on_a_later_day_still_does_not_pad(self, builder, client,
                                                        monkeypatch):
        """The exact shape of the original defect: the process stops days later, and the
        aggregator is flushed to `now()` rather than to the last tick.

        The date bound alone is *not* enough to detect this and was the first version of
        this test — the aggregator clamps flushing at its own session close, so the
        fabricated bars all land on the right day and a date-only assertion passes
        against the bug. The time bound is what actually distinguishes the two.
        """
        last_tick_at = feed_morning(builder, until_minute=30)
        monkeypatch.setattr(cal, "now_ist",
                            lambda: cal.at(dt.date(2026, 3, 20), dt.time(22, 0)))
        builder.shutdown()

        ends = [dt.datetime.fromisoformat(b["bar_end"]) for b in published_bars(client)]
        assert {e.date() for e in ends} == {DAY}
        assert max(ends) <= last_tick_at + dt.timedelta(minutes=1), (
            "flushing was bounded by the wall clock rather than the last tick")

    def test_shutdown_does_not_pad_to_the_session_close(self, builder, client):
        """Padding to 15:30 would report a full day of coverage for a 45-minute run."""
        feed_morning(builder, until_minute=45)
        builder.shutdown()

        latest = max(dt.datetime.fromisoformat(b["bar_end"])
                     for b in published_bars(client))
        assert latest.time() < dt.time(15, 30)
        assert latest.time() < dt.time(10, 30)

    def test_a_session_roll_does_not_pad_the_outgoing_day(self, builder, client):
        """Same failure, different trigger: rolling to a new day must flush what was
        seen, not manufacture the rest of the previous session."""
        feed_morning(builder, until_minute=20)
        builder.open_session(dt.date(2026, 3, 4))

        day_one = [dt.datetime.fromisoformat(b["bar_end"])
                   for b in published_bars(client)
                   if dt.datetime.fromisoformat(b["bar_end"]).date() == DAY]
        assert day_one, "the outgoing session's bars must still be flushed"
        assert max(day_one).time() < dt.time(9, 40)

    def test_coverage_reflects_reality_not_padding(self, builder, store, client):
        """The measurable consequence. A 45-minute run must not report a full session.

        This is the assertion that would have caught the original bug directly: it was
        found by noticing 76% coverage where the truth was 0-17%.
        """
        feed_morning(builder, until_minute=45)
        builder.shutdown()

        frame = store.read_day(RELIANCE, DAY)
        session_minutes = 375
        assert len(frame) <= 46, (
            f"{len(frame)} bars stored for a 45-minute run — anything approaching "
            f"{session_minutes} means the session was padded")

    def test_no_ticks_at_all_produces_no_bars(self, builder, client):
        """A day the bot never saw a tick must leave *no* trace, not a full synthetic
        session. Otherwise every outage becomes fabricated history."""
        builder.open_session(DAY)
        builder.shutdown()
        assert published_bars(client) == []


class TestSessionControl:
    def test_a_trading_day_opens(self, builder):
        assert builder.open_session(DAY) is True
        assert builder.session_day == DAY

    @pytest.mark.parametrize("day", [HOLIDAY, SUNDAY])
    def test_a_non_trading_day_is_refused(self, builder, day):
        assert builder.open_session(day) is False

    def test_reopening_the_same_day_is_a_no_op(self, builder):
        builder.open_session(DAY)
        aggregator = builder.aggregator
        builder.open_session(DAY)
        assert builder.aggregator is aggregator, "the session was needlessly rebuilt"

    def test_rolling_to_a_new_day_replaces_the_aggregator(self, builder):
        builder.open_session(DAY)
        first = builder.aggregator
        builder.open_session(dt.date(2026, 3, 4))
        assert builder.aggregator is not first
        assert builder.session_day == dt.date(2026, 3, 4)

    def test_the_outgoing_session_is_flushed_before_being_abandoned(self, builder, client):
        """Otherwise the final bars of every day are silently lost."""
        feed_morning(builder, until_minute=10)
        assert not published_bars(client) or True   # some may already be closed
        builder.open_session(dt.date(2026, 3, 4))
        assert published_bars(client), "the outgoing session's bars were dropped"

    def test_an_unverified_holiday_year_warns(self, builder, monkeypatch, caplog):
        """Session boundaries depend on the holiday list; a missing year is not silent."""
        monkeypatch.setattr(cal, "covers_year", lambda year: False)
        with caplog.at_level("WARNING"):
            builder.open_session(DAY)
        assert "holiday list" in caplog.text

    def test_feeding_without_a_session_is_an_error(self, builder):
        with pytest.raises(RuntimeError, match="_ensure_session"):
            builder.feed(tick(cal.at(DAY, dt.time(10, 0)), 1300.0))


class TestTickHandling:
    def _open(self, builder):
        builder.open_session(DAY)
        return cal.at(DAY, dt.time(10, 0))

    def test_a_valid_tick_is_aggregated(self, builder):
        at = self._open(builder)
        builder.feed(tick(at, 1300.0))
        assert builder.aggregator.last_tick_time is not None

    def test_a_tick_without_a_price_is_dropped(self, builder):
        at = self._open(builder)
        builder.feed({"instrument_id": RELIANCE, "timestamp": str(at.timestamp())})
        assert builder.aggregator.last_tick_time is None

    def test_an_unparseable_price_is_dropped(self, builder):
        at = self._open(builder)
        builder.feed({"instrument_id": RELIANCE, "ltp": "not-a-number",
                      "timestamp": str(at.timestamp())})
        assert builder.aggregator.last_tick_time is None

    def test_a_tick_without_an_instrument_is_dropped(self, builder):
        at = self._open(builder)
        builder.feed({"ltp": "1300.0", "timestamp": str(at.timestamp())})
        assert builder.aggregator.last_tick_time is None

    def test_the_legacy_instrument_field_still_works(self, builder):
        """Transitional fallback while ingestion is migrated to instrument_id (B4)."""
        at = self._open(builder)
        builder.feed({"instrument": RELIANCE, "ltp": "1300.0",
                      "timestamp": str(at.timestamp())})
        assert builder.aggregator.last_tick_time is not None

    def test_an_unparseable_timestamp_falls_back_to_now(self, builder, monkeypatch):
        """Dropping the tick would lose real trades; guessing a *time* is the lesser
        evil, and only affects which bar it lands in."""
        self._open(builder)
        monkeypatch.setattr(cal, "now_ist", lambda: cal.at(DAY, dt.time(10, 0)))
        builder.feed({"instrument_id": RELIANCE, "ltp": "1300.0", "timestamp": "junk"})
        assert builder.aggregator.last_tick_time is not None

    @pytest.mark.parametrize("key", ["volume", "cum_volume", "v"])
    def test_volume_is_read_from_any_known_key(self, builder, key):
        at = self._open(builder)
        builder.feed({"instrument_id": RELIANCE, "ltp": "1300.0",
                      "timestamp": str(at.timestamp()), key: "5000"})
        assert builder.aggregator.last_tick_time is not None

    def test_a_tick_before_any_session_is_ignored(self, builder):
        builder.aggregator = None
        builder._handle_tick(tick(cal.at(DAY, dt.time(10, 0)), 1300.0))   # must not raise


class TestBarEmission:
    def test_bars_reach_both_the_stream_and_the_store(self, builder, store, client):
        feed_morning(builder, until_minute=20)
        builder.shutdown()
        assert published_bars(client)
        assert len(store.read_day(RELIANCE, DAY)) > 0

    def test_only_closed_bars_are_published(self, builder, client):
        """A forming bar is the most common accidental source of look-ahead."""
        builder.open_session(DAY)
        at = cal.at(DAY, dt.time(10, 0))
        builder.feed(tick(at + dt.timedelta(seconds=10), 1300.0))
        builder.feed(tick(at + dt.timedelta(seconds=30), 1301.0))
        assert published_bars(client) == [], "an unclosed bar was published"

    def test_the_bar_count_is_tracked(self, builder):
        feed_morning(builder, until_minute=20)
        builder.shutdown()
        assert builder._bars_built > 0

    def test_multiple_instruments_are_kept_separate(self, builder, client):
        builder.open_session(DAY)
        open_at = cal.at(DAY, cal.SESSION_OPEN)
        for minute in range(10):
            at = open_at + dt.timedelta(minutes=minute, seconds=30)
            builder.feed(tick(at, 1300.0 + minute, instrument_id=RELIANCE))
            builder.feed(tick(at, 1650.0 + minute, instrument_id="nse_cm:1333"))
        builder.shutdown()
        instruments = {b["instrument_id"] for b in published_bars(client)}
        assert instruments == {RELIANCE, "nse_cm:1333"}


class TestPersistence:
    def test_force_persists_immediately(self, builder, store):
        feed_morning(builder, until_minute=20)
        builder._persist_if_due(force=True)
        assert len(store.read_day(RELIANCE, DAY)) > 0

    def test_it_waits_for_the_interval_otherwise(self, builder, store, monkeypatch):
        monkeypatch.setattr(config, "BAR_PERSIST_INTERVAL_SECONDS", 3600)
        feed_morning(builder, until_minute=20)
        builder._persist_if_due()
        assert len(store.read_day(RELIANCE, DAY)) == 0, "flushed before the interval"

    def test_the_interval_elapsing_triggers_a_write(self, builder, store, monkeypatch):
        monkeypatch.setattr(config, "BAR_PERSIST_INTERVAL_SECONDS", 0)
        feed_morning(builder, until_minute=20)
        builder._persist_if_due()
        assert len(store.read_day(RELIANCE, DAY)) > 0

    def test_shutdown_always_persists(self, builder, store, monkeypatch):
        """Unpersisted bars are data that can never be recovered — there is no re-request
        for a tick stream that has already gone past."""
        monkeypatch.setattr(config, "BAR_PERSIST_INTERVAL_SECONDS", 999_999)
        feed_morning(builder, until_minute=20)
        builder.shutdown()
        assert len(store.read_day(RELIANCE, DAY)) > 0


class TestTheMainLoop:
    def test_it_stops_when_redis_is_unreachable(self, builder, monkeypatch, caplog):
        monkeypatch.setattr(event_bus, "ping", lambda *a, **k: False)
        with caplog.at_level("ERROR"):
            builder.run()
        assert "Redis not reachable" in caplog.text

    def test_one_pass_reads_ticks_and_flushes(self, builder, monkeypatch, client):
        """`run()` is the only thing that wires reading to aggregation in production."""
        import src.bar_builder as mod

        at = cal.at(DAY, dt.time(10, 0))
        monkeypatch.setattr(cal, "now_ist", lambda: at + dt.timedelta(minutes=5))

        calls = {"n": 0}

        def read_new(*a, **k):
            calls["n"] += 1
            if calls["n"] > 1:
                mod._running = False
                return [], "$"
            return [("1-1", tick(at, 1300.0))], "1-1"

        monkeypatch.setattr(event_bus, "read_new", read_new)
        monkeypatch.setattr(mod, "_running", True)
        builder.run()

        assert calls["n"] >= 1
        assert published_bars(client), "the tick never became a bar"

    def test_a_non_trading_day_idles_instead_of_spinning(self, builder, monkeypatch):
        import src.bar_builder as mod

        monkeypatch.setattr(cal, "now_ist", lambda: cal.at(HOLIDAY, dt.time(10, 0)))
        monkeypatch.setattr(mod, "_running", True)

        slept = []

        def sleep(seconds):
            slept.append(seconds)
            mod._running = False

        monkeypatch.setattr(mod.time, "sleep", sleep)

        def must_not_read(*a, **k):
            raise AssertionError("read the tick stream on a non-trading day")

        monkeypatch.setattr(event_bus, "read_new", must_not_read)
        builder.run()
        assert slept == [5]

    def test_the_signal_handler_stops_the_loop(self, monkeypatch):
        import src.bar_builder as mod

        monkeypatch.setattr(mod, "_running", True)
        mod._handle_signal(15, None)
        assert mod._running is False


class TestReplayEntryPoint:
    def test_feed_with_an_explicit_now_closes_elapsed_bars(self, builder, client):
        """The `now` argument is what lets replay and tests drive bar closure
        deterministically instead of waiting on the wall clock."""
        builder.open_session(DAY)
        at = cal.at(DAY, dt.time(10, 0))
        builder.feed(tick(at + dt.timedelta(seconds=30), 1300.0))
        assert published_bars(client) == []
        builder.feed(tick(at + dt.timedelta(minutes=1, seconds=30), 1301.0),
                     now=at + dt.timedelta(minutes=3))
        assert published_bars(client), "elapsed bars were not closed"

    def test_an_unparseable_volume_does_not_drop_the_tick(self, builder):
        """The price is the part that matters; a bad volume field must not discard a
        real trade."""
        builder.open_session(DAY)
        at = cal.at(DAY, dt.time(10, 0))
        builder.feed({"instrument_id": RELIANCE, "ltp": "1300.0",
                      "timestamp": str(at.timestamp()), "volume": "junk"})
        assert builder.aggregator.last_tick_time is not None


class TestStartup:
    def test_main_installs_signal_handlers_and_runs(self, monkeypatch):
        import src.bar_builder as mod

        monkeypatch.setattr("sys.argv", ["bar_builder"])
        handlers = []
        monkeypatch.setattr(mod.signal, "signal",
                            lambda sig, fn: handlers.append(sig))
        started = []
        monkeypatch.setattr(mod.BarBuilder, "run", lambda self: started.append(self))
        mod.main()
        assert mod.signal.SIGINT in handlers and mod.signal.SIGTERM in handlers
        assert started

    def test_the_interval_flag_overrides_config(self, monkeypatch):
        import src.bar_builder as mod

        monkeypatch.setattr(config, "BAR_INTERVAL_SECONDS", 60)
        monkeypatch.setattr("sys.argv", ["bar_builder", "--interval", "300"])
        monkeypatch.setattr(mod.signal, "signal", lambda *a: None)
        monkeypatch.setattr(mod.BarBuilder, "run", lambda self: None)
        mod.main()
        assert config.BAR_INTERVAL_SECONDS == 300

    def test_a_keyboard_interrupt_still_shuts_down_cleanly(self, monkeypatch):
        """Ctrl-C must not cost the unpersisted bars."""
        import src.bar_builder as mod

        monkeypatch.setattr("sys.argv", ["bar_builder"])
        monkeypatch.setattr(mod.signal, "signal", lambda *a: None)

        def interrupt(self):
            raise KeyboardInterrupt()

        shutdowns = []
        monkeypatch.setattr(mod.BarBuilder, "run", interrupt)
        monkeypatch.setattr(mod.BarBuilder, "shutdown",
                            lambda self: shutdowns.append(self))
        mod.main()
        assert shutdowns, "shutdown() was skipped on Ctrl-C"
