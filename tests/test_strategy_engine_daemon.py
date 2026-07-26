"""Strategy engine daemon, the live guard CLI, and the shared auth session.

Three modules that finish the audit, grouped because each is the *last* untested surface of
a subsystem whose core is already covered.

**The strategy engine** should have been in the first tier with ingestion, bar building and
execution. `BarResampler` is well covered by `test_resample.py` — including the session-
boundary bug where the final bucket of a day surfaced *after* `on_session_start` had reset
the indicators, anchoring the new day's VWAP on yesterday's close. But `StrategyEngine`
itself, which is where that fix is wired, sat at 76%.

**The live guard CLI** is how a human authorises real money. `evaluate()` is thoroughly
tested; `main()` — the thing anyone actually types — was not.

**The shared auth session** exists because a TOTP code is single-use within its 30-second
window, so three modules logging in independently guarantees two failures. The paths that
had never run are exactly the contended ones: the lock, the wait, and the give-up.
"""
from __future__ import annotations

import datetime as dt
import json

import fakeredis
import pytest

import config
from src import event_bus
from src import market_calendar as cal
from src.bars import Bar

DAY = dt.date(2026, 3, 2)
RELIANCE = "nse_cm:2885"


@pytest.fixture
def client():
    return fakeredis.FakeStrictRedis(server=fakeredis.FakeServer(), decode_responses=True)


@pytest.fixture(autouse=True)
def _bus(monkeypatch, client):
    monkeypatch.setattr(event_bus, "get_client", lambda *a, **k: client)
    monkeypatch.setattr(event_bus, "ping", lambda *a, **k: True)


def bar_at(minute: int, day: dt.date = DAY, price: float = 1300.0) -> Bar:
    start = cal.at(day, cal.SESSION_OPEN) + dt.timedelta(minutes=minute)
    return Bar(RELIANCE, start, start + dt.timedelta(minutes=1),
               price, price + 1, price - 1, price, volume=5000.0, vwap=price,
               tick_count=50)


def signals(client) -> list[dict]:
    return [row[1] for row in client.xrange(config.STREAM_TRADE_SIGNALS)]


# ---------------------------------------------------------------------------
class TestStrategyEngineDaemon:
    @pytest.fixture(autouse=True)
    def _one_minute_buckets(self, monkeypatch):
        """Match the resampler to the fixture bars, so a bucket actually completes.
        At the default 5-minute setting a handful of 1-minute bars closes nothing."""
        monkeypatch.setattr(config, "STRATEGY_BAR_SECONDS", 60)

    def _engine(self, strategy=None):
        from src.strategy_engine import StrategyEngine
        from src.strategies.sma_crossover import SmaCrossover

        return StrategyEngine(strategy or SmaCrossover(short_period=2, long_period=3))

    def test_it_stops_when_redis_is_unreachable(self, monkeypatch, caplog):
        monkeypatch.setattr(event_bus, "ping", lambda *a, **k: False)
        with caplog.at_level("ERROR"):
            self._engine().run()
        assert "Redis not reachable" in caplog.text

    def test_a_session_start_is_announced_once(self, caplog):
        engine = self._engine()
        with caplog.at_level("INFO"):
            for minute in range(4):
                engine._handle_bar(bar_at(minute).to_event())
        assert caplog.text.count("Session 2026-03-02") == 1

    def test_signals_reach_the_bus(self, client):
        from src.risk_engine import Intent
        from src.strategy_base import Strategy

        class AlwaysBuy(Strategy):
            name = "always_buy"

            def on_bar(self, bar, ctx):
                return [self._signal(bar, Intent.OPEN_LONG, stop_loss=bar.close * 0.99)]

        engine = self._engine(AlwaysBuy())
        for minute in range(3):
            engine._handle_bar(bar_at(minute).to_event())
        rows = signals(client)
        assert rows
        assert rows[0]["instrument_id"] == RELIANCE
        assert rows[0]["intent"] == "OPEN_LONG"

    def test_the_signal_carries_provenance(self, client):
        """A signal whose strategy and params cannot be identified afterwards cannot be
        attributed in a reconciliation."""
        from src.risk_engine import Intent
        from src.strategy_base import Strategy

        class AlwaysBuy(Strategy):
            name = "always_buy"

            def on_bar(self, bar, ctx):
                return [self._signal(bar, Intent.OPEN_LONG, stop_loss=1.0)]

        engine = self._engine(AlwaysBuy())
        engine._handle_bar(bar_at(0).to_event())
        row = signals(client)[0]
        assert row["strategy_name"] == "always_buy"
        assert row["bar_time"]

    def test_the_previous_session_is_flushed_before_the_new_one_starts(self, client):
        """The session-boundary bug: without this the last bucket of day one surfaces
        after on_session_start has reset the indicators, anchoring day two's VWAP on
        day one's close."""
        DAY_TWO = dt.date(2026, 3, 4)
        events: list[tuple[str, dt.date]] = []

        from src.strategy_base import Strategy

        class Recorder(Strategy):
            name = "recorder"

            def on_session_start(self, day):
                events.append(("start", day))

            def on_bar(self, bar, ctx):
                events.append(("bar", bar.bar_start.date()))
                return []

        engine = self._engine(Recorder())
        for minute in range(5):
            engine._handle_bar(bar_at(minute, DAY).to_event())
        engine._handle_bar(bar_at(0, DAY_TWO).to_event())

        assert ("start", DAY) in events and ("start", DAY_TWO) in events
        # The property that matters: no day-one bar is dispatched AFTER day two has been
        # announced. That ordering is what keeps day two's VWAP off day one's close.
        day_two_start = events.index(("start", DAY_TWO))
        assert ("bar", DAY) not in events[day_two_start:], (
            "a day-one bar surfaced after on_session_start reset the indicators")
        assert ("bar", DAY) in events[:day_two_start], (
            "day one's final bucket was dropped instead of flushed")

    def test_position_updates_mirror_into_the_strategy_context(self):
        engine = self._engine()
        engine._handle_position_update({"positions": [
            {"instrument_id": RELIANCE, "quantity": 10, "average_price": 1300.0,
             "stop_loss": 1287.0},
        ]})
        assert engine.positions[RELIANCE].quantity == 10
        assert engine.positions[RELIANCE].stop_loss == 1287.0

    def test_an_update_with_no_positions_clears_the_mirror(self):
        """A stale mirror would hand the strategy a position it no longer has, and
        `in_position` would suppress an entry that should have fired."""
        engine = self._engine()
        engine._handle_position_update({"positions": [
            {"instrument_id": RELIANCE, "quantity": 10, "average_price": 1300.0}]})
        engine._handle_position_update({"positions": []})
        assert engine.positions == {}

    def test_a_string_payload_is_ignored_rather_than_crashing(self):
        """Redis round-trips can hand back a JSON string where a list was published."""
        engine = self._engine()
        engine._handle_position_update({"positions": "[]"})
        assert engine.positions == {}

    def test_malformed_position_rows_are_skipped(self):
        engine = self._engine()
        engine._handle_position_update({"positions": [
            {"quantity": 5},                                   # no instrument_id
            "garbage",
            {"instrument_id": RELIANCE, "quantity": 3, "average_price": 1300.0},
        ]})
        assert list(engine.positions) == [RELIANCE]

    def test_one_loop_pass_wires_every_stage(self, monkeypatch):
        engine = self._engine()
        calls = []

        class Stop(RuntimeError):
            pass

        monkeypatch.setattr(engine.position_updates, "read",
                            lambda **k: calls.append("positions") or [])
        monkeypatch.setattr(engine.bars, "claim_stale",
                            lambda **k: calls.append("claim") or [])

        def stop(**k):
            calls.append("bars")
            raise Stop()

        monkeypatch.setattr(engine.bars, "read", stop)
        with pytest.raises(Stop):
            engine.run()
        assert calls == ["positions", "claim", "bars"]

    def test_main_selects_the_named_strategy(self, monkeypatch):
        import src.strategy_engine as mod

        monkeypatch.setattr("sys.argv", ["strategy_engine", "--strategy", "sma_crossover"])
        started = []
        monkeypatch.setattr(mod.StrategyEngine, "run", lambda self: started.append(self))
        mod.main()
        assert started[0].strategy.name == "sma_crossover"

    def test_the_default_strategy_comes_from_config(self, monkeypatch):
        import src.strategy_engine as mod

        monkeypatch.setattr("sys.argv", ["strategy_engine"])
        monkeypatch.setattr(config, "STRATEGY_NAME", "sma_crossover")
        started = []
        monkeypatch.setattr(mod.StrategyEngine, "run", lambda self: started.append(self))
        mod.main()
        assert started[0].strategy.name == "sma_crossover"


class TestLiveGuardCLI:
    """How a human authorises real money. `evaluate()` was tested; the CLI was not."""

    @pytest.fixture(autouse=True)
    def _isolated_paths(self, monkeypatch, tmp_path):
        monkeypatch.setattr(config, "LIVE_AUTH_PATH", str(tmp_path / "authorisation.json"))
        monkeypatch.setattr(config, "SESSION_STORE_ROOT", str(tmp_path / "sessions"))

    def test_check_prints_a_report(self, monkeypatch, capsys):
        import src.live_guard as mod

        monkeypatch.setattr("sys.argv", ["live_guard", "check"])
        mod.main()
        assert capsys.readouterr().out.strip()

    def test_check_reports_blocked_on_a_fresh_machine(self, monkeypatch, capsys):
        """The default answer must be no. A guard that passes out of the box is not one."""
        import src.live_guard as mod

        monkeypatch.setattr("sys.argv", ["live_guard", "check"])
        mod.main()
        assert "BLOCKED" in capsys.readouterr().out.upper()

    def test_authorise_writes_a_record(self, monkeypatch, capsys):
        import src.live_guard as mod

        monkeypatch.setattr("sys.argv", ["live_guard", "authorise",
                                         "--capital", "50000", "--by", "operator"])
        mod.main()
        out = capsys.readouterr().out
        assert "Authorisation written" in out
        assert json.loads(out[out.index("{"):out.rindex("}") + 1])["capital"] == 50000

    def test_authorise_says_it_is_not_sufficient_on_its_own(self, monkeypatch, capsys):
        """Otherwise someone authorises, sees a success message, and assumes they are
        cleared to trade."""
        import src.live_guard as mod

        monkeypatch.setattr("sys.argv", ["live_guard", "authorise",
                                         "--capital", "50000", "--by", "operator"])
        mod.main()
        assert "intent only" in capsys.readouterr().out

    def test_revoke_removes_the_file(self, monkeypatch, capsys):
        import src.live_guard as mod

        monkeypatch.setattr("sys.argv", ["live_guard", "authorise",
                                         "--capital", "50000", "--by", "operator"])
        mod.main()
        monkeypatch.setattr("sys.argv", ["live_guard", "revoke"])
        mod.main()
        assert "Revoked" in capsys.readouterr().out
        assert not mod.authorisation_path().exists()

    def test_revoking_nothing_is_not_an_error(self, monkeypatch, capsys):
        import src.live_guard as mod

        monkeypatch.setattr("sys.argv", ["live_guard", "revoke"])
        mod.main()
        assert "No authorisation on file" in capsys.readouterr().out


class TestSharedAuthSession:
    """A TOTP code is single-use within its 30-second window, so three modules logging in
    independently guarantees two failures. The untested paths were the contended ones."""

    def test_a_stored_session_is_restored_without_logging_in(self, client, monkeypatch):
        from src import auth_session

        logins = []
        monkeypatch.setattr(auth_session, "restore_session",
                            lambda payload: f"restored:{payload['sid']}")
        client.set(auth_session.session_key(DAY),
                   json.dumps({field: "x" for field in auth_session.SESSION_FIELDS}))
        result = auth_session.get_session(client, day=DAY,
                                          login_fn=lambda: logins.append(1))
        assert result == "restored:x"
        assert logins == [], "a usable session was on file and it logged in anyway"

    def test_the_first_caller_logs_in_and_publishes_for_the_others(self, client,
                                                                   monkeypatch):
        from src import auth_session

        class FakeConfiguration:
            def __init__(self):
                for field in auth_session.SESSION_FIELDS:
                    setattr(self, field, "v")

        class FakeClient:
            """The SDK holds session state on `.configuration`, not on the client."""

            configuration = FakeConfiguration()

        logins = []

        def login():
            logins.append(1)
            return FakeClient()

        auth_session.get_session(client, day=DAY, login_fn=login)
        assert logins == [1]
        assert client.exists(auth_session.session_key(DAY)), (
            "the session was not published, so every other module will log in too")

    def test_a_waiter_gives_up_rather_than_racing_the_holder(self, client):
        """Two logins inside one TOTP window means one of them fails. Waiting and then
        giving up is strictly better than both trying."""
        from src import auth_session

        client.set(auth_session.LOGIN_LOCK_KEY, "someone-else", nx=True, ex=60)
        logins = []
        with pytest.raises(Exception):
            auth_session.get_session(
                client, day=DAY, wait_seconds=0.2,
                login_fn=lambda: logins.append(1))
        assert logins == [], "the waiter attempted its own login"

    def test_the_session_is_keyed_by_day(self):
        """A stale session from yesterday is worse than no session: it looks valid and
        fails on first use."""
        from src import auth_session

        assert auth_session.session_key(DAY) != auth_session.session_key(
            dt.date(2026, 3, 4))

    def test_login_can_be_refused_outright(self, client):
        from src import auth_session

        with pytest.raises(Exception):
            auth_session.get_session(client, day=DAY, allow_login=False)
