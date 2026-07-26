"""Data ingestion tests — the module every tick enters through.

`src/data_ingestion.py` was the last module in the codebase at **0% coverage**: 125
statements, none ever executed by a test. It is the bot's sensory system. Everything
downstream — bars, indicators, signals, risk, orders — is a transformation of what this
module publishes, so a defect here is not contained: it is silently inherited by every
other module and shows up as a bad trade rather than as an error.

Two failure modes drive most of what is tested here.

**Reconnecting forever looks resilient and is not.** The process stays alive, passes every
liveness check, keeps its Redis connection, and never trades. That is strictly worse than
crashing, because nothing pages anyone. `start_live_feed` bounds its retries and halts the
bot loudly when they are exhausted — and that halt is the behaviour most worth pinning,
because it only ever runs on the worst day.

**A display name is not an identifier (B4).** This is where the distinction is born. The
canonical id is built from the exchange token, never from the instrument name, because the
name is not unique and is not what the broker keys on. Sending a name where an id belongs
is what broke the execution path once already.

The rest is the ordinary robustness a tick parser needs: one malformed record out of a
thousand must not take down the feed, and a liveness-key failure must never be able to stop
a tick being published.
"""
from __future__ import annotations

import json

import fakeredis
import pytest

import config
from src import event_bus, feed_health
import src.data_ingestion as ingestion


@pytest.fixture
def client():
    return fakeredis.FakeStrictRedis(server=fakeredis.FakeServer(), decode_responses=True)


@pytest.fixture(autouse=True)
def _bus(monkeypatch, client):
    # `_redis` is bound at import time, so patching event_bus.get_client is not enough.
    monkeypatch.setattr(ingestion, "_redis", client)
    monkeypatch.setattr(event_bus, "ping", lambda *a, **k: True)
    monkeypatch.setattr(ingestion, "_running", True)


def ticks(client) -> list[dict]:
    return [row[1] for row in client.xrange(config.STREAM_MARKET_TICKS)]


# ---------------------------------------------------------------------------
class TestPublishTick:
    def test_a_tick_reaches_the_bus(self, client):
        ingestion._publish_tick("RELIANCE", 1300.5, instrument_id="nse_cm:2885")
        rows = ticks(client)
        assert len(rows) == 1
        assert float(rows[0]["ltp"]) == 1300.5

    def test_the_canonical_id_and_display_name_are_kept_distinct(self, client):
        """B4 in one assertion. Collapsing these is what put a display name into an
        order and made it an order on the wrong instrument."""
        ingestion._publish_tick("RELIANCE", 1300.5, instrument_id="nse_cm:2885")
        row = ticks(client)[0]
        assert row["instrument_id"] == "nse_cm:2885"
        assert row["instrument"] == "RELIANCE"

    def test_without_an_id_the_display_name_is_used(self, client):
        """A documented transitional fallback. It is tested so that if it ever becomes
        the *normal* path again, that shows up as a deliberate change."""
        ingestion._publish_tick("RELIANCE", 1300.5)
        row = ticks(client)[0]
        assert row["instrument_id"] == "RELIANCE"

    def test_a_timestamp_is_always_stamped(self, client):
        ingestion._publish_tick("RELIANCE", 1300.5, instrument_id="nse_cm:2885")
        assert float(ticks(client)[0]["timestamp"]) > 0

    def test_extra_fields_are_carried(self, client):
        ingestion._publish_tick("RELIANCE", 1300.5, {"volume": 5000},
                                instrument_id="nse_cm:2885")
        assert float(ticks(client)[0]["volume"]) == 5000

    def test_liveness_is_refreshed(self, client):
        ingestion._publish_tick("RELIANCE", 1300.5, instrument_id="nse_cm:2885")
        assert feed_health.is_feed_live(client, "nse_cm:2885")

    def test_a_liveness_failure_never_blocks_the_tick(self, client, monkeypatch):
        """Liveness is observability. It must not be able to stop market data."""
        monkeypatch.setattr(feed_health, "publish_liveness",
                            lambda *a, **k: (_ for _ in ()).throw(ConnectionError("down")))
        ingestion._publish_tick("RELIANCE", 1300.5, instrument_id="nse_cm:2885")
        assert len(ticks(client)) == 1


class TestTickParsing:
    def _one(self, client):
        rows = ticks(client)
        assert len(rows) == 1
        return rows[0]

    @pytest.mark.parametrize("field", ["ltp", "last_traded_price", "lp"])
    def test_every_known_price_field_is_read(self, client, field):
        ingestion._on_message({field: "1300.5", "tk": "2885", "e": "nse_cm"})
        assert float(self._one(client)["ltp"]) == 1300.5

    def test_a_single_dict_and_a_list_are_both_accepted(self, client):
        ingestion._on_message({"ltp": "1300.5", "tk": "2885"})
        ingestion._on_message([{"ltp": "1301.5", "tk": "2885"}])
        assert len(ticks(client)) == 2

    def test_the_canonical_id_is_built_from_the_token_not_the_name(self, client):
        """The name is not unique and is not what the broker keys on (B4)."""
        ingestion._on_message({"ltp": "1300.5", "tk": "2885",
                               "trading_symbol": "RELIANCE-EQ", "e": "nse_cm"})
        row = self._one(client)
        assert row["instrument_id"] == "nse_cm:2885"
        assert row["instrument"] == "RELIANCE-EQ"

    def test_the_default_segment_is_used_when_absent(self, client):
        ingestion._on_message({"ltp": "1300.5", "tk": "2885"})
        assert self._one(client)["instrument_id"] == \
            f"{config.DEFAULT_EXCHANGE_SEGMENT}:2885"

    @pytest.mark.parametrize("token_field", ["tk", "instrument_token", "pSymbol"])
    def test_every_known_token_field_is_read(self, client, token_field):
        ingestion._on_message({"ltp": "1300.5", token_field: "2885", "e": "nse_cm"})
        assert self._one(client)["instrument_id"] == "nse_cm:2885"

    @pytest.mark.parametrize("volume_field", ["v", "volume", "cum_volume"])
    def test_every_known_volume_field_is_read(self, client, volume_field):
        ingestion._on_message({"ltp": "1300.5", "tk": "2885", volume_field: "9000"})
        assert float(self._one(client)["volume"]) == 9000

    def test_a_record_without_a_price_is_skipped(self, client):
        ingestion._on_message({"tk": "2885", "trading_symbol": "RELIANCE-EQ"})
        assert ticks(client) == []

    def test_a_non_dict_record_is_skipped(self, client):
        ingestion._on_message(["not-a-dict", 42, None])
        assert ticks(client) == []

    def test_one_bad_record_does_not_discard_the_good_ones(self, client):
        """A thousand ticks a second arrive here. One malformed payload must not cost
        the rest of the batch."""
        ingestion._on_message([
            {"ltp": "1300.5", "tk": "2885"},
            "garbage",
            {"tk": "1333"},                      # no price
            {"ltp": "1650.0", "tk": "1333"},
        ])
        assert len(ticks(client)) == 2

    def test_an_unparseable_price_is_logged_not_raised(self, client, caplog):
        """This runs inside an SDK callback. Raising would propagate into the socket
        thread, and the failure would surface as a dead feed rather than a bad tick."""
        with caplog.at_level("WARNING"):
            ingestion._on_message({"ltp": "not-a-number", "tk": "2885"})
        assert ticks(client) == []
        assert "Could not parse tick" in caplog.text

    def test_an_unknown_instrument_still_publishes(self, client):
        """A tick with no identifiable name is still market data; it is labelled, not
        dropped."""
        ingestion._on_message({"ltp": "1300.5"})
        assert self._one(client)["instrument"] == "UNKNOWN"

    def test_the_raw_keys_are_carried_for_diagnosis(self, client):
        """The broker's payload shape is only partly documented; recording which keys
        actually arrived is how the mapping gets verified against reality."""
        ingestion._on_message({"ltp": "1300.5", "tk": "2885", "xyz": 1})
        assert "xyz" in json.loads(self._one(client)["raw_keys"])


class TestSocketCallbacks:
    def test_an_error_is_logged(self, caplog):
        with caplog.at_level("ERROR"):
            ingestion._on_error("connection reset")
        assert "connection reset" in caplog.text

    def test_a_close_is_logged_as_a_warning(self, caplog):
        """A closed socket is not routine — downstream keeps computing on a price that
        has stopped updating."""
        with caplog.at_level("WARNING"):
            ingestion._on_close("closed")
        assert "WebSocket Close" in caplog.text

    def test_an_open_is_logged(self, caplog):
        with caplog.at_level("INFO"):
            ingestion._on_open("connected")
        assert "WebSocket Open" in caplog.text


class TestConnectAndSubscribe:
    def test_callbacks_are_wired_before_subscribing(self, monkeypatch):
        """Subscribing first would drop every tick that arrives before the handler is
        attached."""
        order = []

        class FakeClient:
            def __init__(self):
                self._on_message = None

            def __setattr__(self, name, value):
                if name.startswith("on_"):
                    order.append(name)
                object.__setattr__(self, name, value)

            def subscribe(self, **kwargs):
                order.append("subscribe")

        monkeypatch.setattr("src.auth_session.get_session", lambda *a, **k: FakeClient())
        ingestion._connect_and_subscribe()
        assert order.index("on_message") < order.index("subscribe")

    def test_the_configured_instruments_are_subscribed(self, monkeypatch):
        sent = {}

        class FakeClient:
            def subscribe(self, **kwargs):
                sent.update(kwargs)

        monkeypatch.setattr("src.auth_session.get_session", lambda *a, **k: FakeClient())
        _, tokens = ingestion._connect_and_subscribe()
        assert len(tokens) == len(config.INSTRUMENTS)
        assert sent["instrument_tokens"] == tokens


class TestReconnection:
    """Bounded retries, then a loud halt.

    Retrying forever is the failure this guards: the process survives, passes liveness
    checks, holds its Redis connection, and never trades. Nothing pages anyone.
    """

    def _no_sleep(self, monkeypatch):
        monkeypatch.setattr(ingestion.time, "sleep", lambda s: None)

    def test_it_halts_the_bot_once_retries_are_exhausted(self, client, monkeypatch):
        self._no_sleep(monkeypatch)
        monkeypatch.setattr(config, "RECONNECT_MAX_ATTEMPTS", 2)
        monkeypatch.setattr(ingestion, "_connect_and_subscribe",
                            lambda: (_ for _ in ()).throw(ConnectionError("no route")))

        ingestion.start_live_feed()

        from src.kill_switch import KillSwitch
        assert KillSwitch(client).state().halted

    def test_the_halt_reason_names_the_feed(self, client, monkeypatch):
        """Whoever reads it at 09:20 needs to know which subsystem died."""
        self._no_sleep(monkeypatch)
        monkeypatch.setattr(config, "RECONNECT_MAX_ATTEMPTS", 1)
        monkeypatch.setattr(ingestion, "_connect_and_subscribe",
                            lambda: (_ for _ in ()).throw(ConnectionError("no route")))
        ingestion.start_live_feed()

        from src.kill_switch import KillSwitch
        assert "feed" in KillSwitch(client).state().reason.lower()

    def test_it_retries_the_configured_number_of_times(self, monkeypatch):
        self._no_sleep(monkeypatch)
        monkeypatch.setattr(config, "RECONNECT_MAX_ATTEMPTS", 3)
        attempts = []

        def fail():
            attempts.append(1)
            raise ConnectionError("no route")

        monkeypatch.setattr(ingestion, "_connect_and_subscribe", fail)
        ingestion.start_live_feed()
        assert len(attempts) == 4, "3 retries after the initial attempt"

    def test_it_does_not_halt_while_retries_remain(self, client, monkeypatch):
        self._no_sleep(monkeypatch)
        monkeypatch.setattr(config, "RECONNECT_MAX_ATTEMPTS", 5)
        calls = []

        def fail_then_stop():
            calls.append(1)
            if len(calls) >= 2:
                ingestion._running = False
            raise ConnectionError("no route")

        monkeypatch.setattr(ingestion, "_connect_and_subscribe", fail_then_stop)
        ingestion.start_live_feed()

        from src.kill_switch import KillSwitch
        assert not KillSwitch(client).state().halted

    def test_a_successful_connection_resets_the_backoff(self, monkeypatch):
        """Otherwise a feed that drops once an hour eventually exhausts its budget and
        halts a perfectly healthy bot."""
        resets = []

        class Policy(feed_health.ReconnectPolicy):
            def reset(self):
                resets.append(1)
                super().reset()

        monkeypatch.setattr(feed_health, "ReconnectPolicy", Policy)
        monkeypatch.setattr(ingestion, "_connect_and_subscribe",
                            lambda: (object(), [{"instrument_token": "1"}]))

        def stop(_seconds):
            ingestion._running = False

        monkeypatch.setattr(ingestion.time, "sleep", stop)
        ingestion.start_live_feed()
        assert resets == [1]

    def test_it_unsubscribes_on_a_clean_stop(self, monkeypatch):
        """Leaving a subscription open wastes a slot against the broker's cap."""
        unsubscribed = []

        class FakeClient:
            def un_subscribe(self, **kwargs):
                unsubscribed.append(kwargs)

        monkeypatch.setattr(ingestion, "_connect_and_subscribe",
                            lambda: (FakeClient(), [{"instrument_token": "1"}]))

        def stop(_seconds):
            ingestion._running = False

        monkeypatch.setattr(ingestion.time, "sleep", stop)
        ingestion.start_live_feed()
        assert unsubscribed

    def test_a_failing_unsubscribe_does_not_mask_the_shutdown(self, monkeypatch, caplog):
        class FakeClient:
            def un_subscribe(self, **kwargs):
                raise RuntimeError("session already gone")

        monkeypatch.setattr(ingestion, "_connect_and_subscribe",
                            lambda: (FakeClient(), [{"instrument_token": "1"}]))

        def stop(_seconds):
            ingestion._running = False

        monkeypatch.setattr(ingestion.time, "sleep", stop)
        with caplog.at_level("INFO"):
            ingestion.start_live_feed()
        assert "Live feed stopped" in caplog.text


class TestSimulatedFeed:
    def test_it_publishes_ticks_without_a_broker(self, client, monkeypatch):
        published = []

        def stop(_seconds):
            published.append(1)
            ingestion._running = False

        monkeypatch.setattr(ingestion.time, "sleep", stop)
        ingestion.start_simulated_feed(interval=0)
        assert ticks(client)

    def test_it_uses_canonical_ids(self, client, monkeypatch):
        """A simulator that shortcuts the identifier path would hide B4-class bugs
        precisely where they are cheapest to find."""
        monkeypatch.setattr(ingestion.time, "sleep",
                            lambda s: setattr(ingestion, "_running", False))
        ingestion.start_simulated_feed(interval=0)
        assert all(":" in row["instrument_id"] for row in ticks(client))

    def test_volume_is_cumulative(self, client, monkeypatch):
        """The broker reports cumulative volume, so the bar builder differences it. A
        simulator emitting per-tick volume would bypass that logic entirely."""
        rounds = {"n": 0}

        def stop(_seconds):
            rounds["n"] += 1
            if rounds["n"] >= 3:
                ingestion._running = False

        monkeypatch.setattr(ingestion.time, "sleep", stop)
        ingestion.start_simulated_feed(interval=0)

        by_instrument: dict[str, list[float]] = {}
        for row in ticks(client):
            by_instrument.setdefault(row["instrument_id"], []).append(float(row["volume"]))
        for series in by_instrument.values():
            assert series == sorted(series), "volume went backwards"

    def test_it_falls_back_to_an_instrument_when_none_are_configured(self, client,
                                                                    monkeypatch):
        monkeypatch.setattr(config, "INSTRUMENTS", [])
        monkeypatch.setattr(ingestion.time, "sleep",
                            lambda s: setattr(ingestion, "_running", False))
        ingestion.start_simulated_feed(interval=0)
        assert ticks(client), "the simulator produced nothing to work with"

    def test_prices_stay_positive(self, client, monkeypatch):
        rounds = {"n": 0}

        def stop(_seconds):
            rounds["n"] += 1
            if rounds["n"] >= 5:
                ingestion._running = False

        monkeypatch.setattr(ingestion.time, "sleep", stop)
        ingestion.start_simulated_feed(interval=0)
        assert all(float(row["ltp"]) > 0 for row in ticks(client))


class TestStartup:
    def test_it_refuses_to_start_without_redis(self, monkeypatch, caplog):
        monkeypatch.setattr("sys.argv", ["data_ingestion"])
        monkeypatch.setattr(event_bus, "ping", lambda *a, **k: False)
        monkeypatch.setattr(ingestion.signal, "signal", lambda *a: None)
        started = []
        monkeypatch.setattr(ingestion, "start_live_feed", lambda: started.append(1))
        with caplog.at_level("ERROR"):
            ingestion.main()
        assert started == []
        assert "Redis is not reachable" in caplog.text

    def test_simulate_selects_the_synthetic_feed(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["data_ingestion", "--simulate"])
        monkeypatch.setattr(ingestion.signal, "signal", lambda *a: None)
        chosen = []
        monkeypatch.setattr(ingestion, "start_simulated_feed",
                            lambda interval: chosen.append(("sim", interval)))
        monkeypatch.setattr(ingestion, "start_live_feed",
                            lambda: chosen.append(("live", None)))
        ingestion.main()
        assert chosen == [("sim", 0.2)]

    def test_the_default_is_the_live_feed(self, monkeypatch):
        """The safe-looking default must not silently be the simulator — a simulated
        feed that looks live is how you paper-trade fictional prices by accident."""
        monkeypatch.setattr("sys.argv", ["data_ingestion"])
        monkeypatch.setattr(ingestion.signal, "signal", lambda *a: None)
        chosen = []
        monkeypatch.setattr(ingestion, "start_live_feed",
                            lambda: chosen.append("live"))
        ingestion.main()
        assert chosen == ["live"]

    def test_the_interval_flag_is_honoured(self, monkeypatch):
        monkeypatch.setattr("sys.argv",
                            ["data_ingestion", "--simulate", "--interval", "0.5"])
        monkeypatch.setattr(ingestion.signal, "signal", lambda *a: None)
        chosen = []
        monkeypatch.setattr(ingestion, "start_simulated_feed",
                            lambda interval: chosen.append(interval))
        ingestion.main()
        assert chosen == [0.5]

    def test_network_calls_are_bounded_before_anything_connects(self, monkeypatch):
        """The SDK sets no per-request timeout; without this a hung broker call blocks
        the process indefinitely."""
        monkeypatch.setattr("sys.argv", ["data_ingestion"])
        monkeypatch.setattr(ingestion.signal, "signal", lambda *a: None)
        monkeypatch.setattr(ingestion, "start_live_feed", lambda: None)
        bounded = []
        monkeypatch.setattr(ingestion.kotak_api, "bound_network_calls",
                            lambda *a, **k: bounded.append(1))
        ingestion.main()
        assert bounded == [1]

    def test_signal_handlers_are_installed(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["data_ingestion"])
        monkeypatch.setattr(ingestion, "start_live_feed", lambda: None)
        installed = []
        monkeypatch.setattr(ingestion.signal, "signal",
                            lambda sig, fn: installed.append(sig))
        ingestion.main()
        assert ingestion.signal.SIGINT in installed
        assert ingestion.signal.SIGTERM in installed

    def test_the_signal_handler_stops_the_feed(self, monkeypatch):
        ingestion._handle_signal(15, None)
        assert ingestion._running is False
