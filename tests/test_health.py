"""Per-module health endpoints (DESIGN.md 3.9).

The naive health check answers *"is the process alive"*. That is worthless here, because
**alive is the failure mode**. Every expensive defect in this project's history has been a
module that ran, logged, held its Redis connection and did nothing useful: the feed that
reconnected forever, the consumer that fell behind until the stream trimmed past it, the
socket that died while downstream kept computing on a price that had stopped updating.

All of those pass a liveness check. That is *why* they were expensive, and it is what this
endpoint has to catch instead.

Four properties are load-bearing:

* **Wedged is distinguished from idle.** A loop that stopped turning is broken; a loop
  turning with nothing to do is a quiet market. Conflating them produces an alert that
  fires every lunchtime, and an alert people learn to ignore is worse than none.
* **A halted bot is healthy.** The kill switch working is the system working. Returning
  503 would tell an orchestrator to restart the process and destroy the halt.
* **The endpoint can never affect trading.** Not when the port is taken, not when a
  handler raises, not when a caller passes rubbish.
* **It does not bind to the world by default.** It reports positions, P&L and equity.
"""
from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest

import config
from src.health import (
    DEGRADED, HALTED, HealthServer, HealthState, OK, STARTING, serve,
)


@pytest.fixture
def state():
    return HealthState("test")


def fetch(server: HealthServer, path: str = "/health"):
    url = f"http://{server.host}:{server.bound_port}{path}"
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


@pytest.fixture
def server(state):
    srv = HealthServer(state, host="127.0.0.1", port=0)
    assert srv.start()
    yield srv
    srv.stop()


# ---------------------------------------------------------------------------
class TestWedgedVersusIdle:
    """The distinction the whole endpoint exists for."""

    def test_a_module_that_has_not_looped_yet_is_starting(self, state):
        assert state.snapshot()["status"] == STARTING
        assert state.snapshot()["healthy"] is True

    def test_a_turning_loop_is_healthy(self, state):
        state.loop_tick()
        assert state.snapshot()["status"] == OK

    def test_a_turning_loop_with_no_events_is_still_healthy(self, state):
        """A quiet market is not a fault. Failing here would page someone every lunchtime,
        and the alert would be ignored by the time it mattered."""
        state.loop_tick()
        snapshot = state.snapshot()
        assert snapshot["status"] == OK
        assert snapshot["events_processed"] == 0
        assert snapshot["seconds_since_last_event"] is None

    def test_a_stalled_loop_is_degraded(self, state, monkeypatch):
        monkeypatch.setattr(config, "HEALTH_LOOP_STALE_SECONDS", -1.0)
        state.loop_tick()
        snapshot = state.snapshot()
        assert snapshot["status"] == DEGRADED
        assert snapshot["healthy"] is False
        assert "wedged, not busy" in snapshot["detail"]

    def test_recent_events_do_not_rescue_a_stalled_loop(self, state, monkeypatch):
        """The precise case: work happened a moment ago, then the loop blocked forever on
        a call that never returned."""
        monkeypatch.setattr(config, "HEALTH_LOOP_STALE_SECONDS", -1.0)
        state.loop_tick()
        state.event_processed()
        assert state.snapshot()["status"] == DEGRADED

    def test_event_age_is_reported_but_never_fails_the_check(self, state):
        state.loop_tick()
        state.event_processed()
        snapshot = state.snapshot()
        assert snapshot["seconds_since_last_event"] is not None
        assert snapshot["status"] == OK


class TestBacklog:
    def test_a_large_backlog_is_degraded(self, state, monkeypatch):
        """Consuming, but not keeping up. The stream will eventually trim past its own
        unacked entries, and on approved_orders that is trades going unplaced."""
        monkeypatch.setattr(config, "BACKLOG_ALERT_THRESHOLD", 100)
        state.loop_tick()
        state.set_backlog(500)
        snapshot = state.snapshot()
        assert snapshot["status"] == DEGRADED
        assert "not keeping up" in snapshot["detail"]

    def test_a_small_backlog_is_fine(self, state, monkeypatch):
        monkeypatch.setattr(config, "BACKLOG_ALERT_THRESHOLD", 100)
        state.loop_tick()
        state.set_backlog(3)
        assert state.snapshot()["status"] == OK

    def test_a_zero_threshold_disables_the_check(self, state, monkeypatch):
        monkeypatch.setattr(config, "BACKLOG_ALERT_THRESHOLD", 0)
        state.loop_tick()
        state.set_backlog(10_000)
        assert state.snapshot()["status"] == OK

    @pytest.mark.parametrize("junk", [None, "many", object()])
    def test_an_unusable_backlog_value_is_ignored_not_raised(self, state, junk):
        """Found by an existing test, not by inspection: an earlier version did a bare
        `int(pending)`, so a None from a consumer that could not read its pending list
        raised TypeError INTO THE CONSUME LOOP. A health call must never be able to take
        a trading module down."""
        state.loop_tick()
        state.set_backlog(junk)          # must not raise
        assert state.snapshot()["consumer_backlog"] == 0


class TestHaltIsHealthy:
    def test_a_halted_module_reports_halted(self, state):
        state.loop_tick()
        state.set_halted(True, "daily loss limit")
        assert state.snapshot()["status"] == HALTED

    def test_a_halted_module_is_still_healthy(self, state):
        """503 here would tell an orchestrator to restart the process, destroying the
        halt. The kill switch working IS the system working."""
        state.loop_tick()
        state.set_halted(True, "daily loss limit")
        assert state.snapshot()["healthy"] is True

    def test_the_halt_reason_is_carried(self, state):
        state.loop_tick()
        state.set_halted(True, "feed outage")
        assert state.snapshot()["halt_reason"] == "feed outage"

    def test_a_wedged_halted_module_is_still_degraded(self, state, monkeypatch):
        """Halted excuses idleness, not a stopped loop."""
        monkeypatch.setattr(config, "HEALTH_LOOP_STALE_SECONDS", -1.0)
        state.loop_tick()
        state.set_halted(True, "x")
        assert state.snapshot()["status"] == DEGRADED

    def test_clearing_the_halt_restores_ok(self, state):
        state.loop_tick()
        state.set_halted(True, "x")
        state.set_halted(False)
        assert state.snapshot()["status"] == OK


class TestTheEndpoint:
    def test_health_returns_200_when_ok(self, server, state):
        state.loop_tick()
        code, payload = fetch(server)
        assert code == 200
        assert payload["status"] == OK

    def test_health_returns_503_when_degraded(self, server, state, monkeypatch):
        """So a monitor does the right thing without parsing the body."""
        monkeypatch.setattr(config, "HEALTH_LOOP_STALE_SECONDS", -1.0)
        state.loop_tick()
        code, payload = fetch(server)
        assert code == 503
        assert payload["healthy"] is False

    def test_a_halted_module_returns_200(self, server, state):
        state.loop_tick()
        state.set_halted(True, "manual")
        assert fetch(server)[0] == 200

    def test_module_specific_detail_is_included(self, server, state):
        state.loop_tick()
        state.set(open_positions=3, net_pnl_today=-1250.5)
        _, payload = fetch(server)
        assert payload["open_positions"] == 3
        assert payload["net_pnl_today"] == -1250.5

    def test_the_trading_mode_is_reported(self, server, state):
        state.loop_tick()
        assert fetch(server)[1]["trading_mode"] == config.TRADING_MODE

    def test_live_is_a_weaker_check_and_says_so(self, server, state, monkeypatch):
        """/live answers only "the process responds". It must stay 200 where /health is
        503, or there is no point having both."""
        monkeypatch.setattr(config, "HEALTH_LOOP_STALE_SECONDS", -1.0)
        state.loop_tick()
        assert fetch(server, "/live")[0] == 200
        assert fetch(server, "/health")[0] == 503

    def test_an_unknown_path_is_404(self, server):
        code, payload = fetch(server, "/nope")
        assert code == 404
        assert "/health" in payload["paths"]

    def test_a_query_string_is_tolerated(self, server, state):
        state.loop_tick()
        assert fetch(server, "/health?verbose=1")[0] == 200


class TestItCannotAffectTrading:
    def test_an_unbindable_port_is_not_fatal(self, state, caplog):
        """Port already taken, or one this user may not bind. Worth saying out loud,
        never worth refusing to trade over.

        This test found a real cross-platform defect. `HTTPServer` sets
        `allow_reuse_address = 1`, and the platforms disagree about what that means: on
        Linux it rebinds a TIME_WAIT socket (wanted), on Windows it binds a port another
        process is actively listening on (not wanted). Two misconfigured modules both
        reported success and one silently answered nobody — the exact class of quiet
        failure this endpoint exists to detect.
        """
        blocker = HealthServer(HealthState("blocker"), host="127.0.0.1", port=0)
        assert blocker.start()
        try:
            clash = HealthServer(state, host="127.0.0.1", port=blocker.bound_port)
            with caplog.at_level("WARNING"):
                assert clash.start() is False
            assert "runs normally without it" in caplog.text
        finally:
            blocker.stop()

    def test_a_snapshot_failure_does_not_escape_the_handler(self, server, state,
                                                            monkeypatch):
        monkeypatch.setattr(type(state), "snapshot",
                            lambda self: (_ for _ in ()).throw(RuntimeError("boom")))
        assert fetch(server)[0] == 500

    def test_the_server_thread_is_a_daemon(self, server):
        """A non-daemon thread would keep the process alive after shutdown."""
        names = [t.name for t in threading.enumerate() if t.name.startswith("health-")]
        assert names
        assert all(t.daemon for t in threading.enumerate()
                   if t.name.startswith("health-"))

    def test_serve_returns_state_even_when_disabled(self, monkeypatch):
        """Modules update the state unconditionally, so switching the endpoint off must
        change nothing about the trading path."""
        monkeypatch.setattr(config, "HEALTH_ENABLED", False)
        state, server = serve("test")
        assert server is None
        state.loop_tick()                      # must not raise
        assert state.snapshot()["status"] == OK

    def test_starting_twice_is_harmless(self, server):
        assert server.start() is True


class TestBindsToLocalhost:
    def test_the_default_host_is_loopback(self):
        """This endpoint reports positions, P&L and equity — operational detail about a
        live trading account. Binding it to every interface because that was the
        convenient default is a real disclosure."""
        assert config.HEALTH_HOST == "127.0.0.1"

    def test_each_module_gets_a_stable_distinct_port(self):
        """A health check whose address moves between restarts is one nobody can wire up."""
        ports = {name: config.health_port(name)
                 for name in ("ingestion", "bars", "strategy", "risk", "execution",
                              "positions", "storage", "paper")}
        assert len(set(ports.values())) == len(ports)
        assert config.health_port("risk") == config.health_port("risk")

    def test_an_unknown_component_still_gets_a_port(self):
        assert config.health_port("something_new") == config.HEALTH_PORT_BASE
