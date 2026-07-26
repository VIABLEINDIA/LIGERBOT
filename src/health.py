"""Per-module health endpoints (DESIGN.md 3.9).

The naive health check answers *"is the process alive"*. That is worthless here, because
**alive is the failure mode**. Every expensive defect in this project's history has been a
module that ran, logged, held its Redis connection, and did nothing useful: the feed that
reconnected forever, the consumer that fell behind until the stream trimmed past it, the
socket that died while downstream kept computing on a price that had stopped updating. All
of those pass a liveness check. That is why they were expensive.

So this reports **whether the module is working**, and the distinction rests on two clocks:

``last_loop_at``
    The consume loop turned. If this goes stale the module is *wedged* — blocked on a
    call that never returned — even if it processed an event a second earlier.

``last_event_at``
    Actual work happened. Staleness here is **not** a failure: a quiet market legitimately
    produces no events for minutes. Reported, never used to fail the check. Conflating the
    two would make the endpoint cry wolf every lunchtime, and an alert people learn to
    ignore is worse than no alert.

**A halted bot is healthy.** The kill switch working is the system working, and returning
503 for it would tell an orchestrator to restart the process — destroying the halt. Halted
returns 200 with the reason attached.

Two deliberate constraints:

* **The server can never affect trading.** It runs on a daemon thread; a port it cannot
  bind is logged and shrugged off; a request handler that raises returns 500 rather than
  propagating. A monitoring feature that can take down execution is not worth having.
* **It binds to localhost by default.** This endpoint reports positions, P&L and equity —
  operational detail about a live trading account. Exposing that on ``0.0.0.0`` because it
  was the convenient default is a real disclosure, so the default is ``127.0.0.1`` and
  widening it is an explicit act.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, Optional

import config

log = logging.getLogger("ligerbot.health")

OK = "ok"
DEGRADED = "degraded"
HALTED = "halted"
STARTING = "starting"


class HealthState:
    """Thread-safe snapshot of what a module is doing.

    Mutated from the trading loop and read from the HTTP thread, so every access is
    lock-guarded. The lock is held only for field access — never across a Redis call —
    because a health endpoint that can block the trading loop is precisely backwards.
    """

    def __init__(self, component: str) -> None:
        self.component = component
        self._lock = threading.Lock()
        self._started_at = time.monotonic()
        self._last_loop_at: Optional[float] = None
        self._last_event_at: Optional[float] = None
        self._events = 0
        self._errors = 0
        self._backlog = 0
        self._halted = False
        self._halt_reason = ""
        self._extra: Dict[str, Any] = {}

    # -- updates -----------------------------------------------------------
    def loop_tick(self) -> None:
        """The consume loop turned. Call once per iteration, events or not."""
        with self._lock:
            self._last_loop_at = time.monotonic()

    def event_processed(self, count: int = 1) -> None:
        with self._lock:
            self._events += count
            self._last_event_at = time.monotonic()

    def error(self) -> None:
        with self._lock:
            self._errors += 1

    def set_backlog(self, pending: Optional[int]) -> None:
        """Record the consumer backlog. Tolerates ``None`` and junk deliberately.

        This is called from inside the trading loop with whatever ``check_backlog()``
        returned. An earlier version did a bare ``int(pending)``, and a ``None`` from a
        consumer that could not read its pending list raised `TypeError` **into the
        consume loop** — a health-reporting call taking the module down, which is the
        exact failure this file exists to avoid. Reporting nothing is always better.
        """
        try:
            value = int(pending)
        except (TypeError, ValueError):
            return
        with self._lock:
            self._backlog = value

    def set_halted(self, halted: bool, reason: str = "") -> None:
        with self._lock:
            self._halted = bool(halted)
            self._halt_reason = reason

    def set(self, **fields: Any) -> None:
        """Attach module-specific detail (session day, open positions, equity)."""
        with self._lock:
            self._extra.update(fields)

    # -- reads -------------------------------------------------------------
    def snapshot(self) -> Dict[str, Any]:
        now = time.monotonic()
        with self._lock:
            loop_age = None if self._last_loop_at is None else now - self._last_loop_at
            event_age = None if self._last_event_at is None else now - self._last_event_at
            halted, reason = self._halted, self._halt_reason
            backlog, events, errors = self._backlog, self._events, self._errors
            uptime = now - self._started_at
            extra = dict(self._extra)

        status, detail = self._classify(loop_age, backlog, halted)
        payload: Dict[str, Any] = {
            "component": self.component,
            "status": status,
            "healthy": status in (OK, HALTED, STARTING),
            "uptime_seconds": round(uptime, 1),
            "events_processed": events,
            "errors": errors,
            "consumer_backlog": backlog,
            "trading_mode": config.TRADING_MODE,
            "checked_at": dt.datetime.now().isoformat(timespec="seconds"),
            # Reported, never used to fail the check: a quiet market legitimately
            # produces no events for minutes at a time.
            "seconds_since_last_event": (None if event_age is None
                                         else round(event_age, 1)),
            "seconds_since_last_loop": (None if loop_age is None else round(loop_age, 1)),
        }
        if detail:
            payload["detail"] = detail
        if halted:
            payload["halt_reason"] = reason
        payload.update(extra)
        return payload

    def _classify(self, loop_age: Optional[float], backlog: int,
                  halted: bool) -> tuple[str, str]:
        if loop_age is None:
            # Started but the loop has not turned yet. Not a failure; not yet a success.
            return STARTING, "the consume loop has not completed an iteration yet"
        if loop_age > config.HEALTH_LOOP_STALE_SECONDS:
            return DEGRADED, (
                f"the consume loop last turned {loop_age:.0f}s ago (limit "
                f"{config.HEALTH_LOOP_STALE_SECONDS:.0f}s) — the module is wedged, not busy")
        if backlog >= config.BACKLOG_ALERT_THRESHOLD > 0:
            return DEGRADED, (
                f"{backlog:,} unacked message(s) — consuming, but not keeping up")
        if halted:
            # Deliberately healthy: the kill switch working IS the system working, and a
            # 503 here would tell an orchestrator to restart away the halt.
            return HALTED, "trading halted; the process is functioning correctly"
        return OK, ""


def _handler_factory(state: HealthState) -> type:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self) -> None:  # noqa: N802 - stdlib API
            try:
                path = self.path.split("?")[0].rstrip("/") or "/"
                if path in ("/", "/health", "/healthz"):
                    payload = state.snapshot()
                    code = 200 if payload["healthy"] else 503
                elif path in ("/live", "/livez"):
                    # Liveness only: the process answers. Deliberately weaker than
                    # /health, and documented as such so nobody wires the weak one up
                    # believing they have the strong one.
                    payload = {"component": state.component, "status": "alive"}
                    code = 200
                else:
                    payload, code = {"error": "not found", "paths": [
                        "/health", "/live"]}, 404
                self._respond(code, payload)
            except Exception as exc:  # noqa: BLE001 - never propagate into the server
                log.debug("Health request failed: %s", exc)
                try:
                    self._respond(500, {"error": "health check failed"})
                except Exception:  # noqa: BLE001
                    pass

        def _respond(self, code: int, payload: Dict[str, Any]) -> None:
            body = json.dumps(payload, default=str).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: Any) -> None:
            """Silence stdlib's stderr access log — a polling monitor would drown the
            module's own output in a request line every few seconds."""


    return Handler


class _Server(HTTPServer):
    """``HTTPServer`` with platform-correct address reuse.

    The stdlib sets ``allow_reuse_address = 1``, and the two platforms mean opposite
    things by it. On Linux it permits rebinding a socket still in ``TIME_WAIT``, which is
    what you want when restarting a module. On Windows it permits binding a port another
    process is **actively listening on**: both servers report success, and one silently
    receives nothing.

    That is not hypothetical — it is how this was found. Two modules misconfigured onto
    the same port would both log "health endpoint on ..." and one would answer nobody,
    which is precisely the class of quiet failure this file exists to detect.
    """

    allow_reuse_address = os.name != "nt"


class HealthServer:
    """An HTTP server on a daemon thread. Failing to start is never fatal."""

    def __init__(self, state: HealthState, *, host: Optional[str] = None,
                 port: Optional[int] = None) -> None:
        self.state = state
        self.host = host if host is not None else config.HEALTH_HOST
        self.port = port if port is not None else config.health_port(state.component)
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    @property
    def bound_port(self) -> Optional[int]:
        return self._server.server_address[1] if self._server else None

    def start(self) -> bool:
        if self._server is not None:
            return True
        try:
            self._server = _Server((self.host, self.port),
                                   _handler_factory(self.state))
        except OSError as exc:
            # A port already taken, or one this user may not bind. Worth saying out loud,
            # never worth refusing to trade over.
            log.warning("Health endpoint unavailable on %s:%s (%s) — the module runs "
                        "normally without it.", self.host, self.port, exc)
            self._server = None
            return False

        self._thread = threading.Thread(
            target=self._server.serve_forever, name=f"health-{self.state.component}",
            daemon=True)
        self._thread.start()
        log.info("Health endpoint on http://%s:%d/health", self.host, self.bound_port)
        return True

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None


def serve(component: str, *, host: Optional[str] = None,
          port: Optional[int] = None) -> tuple[HealthState, Optional[HealthServer]]:
    """Create the state and, if enabled, start its server.

    Returns the state either way, so a module updates it unconditionally and the endpoint
    being switched off changes nothing about the trading path.
    """
    state = HealthState(component)
    if not config.HEALTH_ENABLED:
        return state, None
    server = HealthServer(state, host=host, port=port)
    server.start()
    return state, server
