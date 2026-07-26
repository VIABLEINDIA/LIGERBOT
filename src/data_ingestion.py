"""Module 1 — Data Ingestion Layer.

The bot's sensory system. It connects to Kotak Neo's live WebSocket feed and pushes
every tick onto the ``market_ticks`` Redis Stream. The strategy engine reads from
there — ingestion never talks to strategy directly (that decoupling is the whole point
of the event bus).

Run live:
    python -m src.data_ingestion

Run against a synthetic feed (no broker, no market hours needed):
    python -m src.data_ingestion --simulate
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import signal
import time
from typing import Any, Dict

import config
from src import event_bus
from src import kotak_api
from src import feed_health
from src import logging_setup

logging_setup.configure("ingestion")
log = logging.getLogger("ligerbot.ingestion")

_redis = event_bus.get_client()
_running = True


def _publish_tick(
    instrument: str,
    ltp: float,
    extra: Dict[str, Any] | None = None,
    instrument_id: str | None = None,
) -> None:
    """Normalize a broker tick into our canonical schema and push to the bus.

    ``instrument_id`` is the canonical handle everything downstream keys on; the display
    ``instrument`` name is carried alongside for logs only. Sending a display name where
    an identifier belongs is what broke the execution path (B4), so the two are kept
    firmly distinct from the moment a tick enters the system.
    """
    canonical = instrument_id or instrument
    tick = {
        "instrument_id": canonical,
        "instrument": instrument,
        "ltp": ltp,
        "timestamp": time.time(),
    }
    if extra:
        tick.update(extra)
    event_bus.publish(_redis, config.STREAM_MARKET_TICKS, tick)
    # Refresh the liveness key. It carries a TTL, so an instrument that stops ticking
    # becomes stale on its own — no writer has to mark it dead, and a crashed ingestion
    # process cannot leave a false "live" flag behind (fixes B10).
    try:
        feed_health.publish_liveness(_redis, canonical)
    except Exception as exc:  # noqa: BLE001 - liveness must never break the feed
        log.debug("Could not refresh liveness for %s: %s", canonical, exc)


# ---------------------------------------------------------------------------
# Live feed (Kotak Neo)
# ---------------------------------------------------------------------------
def _on_message(message: Any) -> None:
    """Callback fired by the SDK when live data arrives.

    Kotak's payload shape varies by feed; we defensively pull out the last-traded
    price and instrument label, then forward onto the event bus.
    """
    try:
        records = message if isinstance(message, list) else [message]
        for rec in records:
            if not isinstance(rec, dict):
                continue
            ltp = rec.get("ltp") or rec.get("last_traded_price") or rec.get("lp")
            if ltp is None:
                continue
            instrument = (
                rec.get("trading_symbol")
                or rec.get("name")
                or rec.get("tk")
                or "UNKNOWN"
            )
            # Build the canonical id from the exchange token, never from the display
            # name — the name is not unique and is not what the broker keys on.
            token = rec.get("tk") or rec.get("instrument_token") or rec.get("pSymbol")
            segment = rec.get("e") or rec.get("exchange_segment") or config.DEFAULT_EXCHANGE_SEGMENT
            instrument_id = f"{segment}:{token}" if token else None

            volume = rec.get("v") or rec.get("volume") or rec.get("cum_volume")
            extra: Dict[str, Any] = {"raw_keys": list(rec.keys())}
            if volume is not None:
                extra["volume"] = volume

            _publish_tick(str(instrument), float(ltp), extra, instrument_id=instrument_id)
    except (ValueError, TypeError) as exc:
        log.warning("Could not parse tick %s: %s", message, exc)


def _on_error(message: Any) -> None:
    log.error("[WebSocket Error]: %s", message)


def _on_open(message: Any) -> None:
    log.info("[WebSocket Open]: %s", message)


def _on_close(message: Any) -> None:
    log.warning("[WebSocket Close]: %s", message)


def _connect_and_subscribe():
    """Obtain a session, wire callbacks and subscribe. Returns (client, tokens)."""
    from src.auth_session import get_session

    # Shared session rather than an independent login: a TOTP code is single-use within
    # its 30-second window, and this module starts alongside two others that also need a
    # broker session (DESIGN.md 3.8).
    client = get_session(_redis)
    client.on_message = _on_message
    client.on_error = _on_error
    client.on_open = _on_open
    client.on_close = _on_close

    instrument_tokens = [
        {"instrument_token": inst.token, "exchange_segment": inst.segment}
        for inst in config.INSTRUMENTS
    ]
    log.info("Subscribing to %d instrument(s): %s", len(instrument_tokens),
             ", ".join(i.name for i in config.INSTRUMENTS))
    client.subscribe(instrument_tokens=instrument_tokens, isIndex=False, isDepth=False)
    return client, instrument_tokens


def start_live_feed() -> None:
    """Stream ticks, reconnecting on failure until interrupted or exhausted.

    Fixes **B10**. Previously ``_on_close`` logged a warning and the loop kept running:
    the socket died, the process stayed alive, and every downstream module carried on
    computing indicators from a price that had stopped updating. Silent, and expensive.

    Two behaviours matter here:

    * **The staleness watchdog is independent of the socket.** Liveness keys expire on
      their own, so downstream blocks entries even when the connection *looks* up but no
      data is arriving — the failure mode a connection check would miss entirely.
    * **Reconnection is bounded.** Retrying forever keeps the process alive and passing
      liveness checks while never trading. After the cap it halts loudly instead.
    """
    from src.kill_switch import KillSwitch

    policy = feed_health.ReconnectPolicy()
    kill_switch = KillSwitch(_redis)
    client = None
    instrument_tokens: list = []

    while _running:
        try:
            client, instrument_tokens = _connect_and_subscribe()
            policy.reset()
            log.info("Feed connected.")

            # The SDK streams in the background; keep the process alive.
            while _running:
                time.sleep(1)
            break

        except Exception as exc:  # noqa: BLE001 - any connection failure
            delay = policy.next_delay()
            if delay is None:
                message = (f"market feed reconnection exhausted after "
                           f"{policy.max_attempts} attempts: {exc}")
                log.error("%s — halting.", message)
                kill_switch.halt(message, source="data_ingestion")
                break
            log.error("Feed connection failed (%s). Retrying in %.1fs "
                      "(attempt %d/%d).", exc, delay, policy.attempts, policy.max_attempts)
            time.sleep(delay)

    if client is not None and instrument_tokens:
        try:
            client.un_subscribe(instrument_tokens=instrument_tokens)
        except Exception as exc:  # pragma: no cover - depends on SDK/session state
            log.warning("Error during un_subscribe: %s", exc)
    log.info("Live feed stopped.")


# ---------------------------------------------------------------------------
# Simulated feed (no broker required)
# ---------------------------------------------------------------------------
def start_simulated_feed(interval: float = 0.2) -> None:
    """Emit a synthetic random-walk price series so the full pipeline can be
    exercised offline. Great for developing the strategy/risk/execution modules."""
    log.info("Starting SIMULATED feed (no broker). Ctrl-C to stop.")
    # Keyed by canonical id so the simulated feed exercises the same identifier path as
    # the live one — a simulator that takes a shortcut here would hide B4-class bugs.
    tracked = {
        f"{inst.segment}:{inst.token}": (inst.name, 20_000.0 + random.random() * 100)
        for inst in config.INSTRUMENTS
    }
    if not tracked:
        tracked = {"nse_cm:11536": ("Nifty 50", 24_500.0)}

    cumulative_volume = {key: 0.0 for key in tracked}

    while _running:
        for instrument_id, (name, last) in list(tracked.items()):
            # Gentle random walk with slight mean drift so crossovers actually occur.
            drift = random.uniform(-1.0, 1.0) * (last * 0.0004)
            new_price = round(max(1.0, last + drift), 2)
            tracked[instrument_id] = (name, new_price)
            # Cumulative, matching how the broker reports volume, so the bar builder's
            # differencing logic is exercised rather than bypassed.
            cumulative_volume[instrument_id] += random.randint(50, 500)
            _publish_tick(
                name, new_price,
                {"source": "simulator", "volume": cumulative_volume[instrument_id]},
                instrument_id=instrument_id,
            )
        time.sleep(interval)
    log.info("Simulated feed stopped.")


def _handle_signal(signum, _frame) -> None:
    global _running
    log.info("Received signal %s — shutting down.", signum)
    _running = False


def main() -> None:
    # The SDK sets no per-request timeout; bound it before any network call.
    kotak_api.bound_network_calls()
    parser = argparse.ArgumentParser(description="LIGERBOT data ingestion layer")
    parser.add_argument("--simulate", action="store_true",
                        help="emit a synthetic feed instead of connecting to Kotak Neo")
    parser.add_argument("--interval", type=float, default=0.2,
                        help="seconds between simulated ticks (default 0.2)")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    log.info(config.summary())
    if not event_bus.ping(_redis):
        log.error("Redis is not reachable at %s:%s — start it with `docker compose up -d`.",
                  config.REDIS_HOST, config.REDIS_PORT)
        return

    if args.simulate:
        start_simulated_feed(interval=args.interval)
    else:
        start_live_feed()


if __name__ == "__main__":
    main()
