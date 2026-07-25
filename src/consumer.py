"""Test consumer — the smallest possible reader of the event bus.

Represents the "hello world" of your Strategy Engine: it subscribes to the
``market_ticks`` stream and prints each tick. Run this alongside
``data_ingestion --simulate`` to confirm the ingestion -> event-bus wiring works
before you layer on the real strategy logic.

    python -m src.consumer
"""
from __future__ import annotations

import logging

import config
from src import event_bus

logging.basicConfig(level=logging.INFO, format="%(asctime)s [consumer] %(message)s")
log = logging.getLogger("ligerbot.consumer")


def main() -> None:
    client = event_bus.get_client()
    if not event_bus.ping(client):
        log.error("Redis not reachable — start it with `docker compose up -d`.")
        return

    # Start reading only NEW messages ("$"); switch to "0" to replay history.
    last_id = "$"
    log.info("Test consumer listening on '%s'...", config.STREAM_MARKET_TICKS)
    while True:
        entries, last_id = event_bus.read_new(
            client, config.STREAM_MARKET_TICKS, last_id, count=100, block_ms=5000
        )
        for entry_id, tick in entries:
            log.info("tick %s -> %s @ %s", entry_id, tick.get("instrument"), tick.get("ltp"))


if __name__ == "__main__":
    main()
