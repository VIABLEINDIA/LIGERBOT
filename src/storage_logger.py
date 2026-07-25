"""Module 6 — Storage & Logging.

A passive observer on the event bus. It reads from every stream (market_ticks,
trade_signals, approved_orders, filled_orders) and archives each record into
InfluxDB, a time-series database optimized for high-write, time-stamped data.

With everything in InfluxDB you can point Grafana at it and plot price as a
candlestick chart, overlay signals as arrows, and mark executions — the whole
lifecycle of the bot on one dashboard.

If the InfluxDB client isn't installed or the DB is unreachable, this module still
runs and logs to the console (so it never blocks the pipeline in development).

    python -m src.storage_logger
"""
from __future__ import annotations

import logging
import time
from typing import Optional

import config
from src import event_bus

logging.basicConfig(level=logging.INFO, format="%(asctime)s [storage] %(message)s")
log = logging.getLogger("ligerbot.storage")

# Every stream we archive, and the InfluxDB measurement each maps to.
STREAM_MEASUREMENTS = {
    config.STREAM_MARKET_TICKS: "market_ticks",
    config.STREAM_MARKET_BARS: "market_bars",
    config.STREAM_TRADE_SIGNALS: "trade_signals",
    config.STREAM_APPROVED_ORDERS: "approved_orders",
    config.STREAM_FILLED_ORDERS: "filled_orders",
    config.STREAM_POSITION_UPDATES: "position_updates",
    config.STREAM_DEAD_LETTER: "dead_letter",
}


class StorageLogger:
    def __init__(self, writer=None) -> None:
        self.client = event_bus.get_client()
        # Consumer groups rather than "$" cursors (fixes B6 for the archive too): an
        # archiver that silently skipped events during a restart would leave gaps in the
        # record precisely around the incidents worth investigating.
        self.consumer = event_bus.MultiStreamConsumer(
            self.client, list(STREAM_MEASUREMENTS),
            f"{config.CONSUMER_GROUP_PREFIX}.storage", "storage-1",
            max_deliveries=config.MAX_DELIVERIES,
        )
        from src.alerting import get_alerter
        from src.influx_writer import BatchingInfluxWriter

        # Batched and non-blocking (defect B12). One synchronous write per event could not
        # keep up with real tick volume: the archiver fell behind, stopped acking, and the
        # stream trimmed past its own unacked messages — so the archive developed holes
        # exactly when the most was happening.
        self.writer = writer or BatchingInfluxWriter(alerter=get_alerter(self.client))
        self.writer.start()

    def _write(self, measurement: str, fields: dict) -> None:
        """Queue one point. Returns immediately — never blocks on the network."""
        self.writer.write(measurement, fields)

    def run(self) -> None:
        if not event_bus.ping(self.client):
            log.error("Redis not reachable — start it with `docker compose up -d`.")
            return
        log.info("Storage Logger observing %d streams → %s",
                 len(STREAM_MEASUREMENTS),
                 "InfluxDB (batched)" if self.writer.connected else "console")
        last_report = time.monotonic()

        while True:
            for stream_name, entry_id, fields in self.consumer.read(
                    count=500, block_ms=2000):
                measurement = STREAM_MEASUREMENTS.get(stream_name, stream_name)
                self._write(measurement, fields)
                # Acked once the point is QUEUED, not once it is written. The archive is
                # best-effort by design: holding the ack until a possibly-dead backend
                # confirms would grow the pending list until the stream trimmed past it,
                # losing far more than the occasional dropped point — and stalling the
                # consumer group along the way.
                self.consumer.ack(stream_name, entry_id)

            if time.monotonic() - last_report >= 60.0:
                snapshot = self.writer.snapshot()
                if snapshot["dropped"]:
                    log.warning("Archive: %(written)s written, %(dropped)s dropped "
                                "(%(drop_ratio).2%% ), %(queued)s queued.", snapshot)
                else:
                    log.debug("Archive: %(written)s written, %(queued)s queued.", snapshot)
                last_report = time.monotonic()

    def close(self) -> None:
        """Drain the queue before exiting — unflushed points are lost otherwise."""
        self.writer.close()


def main() -> None:
    logger = StorageLogger()
    try:
        logger.run()
    finally:
        logger.close()


if __name__ == "__main__":
    main()
