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
    def __init__(self) -> None:
        self.client = event_bus.get_client()
        # Consumer groups rather than "$" cursors (fixes B6 for the archive too): an
        # archiver that silently skipped events during a restart would leave gaps in the
        # record precisely around the incidents worth investigating.
        self.consumer = event_bus.MultiStreamConsumer(
            self.client, list(STREAM_MEASUREMENTS),
            f"{config.CONSUMER_GROUP_PREFIX}.storage", "storage-1",
            max_deliveries=config.MAX_DELIVERIES,
        )
        self.write_api = None
        self.influx = None
        self._init_influx()

    def _init_influx(self) -> None:
        if not config.INFLUX_TOKEN or config.INFLUX_TOKEN == "YOUR_INFLUX_API_TOKEN":
            log.warning("INFLUX_TOKEN not set — running in console-only mode "
                        "(events logged, not persisted).")
            return
        try:
            from influxdb_client import InfluxDBClient
            from influxdb_client.client.write_api import SYNCHRONOUS

            self.influx = InfluxDBClient(
                url=config.INFLUX_URL, token=config.INFLUX_TOKEN, org=config.INFLUX_ORG
            )
            self.write_api = self.influx.write_api(write_options=SYNCHRONOUS)
            log.info("Connected to InfluxDB at %s (bucket=%s).",
                     config.INFLUX_URL, config.INFLUX_BUCKET)
        except ImportError:
            log.warning("influxdb-client not installed — console-only mode.")
        except Exception as exc:  # pragma: no cover - network dependent
            log.warning("Could not connect to InfluxDB (%s) — console-only mode.", exc)

    def _to_float(self, value) -> Optional[float]:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _write(self, measurement: str, fields: dict) -> None:
        if self.write_api is None:
            log.info("archive[%s] %s", measurement, fields)
            return
        try:
            from influxdb_client import Point, WritePrecision

            point = Point(measurement)
            instrument = fields.get("instrument")
            if instrument:
                point = point.tag("instrument", str(instrument))
            action = fields.get("action") or fields.get("signal")
            if action:
                point = point.tag("action", str(action))

            wrote_field = False
            for key in ("ltp", "price", "quantity", "short_ma", "long_ma"):
                num = self._to_float(fields.get(key))
                if num is not None:
                    point = point.field(key, num)
                    wrote_field = True
            for key in ("status", "order_no", "strategy_name"):
                if fields.get(key) is not None:
                    point = point.field(key, str(fields[key]))
                    wrote_field = True
            if not wrote_field:
                point = point.field("event", 1)

            ts = self._to_float(fields.get("timestamp"))
            if ts is not None:
                point = point.time(int(ts * 1_000_000_000), WritePrecision.NS)

            self.write_api.write(bucket=config.INFLUX_BUCKET, org=config.INFLUX_ORG, record=point)
        except Exception as exc:  # pragma: no cover - network dependent
            log.warning("InfluxDB write failed (%s) — logging instead: %s", exc, fields)

    def run(self) -> None:
        if not event_bus.ping(self.client):
            log.error("Redis not reachable — start it with `docker compose up -d`.")
            return
        log.info("Storage Logger observing %d streams → %s",
                 len(STREAM_MEASUREMENTS), "InfluxDB" if self.write_api else "console")
        while True:
            for stream_name, entry_id, fields in self.consumer.read(
                    count=500, block_ms=2000):
                measurement = STREAM_MEASUREMENTS.get(stream_name, stream_name)
                self._write(measurement, fields)
                # Acked after the write: archiving must never lose an event, but it must
                # also never backpressure the trading path, so failures are logged inside
                # _write rather than raised.
                self.consumer.ack(stream_name, entry_id)

    def close(self) -> None:
        if self.influx is not None:
            self.influx.close()


def main() -> None:
    logger = StorageLogger()
    try:
        logger.run()
    finally:
        logger.close()


if __name__ == "__main__":
    main()
