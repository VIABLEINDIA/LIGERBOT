"""Batched, non-blocking InfluxDB writes (DESIGN.md 3.9 — the second half of B12).

The original archiver used ``write_options=SYNCHRONOUS`` and wrote **one point per event**.
That cannot keep up with real tick volume, and the failure is indirect enough to be worth
spelling out:

    archiver blocks on Influx → it stops acking → its pending list grows → the stream
    reaches MAXLEN and trims entries that were never archived → the archive develops
    holes, silently.

So the fix is not only "write in batches" but a decision about what to sacrifice when the
archive cannot keep up. **The archive is best-effort by design.** Falling behind must cost
archive completeness, never consumer progress — because a stalled archiver eventually
costs both. Points are enqueued and the caller returns immediately; on overflow the
*oldest* are dropped, and the drop count is reported rather than hidden.

Dropping oldest rather than newest is deliberate: during a burst, recent data describes the
state you are currently in, and old data describes a state that has already passed.

If InfluxDB is absent or unreachable the writer degrades to counting, so the archiver keeps
consuming and acking rather than stalling the stream behind a dead dependency.
"""
from __future__ import annotations

import logging
import threading
from collections import deque
from typing import Any, Deque, Dict, Optional, Tuple

import config

log = logging.getLogger("ligerbot.influx")

# Fields worth storing as numbers, in the order they are looked for.
#
# These lists are an allow-list, and anything absent is discarded silently. That is a
# reasonable design — it keeps the archive to fields worth charting — but it made the
# writer a place where data disappeared without a word. Building the Grafana dashboards
# found seven of the nine fields in a `position_updates` snapshot being dropped, including
# `open_positions` and `total_open_risk`, both of which §3.9 asks the dashboard to plot.
# The panels would have been permanently, silently empty.
#
# `tests/test_grafana_dashboards.py` now asserts that every field a dashboard queries is
# actually archived, so the two cannot drift apart again.
NUMERIC_FIELDS = (
    # market data
    "ltp", "price", "close", "open", "high", "low", "volume", "vwap",
    # orders
    "quantity", "filled_quantity", "average_fill_price", "risk_amount", "costs",
    "stop_loss", "take_profit", "ref_price", "slippage", "r_multiple",
    # the position book's snapshot (§3.2)
    "net_pnl_today", "realized_pnl_today", "costs_today", "open_positions",
    "total_open_risk", "gross_exposure", "trade_count",
    # indicators
    "short_ma", "long_ma",
)
TEXT_FIELDS = (
    "status", "order_no", "client_order_id", "broker_order_id", "strategy_name",
    "strategy_version", "reason", "error", "fill_reason", "mode", "order_type",
    # A field rather than a tag, deliberately: Influx tags are indexed, and one distinct
    # value per order would explode series cardinality. Filterable, just not indexed.
    "correlation_id",
)
TAG_FIELDS = ("instrument_id", "instrument", "action", "signal", "side", "intent")


class BatchingInfluxWriter:
    """Queues points and flushes them in batches on a background thread."""

    def __init__(
        self,
        *,
        url: Optional[str] = None,
        token: Optional[str] = None,
        org: Optional[str] = None,
        bucket: Optional[str] = None,
        batch_size: Optional[int] = None,
        flush_interval_ms: Optional[int] = None,
        queue_maxlen: Optional[int] = None,
        alerter=None,
    ) -> None:
        self.url = url or config.INFLUX_URL
        self.token = token or config.INFLUX_TOKEN
        self.org = org or config.INFLUX_ORG
        self.bucket = bucket or config.INFLUX_BUCKET
        self.batch_size = batch_size or config.INFLUX_BATCH_SIZE
        self.flush_interval = (flush_interval_ms or config.INFLUX_FLUSH_INTERVAL_MS) / 1000.0
        self.alerter = alerter

        # `deque(maxlen=...)` gives drop-oldest for free and, crucially, atomically —
        # appending to a full deque evicts from the left without a separate check.
        self._queue: Deque[Tuple[str, Dict[str, Any]]] = deque(
            maxlen=queue_maxlen or config.INFLUX_QUEUE_MAXLEN)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self.enqueued = 0
        self.written = 0
        self.dropped = 0
        self.write_failures = 0
        self._alerted_drops = False

        self._client = None
        self._write_api = None
        self._connect()

    # -- connection --------------------------------------------------------
    def _connect(self) -> None:
        if not self.token or self.token.startswith("YOUR_"):
            log.warning("INFLUX_TOKEN not set — archiving to the log only. Events are "
                        "still consumed and acked; nothing blocks.")
            return
        try:
            from influxdb_client import InfluxDBClient
            from influxdb_client.client.write_api import ASYNCHRONOUS

            self._client = InfluxDBClient(url=self.url, token=self.token, org=self.org)
            # Asynchronous rather than batching mode: we do our own batching above so the
            # bounded queue and drop accounting stay ours, rather than hidden inside the
            # client where overflow behaviour is not visible.
            self._write_api = self._client.write_api(write_options=ASYNCHRONOUS)
            log.info("InfluxDB connected (%s, bucket=%s), batching %d points / %.1fs.",
                     self.url, self.bucket, self.batch_size, self.flush_interval)
        except ImportError:
            log.warning("influxdb-client not installed — archiving to the log only.")
        except Exception as exc:  # noqa: BLE001 - network dependent
            log.warning("InfluxDB unreachable (%s) — archiving to the log only. The "
                        "archiver keeps consuming rather than stalling behind it.", exc)

    @property
    def connected(self) -> bool:
        return self._write_api is not None

    # -- enqueue -----------------------------------------------------------
    def write(self, measurement: str, fields: Dict[str, Any]) -> None:
        """Queue one point. Returns immediately; never blocks on the network."""
        with self._lock:
            was_full = len(self._queue) == self._queue.maxlen
            self._queue.append((measurement, fields))
            self.enqueued += 1
            if was_full:
                # deque evicted the oldest to make room.
                self.dropped += 1
        if self.dropped and not self._alerted_drops:
            self._maybe_alert_drops()

    def _maybe_alert_drops(self) -> None:
        if not self.enqueued:
            return
        ratio = self.dropped / self.enqueued
        if ratio < config.INFLUX_DROP_ALERT_RATIO:
            return
        self._alerted_drops = True
        message = (f"Archive queue overflowing: {self.dropped:,} of {self.enqueued:,} "
                   f"points dropped ({ratio:.1%}). The archive has gaps; trading is "
                   f"unaffected.")
        log.error(message)
        if self.alerter is not None:
            try:
                self.alerter.warning("Archive falling behind", message,
                                     source="influx_writer", dedup_key="influx_drops")
            except Exception:  # noqa: BLE001
                pass

    # -- flushing ----------------------------------------------------------
    def _drain(self, limit: int) -> list:
        with self._lock:
            batch = []
            while self._queue and len(batch) < limit:
                batch.append(self._queue.popleft())
            return batch

    def flush(self) -> int:
        """Write everything currently queued. Returns points written."""
        total = 0
        while True:
            batch = self._drain(self.batch_size)
            if not batch:
                break
            total += self._write_batch(batch)
        return total

    def _write_batch(self, batch: list) -> int:
        if not self.connected:
            # Log-only mode: count them so coverage is still measurable.
            self.written += len(batch)
            return len(batch)
        try:
            from influxdb_client import Point, WritePrecision

            points = []
            for measurement, fields in batch:
                point = self._build_point(measurement, fields, Point, WritePrecision)
                if point is not None:
                    points.append(point)
            if points:
                self._write_api.write(bucket=self.bucket, org=self.org, record=points)
            self.written += len(points)
            return len(points)
        except Exception as exc:  # noqa: BLE001 - network dependent
            self.write_failures += 1
            # Deliberately not requeued: retrying a failing backend indefinitely is how
            # the queue fills and the archiver falls behind in the first place.
            log.warning("Influx write failed (%s) — %d point(s) lost. Archive only.",
                        exc, len(batch))
            return 0

    @staticmethod
    def _build_point(measurement: str, fields: Dict[str, Any], Point, WritePrecision):
        point = Point(measurement)
        wrote_field = False

        for tag in TAG_FIELDS:
            value = fields.get(tag)
            if value not in (None, ""):
                point = point.tag(tag, str(value))

        for key in NUMERIC_FIELDS:
            raw = fields.get(key)
            if raw in (None, ""):
                continue
            try:
                point = point.field(key, float(raw))
                wrote_field = True
            except (TypeError, ValueError):
                continue

        for key in TEXT_FIELDS:
            value = fields.get(key)
            if value not in (None, ""):
                point = point.field(key, str(value)[:500])
                wrote_field = True

        if not wrote_field:
            # A point with tags but no fields is rejected by Influx; a counter keeps the
            # event visible.
            point = point.field("event", 1)

        timestamp = fields.get("timestamp")
        if timestamp not in (None, ""):
            try:
                point = point.time(int(float(timestamp) * 1_000_000_000),
                                   WritePrecision.NS)
            except (TypeError, ValueError):
                pass
        return point

    # -- background thread -------------------------------------------------
    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="influx-flush", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self.flush_interval):
            try:
                self.flush()
            except Exception as exc:  # noqa: BLE001 - the thread must never die
                log.error("Flush loop error: %s", exc)

    def close(self) -> None:
        """Stop the flusher and drain what is left."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        self.flush()
        if self._client is not None:
            try:
                self._client.close()
            except Exception:  # noqa: BLE001
                pass

    # -- observability -----------------------------------------------------
    def snapshot(self) -> Dict[str, Any]:
        return {
            "connected": self.connected,
            "queued": len(self._queue),
            "queue_capacity": self._queue.maxlen,
            "enqueued": self.enqueued,
            "written": self.written,
            "dropped": self.dropped,
            "write_failures": self.write_failures,
            "drop_ratio": round(self.dropped / self.enqueued, 5) if self.enqueued else 0.0,
        }
