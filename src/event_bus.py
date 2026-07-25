"""Redis Streams helpers — the shared spine every module talks through.

We use **Redis Streams** (not Pub/Sub) deliberately. Pub/Sub is fire-and-forget:
if a consumer is down when a message is published, it is lost forever. Streams are
an append-only log with per-consumer cursors, so a module that restarts can resume
exactly where it left off — critical when the messages are trading signals and orders.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple

import redis

import config

log = logging.getLogger("ligerbot.event_bus")


def get_client() -> "redis.Redis":
    """Return a Redis client configured from the central config.

    ``decode_responses=True`` gives us plain ``str`` keys/values instead of bytes.
    """
    return redis.Redis(
        host=config.REDIS_HOST,
        port=config.REDIS_PORT,
        db=config.REDIS_DB,
        decode_responses=True,
    )


def publish(
    client: "redis.Redis",
    stream: str,
    payload: Dict[str, Any],
    *,
    maxlen: Optional[int] = None,
) -> str:
    """Append one event to ``stream``.

    Nested/complex values are JSON-encoded so Redis (which stores flat string
    field maps) can hold them; :func:`decode_fields` reverses this on read.

    ``maxlen`` caps the stream's length approximately (``~`` trimming, which Redis can
    do cheaply at node boundaries). Without it a tick stream grows without bound until
    Redis exhausts memory and starts refusing writes — at which point the bot stops
    receiving market data mid-session. Defaults to :data:`config.STREAM_MAXLEN`.
    """
    flat: Dict[str, str] = {}
    for key, value in payload.items():
        if isinstance(value, (dict, list, bool)) or value is None:
            flat[key] = json.dumps(value)
        else:
            flat[key] = str(value)
    limit = maxlen if maxlen is not None else getattr(config, "STREAM_MAXLEN", None)
    if limit:
        return client.xadd(stream, flat, maxlen=limit, approximate=True)
    return client.xadd(stream, flat)


def decode_fields(fields: Dict[str, str]) -> Dict[str, Any]:
    """Best-effort reverse of :func:`publish` — decode any JSON-looking values."""
    out: Dict[str, Any] = {}
    for key, value in fields.items():
        if isinstance(value, str) and value and value[0] in "[{tfn\"-0123456789":
            try:
                out[key] = json.loads(value)
                continue
            except (json.JSONDecodeError, ValueError):
                pass
        out[key] = value
    return out


def read_new(
    client: "redis.Redis",
    stream: str,
    last_id: str,
    *,
    count: int = 100,
    block_ms: int = 2000,
) -> Tuple[List[Tuple[str, Dict[str, Any]]], str]:
    """Blocking read of new entries on ``stream`` after ``last_id``.

    Returns ``(entries, new_last_id)`` where ``entries`` is a list of
    ``(entry_id, decoded_fields)``. Pass the returned id back on the next call so
    each event is processed exactly once — this is what makes restarts safe.
    """
    response = client.xread({stream: last_id}, count=count, block=block_ms)
    if not response:
        return [], last_id

    entries: List[Tuple[str, Dict[str, Any]]] = []
    new_last_id = last_id
    for _stream_name, messages in response:
        for entry_id, fields in messages:
            entries.append((entry_id, decode_fields(fields)))
            new_last_id = entry_id
    return entries, new_last_id


def read_new_multi(
    client: "redis.Redis",
    cursors: Dict[str, str],
    *,
    count: int = 100,
    block_ms: int = 2000,
) -> Tuple[List[Tuple[str, str, Dict[str, Any]]], Dict[str, str]]:
    """Read new entries across several streams at once (used by the logger).

    ``cursors`` maps stream name -> last-seen id. Returns
    ``(entries, updated_cursors)`` where each entry is
    ``(stream_name, entry_id, decoded_fields)``.
    """
    response = client.xread(cursors, count=count, block=block_ms)
    entries: List[Tuple[str, str, Dict[str, Any]]] = []
    updated = dict(cursors)
    if not response:
        return entries, updated
    for stream_name, messages in response:
        for entry_id, fields in messages:
            entries.append((stream_name, entry_id, decode_fields(fields)))
            updated[stream_name] = entry_id
    return entries, updated


# ---------------------------------------------------------------------------
# Consumer groups — the fix for B6
# ---------------------------------------------------------------------------
class StreamConsumer:
    """At-least-once consumption via a Redis consumer group.

    Replaces the ``last_id = "$"`` pattern every module used, which meant "only messages
    published from now on" and persisted nothing. A module that restarted **silently
    discarded every queued event**, including approved orders — while the README claimed
    the opposite (defect B6).

    The contract here is different in a way that matters: a message stays in the group's
    Pending Entries List until it is explicitly :meth:`ack`-ed. Crash before the ack and
    the message is redelivered rather than lost. That makes delivery *at-least-once*
    rather than at-most-once, which is the right trade for orders — a duplicate can be
    caught by an idempotency key, whereas a dropped order is simply gone.

    Failure handling has three stages:

    1. **Redelivery.** Unacked work is reclaimed by :meth:`claim_stale` after an idle
       timeout, so a dead consumer's messages are picked up by a live one.
    2. **Dead-letter.** A message that has been delivered ``max_deliveries`` times is
       poison — it will fail forever and block nothing else usefully. It goes to a
       dead-letter stream with its error, and an alert is logged.
    3. **Never silent.** Every drop is recorded somewhere a human can find it.
    """

    def __init__(
        self,
        client: "redis.Redis",
        stream: str,
        group: str,
        consumer: str,
        *,
        max_deliveries: int = 5,
        dead_letter_stream: Optional[str] = None,
    ) -> None:
        self.client = client
        self.stream = stream
        self.group = group
        self.consumer = consumer
        self.max_deliveries = max_deliveries
        self.dead_letter_stream = dead_letter_stream or getattr(
            config, "STREAM_DEAD_LETTER", "dead_letter")
        self.ensure_group()

    def ensure_group(self) -> None:
        """Create the group if absent, starting from the beginning of the stream.

        ``id="0"`` rather than ``"$"`` deliberately: a brand-new consumer group should
        process the backlog, not skip it. Skipping is what B6 did.
        """
        try:
            self.client.xgroup_create(self.stream, self.group, id="0", mkstream=True)
            log.info("Created consumer group %r on stream %r", self.group, self.stream)
        except redis.exceptions.ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise

    # -- reading -----------------------------------------------------------
    def read(
        self, *, count: int = 100, block_ms: int = 2000
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """Read undelivered messages. Each must be :meth:`ack`-ed once handled."""
        response = self.client.xreadgroup(
            self.group, self.consumer, {self.stream: ">"}, count=count, block=block_ms)
        if not response:
            return []
        out: List[Tuple[str, Dict[str, Any]]] = []
        for _stream, messages in response:
            for entry_id, fields in messages:
                out.append((entry_id, decode_fields(fields)))
        return out

    def claim_stale(
        self, *, min_idle_ms: int = 60_000, count: int = 100
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """Take over messages another consumer read but never acked.

        This is how a crashed module's in-flight work gets picked up rather than sitting
        in the pending list forever.
        """
        try:
            result = self.client.xautoclaim(
                self.stream, self.group, self.consumer,
                min_idle_time=min_idle_ms, count=count)
        except redis.exceptions.ResponseError as exc:
            log.warning("XAUTOCLAIM failed on %s: %s", self.stream, exc)
            return []

        messages = result[1] if len(result) > 1 else []
        claimed: List[Tuple[str, Dict[str, Any]]] = []
        for entry_id, fields in messages:
            if not fields:
                continue
            claimed.append((entry_id, decode_fields(fields)))
        if claimed:
            log.info("Reclaimed %d stale message(s) from %s", len(claimed), self.stream)
        return claimed

    # -- completion --------------------------------------------------------
    def ack(self, entry_id: str) -> None:
        """Mark a message fully handled. Only call this after the work is durable."""
        self.client.xack(self.stream, self.group, entry_id)

    def pending_count(self) -> int:
        try:
            summary = self.client.xpending(self.stream, self.group)
        except redis.exceptions.ResponseError:
            return 0
        if isinstance(summary, dict):
            return int(summary.get("pending", 0))
        return int(summary[0]) if summary else 0

    def delivery_count(self, entry_id: str) -> int:
        """How many times this message has been delivered. Drives poison detection."""
        try:
            entries = self.client.xpending_range(
                self.stream, self.group, min=entry_id, max=entry_id, count=1)
        except redis.exceptions.ResponseError:
            return 0
        return int(entries[0]["times_delivered"]) if entries else 0

    def is_poison(self, entry_id: str) -> bool:
        return self.delivery_count(entry_id) >= self.max_deliveries

    def dead_letter(
        self, entry_id: str, fields: Dict[str, Any], error: str
    ) -> None:
        """Route a poison message aside and ack it, so it stops blocking progress.

        Acked because leaving it pending would have it reclaimed forever. Recorded
        because a message that vanishes without trace is indistinguishable from one that
        was handled.
        """
        payload = {
            "original_stream": self.stream,
            "original_id": entry_id,
            "group": self.group,
            "error": str(error)[:500],
            "deliveries": self.delivery_count(entry_id),
            "payload": json.dumps(fields, default=str)[:2000],
        }
        publish(self.client, self.dead_letter_stream, payload)
        self.ack(entry_id)
        log.error("DEAD-LETTER %s:%s after %d deliveries — %s",
                  self.stream, entry_id, payload["deliveries"], error)

        # A dead-lettered message is work the system has given up on. That must reach a
        # human, not just the log — especially on approved_orders, where it means a trade
        # was never placed.
        try:
            from src.alerting import get_alerter

            get_alerter(self.client).critical(
                "Message dead-lettered",
                f"{self.stream} gave up on a message after "
                f"{payload['deliveries']} deliveries: {error}",
                source=self.group,
                dedup_key=f"dead_letter:{self.stream}",
                context={"stream": self.stream, "entry_id": entry_id},
            )
        except Exception as exc:  # noqa: BLE001 - alerting must not break consumption
            log.error("Could not raise the dead-letter alert: %s", exc)

    def handle(
        self, entry_id: str, fields: Dict[str, Any], handler
    ) -> bool:
        """Run ``handler(fields)``, acking on success and dead-lettering poison.

        Returns True if handled. A failing message is left unacked so it is retried,
        until it exceeds ``max_deliveries`` — at which point retrying it forever is
        just a slow way of never making progress.
        """
        try:
            handler(fields)
        except Exception as exc:  # noqa: BLE001 - handler failures must not kill the loop
            if self.is_poison(entry_id):
                self.dead_letter(entry_id, fields, str(exc))
            else:
                log.warning("Handler failed for %s:%s (delivery %d) — will retry: %s",
                            self.stream, entry_id, self.delivery_count(entry_id), exc)
            return False
        self.ack(entry_id)
        return True


class MultiStreamConsumer:
    """One consumer group per stream, read together. Used by the archiver."""

    def __init__(
        self,
        client: "redis.Redis",
        streams: Iterable[str],
        group: str,
        consumer: str,
        **kwargs,
    ) -> None:
        self.consumers = {
            stream: StreamConsumer(client, stream, group, consumer, **kwargs)
            for stream in streams
        }

    def read(
        self, *, count: int = 100, block_ms: int = 2000
    ) -> List[Tuple[str, str, Dict[str, Any]]]:
        out: List[Tuple[str, str, Dict[str, Any]]] = []
        for stream, consumer in self.consumers.items():
            for entry_id, fields in consumer.read(count=count, block_ms=block_ms):
                out.append((stream, entry_id, fields))
        return out

    def ack(self, stream: str, entry_id: str) -> None:
        self.consumers[stream].ack(entry_id)

    def pending_total(self) -> int:
        return sum(c.pending_count() for c in self.consumers.values())


def ping(client: Optional["redis.Redis"] = None) -> bool:
    """Return True if Redis is reachable — handy for startup health checks."""
    client = client or get_client()
    try:
        return bool(client.ping())
    except redis.exceptions.RedisError as exc:  # pragma: no cover - network dependent
        log.error("Redis ping failed: %s", exc)
        return False
