"""Consumer-group tests — the fix for B6.

B6 was the highest-severity defect in the register: every module read with
``last_id = "$"`` ("only messages from now on") and persisted nothing, so a restart
silently discarded queued events including approved orders. These tests pin the
replacement behaviour.
"""
from __future__ import annotations

import fakeredis
import pytest

import config
from src import event_bus


@pytest.fixture
def client():
    return fakeredis.FakeStrictRedis(server=fakeredis.FakeServer(), decode_responses=True)


def consumer(client, stream="orders", group="risk", name="w1", **kw):
    return event_bus.StreamConsumer(client, stream, group, name, **kw)


class TestB6Regression:
    """The specific failure that shipped: events published while a module was down."""

    def test_backlog_is_delivered_on_first_connect(self, client):
        event_bus.publish(client, "orders", {"id": "1"})
        event_bus.publish(client, "orders", {"id": "2"})
        # The old code started at "$" and would have seen zero of these.
        assert len(consumer(client).read(block_ms=10)) == 2

    def test_restart_redelivers_unacked_work(self, client):
        event_bus.publish(client, "orders", {"id": "1"})
        first = consumer(client, name="w1")
        messages = first.read(block_ms=10)
        assert len(messages) == 1
        # Process crashes here — no ack.

        second = consumer(client, name="w2")
        reclaimed = second.claim_stale(min_idle_ms=0)
        assert len(reclaimed) == 1
        assert reclaimed[0][1]["id"] == 1

    def test_acked_messages_are_not_redelivered(self, client):
        event_bus.publish(client, "orders", {"id": "1"})
        first = consumer(client, name="w1")
        entry_id, _ = first.read(block_ms=10)[0]
        first.ack(entry_id)

        second = consumer(client, name="w2")
        assert second.claim_stale(min_idle_ms=0) == []
        assert first.pending_count() == 0

    def test_at_least_once_beats_at_most_once_for_orders(self, client):
        """A duplicate is detectable and reversible; a dropped order is neither."""
        event_bus.publish(client, "orders", {"id": "critical"})
        c = consumer(client)
        c.read(block_ms=10)
        assert c.pending_count() == 1  # still owed, not lost


class TestGroupIsolation:
    def test_each_group_receives_its_own_copy(self, client):
        event_bus.publish(client, "bars", {"b": "1"})
        strategy = consumer(client, stream="bars", group="strategy", name="s1")
        archiver = consumer(client, stream="bars", group="archiver", name="a1")
        assert len(strategy.read(block_ms=10)) == 1
        assert len(archiver.read(block_ms=10)) == 1

    def test_consumers_in_one_group_share_the_work(self, client):
        for i in range(6):
            event_bus.publish(client, "bars", {"i": i})
        a = consumer(client, stream="bars", group="g", name="a")
        b = consumer(client, stream="bars", group="g", name="b")
        first = a.read(count=3, block_ms=10)
        second = b.read(count=10, block_ms=10)
        assert len(first) == 3
        assert len(second) == 3
        assert {m[0] for m in first}.isdisjoint({m[0] for m in second})

    def test_recreating_a_group_is_idempotent(self, client):
        consumer(client)
        consumer(client)  # must not raise BUSYGROUP


class TestPoisonMessages:
    def test_failing_handler_leaves_the_message_unacked(self, client):
        event_bus.publish(client, "orders", {"id": "bad"})
        c = consumer(client, max_deliveries=5)
        entry_id, fields = c.read(block_ms=10)[0]

        def boom(_):
            raise ValueError("nope")

        assert c.handle(entry_id, fields, boom) is False
        assert c.pending_count() == 1

    def test_poison_goes_to_dead_letter_and_stops_blocking(self, client):
        event_bus.publish(client, "orders", {"id": "bad"})
        c = consumer(client, max_deliveries=2)

        def boom(_):
            raise ValueError("cannot parse")

        for _ in range(4):
            batch = c.read(block_ms=10) or c.claim_stale(min_idle_ms=0)
            for entry_id, fields in batch:
                c.handle(entry_id, fields, boom)

        assert client.xlen(config.STREAM_DEAD_LETTER) >= 1
        assert c.pending_count() == 0  # acked, so it no longer blocks progress

    def test_dead_letter_records_enough_to_diagnose(self, client):
        event_bus.publish(client, "orders", {"id": "bad", "qty": 5})
        c = consumer(client, max_deliveries=1)
        entry_id, fields = c.read(block_ms=10)[0]
        c.dead_letter(entry_id, fields, "boom")

        record = client.xrange(config.STREAM_DEAD_LETTER)[-1][1]
        assert record["original_stream"] == "orders"
        assert "boom" in record["error"]
        assert "qty" in record["payload"]

    def test_successful_handler_acks(self, client):
        event_bus.publish(client, "orders", {"id": "1"})
        c = consumer(client)
        entry_id, fields = c.read(block_ms=10)[0]
        assert c.handle(entry_id, fields, lambda _: None) is True
        assert c.pending_count() == 0


class TestTrimming:
    def test_maxlen_bounds_the_stream(self, client):
        for i in range(500):
            event_bus.publish(client, "ticks", {"i": i}, maxlen=100)
        # Approximate trimming, so allow slack — the point is that it is bounded.
        assert client.xlen("ticks") <= 200

    def test_config_default_is_applied(self, client, monkeypatch):
        monkeypatch.setattr(config, "STREAM_MAXLEN", 50)
        for i in range(300):
            event_bus.publish(client, "ticks", {"i": i})
        assert client.xlen("ticks") <= 150

    def test_trimming_can_be_disabled(self, client, monkeypatch):
        monkeypatch.setattr(config, "STREAM_MAXLEN", 0)
        for i in range(120):
            event_bus.publish(client, "ticks", {"i": i})
        assert client.xlen("ticks") == 120


class TestMultiStreamConsumer:
    def test_reads_across_streams(self, client):
        event_bus.publish(client, "a", {"x": 1})
        event_bus.publish(client, "b", {"x": 2})
        multi = event_bus.MultiStreamConsumer(client, ["a", "b"], "archiver", "w")
        results = multi.read(block_ms=10)
        assert {stream for stream, _, _ in results} == {"a", "b"}

    def test_ack_targets_the_right_stream(self, client):
        event_bus.publish(client, "a", {"x": 1})
        multi = event_bus.MultiStreamConsumer(client, ["a", "b"], "archiver", "w")
        stream, entry_id, _ = multi.read(block_ms=10)[0]
        multi.ack(stream, entry_id)
        assert multi.pending_total() == 0
