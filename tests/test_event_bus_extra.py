"""Event bus — the paths the existing group tests did not reach.

`test_event_bus_groups.py` covers the happy path of consumer groups. What it did not cover
is everything the bus does when Redis, a handler, or the alerter misbehaves — and the event
bus is the spine of the system: every module talks through it, so a defect here is not
local to anything.

The theme is that **the bus must degrade rather than propagate**. A consumer that cannot
read its pending list, an alerter that is down, an `XAUTOCLAIM` that fails on an old Redis
— none of these are reasons to take a trading module down with them. Equally, none of them
may be allowed to *look* like success: a poison message that quietly disappears is
indistinguishable from one that was handled correctly, which is why dead-lettering records
and alerts rather than dropping.
"""
from __future__ import annotations

import fakeredis
import pytest
import redis as redis_lib

import config
from src import event_bus
from src.event_bus import MultiStreamConsumer, StreamConsumer


@pytest.fixture
def client():
    return fakeredis.FakeStrictRedis(server=fakeredis.FakeServer(), decode_responses=True)


@pytest.fixture(autouse=True)
def _isolate_alerter():
    from src.alerting import reset_alerter

    reset_alerter()
    yield
    reset_alerter()


class TestReadNew:
    def test_it_returns_entries_and_advances_the_cursor(self, client):
        for i in range(3):
            event_bus.publish(client, "s1", {"i": i})
        entries, cursor = event_bus.read_new(client, "s1", "0", block_ms=10)
        assert len(entries) == 3
        assert cursor != "0"

    def test_the_cursor_makes_reads_exactly_once(self, client):
        """Passing the returned id back is what makes a restart safe rather than a
        source of duplicate processing."""
        event_bus.publish(client, "s1", {"i": 1})
        _, cursor = event_bus.read_new(client, "s1", "0", block_ms=10)
        entries, _ = event_bus.read_new(client, "s1", cursor, block_ms=10)
        assert entries == []

    def test_an_empty_stream_returns_the_cursor_unchanged(self, client):
        entries, cursor = event_bus.read_new(client, "empty", "5-5", block_ms=10)
        assert entries == []
        assert cursor == "5-5"

    def test_fields_are_decoded(self, client):
        event_bus.publish(client, "s1", {"n": 42, "flag": True, "s": "x"})
        entries, _ = event_bus.read_new(client, "s1", "0", block_ms=10)
        assert entries[0][1]["n"] == 42
        assert entries[0][1]["flag"] is True


class TestReadNewMulti:
    def test_it_reads_across_several_streams(self, client):
        event_bus.publish(client, "a", {"i": 1})
        event_bus.publish(client, "b", {"i": 2})
        entries, cursors = event_bus.read_new_multi(
            client, {"a": "0", "b": "0"}, block_ms=10)
        assert {e[0] for e in entries} == {"a", "b"}
        assert cursors["a"] != "0" and cursors["b"] != "0"

    def test_cursors_are_returned_unchanged_when_nothing_is_new(self, client):
        entries, cursors = event_bus.read_new_multi(
            client, {"a": "9-9", "b": "8-8"}, block_ms=10)
        assert entries == []
        assert cursors == {"a": "9-9", "b": "8-8"}

    def test_only_the_streams_with_data_advance(self, client):
        event_bus.publish(client, "a", {"i": 1})
        _, cursors = event_bus.read_new_multi(client, {"a": "0", "b": "0"}, block_ms=10)
        assert cursors["b"] == "0"


class TestDegradingRatherThanPropagating:
    def _consumer(self, client, stream="s1"):
        return StreamConsumer(client, stream, "g1", "c1", max_deliveries=3)

    def test_a_failed_xautoclaim_returns_nothing_rather_than_raising(self, client,
                                                                    monkeypatch, caplog):
        """Older Redis builds lack XAUTOCLAIM. Missing a reclaim is survivable; taking
        the module down over it is not."""
        consumer = self._consumer(client)

        def boom(*a, **k):
            raise redis_lib.exceptions.ResponseError("unknown command")

        monkeypatch.setattr(client, "xautoclaim", boom)
        with caplog.at_level("WARNING"):
            assert consumer.claim_stale() == []
        assert "XAUTOCLAIM failed" in caplog.text

    def test_claimed_entries_with_no_fields_are_skipped(self, client, monkeypatch):
        """A claimed-but-empty entry means the original was trimmed away underneath us."""
        consumer = self._consumer(client)
        monkeypatch.setattr(client, "xautoclaim",
                            lambda *a, **k: ("0-0", [("1-1", {}), ("1-2", {"a": "1"})]))
        assert len(consumer.claim_stale()) == 1

    def test_pending_count_survives_a_missing_group(self, client, monkeypatch):
        """Real Redis answers NOGROUP with a ResponseError; fakeredis raises IndexError
        instead, so the error is injected rather than provoked. Provoking it would test
        the fake's quirk, not the guard that runs in production."""
        consumer = self._consumer(client)

        def nogroup(*a, **k):
            raise redis_lib.exceptions.ResponseError("NOGROUP No such consumer group")

        monkeypatch.setattr(client, "xpending", nogroup)
        assert consumer.pending_count() == 0

    def test_delivery_count_survives_a_missing_group(self, client, monkeypatch):
        consumer = self._consumer(client)

        def nogroup(*a, **k):
            raise redis_lib.exceptions.ResponseError("NOGROUP No such consumer group")

        monkeypatch.setattr(client, "xpending_range", nogroup)
        assert consumer.delivery_count("1-1") == 0

    def test_pending_count_reads_the_summary_tuple_form(self, client):
        consumer = self._consumer(client)
        for i in range(4):
            event_bus.publish(client, "s1", {"i": i})
        consumer.read(count=10, block_ms=10)
        assert consumer.pending_count() == 4

    def test_a_broken_alerter_does_not_break_backlog_checking(self, client, monkeypatch):
        consumer = self._consumer(client)
        for i in range(5):
            event_bus.publish(client, "s1", {"i": i})
        consumer.read(count=10, block_ms=10)

        class Broken:
            def warning(self, *a, **k):
                raise ConnectionError("sink down")

        assert consumer.check_backlog(threshold=1, alerter=Broken()) == 5

    def test_an_unavailable_alerter_module_does_not_break_backlog_checking(
            self, client, monkeypatch):
        consumer = self._consumer(client)
        for i in range(5):
            event_bus.publish(client, "s1", {"i": i})
        consumer.read(count=10, block_ms=10)
        monkeypatch.setattr("src.alerting.get_alerter",
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no redis")))
        assert consumer.check_backlog(threshold=1) == 5


class TestPoisonHandling:
    def _consumer(self, client):
        return StreamConsumer(client, "s1", "g1", "c1", max_deliveries=2)

    def test_a_failing_handler_is_retried_before_being_given_up_on(self, client, caplog):
        consumer = self._consumer(client)
        event_bus.publish(client, "s1", {"bad": 1})
        entry_id, fields = consumer.read(count=1, block_ms=10)[0]

        with caplog.at_level("WARNING"):
            handled = consumer.handle(entry_id, fields,
                                      lambda f: (_ for _ in ()).throw(ValueError("nope")))
        assert handled is False
        assert "will retry" in caplog.text
        assert client.xlen(config.STREAM_DEAD_LETTER) == 0

    def test_it_is_dead_lettered_once_deliveries_are_exhausted(self, client):
        """Retrying forever is just a slow way of never making progress."""
        consumer = self._consumer(client)
        event_bus.publish(client, "s1", {"bad": 1})

        def always_fails(fields):
            raise ValueError("nope")

        # First delivery via read(); subsequent ones via the reclaim path, which is how a
        # message that is never acked actually comes back around in production.
        for entry_id, fields in consumer.read(count=1, block_ms=10):
            consumer.handle(entry_id, fields, always_fails)
        for _ in range(4):
            for entry_id, fields in consumer.claim_stale(min_idle_ms=0):
                consumer.handle(entry_id, fields, always_fails)

        assert client.xlen(config.STREAM_DEAD_LETTER) >= 1

    def test_the_dead_letter_records_enough_to_reconstruct(self, client):
        """A message that vanishes without trace is indistinguishable from one that was
        handled."""
        consumer = self._consumer(client)
        event_bus.publish(client, "s1", {"instrument_id": "nse_cm:2885"})
        entry_id, fields = consumer.read(count=1, block_ms=10)[0]
        consumer.dead_letter(entry_id, fields, "boom")

        row = client.xrange(config.STREAM_DEAD_LETTER)[-1][1]
        assert row["original_stream"] == "s1"
        assert row["original_id"] == entry_id
        assert row["error"] == "boom"
        assert "nse_cm:2885" in row["payload"]

    def test_a_dead_letter_is_acked_so_it_stops_blocking(self, client):
        """Left pending it would be reclaimed forever, and the group would never advance."""
        consumer = self._consumer(client)
        event_bus.publish(client, "s1", {"bad": 1})
        entry_id, fields = consumer.read(count=1, block_ms=10)[0]
        consumer.dead_letter(entry_id, fields, "boom")
        assert consumer.pending_count() == 0

    def test_a_dead_letter_raises_a_critical_alert(self, client):
        consumer = self._consumer(client)
        event_bus.publish(client, "s1", {"bad": 1})
        entry_id, fields = consumer.read(count=1, block_ms=10)[0]
        consumer.dead_letter(entry_id, fields, "boom")
        alerts = client.xrange(config.STREAM_ALERTS)
        assert any(row[1]["severity"] == "critical" for row in alerts)

    def test_a_broken_alerter_does_not_stop_the_dead_letter(self, client, monkeypatch,
                                                            caplog):
        """The routing is the safety action; the alert is observability on top of it."""
        consumer = self._consumer(client)
        event_bus.publish(client, "s1", {"bad": 1})
        entry_id, fields = consumer.read(count=1, block_ms=10)[0]
        monkeypatch.setattr("src.alerting.get_alerter",
                            lambda *a, **k: (_ for _ in ()).throw(ConnectionError("down")))
        with caplog.at_level("ERROR"):
            consumer.dead_letter(entry_id, fields, "boom")
        assert client.xlen(config.STREAM_DEAD_LETTER) == 1
        assert "Could not raise the dead-letter alert" in caplog.text

    def test_a_successful_handler_acks(self, client):
        consumer = self._consumer(client)
        event_bus.publish(client, "s1", {"ok": 1})
        entry_id, fields = consumer.read(count=1, block_ms=10)[0]
        assert consumer.handle(entry_id, fields, lambda f: None) is True
        assert consumer.pending_count() == 0


class TestMultiStreamConsumer:
    def test_it_reads_from_every_stream(self, client):
        consumer = MultiStreamConsumer(client, ["a", "b"], "g1", "c1")
        event_bus.publish(client, "a", {"i": 1})
        event_bus.publish(client, "b", {"i": 2})
        entries = consumer.read(count=10, block_ms=10)
        assert {e[0] for e in entries} == {"a", "b"}

    def test_acking_clears_the_pending_total(self, client):
        consumer = MultiStreamConsumer(client, ["a", "b"], "g1", "c1")
        event_bus.publish(client, "a", {"i": 1})
        for stream, entry_id, _ in consumer.read(count=10, block_ms=10):
            consumer.ack(stream, entry_id)
        assert consumer.pending_total() == 0

    def test_unacked_entries_are_counted(self, client):
        consumer = MultiStreamConsumer(client, ["a"], "g1", "c1")
        event_bus.publish(client, "a", {"i": 1})
        consumer.read(count=10, block_ms=10)
        assert consumer.pending_total() == 1


class TestPing:
    def test_a_reachable_client_pings(self, client):
        assert event_bus.ping(client) is True

    def test_an_unreachable_client_reports_false_rather_than_raising(self, monkeypatch):
        """Every module's startup check depends on this returning, not raising."""
        class Dead:
            def ping(self):
                raise redis_lib.exceptions.ConnectionError("refused")

        assert event_bus.ping(Dead()) is False
