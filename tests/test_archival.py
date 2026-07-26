"""The archive layer: batching Influx writer and the storage logger.

Lowest stakes in the codebase and tested last on purpose — if this breaks you lose
observability, not money. But it earns tests for one specific reason: **it is the only
subsystem deliberately designed to lose data**, and the correctness question is therefore
inverted. Everywhere else the property is "nothing is dropped". Here it is *"the right
things are dropped, the loss is bounded, and somebody is told"*.

Three design decisions that only tests can hold in place:

* **Points are acked once queued, not once written.** Holding the ack until a possibly-dead
  backend confirms would grow the pending list until the stream trimmed past it — losing
  far more than the occasional dropped point, and stalling the consumer group along the way.
* **The queue is bounded and drops the oldest.** An unbounded queue in front of a dead
  backend is a memory leak that takes the trading process down with it. Archival must never
  be able to do that.
* **A failed batch is not requeued.** Retrying a failing backend indefinitely is precisely
  how the queue fills in the first place.

The writer must also work with no Influx installed at all, which is the normal state of a
developer machine and of CI.
"""
from __future__ import annotations

import fakeredis
import pytest

import config
from src import event_bus
from src.influx_writer import BatchingInfluxWriter


@pytest.fixture
def client():
    return fakeredis.FakeStrictRedis(server=fakeredis.FakeServer(), decode_responses=True)


@pytest.fixture(autouse=True)
def _bus(monkeypatch, client):
    monkeypatch.setattr(event_bus, "get_client", lambda *a, **k: client)
    monkeypatch.setattr(event_bus, "ping", lambda *a, **k: True)


@pytest.fixture(autouse=True)
def _isolate_alerter():
    from src.alerting import reset_alerter

    reset_alerter()
    yield
    reset_alerter()


def writer(max_queue: int | None = None, **kwargs) -> BatchingInfluxWriter:
    # A placeholder token forces log-only mode deterministically: an empty string would
    # fall through to config.INFLUX_TOKEN and could pick up a real one from .env.
    kwargs.setdefault("token", "YOUR_TOKEN_HERE")
    if max_queue is not None:
        kwargs["queue_maxlen"] = max_queue
    return BatchingInfluxWriter(**kwargs)


class FakePoint:
    """Stands in for influxdb_client.Point, which is not installed here."""

    def __init__(self, measurement):
        self.measurement = measurement
        self.tags: dict = {}
        self.fields: dict = {}
        self.timestamp = None

    def tag(self, key, value):
        self.tags[key] = value
        return self

    def field(self, key, value):
        self.fields[key] = value
        return self

    def time(self, value, precision):
        self.timestamp = value
        return self


class FakePrecision:
    NS = "ns"


# ---------------------------------------------------------------------------
class TestBoundedQueue:
    def test_writing_without_influx_still_counts(self):
        """Log-only is the normal developer and CI state. It must not look like failure."""
        w = writer()
        w.write("ticks", {"ltp": 1300.0})
        assert w.snapshot()["queued"] == 1
        assert w.connected is False

    def test_the_queue_is_bounded(self):
        w = writer(max_queue=10)
        for i in range(25):
            w.write("ticks", {"ltp": i})
        assert w.snapshot()["queued"] == 10

    def test_overflow_drops_the_oldest_and_counts_it(self):
        """An unbounded queue in front of a dead backend is a memory leak that takes the
        trading process down with it."""
        w = writer(max_queue=5)
        for i in range(12):
            w.write("ticks", {"ltp": i})
        assert w.snapshot()["dropped"] == 7

    def test_the_newest_points_are_the_ones_kept(self):
        w = writer(max_queue=3)
        for i in range(6):
            w.write("ticks", {"ltp": i})
        remaining = [fields["ltp"] for _, fields in w._drain(10)]
        assert remaining == [3, 4, 5]

    def test_dropping_raises_an_alert_once(self):
        received = []

        class Alerter:
            def warning(self, *a, **k):
                received.append(a)

        w = writer(max_queue=2, alerter=Alerter())
        for i in range(50):
            w.write("ticks", {"ltp": i})
        assert len(received) == 1, "the drop alert must not flood"

    def test_a_broken_alerter_does_not_break_writing(self):
        class Broken:
            def warning(self, *a, **k):
                raise ConnectionError("sink down")

        w = writer(max_queue=2, alerter=Broken())
        for i in range(10):
            w.write("ticks", {"ltp": i})
        assert w.snapshot()["dropped"] > 0

    def test_no_alert_below_the_drop_ratio(self, monkeypatch):
        monkeypatch.setattr(config, "INFLUX_DROP_ALERT_RATIO", 0.99)
        received = []

        class Alerter:
            def warning(self, *a, **k):
                received.append(a)

        w = writer(max_queue=5, alerter=Alerter())
        for i in range(6):
            w.write("ticks", {"ltp": i})
        assert received == []


class TestFlushing:
    def test_flush_empties_the_queue(self):
        w = writer(batch_size=3)
        for i in range(7):
            w.write("ticks", {"ltp": i})
        assert w.flush() == 7
        assert w.snapshot()["queued"] == 0

    def test_flushing_an_empty_queue_is_harmless(self):
        assert writer().flush() == 0

    def test_a_failed_batch_is_not_requeued(self, monkeypatch):
        """Retrying a failing backend indefinitely is how the queue fills in the first
        place. The points are lost, the failure is counted, and the archiver moves on."""
        w = writer()
        monkeypatch.setattr(type(w), "connected", property(lambda self: True))

        class DeadApi:
            def write(self, **kwargs):
                raise ConnectionError("influx down")

        w._write_api = DeadApi()
        w.write("ticks", {"ltp": 1.0})
        assert w.flush() == 0
        assert w.snapshot()["queued"] == 0
        assert w.write_failures == 1

    def test_the_snapshot_reports_a_drop_ratio(self):
        w = writer(max_queue=2)
        for i in range(10):
            w.write("ticks", {"ltp": i})
        snapshot = w.snapshot()
        assert 0.0 < snapshot["drop_ratio"] <= 100.0


class TestPointConstruction:
    def _build(self, fields, measurement="ticks"):
        return BatchingInfluxWriter._build_point(
            measurement, fields, FakePoint, FakePrecision)

    def test_numeric_fields_are_coerced(self):
        point = self._build({"ltp": "1300.5"})
        assert point.fields["ltp"] == 1300.5

    def test_unparseable_numerics_are_skipped_not_fatal(self):
        point = self._build({"ltp": "not-a-number", "quantity": "10"})
        assert "ltp" not in point.fields
        assert point.fields["quantity"] == 10.0

    def test_text_fields_are_truncated(self):
        point = self._build({"reason": "x" * 900})
        assert len(point.fields["reason"]) == 500

    def test_a_point_with_no_fields_gets_a_counter(self):
        """Influx rejects a point carrying tags but no fields; the counter keeps the
        event visible rather than silently discarding it."""
        point = self._build({"instrument_id": "nse_cm:2885"})
        assert point.fields == {"event": 1}
        assert point.tags["instrument_id"] == "nse_cm:2885"

    def test_empty_values_are_ignored(self):
        point = self._build({"instrument_id": "", "ltp": ""})
        assert point.tags == {}
        assert point.fields == {"event": 1}

    def test_a_timestamp_is_converted_to_nanoseconds(self):
        point = self._build({"ltp": "1.0", "timestamp": "1700000000"})
        assert point.timestamp == 1_700_000_000 * 1_000_000_000

    def test_an_unparseable_timestamp_is_ignored(self):
        point = self._build({"ltp": "1.0", "timestamp": "yesterday"})
        assert point.timestamp is None


class TestBackgroundThread:
    def test_start_is_idempotent(self):
        w = writer()
        w.start()
        first = w._thread
        w.start()
        assert w._thread is first
        w.close()

    def test_close_drains_what_is_queued(self):
        """Unflushed points are simply lost, so shutdown has to drain."""
        w = writer()
        w.start()
        for i in range(5):
            w.write("ticks", {"ltp": i})
        w.close()
        assert w.snapshot()["queued"] == 0

    def test_close_without_start_is_harmless(self):
        writer().close()

    def test_no_token_means_log_only(self, caplog):
        with caplog.at_level("WARNING"):
            w = BatchingInfluxWriter(token="YOUR_TOKEN_HERE")
            w._connect()
        assert w.connected is False
        assert "archiving to the log only" in caplog.text


class TestStorageLogger:
    def test_it_stops_when_redis_is_unreachable(self, monkeypatch, caplog):
        import src.storage_logger as mod

        monkeypatch.setattr(event_bus, "ping", lambda *a, **k: False)
        logger = mod.StorageLogger(writer=writer())
        with caplog.at_level("ERROR"):
            logger.run()
        assert "Redis not reachable" in caplog.text

    def test_entries_are_acked_once_queued(self, monkeypatch):
        """The archive is best-effort by design. Holding the ack until a possibly-dead
        backend confirms would grow the pending list until the stream trimmed past it."""
        import src.storage_logger as mod

        w = writer()
        logger = mod.StorageLogger(writer=w)

        class Stop(RuntimeError):
            pass

        acked = []
        rounds = {"n": 0}

        def read(**kwargs):
            rounds["n"] += 1
            if rounds["n"] > 1:
                raise Stop()
            return [(config.STREAM_MARKET_TICKS, "1-1", {"ltp": 1300.0})]

        monkeypatch.setattr(logger.consumer, "read", read)
        monkeypatch.setattr(logger.consumer, "ack",
                            lambda stream, entry_id: acked.append(entry_id))
        with pytest.raises(Stop):
            logger.run()
        assert acked == ["1-1"]
        assert w.snapshot()["queued"] == 1

    def test_streams_map_to_measurements(self, monkeypatch):
        import src.storage_logger as mod

        assert mod.STREAM_MEASUREMENTS
        logger = mod.StorageLogger(writer=writer())
        logger._write("ticks", {"ltp": 1.0})
        assert logger.writer.snapshot()["queued"] == 1

    def test_close_drains(self):
        import src.storage_logger as mod

        w = writer()
        logger = mod.StorageLogger(writer=w)
        logger._write("ticks", {"ltp": 1.0})
        logger.close()
        assert w.snapshot()["queued"] == 0

    def test_main_always_closes_even_when_the_loop_raises(self, monkeypatch):
        """Otherwise a crash costs whatever was still queued."""
        import src.storage_logger as mod

        closed = []
        monkeypatch.setattr(mod.StorageLogger, "run",
                            lambda self: (_ for _ in ()).throw(RuntimeError("boom")))
        monkeypatch.setattr(mod.StorageLogger, "close", lambda self: closed.append(1))
        with pytest.raises(RuntimeError):
            mod.main()
        assert closed == [1]
