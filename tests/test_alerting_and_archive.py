"""Alerting and batched archiving (DESIGN.md 3.9, second half of B12).

Two gaps closed here, with a shared theme: **neither may ever damage trading.**

Before this, a halt logged ``ERROR`` to stdout and nothing else. During a six-hour session
nobody is watching that terminal, so the halt was effectively invisible until someone next
looked.

And the archiver wrote one point per event synchronously. The failure was indirect: it
blocks on Influx, stops acking, its pending list grows, the stream reaches MAXLEN and trims
entries that were never archived — so the archive develops holes exactly when the most is
happening.
"""
from __future__ import annotations

import time

import fakeredis
import pytest

import config
from src.alerting import (
    Alert, Alerter, Severity, build_alerter, log_sink, redis_sink, reset_alerter,
)
from src.influx_writer import BatchingInfluxWriter


@pytest.fixture
def client():
    return fakeredis.FakeStrictRedis(server=fakeredis.FakeServer(), decode_responses=True)


@pytest.fixture(autouse=True)
def _reset():
    reset_alerter()
    yield
    reset_alerter()


def collector():
    received = []
    return received, lambda alert: received.append(alert)


# ---------------------------------------------------------------------------
class TestAlertRouting:
    def test_alert_reaches_every_sink(self):
        a, sink_a = collector()
        b, sink_b = collector()
        Alerter([sink_a, sink_b]).warning("t", "m")
        assert len(a) == 1 and len(b) == 1

    def test_a_failing_sink_does_not_break_the_caller(self):
        """Losing an alert is bad; losing the trading loop because one failed is worse."""
        received, good = collector()

        def broken(alert):
            raise ConnectionError("webhook down")

        assert Alerter([broken, good]).critical("t", "m") is True
        assert len(received) == 1, "a later sink must still run after an earlier failure"

    def test_severity_helpers(self):
        received, sink = collector()
        alerter = Alerter([sink], cooldown_seconds=0)
        alerter.info("a", "m")
        alerter.warning("b", "m")
        alerter.critical("c", "m")
        assert [x.severity for x in received] == [
            Severity.INFO, Severity.WARNING, Severity.CRITICAL]

    def test_min_severity_filters(self):
        received, sink = collector()
        alerter = Alerter([sink], cooldown_seconds=0, min_severity=Severity.CRITICAL)
        alerter.warning("ignored", "m")
        alerter.critical("kept", "m")
        assert [x.title for x in received] == ["kept"]


class TestDeduplication:
    def test_repeats_are_suppressed(self):
        """A stale feed evaluated every second would otherwise alert every second."""
        received, sink = collector()
        alerter = Alerter([sink], cooldown_seconds=3600)
        for _ in range(50):
            alerter.warning("Feed stale", "no tick for nse_cm:1")
        assert len(received) == 1
        assert alerter.suppressed_total == 49

    def test_the_suppressed_count_is_reported_when_it_refires(self):
        """The volume stays visible without the flood."""
        received, sink = collector()
        alerter = Alerter([sink], cooldown_seconds=0.15)
        alerter.warning("Feed stale", "m")
        for _ in range(9):
            alerter.warning("Feed stale", "m")
        time.sleep(0.2)
        alerter.warning("Feed stale", "m")
        assert len(received) == 2
        assert received[1].suppressed_since_last == 9

    def test_different_conditions_are_not_deduped_together(self):
        received, sink = collector()
        alerter = Alerter([sink], cooldown_seconds=3600)
        alerter.warning("Feed stale", "m", dedup_key="stale:nse_cm:1")
        alerter.warning("Feed stale", "m", dedup_key="stale:nse_cm:2")
        assert len(received) == 2

    def test_critical_is_deduped_too(self):
        """A condition that keeps tripping is still one condition."""
        received, sink = collector()
        alerter = Alerter([sink], cooldown_seconds=3600)
        for _ in range(5):
            alerter.critical("Halted", "drawdown")
        assert len(received) == 1

    def test_dedup_key_defaults_to_the_title(self):
        assert Alert("Some title", "m").dedup_key == "Some title"


class TestPersistence:
    def test_alerts_land_on_a_redis_stream(self, client):
        """So the evening briefing can report a session nobody watched."""
        Alerter([redis_sink(client)]).critical("Halted", "drawdown breached")
        rows = client.xrange(config.STREAM_ALERTS)
        assert len(rows) == 1
        assert rows[0][1]["title"] == "Halted"
        assert rows[0][1]["severity"] == "critical"

    def test_build_alerter_always_includes_the_log_sink(self, monkeypatch):
        """A misconfigured webhook must not leave a module with no alerting at all."""
        monkeypatch.setattr(config, "ALERT_WEBHOOK_URL", "")
        assert log_sink in build_alerter(None).sinks

    def test_webhook_is_added_when_configured(self, client, monkeypatch):
        monkeypatch.setattr(config, "ALERT_WEBHOOK_URL", "https://example.com/hook")
        assert len(build_alerter(client).sinks) == 3


class TestHaltRaisesAnAlert:
    def test_halting_emits_a_critical_alert(self, client):
        from src.kill_switch import KillSwitch

        KillSwitch(client).halt("daily drawdown breached", source="risk_engine")
        rows = client.xrange(config.STREAM_ALERTS)
        assert rows, "a halt reaching only stdout is invisible mid-session"
        assert rows[-1][1]["severity"] == "critical"
        assert "drawdown" in rows[-1][1]["message"]

    def test_alert_failure_does_not_prevent_the_halt(self, client, monkeypatch):
        from src import alerting
        from src.kill_switch import KillSwitch

        monkeypatch.setattr(alerting, "get_alerter",
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
        switch = KillSwitch(client)
        switch.halt("something bad")
        assert switch.is_halted(), "the halt must take effect even if alerting fails"


class TestDeadLetterAlerts:
    def test_dead_lettering_raises_an_alert(self, client):
        from src import event_bus

        event_bus.publish(client, "approved_orders", {"id": "bad"})
        consumer = event_bus.StreamConsumer(
            client, "approved_orders", "g", "w", max_deliveries=1)
        entry_id, fields = consumer.read(block_ms=10)[0]
        consumer.dead_letter(entry_id, fields, "unparseable")

        rows = client.xrange(config.STREAM_ALERTS)
        assert rows, "giving up on an order must reach a human"
        assert rows[-1][1]["severity"] == "critical"


# ---------------------------------------------------------------------------
class TestArchiveNeverBlocks:
    def test_write_returns_without_a_backend(self):
        writer = BatchingInfluxWriter(token="", queue_maxlen=10)
        writer.write("ticks", {"ltp": 100.0})
        assert writer.enqueued == 1
        assert not writer.connected

    def test_queue_is_bounded_and_drops_oldest(self):
        """Falling behind must cost archive completeness, never consumer progress."""
        writer = BatchingInfluxWriter(token="", queue_maxlen=100)
        for i in range(1000):
            writer.write("ticks", {"ltp": float(i)})
        assert writer.enqueued == 1000
        assert writer.dropped == 900
        assert len(writer._queue) == 100

    def test_the_newest_points_survive_an_overflow(self):
        """During a burst, recent data describes the state you are in now."""
        writer = BatchingInfluxWriter(token="", queue_maxlen=5)
        for i in range(20):
            writer.write("ticks", {"ltp": float(i)})
        kept = [fields["ltp"] for _, fields in writer._queue]
        assert kept == [15.0, 16.0, 17.0, 18.0, 19.0]

    def test_drop_ratio_is_reported(self):
        writer = BatchingInfluxWriter(token="", queue_maxlen=10)
        for i in range(100):
            writer.write("ticks", {"ltp": float(i)})
        snapshot = writer.snapshot()
        assert snapshot["dropped"] == 90
        assert snapshot["drop_ratio"] == pytest.approx(0.9)

    def test_overflow_raises_an_alert(self):
        received, sink = collector()
        writer = BatchingInfluxWriter(token="", queue_maxlen=5,
                                      alerter=Alerter([sink], cooldown_seconds=0))
        for i in range(500):
            writer.write("ticks", {"ltp": float(i)})
        assert received, "silent archive gaps are worse than noisy ones"
        assert "dropped" in received[0].message


class TestBatching:
    def test_flush_drains_the_queue(self):
        writer = BatchingInfluxWriter(token="", batch_size=10)
        for i in range(35):
            writer.write("ticks", {"ltp": float(i)})
        assert writer.flush() == 35
        assert len(writer._queue) == 0

    def test_flush_on_an_empty_queue_is_a_no_op(self):
        assert BatchingInfluxWriter(token="").flush() == 0

    def test_background_thread_flushes(self):
        writer = BatchingInfluxWriter(token="", flush_interval_ms=50)
        writer.start()
        try:
            for i in range(10):
                writer.write("ticks", {"ltp": float(i)})
            time.sleep(0.25)
            assert writer.written >= 10
        finally:
            writer.close()

    def test_close_drains_what_is_left(self):
        writer = BatchingInfluxWriter(token="", flush_interval_ms=60_000)
        writer.start()
        for i in range(5):
            writer.write("ticks", {"ltp": float(i)})
        writer.close()
        assert writer.written == 5, "unflushed points would otherwise be lost on exit"

    def test_snapshot_reports_state(self):
        writer = BatchingInfluxWriter(token="", queue_maxlen=50)
        writer.write("ticks", {"ltp": 1.0})
        snapshot = writer.snapshot()
        assert snapshot["queued"] == 1
        assert snapshot["queue_capacity"] == 50
        assert snapshot["connected"] is False


class TestStorageLoggerIntegration:
    def test_events_are_queued_not_written_synchronously(self, client, monkeypatch):
        from src import event_bus
        from src.storage_logger import StorageLogger

        monkeypatch.setattr(event_bus, "get_client", lambda *a, **k: client)
        monkeypatch.setattr(event_bus, "ping", lambda *a, **k: True)

        writer = BatchingInfluxWriter(token="", queue_maxlen=100)
        logger = StorageLogger(writer=writer)
        logger._write("market_ticks", {"ltp": 100.0, "instrument_id": "nse_cm:1"})
        assert writer.enqueued == 1
        logger.close()
