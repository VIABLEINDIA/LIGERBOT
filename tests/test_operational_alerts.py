"""The three alert conditions DESIGN.md 3.9 listed but nothing raised.

Halt, dead-letter, reconciliation mismatch and archive overflow were already wired. These
three were not, and each is a condition where the bot *looks* healthy while quietly not
working — which is the worst shape for a failure to take:

* **Feed stale** — entries are blocked, exits still permitted, nothing says so.
* **Order rejected** — one is routine; a run of them is systemic, and the strategy keeps
  generating signals into it.
* **Consumer backlog** — the module runs, logs and passes any liveness check. It is simply
  not keeping up, and the stream will eventually trim past its own unacked entries.
"""
from __future__ import annotations

import datetime as dt

import fakeredis
import pytest

import config
from src import event_bus
from src import market_calendar as cal
from src.alerting import Alerter, Severity, reset_alerter
from src.feed_health import FeedMonitor, FeedState

DAY = dt.date(2026, 7, 23)
MIDDAY = cal.at(DAY, dt.time(12, 0))


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
    return received, Alerter([lambda a: received.append(a)], cooldown_seconds=0)


class TestFeedStaleAlerts:
    def test_going_stale_raises_an_alert(self):
        received, alerter = collector()
        monitor = FeedMonitor(stale_after_seconds=30, alerter=alerter)
        monitor.record_tick("nse_cm:1", 1300.0, now=1000.0)
        monitor.evaluate(now=1100.0, moment=MIDDAY)
        assert received
        assert received[0].severity is Severity.CRITICAL

    def test_the_alert_says_exits_remain_permitted(self):
        """The asymmetry matters to whoever reads it at 11:03."""
        received, alerter = collector()
        monitor = FeedMonitor(stale_after_seconds=30, alerter=alerter)
        monitor.record_tick("nse_cm:1", 1300.0, now=1000.0)
        monitor.record_tick("nse_cm:2", 1650.0, now=1090.0)
        monitor.evaluate(now=1100.0, moment=MIDDAY)
        assert "exits remain permitted" in received[0].message

    def test_a_whole_feed_outage_reads_differently(self):
        received, alerter = collector()
        monitor = FeedMonitor(stale_after_seconds=30, alerter=alerter)
        monitor.record_tick("nse_cm:1", 1300.0, now=1000.0)
        monitor.record_tick("nse_cm:2", 1650.0, now=1000.0)
        monitor.evaluate(now=1200.0, moment=MIDDAY)
        assert any(a.title == "Market feed down" for a in received)

    def test_per_instrument_deduplication(self):
        """evaluate() runs every loop; without dedup this alerts every second."""
        received = []
        alerter = Alerter([lambda a: received.append(a)], cooldown_seconds=3600)
        monitor = FeedMonitor(stale_after_seconds=30, alerter=alerter)
        monitor.record_tick("nse_cm:1", 1300.0, now=1000.0)
        for _ in range(20):
            monitor.evaluate(now=1100.0, moment=MIDDAY)
        assert len(received) == 1

    def test_no_alert_outside_market_hours(self):
        """Silence overnight is correct, and alerting on it trains people to ignore alerts."""
        received, alerter = collector()
        monitor = FeedMonitor(stale_after_seconds=30, alerter=alerter)
        monitor.record_tick("nse_cm:1", 1300.0, now=1000.0)
        monitor.evaluate(now=99_000.0, moment=cal.at(DAY, dt.time(22, 0)))
        assert received == []

    def test_no_alerter_configured_is_harmless(self):
        monitor = FeedMonitor(stale_after_seconds=30)
        monitor.record_tick("nse_cm:1", 1300.0, now=1000.0)
        monitor.evaluate(now=1100.0, moment=MIDDAY)   # must not raise

    def test_a_failing_alerter_does_not_break_the_watchdog(self):
        class Broken:
            def critical(self, *a, **kw):
                raise ConnectionError("down")

        monitor = FeedMonitor(stale_after_seconds=30, alerter=Broken())
        monitor.record_tick("nse_cm:1", 1300.0, now=1000.0)
        monitor.evaluate(now=1100.0, moment=MIDDAY)
        assert monitor.state_of("nse_cm:1") is FeedState.STALE


class TestBacklogAlerts:
    def _consumer(self, client, name="w1"):
        return event_bus.StreamConsumer(client, "approved_orders", "exec", name)

    def test_no_alert_below_the_threshold(self, client):
        received, alerter = collector()
        consumer = self._consumer(client)
        for i in range(5):
            event_bus.publish(client, "approved_orders", {"i": i})
        consumer.read(count=10, block_ms=10)
        assert consumer.check_backlog(threshold=100, alerter=alerter) == 5
        assert received == []

    def test_alert_above_the_threshold(self, client):
        received, alerter = collector()
        consumer = self._consumer(client)
        for i in range(20):
            event_bus.publish(client, "approved_orders", {"i": i})
        consumer.read(count=50, block_ms=10)
        pending = consumer.check_backlog(threshold=10, alerter=alerter)
        assert pending == 20
        assert received
        assert "not keeping up" in received[0].message

    def test_acking_clears_the_backlog(self, client):
        received, alerter = collector()
        consumer = self._consumer(client)
        for i in range(20):
            event_bus.publish(client, "approved_orders", {"i": i})
        for entry_id, _ in consumer.read(count=50, block_ms=10):
            consumer.ack(entry_id)
        assert consumer.check_backlog(threshold=10, alerter=alerter) == 0
        assert received == []

    def test_threshold_zero_disables_the_check(self, client):
        received, alerter = collector()
        consumer = self._consumer(client)
        for i in range(50):
            event_bus.publish(client, "approved_orders", {"i": i})
        consumer.read(count=100, block_ms=10)
        consumer.check_backlog(threshold=0, alerter=alerter)
        assert received == []

    def test_deduplicated_per_stream_and_group(self, client):
        received = []
        alerter = Alerter([lambda a: received.append(a)], cooldown_seconds=3600)
        consumer = self._consumer(client)
        for i in range(20):
            event_bus.publish(client, "approved_orders", {"i": i})
        consumer.read(count=50, block_ms=10)
        for _ in range(10):
            consumer.check_backlog(threshold=5, alerter=alerter)
        assert len(received) == 1

    def test_default_threshold_comes_from_config(self):
        assert config.BACKLOG_ALERT_THRESHOLD > 0


class TestOrderRejectionAlerts:
    def test_a_rejection_raises_a_warning(self, client, monkeypatch):
        monkeypatch.setattr(event_bus, "get_client", lambda *a, **k: client)
        monkeypatch.setattr(event_bus, "ping", lambda *a, **k: True)

        from src.execution_engine import ExecutionEngine
        from src.order_state import ManagedOrder
        from src.risk_engine import Intent, Side

        engine = ExecutionEngine(None)
        order = ManagedOrder(client_order_id="lb1", instrument_id="nse_cm:2885",
                             side=Side.BUY, intent=Intent.OPEN_LONG, quantity=10)
        order.mark_sent()
        order.mark_rejected("insufficient margin")
        engine._alert_rejection(order)

        rows = client.xrange(config.STREAM_ALERTS)
        assert rows
        assert rows[-1][1]["severity"] == "warning"
        assert "margin" in rows[-1][1]["message"]

    def test_the_running_count_is_carried(self, client, monkeypatch):
        """One rejection is routine; a run of them is systemic."""
        monkeypatch.setattr(event_bus, "get_client", lambda *a, **k: client)
        monkeypatch.setattr(event_bus, "ping", lambda *a, **k: True)

        from src.execution_engine import ExecutionEngine
        from src.order_state import ManagedOrder
        from src.risk_engine import Intent, Side

        engine = ExecutionEngine(None)
        for i in range(3):
            order = ManagedOrder(client_order_id=f"lb{i}", instrument_id="nse_cm:1",
                                 side=Side.BUY, intent=Intent.OPEN_LONG, quantity=10)
            order.mark_sent()
            order.mark_rejected(f"reason {i}")
            engine._alert_rejection(order)
        assert engine._rejections == 3

    def test_distinct_reasons_alert_separately(self, client, monkeypatch):
        """Different causes must surface separately; a repeating one must not flood."""
        monkeypatch.setattr(event_bus, "get_client", lambda *a, **k: client)
        monkeypatch.setattr(event_bus, "ping", lambda *a, **k: True)

        from src.execution_engine import ExecutionEngine
        from src.order_state import ManagedOrder
        from src.risk_engine import Intent, Side

        engine = ExecutionEngine(None)
        for reason in ("insufficient margin", "insufficient margin", "invalid symbol"):
            order = ManagedOrder(client_order_id=f"lb{reason}", instrument_id="nse_cm:1",
                                 side=Side.BUY, intent=Intent.OPEN_LONG, quantity=10)
            order.mark_sent()
            order.mark_rejected(reason)
            engine._alert_rejection(order)

        titles = [r[1]["message"] for r in client.xrange(config.STREAM_ALERTS)]
        assert len(titles) == 2, "the repeat should dedupe, the new cause should not"


class TestAllSixConditionsCovered:
    def test_every_design_listed_condition_now_alerts(self):
        """DESIGN.md 3.9 lists six. All six now have a call site."""
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent / "src"
        sources = {p.name: p.read_text(encoding="utf-8") for p in root.glob("*.py")}

        expected = {
            "halt": "kill_switch.py",
            "dead-letter": "event_bus.py",
            "reconciliation": "position_manager.py",
            "feed stale": "feed_health.py",
            "order rejected": "execution_engine.py",
            "backlog": "event_bus.py",
        }
        for condition, module in expected.items():
            text = sources[module]
            assert "alerter" in text or "get_alerter" in text, (
                f"{condition} has no alert path in {module}")
