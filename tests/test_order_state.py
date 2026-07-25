"""Order lifecycle and idempotency tests (DESIGN.md 3.3) — the fix for B3."""
from __future__ import annotations

import datetime as dt

import fakeredis
import pytest

from src.order_state import (
    Fill, IllegalTransition, ManagedOrder, OrderRegistry, OrderStatus, client_order_id,
)
from src.risk_engine import Intent, OrderRequest, Side

SIGNAL_TIME = dt.datetime(2026, 7, 23, 10, 30)


def order(quantity: int = 100, **kw) -> ManagedOrder:
    defaults = dict(
        client_order_id="lbtest01", instrument_id="nse_cm:2885",
        side=Side.BUY, intent=Intent.OPEN_LONG, quantity=quantity,
    )
    defaults.update(kw)
    return ManagedOrder(**defaults)


class TestClientOrderId:
    def test_deterministic_for_the_same_signal(self):
        """The property that makes a redelivered order recognisable as a duplicate."""
        first = client_order_id("nse_cm:1", Intent.OPEN_LONG, SIGNAL_TIME, 100)
        second = client_order_id("nse_cm:1", Intent.OPEN_LONG, SIGNAL_TIME, 100)
        assert first == second

    def test_differs_across_instruments_intents_times_and_sizes(self):
        base = client_order_id("nse_cm:1", Intent.OPEN_LONG, SIGNAL_TIME, 100)
        assert base != client_order_id("nse_cm:2", Intent.OPEN_LONG, SIGNAL_TIME, 100)
        assert base != client_order_id("nse_cm:1", Intent.CLOSE_LONG, SIGNAL_TIME, 100)
        assert base != client_order_id("nse_cm:1", Intent.OPEN_LONG, SIGNAL_TIME, 200)
        assert base != client_order_id(
            "nse_cm:1", Intent.OPEN_LONG, dt.datetime(2026, 7, 23, 10, 31), 100)

    def test_short_enough_for_a_broker_tag(self):
        assert len(client_order_id("nse_cm:1", Intent.OPEN_LONG, SIGNAL_TIME, 100)) <= 20


class TestStateMachine:
    def test_happy_path(self):
        managed = order()
        assert managed.status is OrderStatus.PENDING
        managed.mark_sent()
        assert managed.status is OrderStatus.SENT
        managed.mark_acked("EX123")
        assert managed.status is OrderStatus.ACKED
        managed.add_fill(Fill(100, 1300.0, dt.datetime.now()))
        assert managed.status is OrderStatus.FILLED
        assert managed.is_terminal

    def test_acceptance_is_not_a_fill(self):
        """B3 in one assertion: ACKED means resting at the broker, not executed."""
        managed = order()
        managed.mark_sent()
        managed.mark_acked("EX1")
        assert managed.status is OrderStatus.ACKED
        assert managed.filled_quantity == 0
        assert managed.average_fill_price is None
        assert not managed.is_terminal

    def test_partial_fills_accumulate(self):
        managed = order(quantity=100)
        managed.mark_sent()
        managed.mark_acked("EX1")
        managed.add_fill(Fill(30, 1300.0, dt.datetime.now()))
        assert managed.status is OrderStatus.PARTIAL
        assert managed.remaining_quantity == 70
        managed.add_fill(Fill(70, 1302.0, dt.datetime.now()))
        assert managed.status is OrderStatus.FILLED
        assert managed.average_fill_price == pytest.approx((30 * 1300 + 70 * 1302) / 100)

    def test_illegal_transitions_raise(self):
        managed = order()
        managed.mark_sent()
        managed.mark_acked("EX1")
        managed.add_fill(Fill(100, 1300.0, dt.datetime.now()))
        with pytest.raises(IllegalTransition):
            managed.mark_acked("EX2")  # already terminal

    def test_rejection_is_terminal(self):
        managed = order()
        managed.mark_sent()
        managed.mark_rejected("insufficient margin")
        assert managed.status is OrderStatus.REJECTED
        assert managed.is_terminal
        assert "margin" in managed.error

    def test_overfill_is_clamped_not_trusted(self):
        """A broker reporting more filled than ordered means we misread the response."""
        managed = order(quantity=100)
        managed.mark_sent()
        managed.mark_acked("EX1")
        managed.add_fill(Fill(150, 1300.0, dt.datetime.now()))
        assert managed.filled_quantity == 100
        assert managed.status is OrderStatus.FILLED

    def test_history_is_recorded(self):
        managed = order()
        managed.mark_sent()
        managed.mark_acked("EX1")
        assert len(managed.history) == 2
        assert "PENDING->SENT" in managed.history[0]

    def test_terminal_statuses_are_flagged(self):
        for status in (OrderStatus.FILLED, OrderStatus.REJECTED,
                       OrderStatus.CANCELLED, OrderStatus.EXPIRED):
            assert status.is_terminal
            assert not status.is_live
        for status in (OrderStatus.SENT, OrderStatus.ACKED, OrderStatus.PARTIAL):
            assert status.is_live


class TestStaleOrders:
    def test_sent_order_goes_stale_without_an_ack(self):
        managed = order()
        managed.mark_sent()
        managed.sent_at = dt.datetime.now() - dt.timedelta(seconds=60)
        assert managed.is_stale(15.0)

    def test_acked_order_is_never_stale(self):
        managed = order()
        managed.mark_sent()
        managed.sent_at = dt.datetime.now() - dt.timedelta(seconds=600)
        managed.mark_acked("EX1")
        assert not managed.is_stale(15.0)

    def test_registry_expires_stale_orders(self):
        registry = OrderRegistry()
        managed = registry.register(order())
        managed.mark_sent()
        managed.sent_at = dt.datetime.now() - dt.timedelta(seconds=60)
        expired = registry.expire_stale(15.0)
        assert expired and expired[0].status is OrderStatus.EXPIRED


class TestIdempotency:
    @pytest.fixture
    def client(self):
        return fakeredis.FakeStrictRedis(decode_responses=True)

    def test_duplicate_send_is_detected_across_restarts(self, client):
        """The crash-mid-send case.

        At-least-once delivery means a redelivered order recomputes the same client id.
        The dedupe set must live in Redis, because an in-memory one would be empty
        exactly when the redelivery arrives.
        """
        registry = OrderRegistry(client)
        coid = client_order_id("nse_cm:1", Intent.OPEN_LONG, SIGNAL_TIME, 100)
        assert not registry.already_sent(coid)
        registry.mark_sent(coid)

        fresh = OrderRegistry(client)  # simulates a process restart
        assert fresh.already_sent(coid)

    def test_different_signals_are_not_confused(self, client):
        registry = OrderRegistry(client)
        first = client_order_id("nse_cm:1", Intent.OPEN_LONG, SIGNAL_TIME, 100)
        second = client_order_id("nse_cm:2", Intent.OPEN_LONG, SIGNAL_TIME, 100)
        registry.mark_sent(first)
        assert not registry.already_sent(second)

    def test_works_without_redis(self):
        registry = OrderRegistry(None)
        managed = registry.register(order(client_order_id="abc"))
        assert registry.already_sent("abc")
        assert not registry.already_sent("other")


class TestRegistry:
    def test_lookup_by_broker_id(self):
        registry = OrderRegistry()
        managed = registry.register(order())
        managed.mark_sent()
        managed.mark_acked("EX999")
        assert registry.by_broker_id("EX999") is managed
        assert registry.by_broker_id("nope") is None

    def test_live_orders_excludes_terminal(self):
        registry = OrderRegistry()
        live = registry.register(order(client_order_id="a"))
        live.mark_sent()
        done = registry.register(order(client_order_id="b"))
        done.mark_sent()
        done.mark_rejected("no")
        assert registry.live_orders() == [live]

    def test_from_request_builds_a_deterministic_order(self):
        request = OrderRequest(
            instrument_id="nse_cm:2885", side=Side.BUY, quantity=50,
            intent=Intent.OPEN_LONG, ref_price=1300.0, stop_loss=1274.0,
        )
        first = ManagedOrder.from_request(request, SIGNAL_TIME)
        second = ManagedOrder.from_request(request, SIGNAL_TIME)
        assert first.client_order_id == second.client_order_id
        assert first.stop_loss == 1274.0

    def test_event_payload_distinguishes_filled_from_ordered(self):
        managed = order(quantity=100)
        managed.mark_sent()
        managed.mark_acked("EX1")
        managed.add_fill(Fill(40, 1300.0, dt.datetime.now()))
        event = managed.to_event()
        assert event["status"] == "PARTIAL"
        assert event["quantity"] == 100
        assert event["filled_quantity"] == 40
