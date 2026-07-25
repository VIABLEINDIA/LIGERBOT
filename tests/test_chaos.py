"""Chaos and fault-injection tests — the Phase 3 exit criterion (DESIGN.md 4).

Two things must be demonstrated, not asserted:

1. **Kill a module mid-flight and no order is lost *or* duplicated.** These are separate
   failures with separate causes: loss comes from at-most-once delivery (defect B6),
   duplication from at-least-once delivery without idempotency (§3.3). Fixing one and not
   the other just trades a silent failure for a loud one.

2. **The drawdown circuit breaker actually trips.** It was dead code (B2) because nothing
   ever wrote realised P&L. Here the full path is exercised — fills into the position
   book, P&L into the risk engine, breaker fires.

The crashes are injected at the genuinely dangerous point: *after* the broker call and
*before* the acknowledgement, which is the window where the system's belief and the
exchange's reality can diverge.
"""
from __future__ import annotations

import datetime as dt

import fakeredis
import pytest

import config
from src import event_bus
from src.kill_switch import KillSwitch
from src.order_state import Fill, ManagedOrder, OrderRegistry, OrderStatus
from src.position_manager import PositionBook
from src.risk_engine import Intent, RiskEngine, RiskLimits, Side, Signal

DAY = dt.date(2026, 7, 23)
SIGNAL_TIME = dt.datetime(2026, 7, 23, 10, 30)
GROUP = "chaos.exec"


@pytest.fixture
def client():
    return fakeredis.FakeStrictRedis(server=fakeredis.FakeServer(), decode_responses=True)


class FakeBroker:
    """Records every order that reached it, so duplicates are detectable."""

    def __init__(self) -> None:
        self.sent: list[str] = []

    def place_order(self, client_order_id: str) -> str:
        self.sent.append(client_order_id)
        return f"EX{len(self.sent):04d}"

    def count_for(self, client_order_id: str) -> int:
        return self.sent.count(client_order_id)


class ExecutionWorker:
    """A minimal execution engine with the Phase 3 machinery, and a crash injector."""

    def __init__(self, client, broker: FakeBroker, name: str, *, crash_on: set | None = None):
        self.client = client
        self.broker = broker
        self.registry = OrderRegistry(client)
        self.crash_on = crash_on or set()
        self.consumer = event_bus.StreamConsumer(
            client, config.STREAM_APPROVED_ORDERS, GROUP, name, max_deliveries=10)
        self.processed: list[str] = []

    def _execute(self, fields: dict) -> None:
        coid = fields["client_order_id"]

        # Idempotency check happens BEFORE the send — this is what makes redelivery safe.
        if self.registry.already_sent(coid):
            self.processed.append(coid)
            return

        self.registry.mark_sent(coid)     # recorded before the call, deliberately
        broker_id = self.broker.place_order(coid)

        # Crash window: the order is at the exchange, but nothing has been acked yet.
        if coid in self.crash_on:
            raise RuntimeError(f"simulated crash after sending {coid}")

        event_bus.publish(self.client, config.STREAM_FILLED_ORDERS, {
            "client_order_id": coid, "broker_order_id": broker_id,
            "instrument_id": fields["instrument_id"], "side": fields["side"],
            "status": "FILLED", "quantity": fields["quantity"],
            "filled_quantity": fields["quantity"],
            "average_fill_price": fields["price"],
        })
        self.processed.append(coid)

    def drain(self, *, rounds: int = 3) -> None:
        for _ in range(rounds):
            batch = self.consumer.read(count=50, block_ms=10)
            batch += self.consumer.claim_stale(min_idle_ms=0, count=50)
            if not batch:
                break
            for entry_id, fields in batch:
                self.consumer.handle(entry_id, fields, self._execute)


def approve(client, index: int, quantity: int = 10) -> str:
    coid = f"lborder{index:03d}"
    event_bus.publish(client, config.STREAM_APPROVED_ORDERS, {
        "client_order_id": coid, "instrument_id": f"nse_cm:{index}",
        "side": "BUY", "quantity": quantity, "price": 1000.0 + index,
    })
    return coid


class TestNoOrderLost:
    """Defect B6: a restart must not silently discard queued orders."""

    def test_orders_queued_while_down_are_processed_on_restart(self, client):
        coids = [approve(client, i) for i in range(5)]
        # Nothing is consuming yet — this is the module-is-down window.
        broker = FakeBroker()
        ExecutionWorker(client, broker, "w1").drain()
        assert sorted(broker.sent) == sorted(coids)

    def test_crash_mid_flight_loses_nothing(self, client):
        coids = [approve(client, i) for i in range(5)]
        broker = FakeBroker()

        first = ExecutionWorker(client, broker, "w1", crash_on={coids[2]})
        first.drain()

        # A fresh worker takes over the dead one's pending messages.
        second = ExecutionWorker(client, broker, "w2")
        second.drain()

        assert set(broker.sent) == set(coids), "an order was lost across the crash"

    def test_nothing_remains_pending_after_recovery(self, client):
        coids = [approve(client, i) for i in range(4)]
        broker = FakeBroker()
        ExecutionWorker(client, broker, "w1", crash_on={coids[1]}).drain()
        recovered = ExecutionWorker(client, broker, "w2")
        recovered.drain(rounds=5)
        assert recovered.consumer.pending_count() == 0


class TestNoOrderDuplicated:
    """§3.3: at-least-once delivery without idempotency means double-firing."""

    def test_redelivered_order_is_not_sent_twice(self, client):
        coid = approve(client, 1)
        broker = FakeBroker()

        # Crashes after sending, so the message is redelivered.
        ExecutionWorker(client, broker, "w1", crash_on={coid}).drain()
        assert broker.count_for(coid) == 1

        ExecutionWorker(client, broker, "w2").drain()
        assert broker.count_for(coid) == 1, (
            "the redelivered order was sent to the broker a second time — "
            "idempotency is not holding")

    def test_every_order_reaches_the_broker_exactly_once(self, client):
        coids = [approve(client, i) for i in range(8)]
        broker = FakeBroker()

        # Three of the eight crash mid-flight.
        crashing = {coids[1], coids[4], coids[6]}
        for coid in crashing:
            ExecutionWorker(client, broker, f"w-{coid}", crash_on={coid}).drain(rounds=1)

        ExecutionWorker(client, broker, "final").drain(rounds=6)

        for coid in coids:
            assert broker.count_for(coid) == 1, (
                f"{coid} reached the broker {broker.count_for(coid)} times")

    def test_dedupe_survives_a_process_restart(self, client):
        """The dedupe set must be in Redis: an in-memory one would be empty exactly
        when the redelivered message arrives."""
        coid = approve(client, 1)
        broker = FakeBroker()
        ExecutionWorker(client, broker, "w1", crash_on={coid}).drain()

        # Brand-new registry, as after a restart.
        assert OrderRegistry(client).already_sent(coid)


class TestDrawdownBreakerFires:
    """Defect B2: the breaker could never trip because nothing wrote realised P&L."""

    def test_full_path_from_fills_to_halt(self, client):
        equity = 500_000.0
        book = PositionBook()
        book.start_session(DAY)
        risk = RiskEngine(RiskLimits())
        risk.start_session(DAY, equity)
        switch = KillSwitch(client)

        # Four full-risk losing round trips: 4 x 0.5% = 2.0%, the daily limit.
        for i in range(4):
            instrument = f"nse_cm:{i}"
            entry, stop = 1000.0, 990.0
            decision = risk.evaluate(
                Signal(instrument_id=instrument, intent=Intent.OPEN_LONG,
                       ref_price=entry, stop_loss=stop, bar_time=SIGNAL_TIME),
                allows_entry=True, allows_exit=True)
            if not decision.approved:
                break
            quantity = decision.order.quantity

            book.apply_fill(instrument, quantity, entry, stop_loss=stop)
            risk.on_open_fill(instrument, quantity, entry, stop)

            book.apply_fill(instrument, -quantity, stop)   # stopped out
            risk.on_close_fill(instrument, stop)

        assert risk.halted, "the drawdown breaker did not fire on real losses"
        assert "daily drawdown breached" in risk.halt_reason

        # And the halt propagates to every module via the shared switch.
        switch.halt(risk.halt_reason, source="risk_engine")
        assert KillSwitch(client).is_halted()

    def test_book_and_risk_engine_agree_on_pnl(self, client):
        """Two independent ledgers must not drift; a divergence would hide the breach."""
        book = PositionBook()
        book.start_session(DAY)
        risk = RiskEngine(RiskLimits())
        risk.start_session(DAY, 500_000.0)

        book.apply_fill("nse_cm:1", 100, 1000.0, stop_loss=990.0)
        risk.on_open_fill("nse_cm:1", 100, 1000.0, 990.0)
        book.apply_fill("nse_cm:1", -100, 990.0)
        risk.on_close_fill("nse_cm:1", 990.0)

        assert book.realized_pnl_today == pytest.approx(risk.realized_pnl_today)

    def test_halt_blocks_new_entries_but_not_exits(self, client):
        risk = RiskEngine(RiskLimits())
        risk.start_session(DAY, 500_000.0)
        decision = risk.evaluate(
            Signal(instrument_id="nse_cm:1", intent=Intent.OPEN_LONG,
                   ref_price=1000.0, stop_loss=990.0, bar_time=SIGNAL_TIME),
            allows_entry=True, allows_exit=True)
        risk.on_open_fill("nse_cm:1", decision.order.quantity, 1000.0, 990.0)

        risk.halt("chaos test")

        blocked = risk.evaluate(
            Signal(instrument_id="nse_cm:2", intent=Intent.OPEN_LONG,
                   ref_price=1000.0, stop_loss=990.0, bar_time=SIGNAL_TIME),
            allows_entry=True, allows_exit=True)
        assert not blocked.approved

        exit_decision = risk.evaluate(
            Signal(instrument_id="nse_cm:1", intent=Intent.CLOSE_LONG,
                   ref_price=995.0, bar_time=SIGNAL_TIME),
            allows_entry=True, allows_exit=True)
        assert exit_decision.approved, "a halt must never trap an open position"


class TestReconciliationHalt:
    def test_broker_disagreement_halts_trading(self, client):
        """A bot that does not know what it owns cannot size its next trade."""
        book = PositionBook()
        book.start_session(DAY)
        book.apply_fill("nse_cm:1", 100, 1300.0, stop_loss=1274.0)

        result = book.reconcile({"nse_cm:1": 40})
        assert not result.clean

        switch = KillSwitch(client)
        if result.discrepancy_count >= config.RECONCILE_HALT_THRESHOLD:
            switch.halt(f"reconciliation: {result.summary()}", source="position_manager")
        assert switch.is_halted()
        assert book.positions["nse_cm:1"].quantity == 40  # broker wins

    def test_unknown_broker_position_is_a_discrepancy(self, client):
        book = PositionBook()
        book.start_session(DAY)
        result = book.reconcile({"nse_cm:99": 50})
        assert result.discrepancy_count >= 1
        assert "nse_cm:99" in book.positions


class TestPoisonDoesNotStopTheLine:
    def test_one_bad_message_does_not_block_the_rest(self, client):
        """A message that fails forever must be set aside, not retried forever."""
        event_bus.publish(client, config.STREAM_APPROVED_ORDERS, {"broken": "yes"})
        good = [approve(client, i) for i in range(3)]

        broker = FakeBroker()
        worker = ExecutionWorker(client, broker, "w1")
        worker.consumer.max_deliveries = 2
        worker.drain(rounds=8)

        assert set(broker.sent) == set(good)
        assert client.xlen(config.STREAM_DEAD_LETTER) >= 1
