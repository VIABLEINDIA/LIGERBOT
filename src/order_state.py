"""Order lifecycle and idempotency (DESIGN.md 3.3).

Fixes **B3**: the execution engine published to ``filled_orders`` the moment the broker
*accepted* an order, and nothing consumed that stream. So the bot never learned what it
actually owned or what it made — which is also why the drawdown breaker was dead code
(B2). Acceptance is not a fill. An accepted order can rest unfilled, fill partially, or be
rejected by the exchange seconds later.

Two mechanisms:

**A real state machine.** ``PENDING -> SENT -> ACKED -> PARTIAL -> FILLED`` with terminal
branches for rejection, cancellation and expiry. Only genuine fills emit fill events, and
each partial emits its own — a position built from three partials is three fills, not one.

**Idempotency.** Every order carries a deterministic client id derived from the signal that
caused it. Combined with at-least-once delivery from the consumer groups (§3.1), this is
what makes a crash mid-send safe: the redelivered order computes the same id, the dedupe
set recognises it, and it is not sent twice. Without it, at-least-once delivery would mean
*duplicate orders* — trading the same signal twice.

The asymmetry is deliberate: at-least-once delivery plus idempotency gives exactly-once
*effects*. At-most-once delivery (what B6 did) gives silently missing orders, which is
strictly worse — a duplicate is detectable and reversible, a missing order is neither.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import config
from src.risk_engine import Intent, OrderRequest, Side

log = logging.getLogger("ligerbot.order_state")


class OrderStatus(Enum):
    PENDING = "PENDING"      # created locally, not yet sent
    SENT = "SENT"            # handed to the broker, no response yet
    ACKED = "ACKED"          # broker accepted it; resting, not filled
    PARTIAL = "PARTIAL"      # partially filled
    FILLED = "FILLED"        # fully filled — terminal
    REJECTED = "REJECTED"    # terminal
    CANCELLED = "CANCELLED"  # terminal
    EXPIRED = "EXPIRED"      # no ack within the timeout — terminal

    @property
    def is_terminal(self) -> bool:
        return self in (OrderStatus.FILLED, OrderStatus.REJECTED,
                        OrderStatus.CANCELLED, OrderStatus.EXPIRED)

    @property
    def is_live(self) -> bool:
        """Working at the broker — must be polled and reconciled."""
        return self in (OrderStatus.SENT, OrderStatus.ACKED, OrderStatus.PARTIAL)


# Which transitions are legal. An illegal one means our view of the order disagrees with
# the broker's, and quietly applying it would hide the disagreement.
_ALLOWED: Dict[OrderStatus, set] = {
    OrderStatus.PENDING: {OrderStatus.SENT, OrderStatus.REJECTED, OrderStatus.CANCELLED},
    OrderStatus.SENT: {OrderStatus.ACKED, OrderStatus.REJECTED, OrderStatus.PARTIAL,
                       OrderStatus.FILLED, OrderStatus.EXPIRED, OrderStatus.CANCELLED},
    OrderStatus.ACKED: {OrderStatus.PARTIAL, OrderStatus.FILLED, OrderStatus.CANCELLED,
                        OrderStatus.REJECTED, OrderStatus.EXPIRED},
    OrderStatus.PARTIAL: {OrderStatus.PARTIAL, OrderStatus.FILLED,
                          OrderStatus.CANCELLED, OrderStatus.EXPIRED},
    OrderStatus.FILLED: set(),
    OrderStatus.REJECTED: set(),
    OrderStatus.CANCELLED: set(),
    OrderStatus.EXPIRED: set(),
}


class IllegalTransition(RuntimeError):
    """Raised on a transition the state machine forbids."""


def client_order_id(
    instrument_id: str, intent: Intent, signal_time: Any, quantity: int
) -> str:
    """Deterministic id for an order, derived from the signal that caused it.

    The same signal always produces the same id, so a redelivered order is recognisable
    as a duplicate. Kept short because brokers cap the tag field — Kotak's ``tag`` is
    limited, so this is 20 characters rather than a full hash.
    """
    material = f"{instrument_id}|{intent.value}|{signal_time}|{quantity}"
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:12]
    return f"lb{digest}"


@dataclass
class Fill:
    """One execution against an order. Partials each produce their own."""

    quantity: int
    price: float
    at: dt.datetime
    exchange_fill_id: str = ""


@dataclass
class ManagedOrder:
    """An order and everything known about its life."""

    client_order_id: str
    instrument_id: str
    side: Side
    intent: Intent
    quantity: int
    order_type: str = "MARKET"
    limit_price: Optional[float] = None
    stop_loss: Optional[float] = None
    strategy_name: str = ""
    correlation_id: str = ""

    status: OrderStatus = OrderStatus.PENDING
    broker_order_id: str = ""
    fills: List[Fill] = field(default_factory=list)
    created_at: dt.datetime = field(default_factory=dt.datetime.now)
    sent_at: Optional[dt.datetime] = None
    last_update: dt.datetime = field(default_factory=dt.datetime.now)
    error: str = ""
    history: List[str] = field(default_factory=list)

    # -- derived -----------------------------------------------------------
    @property
    def filled_quantity(self) -> int:
        return sum(f.quantity for f in self.fills)

    @property
    def remaining_quantity(self) -> int:
        return max(0, self.quantity - self.filled_quantity)

    @property
    def average_fill_price(self) -> Optional[float]:
        filled = self.filled_quantity
        if filled <= 0:
            return None
        return sum(f.price * f.quantity for f in self.fills) / filled

    @property
    def is_terminal(self) -> bool:
        return self.status.is_terminal

    # -- transitions -------------------------------------------------------
    def transition(self, new_status: OrderStatus, *, note: str = "") -> None:
        if new_status not in _ALLOWED[self.status]:
            raise IllegalTransition(
                f"{self.client_order_id}: cannot go {self.status.value} -> "
                f"{new_status.value}. This means our view of the order disagrees with "
                f"the broker's; applying it silently would hide the disagreement."
            )
        self.history.append(
            f"{dt.datetime.now().isoformat(timespec='seconds')} "
            f"{self.status.value}->{new_status.value}" + (f" ({note})" if note else "")
        )
        self.status = new_status
        self.last_update = dt.datetime.now()

    def mark_sent(self) -> None:
        self.transition(OrderStatus.SENT)
        self.sent_at = dt.datetime.now()

    def mark_acked(self, broker_order_id: str) -> None:
        self.broker_order_id = broker_order_id
        self.transition(OrderStatus.ACKED, note=f"broker id {broker_order_id}")

    def mark_rejected(self, error: str) -> None:
        self.error = error
        self.transition(OrderStatus.REJECTED, note=error[:80])

    def mark_cancelled(self, reason: str = "") -> None:
        self.transition(OrderStatus.CANCELLED, note=reason[:80])

    def mark_expired(self) -> None:
        self.error = "no broker acknowledgement within the timeout"
        self.transition(OrderStatus.EXPIRED)

    def add_fill(self, fill: Fill) -> OrderStatus:
        """Record an execution and move to PARTIAL or FILLED.

        Over-fills are clamped rather than trusted: a broker reporting more filled than
        ordered means we have misread the response, and inventing extra shares would
        corrupt the position.
        """
        if fill.quantity <= 0:
            return self.status
        if self.filled_quantity + fill.quantity > self.quantity:
            clamped = self.quantity - self.filled_quantity
            log.error("%s: fill of %d exceeds remaining %d — clamping. Check the "
                      "broker response mapping.",
                      self.client_order_id, fill.quantity, clamped)
            if clamped <= 0:
                return self.status
            fill = Fill(clamped, fill.price, fill.at, fill.exchange_fill_id)

        self.fills.append(fill)
        target = (OrderStatus.FILLED if self.remaining_quantity == 0
                  else OrderStatus.PARTIAL)
        self.transition(target, note=f"{self.filled_quantity}/{self.quantity} @ "
                                     f"{fill.price:.2f}")
        return self.status

    def is_stale(self, timeout_seconds: float) -> bool:
        """True if sent but never acknowledged within the timeout."""
        if self.status is not OrderStatus.SENT or self.sent_at is None:
            return False
        return (dt.datetime.now() - self.sent_at).total_seconds() > timeout_seconds

    def to_event(self) -> Dict[str, Any]:
        return {
            "client_order_id": self.client_order_id,
            "broker_order_id": self.broker_order_id,
            "instrument_id": self.instrument_id,
            "side": self.side.value,
            "intent": self.intent.value,
            "status": self.status.value,
            "quantity": self.quantity,
            "filled_quantity": self.filled_quantity,
            "average_fill_price": self.average_fill_price or 0.0,
            "stop_loss": self.stop_loss or 0.0,
            "strategy_name": self.strategy_name,
            "correlation_id": self.correlation_id,
            "error": self.error,
            "timestamp": dt.datetime.now().timestamp(),
        }

    @classmethod
    def from_request(
        cls, request: OrderRequest, signal_time: Any, *, correlation_id: str = ""
    ) -> "ManagedOrder":
        return cls(
            client_order_id=client_order_id(
                request.instrument_id, request.intent, signal_time, request.quantity),
            instrument_id=request.instrument_id,
            side=request.side,
            intent=request.intent,
            quantity=request.quantity,
            order_type=request.order_type,
            limit_price=request.ref_price if request.order_type != "MARKET" else None,
            stop_loss=request.stop_loss,
            strategy_name=request.strategy_name,
            correlation_id=correlation_id or client_order_id(
                request.instrument_id, request.intent, signal_time, request.quantity),
        )


class OrderRegistry:
    """Tracks live orders and enforces idempotency across restarts.

    The dedupe set lives in Redis rather than in memory, because the failure it guards
    against *is* the process dying: an in-memory set would be empty exactly when the
    redelivered order arrives.
    """

    def __init__(self, client=None, *, dedupe_ttl: Optional[int] = None) -> None:
        self.client = client
        self.dedupe_ttl = dedupe_ttl or config.ORDER_DEDUPE_TTL_SECONDS
        self.orders: Dict[str, ManagedOrder] = {}

    # -- idempotency -------------------------------------------------------
    def _dedupe_key(self, client_order_id: str) -> str:
        return f"ligerbot:sent:{client_order_id}"

    def already_sent(self, client_order_id: str) -> bool:
        """True if this exact order has been sent before.

        Checked immediately *before* sending, so a redelivered message after a crash
        does not fire a second order for the same signal.
        """
        if self.client is None:
            return client_order_id in self.orders
        return bool(self.client.exists(self._dedupe_key(client_order_id)))

    def mark_sent(self, client_order_id: str) -> None:
        """Record the send. Written before the broker call, not after.

        Order matters: if the process dies between the API call and the bookkeeping,
        recording afterwards would leave no trace of an order that may well have
        reached the exchange. Recording first risks marking an order that never
        actually went out — a missed trade, which is the cheaper failure.
        """
        if self.client is not None:
            self.client.set(self._dedupe_key(client_order_id), "1", ex=self.dedupe_ttl)

    # -- registry ----------------------------------------------------------
    def register(self, order: ManagedOrder) -> ManagedOrder:
        self.orders[order.client_order_id] = order
        return order

    def get(self, client_order_id: str) -> Optional[ManagedOrder]:
        return self.orders.get(client_order_id)

    def by_broker_id(self, broker_order_id: str) -> Optional[ManagedOrder]:
        for order in self.orders.values():
            if order.broker_order_id and order.broker_order_id == broker_order_id:
                return order
        return None

    def live_orders(self) -> List[ManagedOrder]:
        return [o for o in self.orders.values() if o.status.is_live]

    def expire_stale(self, timeout_seconds: Optional[float] = None) -> List[ManagedOrder]:
        """Expire orders the broker never acknowledged.

        An order stuck in SENT is the genuinely ambiguous case — it may have reached the
        exchange or not. Marking it EXPIRED surfaces that for reconciliation instead of
        leaving it live forever.
        """
        timeout = timeout_seconds or config.ORDER_ACK_TIMEOUT_SECONDS
        expired = []
        for order in list(self.orders.values()):
            if order.is_stale(timeout):
                order.mark_expired()
                log.error("Order %s expired with no ack after %.0fs — reconcile against "
                          "the broker before assuming it did not reach the exchange.",
                          order.client_order_id, timeout)
                expired.append(order)
        return expired

    def purge_terminal(self, keep: int = 500) -> None:
        """Drop old terminal orders, keeping the most recent for inspection."""
        terminal = sorted(
            (o for o in self.orders.values() if o.is_terminal),
            key=lambda o: o.last_update,
        )
        for order in terminal[:-keep] if len(terminal) > keep else []:
            self.orders.pop(order.client_order_id, None)
