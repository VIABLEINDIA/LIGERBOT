"""Module 5 — Execution Engine (the final mile).

Consumes ``approved_orders`` and routes each to Kotak Neo. Rewritten in Phase 3 around the
order state machine (DESIGN.md 3.3); the original had three defects that mattered:

* **B3** — it published to ``filled_orders`` the moment the broker *accepted* an order and
  called that a fill. Acceptance is not execution: an accepted order can rest unfilled,
  fill partially, or be rejected seconds later. Now only genuine fills emit fill events,
  and each partial emits its own.
* **B6** — it read with ``last_id = "$"`` and persisted nothing, so a restart silently
  discarded queued approved orders. Now a consumer group with explicit acks.
* **B4** — it sent the display name as ``trading_symbol``. Now the instrument master
  resolves the canonical id at this boundary, and an unknown id raises rather than
  guessing.

Safety, in the order it applies:
  1. Kill switch — checked per order, effective without a restart.
  2. ``DRY_RUN`` — logs instead of sending. Start here. Always.
  3. Idempotency — a redelivered order is recognised and not sent twice.
  4. Rate limiting — stays under the broker's orders/second cap.

    python -m src.execution_engine
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import time
from typing import Any, Dict, Optional

import config
from src import event_bus
from src.instruments import InstrumentMaster
from src.kill_switch import KillSwitch
from src.order_state import (
    Fill, IllegalTransition, ManagedOrder, OrderRegistry, OrderStatus,
    parse_order_report,
)
from src.risk_engine import Intent, Side

logging.basicConfig(level=logging.INFO, format="%(asctime)s [execution] %(message)s")
log = logging.getLogger("ligerbot.execution")


class RateLimiter:
    """Simple token-spacing limiter: enforce a minimum gap between orders."""

    def __init__(self, max_per_second: int) -> None:
        self.min_interval = 1.0 / max(1, max_per_second)
        self._last = 0.0

    def wait(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last = time.monotonic()


class ExecutionEngine:
    def __init__(self, instrument_master: Optional[InstrumentMaster] = None) -> None:
        self.client = event_bus.get_client()
        self.rate_limiter = RateLimiter(config.MAX_ORDERS_PER_SECOND)
        self.registry = OrderRegistry(self.client)
        self.kill_switch = KillSwitch(self.client)
        self.instruments = instrument_master
        self.neo = None  # authenticated lazily, only for a real order
        self.consumer = event_bus.StreamConsumer(
            self.client, config.STREAM_APPROVED_ORDERS,
            f"{config.CONSUMER_GROUP_PREFIX}.execution", "exec-1",
            max_deliveries=config.MAX_DELIVERIES,
        )

    def _ensure_broker(self):
        if self.neo is None:
            from src.auth import authenticate_neo

            self.neo = authenticate_neo()
        return self.neo

    def _trading_symbol(self, instrument_id: str) -> str:
        """Resolve the canonical id to the broker's symbol (fixes B4).

        Raises rather than falling back to the raw id: a guessed symbol becomes a
        rejected order at best, and an order on the wrong instrument at worst.
        """
        if self.instruments is None:
            raise RuntimeError(
                f"No instrument master loaded, so {instrument_id!r} cannot be resolved "
                f"to a trading symbol. Refusing to guess (defect B4)."
            )
        return self.instruments.require(instrument_id).trading_symbol

    # -- sending -----------------------------------------------------------
    def _send(self, order: ManagedOrder) -> None:
        """Send one order and record the broker's response."""
        if config.DRY_RUN:
            order.mark_sent()
            order.mark_acked(f"DRYRUN-{int(time.time() * 1000)}")
            log.info("   DRY_RUN — not sent. Would place: %s %d %s",
                     order.side.value, order.quantity, order.instrument_id)
            # In dry run we simulate the fill so the downstream pipeline is exercised
            # end to end; it is explicitly flagged so nothing mistakes it for real.
            order.add_fill(Fill(order.quantity, order.limit_price or 0.0,
                                __import__("datetime").datetime.now()))
            self._publish(order, dry_run=True)
            return

        symbol = self._trading_symbol(order.instrument_id)
        self.rate_limiter.wait()
        order.mark_sent()

        try:
            response = self._ensure_broker().place_order(
                exchange_segment=config.DEFAULT_EXCHANGE_SEGMENT,
                product=config.DEFAULT_PRODUCT,
                price="0" if order.order_type == "MARKET" else str(order.limit_price or 0),
                order_type="MKT" if order.order_type == "MARKET" else "L",
                quantity=str(order.quantity),
                validity="DAY",
                trading_symbol=symbol,
                transaction_type="B" if order.side is Side.BUY else "S",
                amo="NO",
                disclosed_quantity="0",
                market_protection="0",
                pf="N",
                trigger_price="0",
                tag=order.client_order_id,  # carries idempotency to the broker
            )
        except Exception as exc:  # noqa: BLE001 - broker SDKs raise widely
            # Deliberately NOT marked rejected: the order may well have reached the
            # exchange. It stays SENT so the ack timeout expires it and reconciliation
            # can establish the truth.
            log.error("Order %s failed in transit (%s). State left SENT — reconcile "
                      "before assuming it did not reach the exchange.",
                      order.client_order_id, exc)
            order.error = str(exc)
            self._publish(order)
            return

        payload = response if isinstance(response, dict) else {"stat": "Ok"}
        if str(payload.get("stat", "")).lower() == "ok":
            order.mark_acked(str(payload.get("nOrdNo", "")))
            log.info("Order accepted — broker id %s (ACKED, not yet filled)",
                     order.broker_order_id)
        else:
            order.mark_rejected(str(payload.get("errMsg", "unknown rejection")))
            log.error("Exchange rejected %s: %s", order.client_order_id, order.error)
        self._publish(order)

    def _publish(self, order: ManagedOrder, *, dry_run: bool = False) -> None:
        """Emit the order's current state.

        Downstream keys on ``status``: only FILLED and PARTIAL move the position book.
        This is the distinction B3 collapsed.
        """
        payload = order.to_event()
        payload["dry_run"] = dry_run or config.DRY_RUN
        event_bus.publish(self.client, config.STREAM_FILLED_ORDERS, payload)

    # -- consumption -------------------------------------------------------
    def _handle(self, fields: Dict[str, Any]) -> None:
        instrument_id = fields.get("instrument_id") or fields.get("instrument")
        if not instrument_id:
            raise ValueError(f"approved order without an instrument: {fields}")

        client_order_id = fields.get("client_order_id")
        if not client_order_id:
            raise ValueError(
                f"approved order without a client_order_id: {fields}. The risk manager "
                f"must assign one — without it, a redelivered order cannot be "
                f"recognised as a duplicate.")

        # Idempotency before anything irreversible. This is what makes at-least-once
        # delivery safe rather than dangerous.
        if self.registry.already_sent(client_order_id):
            log.warning("Order %s was already sent — skipping the redelivery.",
                        client_order_id)
            return

        halt = self.kill_switch.state()
        intent = Intent(fields.get("intent", "OPEN_LONG"))
        if halt.halted and intent.is_open:
            # Halts stop new risk, never exits.
            log.error("HALTED (%s) — refusing to open %s.", halt.reason, instrument_id)
            return

        order = ManagedOrder(
            client_order_id=str(client_order_id),
            instrument_id=str(instrument_id),
            side=Side(fields.get("side", "BUY")),
            intent=intent,
            quantity=int(float(fields.get("quantity", 0))),
            order_type=str(fields.get("order_type", "MARKET")),
            limit_price=float(fields.get("price") or 0.0) or None,
            stop_loss=float(fields.get("stop_loss") or 0.0) or None,
            strategy_name=str(fields.get("strategy_name", "")),
            correlation_id=str(fields.get("correlation_id", client_order_id)),
        )
        if order.quantity <= 0:
            raise ValueError(f"approved order with quantity {order.quantity}")

        self.registry.register(order)
        self.registry.mark_sent(order.client_order_id)
        log.info("Executing %s %d %s (%s)", order.side.value, order.quantity,
                 order.instrument_id, order.client_order_id)
        self._send(order)

    def poll_open_orders(self) -> None:
        """Reconcile live orders against the broker, then expire the unacknowledged.

        Fixes the second half of **B3**. Previously this only expired stale ``SENT``
        orders, so a partial fill or a post-acknowledgement rejection was never detected
        — the bot would believe an order was still working long after the exchange had
        finished with it.

        Ordering matters: reconcile first, expire second. An order the broker has already
        filled must not be expired just because our own ack timeout elapsed.
        """
        if self.registry.live_orders() and not config.DRY_RUN:
            self._reconcile_live_orders()
        for order in self.registry.expire_stale():
            self._publish(order)

    def _reconcile_live_orders(self) -> None:
        """Apply the broker's view of every working order."""
        try:
            report = parse_order_report(self._ensure_broker().order_report())
        except Exception as exc:  # noqa: BLE001 - broker SDKs raise widely
            log.error("order_report() failed (%s) — cannot verify working orders.", exc)
            return

        for order in self.registry.live_orders():
            if not order.broker_order_id:
                continue                      # not acked yet; the timeout handles it
            status = report.get(order.broker_order_id)
            if status is None:
                # Acked but absent from the report: a real disagreement, not a
                # transient. Surfaced rather than assumed either way.
                log.error("Order %s (broker %s) is not in order_report(). Our view and "
                          "the broker's disagree — investigate before trading further.",
                          order.client_order_id, order.broker_order_id)
                continue
            self._apply_broker_status(order, status)

    def _apply_broker_status(self, order: ManagedOrder, status) -> None:
        """Move one order to match the broker, emitting fills for new quantity."""
        newly_filled = status.filled_quantity - order.filled_quantity
        if newly_filled > 0:
            # Each increment is its own fill event, so a position built from three
            # partials produces three fills rather than one aggregate.
            order.add_fill(Fill(
                quantity=newly_filled,
                price=status.average_price or order.limit_price or 0.0,
                at=dt.datetime.now(),
                exchange_fill_id=status.broker_order_id,
            ))
            log.info("Order %s filled %d more (%d/%d @ %.2f)",
                     order.client_order_id, newly_filled, order.filled_quantity,
                     order.quantity, status.average_price)
            self._publish(order)
            return

        mapped = status.mapped_status
        if mapped is None:
            log.warning("Order %s has an unrecognised broker status %r — leaving it "
                        "untouched rather than guessing.",
                        order.client_order_id, status.raw_status)
            return
        if mapped is order.status or mapped is OrderStatus.PARTIAL:
            return

        try:
            if mapped is OrderStatus.REJECTED:
                order.mark_rejected(status.rejection_reason or status.raw_status)
            elif mapped is OrderStatus.CANCELLED:
                order.mark_cancelled(status.raw_status)
            elif mapped is OrderStatus.ACKED and order.status is OrderStatus.SENT:
                order.mark_acked(status.broker_order_id)
            else:
                return
        except IllegalTransition as exc:
            # Our state machine and the broker disagree about what is possible. Logged
            # loudly rather than forced, because forcing it would hide the disagreement.
            log.error("Cannot apply broker status to %s: %s", order.client_order_id, exc)
            return
        self._publish(order)

    def _check_live_clearance(self) -> bool:
        """Refuse to arm live trading unless every prerequisite is met.

        ``TRADING_MODE=live`` is deliberately *not* sufficient (DESIGN.md Phase 5). An
        environment variable is one typo away from committing real capital, so the guard
        additionally requires demonstrated backtest and paper evidence, a verified broker
        field mapping, and a human-written authorisation file.
        """
        if config.TRADING_MODE != "live":
            return True

        from src.live_guard import LiveTradingBlocked, evaluate
        from src.session_recorder import SessionStore

        store = SessionStore(config.SESSION_STORE_ROOT)
        report = evaluate(
            paper_sessions=len(store.days("paper")),
            instrument_master_loaded=self.instruments is not None,
        )
        if report.cleared:
            log.warning("Live trading cleared by the guard.")
            return True

        log.error("LIVE TRADING BLOCKED — refusing to arm.\n%s", report.render())
        log.error("The execution engine will not send orders. Fix the blockers above, "
                  "or run in paper mode: TRADING_MODE=paper")
        return False

    def run(self) -> None:
        if not event_bus.ping(self.client):
            log.error("Redis not reachable — start it with `docker compose up -d`.")
            return
        if not self._check_live_clearance():
            return

        mode = {"live": "LIVE (real money!)",
                "paper": "PAPER (simulated fills)",
                "dry_run": "DRY_RUN (nothing fills)"}.get(
                    config.TRADING_MODE, config.TRADING_MODE)
        log.info("Execution Engine armed — mode: %s", mode)
        if self.instruments is None and not config.DRY_RUN:
            log.error("No instrument master loaded. Live orders cannot resolve trading "
                      "symbols and will be refused (B4).")

        while True:
            # Reclaim work a previous instance read but never acked, so a crash
            # mid-send does not strand an order.
            for entry_id, fields in self.consumer.claim_stale(
                    min_idle_ms=config.CLAIM_IDLE_MS):
                self.consumer.handle(entry_id, fields, self._handle)

            for entry_id, fields in self.consumer.read(count=50, block_ms=1000):
                self.consumer.handle(entry_id, fields, self._handle)

            self.poll_open_orders()
            self.registry.purge_terminal()


def main() -> None:
    master = None
    try:
        from pathlib import Path

        import datetime as dt
        cached = InstrumentMaster.cache_path(
            config.SCRIP_MASTER_DIR, dt.date.today(), config.DEFAULT_EXCHANGE_SEGMENT)
        master = InstrumentMaster.load_cache(cached)
        if master:
            log.info("Loaded %d instruments from %s", len(master), cached)
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not load the instrument master: %s", exc)

    ExecutionEngine(master).run()


if __name__ == "__main__":
    main()
