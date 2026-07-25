"""Paper trading with realistic fills (DESIGN.md 4, Phase 4).

Phase 4's whole purpose is comparing paper results against a backtest over the same
sessions. That comparison is only meaningful if both sides fill the same way — so this
module drives the **backtester's own** :class:`~src.backtest.sim_broker.SimBroker` from the
live bar stream. Same next-bar-open execution, same pessimistic intrabar resolution, same
slippage, same cost model.

**Why this module exists at all.** ``DRY_RUN`` filled instantly at the signal's reference
price: no slippage, no next-bar delay, no liquidity check. Paper trading on that would
have beaten the backtest for entirely artificial reasons, and the reconciliation in
:mod:`src.reconciliation` would have read that as "the backtest is conservative" when the
truth was "paper mode is lying". An optimistic paper mode is worse than no paper mode,
because it manufactures confidence rather than merely failing to provide it.

Three trading modes, replacing the old ``DRY_RUN`` boolean:

``dry_run``
    Nothing is filled at all. Orders are logged and discarded. For smoke-testing wiring.
``paper``
    Realistic simulated fills against live market data. What Phase 4 runs.
``live``
    Real orders to the broker.

    python -m src.paper_broker
"""
from __future__ import annotations

import datetime as dt
import logging
from typing import Any, Dict, Optional

import config
from src import event_bus
from src import market_calendar as cal
from src.backtest.costs import CostModel, SlippageModel
from src.backtest.sim_broker import Fill as SimFill
from src.backtest.sim_broker import FillReason, SimBroker
from src.bars import Bar
from src.risk_engine import Intent, OrderRequest, Side

logging.basicConfig(level=logging.INFO, format="%(asctime)s [paper] %(message)s")
log = logging.getLogger("ligerbot.paper")


class PaperBroker:
    """Redis adapter around the backtester's fill model.

    Deliberately thin: every fill decision belongs to ``SimBroker``, which the backtester
    also uses. If this module reimplemented any of it, paper and backtest would drift and
    the reconciliation would be measuring the drift rather than the strategy.
    """

    def __init__(
        self,
        cost_model: Optional[CostModel] = None,
        slippage: Optional[SlippageModel] = None,
    ) -> None:
        self.client = event_bus.get_client()
        self.broker = SimBroker(
            cost_model or CostModel(),
            slippage or SlippageModel(),
            max_volume_participation=config.MAX_VOLUME_PARTICIPATION,
            enforce_liquidity=True,
        )
        group = f"{config.CONSUMER_GROUP_PREFIX}.paper"
        self.orders = event_bus.StreamConsumer(
            self.client, config.STREAM_APPROVED_ORDERS, group, "paper-1",
            max_deliveries=config.MAX_DELIVERIES)
        self.bars = event_bus.StreamConsumer(
            self.client, config.STREAM_MARKET_BARS, group, "paper-1",
            max_deliveries=config.MAX_DELIVERIES)
        # Client order ids, so published fills carry the same identity live fills would.
        self._order_ids: Dict[str, str] = {}
        self._entry_costs: Dict[str, Any] = {}
        self.session_day: Optional[dt.date] = None

    # -- orders ------------------------------------------------------------
    def _handle_order(self, fields: Dict[str, Any]) -> None:
        instrument_id = fields.get("instrument_id")
        if not instrument_id:
            raise ValueError(f"approved order without an instrument: {fields}")

        request = OrderRequest(
            instrument_id=str(instrument_id),
            side=Side(fields.get("side", "BUY")),
            quantity=int(float(fields.get("quantity", 0))),
            intent=Intent(fields.get("intent", "OPEN_LONG")),
            ref_price=float(fields.get("price") or 0.0),
            stop_loss=float(fields.get("stop_loss") or 0.0) or None,
            take_profit=float(fields.get("take_profit") or 0.0) or None,
            strategy_name=str(fields.get("strategy_name", "")),
        )
        if request.quantity <= 0:
            raise ValueError(f"approved order with quantity {request.quantity}")

        self._order_ids[request.instrument_id] = str(
            fields.get("client_order_id") or "")
        # Queued, not filled. The next bar for this instrument fills it at its open —
        # the same next-bar rule the backtester enforces.
        self.broker.submit(request, cal.now_ist())
        log.info("queued %s %d %s (fills at the next bar's open)",
                 request.side.value, request.quantity, request.instrument_id)

    # -- bars --------------------------------------------------------------
    def _handle_bar(self, fields: Dict[str, Any]) -> None:
        bar = Bar.from_event(fields)
        day = bar.bar_start.date()
        if day != self.session_day:
            self.session_day = day

        for fill in self.broker.process_bar(bar):
            self._publish_fill(fill, bar)

        # Own the exit at the session deadline rather than leaving a position open
        # overnight in the paper book (DESIGN.md 1.6).
        if cal.phase(bar.bar_end).requires_flat and self.broker.has_position(
                bar.instrument_id):
            forced = self.broker.force_close(bar.instrument_id, bar)
            if forced is not None:
                self._publish_fill(forced, bar)

    def _publish_fill(self, fill: SimFill, bar: Bar) -> None:
        """Emit a fill in the same shape a live fill would take."""
        client_order_id = self._order_ids.get(fill.instrument_id, "")
        payload = {
            "client_order_id": client_order_id,
            "broker_order_id": f"PAPER-{int(fill.at.timestamp() * 1000)}",
            "instrument_id": fill.instrument_id,
            "side": fill.side.value,
            "intent": fill.intent.value,
            "status": "FILLED",
            "quantity": fill.quantity,
            "filled_quantity": fill.quantity,
            "average_fill_price": round(fill.price, 4),
            "costs": round(fill.costs.total, 4),
            "slippage_per_share": round(fill.slippage_per_share, 4),
            "fill_reason": fill.reason.value,
            "mode": "paper",
            "timestamp": fill.at.timestamp(),
        }
        position = self.broker.positions.get(fill.instrument_id)
        payload["stop_loss"] = round(position.stop_loss, 4) if position else 0.0
        event_bus.publish(self.client, config.STREAM_FILLED_ORDERS, payload)

        log.info("FILL %s %s %d @ %.2f (%s) slip %.3f cost %.2f",
                 fill.side.value, fill.instrument_id, fill.quantity, fill.price,
                 fill.reason.value, fill.slippage_per_share, fill.costs.total)

    # -- loop --------------------------------------------------------------
    def run(self) -> None:
        if not event_bus.ping(self.client):
            log.error("Redis not reachable — start it with `docker compose up -d`.")
            return
        log.info("Paper broker online. Fills use the backtester's model: next-bar open, "
                 "pessimistic intrabar, %s", self.broker.slippage.describe())

        while True:
            for entry_id, fields in self.orders.read(count=50, block_ms=10):
                self.orders.handle(entry_id, fields, self._handle_order)
            for entry_id, fields in self.bars.claim_stale(
                    min_idle_ms=config.CLAIM_IDLE_MS):
                self.bars.handle(entry_id, fields, self._handle_bar)
            for entry_id, fields in self.bars.read(count=200, block_ms=1000):
                self.bars.handle(entry_id, fields, self._handle_bar)


def main() -> None:
    if not config.simulates_fills():
        # Refuse rather than warn. In dry_run and live the execution engine is the filler,
        # and both consume approved_orders from separate consumer groups — so both would
        # receive every order and both would fill it, double-counting every trade while
        # looking entirely healthy.
        log.error("TRADING_MODE=%r — the paper broker must not run. The execution engine "
                  "fills orders in this mode; running both would fill every order twice. "
                  "Set TRADING_MODE=paper, or start src.execution_engine instead.",
                  config.TRADING_MODE)
        return
    PaperBroker().run()


if __name__ == "__main__":
    main()
