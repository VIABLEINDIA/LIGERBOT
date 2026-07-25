"""Module 7 — Position Manager (DESIGN.md 3.2).

The single source of truth for positions, equity and realised P&L, and the fix for the
pair of defects that made the safety features theatre:

* **B2** — ``realized_pnl_today`` was initialised to zero and never written to by
  anything, so the max-daily-drawdown halt could not fire.
* **B3** — the execution engine published on broker *acceptance*, never on actual fills,
  and nothing consumed the stream. The loop was open: the bot never learned what it owned.

This module closes it. Fills flow in, positions and P&L are maintained, and
``position_updates`` is published for the risk manager to consume — which is what finally
makes the drawdown breaker load-bearing rather than decorative.

**The broker is authoritative.** Local state is a cache, reconciled on startup and on a
timer. When the two disagree the broker wins, and a disagreement beyond a threshold halts
trading rather than being papered over: a bot that does not know what it owns cannot size
its next trade, and guessing is how a small bookkeeping bug becomes a large position.

    python -m src.position_manager
"""
from __future__ import annotations

import datetime as dt
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import config
from src import event_bus
from src import kotak_api
from src import kotak_api
from src import market_calendar as cal
from src.kill_switch import KillSwitch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [positions] %(message)s")
log = logging.getLogger("ligerbot.positions")


@dataclass
class TrackedPosition:
    """A position as we believe it to be."""

    instrument_id: str
    quantity: int                    # signed: + long, - short
    average_price: float
    stop_loss: float = 0.0
    opened_at: Optional[str] = None
    realized_pnl: float = 0.0
    costs: float = 0.0

    @property
    def is_long(self) -> bool:
        return self.quantity > 0

    @property
    def risk_amount(self) -> float:
        if self.stop_loss <= 0:
            return 0.0
        return abs(self.quantity) * abs(self.average_price - self.stop_loss)

    @property
    def notional(self) -> float:
        return abs(self.quantity) * self.average_price

    def unrealized(self, mark_price: float) -> float:
        return (mark_price - self.average_price) * self.quantity


@dataclass
class ReconciliationResult:
    """Outcome of comparing local belief against the broker."""

    checked_at: str
    matched: List[str] = field(default_factory=list)
    mismatched: List[str] = field(default_factory=list)
    missing_locally: List[str] = field(default_factory=list)
    missing_at_broker: List[str] = field(default_factory=list)
    details: List[str] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not (self.mismatched or self.missing_locally or self.missing_at_broker)

    @property
    def discrepancy_count(self) -> int:
        return len(self.mismatched) + len(self.missing_locally) + len(self.missing_at_broker)

    def summary(self) -> str:
        if self.clean:
            return f"reconciled clean ({len(self.matched)} position(s))"
        return (f"{self.discrepancy_count} discrepancy(ies): "
                f"{len(self.mismatched)} mismatched, "
                f"{len(self.missing_locally)} only at broker, "
                f"{len(self.missing_at_broker)} only local")


class PositionBook:
    """Pure position and P&L bookkeeping — no Redis, no broker, no clock.

    Extracted for the same reason as ``RiskEngine``: the backtester and the live system
    must apply identical logic, and logic that only runs in production is logic that was
    never tested.
    """

    def __init__(self) -> None:
        self.positions: Dict[str, TrackedPosition] = {}
        self.realized_pnl_today: float = 0.0
        self.costs_today: float = 0.0
        self.trade_count: int = 0
        self.session_day: Optional[dt.date] = None

    def start_session(self, day: dt.date) -> None:
        self.session_day = day
        self.realized_pnl_today = 0.0
        self.costs_today = 0.0
        self.trade_count = 0

    # -- fills -------------------------------------------------------------
    def apply_fill(
        self,
        instrument_id: str,
        signed_quantity: int,
        price: float,
        *,
        stop_loss: float = 0.0,
        costs: float = 0.0,
        at: Optional[str] = None,
    ) -> float:
        """Apply one fill. Returns realised P&L from this fill (0 when opening).

        Handles the four cases explicitly rather than assuming every fill opens or
        closes: opening, adding, reducing, and reversing through zero. Reversal is the
        one that bites — a fill larger than the open position both closes it and opens
        the other side, and treating that as a simple close loses the remainder.
        """
        self.costs_today += costs
        existing = self.positions.get(instrument_id)
        realized = 0.0

        if existing is None or existing.quantity == 0:
            self.positions[instrument_id] = TrackedPosition(
                instrument_id=instrument_id, quantity=signed_quantity,
                average_price=price, stop_loss=stop_loss,
                opened_at=at or dt.datetime.now().isoformat(timespec="seconds"),
                costs=costs,
            )
            return 0.0

        same_direction = (existing.quantity > 0) == (signed_quantity > 0)
        if same_direction:
            # Adding: weighted-average the entry price.
            total = existing.quantity + signed_quantity
            existing.average_price = (
                (existing.average_price * abs(existing.quantity) + price * abs(signed_quantity))
                / abs(total)
            )
            existing.quantity = total
            existing.costs += costs
            if stop_loss:
                existing.stop_loss = stop_loss
            return 0.0

        # Opposite direction: close some or all of the position.
        closing = min(abs(signed_quantity), abs(existing.quantity))
        direction = 1 if existing.quantity > 0 else -1
        realized = (price - existing.average_price) * closing * direction
        self.realized_pnl_today += realized
        existing.realized_pnl += realized
        existing.costs += costs
        self.trade_count += 1

        remaining = existing.quantity + signed_quantity
        if remaining == 0:
            self.positions.pop(instrument_id, None)
        elif (remaining > 0) == (existing.quantity > 0):
            existing.quantity = remaining
        else:
            # Reversed through zero: the excess opens a new position the other way.
            self.positions[instrument_id] = TrackedPosition(
                instrument_id=instrument_id, quantity=remaining,
                average_price=price, stop_loss=stop_loss,
                opened_at=at or dt.datetime.now().isoformat(timespec="seconds"),
            )
        return realized

    # -- views -------------------------------------------------------------
    def total_open_risk(self) -> float:
        return sum(p.risk_amount for p in self.positions.values())

    def gross_exposure(self) -> float:
        return sum(p.notional for p in self.positions.values())

    def net_pnl_today(self) -> float:
        """Realised P&L after costs — what the drawdown breaker must see."""
        return self.realized_pnl_today - self.costs_today

    def snapshot(self) -> Dict[str, Any]:
        return {
            "session_day": self.session_day.isoformat() if self.session_day else None,
            "open_positions": len(self.positions),
            "realized_pnl_today": round(self.realized_pnl_today, 2),
            "costs_today": round(self.costs_today, 2),
            "net_pnl_today": round(self.net_pnl_today(), 2),
            "total_open_risk": round(self.total_open_risk(), 2),
            "gross_exposure": round(self.gross_exposure(), 2),
            "trade_count": self.trade_count,
            "positions": [asdict(p) for p in self.positions.values()],
        }

    # -- reconciliation ----------------------------------------------------
    def reconcile(self, broker_positions: Dict[str, int]) -> ReconciliationResult:
        """Compare local belief against the broker's quantities.

        The broker wins on every disagreement. Local state is corrected, and the
        discrepancy is reported so a threshold check can decide whether to halt.
        """
        result = ReconciliationResult(
            checked_at=dt.datetime.now().isoformat(timespec="seconds"))

        for instrument_id, position in list(self.positions.items()):
            broker_quantity = broker_positions.get(instrument_id)
            if broker_quantity is None:
                result.missing_at_broker.append(instrument_id)
                result.details.append(
                    f"{instrument_id}: local {position.quantity}, broker has none — "
                    f"dropping the local position")
                self.positions.pop(instrument_id, None)
            elif broker_quantity != position.quantity:
                result.mismatched.append(instrument_id)
                result.details.append(
                    f"{instrument_id}: local {position.quantity} vs broker "
                    f"{broker_quantity} — taking the broker's")
                if broker_quantity == 0:
                    self.positions.pop(instrument_id, None)
                else:
                    position.quantity = broker_quantity
            else:
                result.matched.append(instrument_id)

        for instrument_id, quantity in broker_positions.items():
            if quantity != 0 and instrument_id not in self.positions:
                result.missing_locally.append(instrument_id)
                result.details.append(
                    f"{instrument_id}: broker holds {quantity}, we had no record — "
                    f"adopting it with an unknown entry price")
                self.positions[instrument_id] = TrackedPosition(
                    instrument_id=instrument_id, quantity=quantity,
                    average_price=0.0, stop_loss=0.0,
                )
        return result


def parse_broker_positions(payload: Any) -> Dict[str, int]:
    """Normalise a Kotak ``positions()`` response into ``{instrument_id: net_qty}``.

    .. warning::
       The field names are best-effort, exactly like ``account.py`` (DESIGN.md 5.3).
       The probe must pin them down against a real response. Unrecognised rows are
       skipped **and counted**, never silently treated as flat — a position we fail to
       parse is a position we would otherwise believe we do not have.
    """
    rows: List[dict] = []
    if isinstance(payload, dict):
        for key in ("data", "Data", "result", "positions"):
            value = payload.get(key)
            if isinstance(value, list):
                rows = [r for r in value if isinstance(r, dict)]
                break
    elif isinstance(payload, list):
        rows = [r for r in payload if isinstance(r, dict)]

    out: Dict[str, int] = {}
    skipped = 0
    for row in rows:
        token = (row.get("tok") or row.get("token") or row.get("instrument_token")
                 or row.get("pSymbol"))
        segment = (row.get("exSeg") or row.get("exchange_segment")
                   or config.DEFAULT_EXCHANGE_SEGMENT)
        if not token:
            skipped += 1
            continue
        try:
            buy = float(row.get("flBuyQty", row.get("buy_quantity", 0)) or 0)
            sell = float(row.get("flSellQty", row.get("sell_quantity", 0)) or 0)
            net = int(buy - sell)
        except (TypeError, ValueError):
            skipped += 1
            continue
        out[f"{segment}:{token}"] = net

    if skipped:
        log.error("Could not parse %d position row(s) from the broker response. "
                  "Verify the field mapping — an unparsed position reads as flat.",
                  skipped)
    return out


class PositionManager:
    """Consumes fills, maintains the book, reconciles, and publishes updates."""

    def __init__(self, neo_client: Any = None) -> None:
        self.client = event_bus.get_client()
        self.book = PositionBook()
        self.neo = neo_client
        self.kill_switch = KillSwitch(self.client)
        self.consumer = event_bus.StreamConsumer(
            self.client, config.STREAM_FILLED_ORDERS,
            f"{config.CONSUMER_GROUP_PREFIX}.positions", "pm-1",
            max_deliveries=config.MAX_DELIVERIES,
        )
        self._last_reconcile = 0.0

    # -- fills -------------------------------------------------------------
    def _handle_fill(self, fill: Dict[str, Any]) -> None:
        instrument_id = fill.get("instrument_id") or fill.get("instrument")
        if not instrument_id:
            raise ValueError(f"fill without an instrument: {fill}")
        # Only genuine fills move the book. Acceptance-only events are exactly what B3
        # mistook for fills.
        if str(fill.get("status", "")).upper() not in ("FILLED", "PARTIAL"):
            return

        quantity = int(float(fill.get("filled_quantity") or fill.get("quantity") or 0))
        if quantity <= 0:
            return
        price = float(fill.get("average_fill_price") or fill.get("price") or 0.0)
        if price <= 0:
            raise ValueError(f"fill for {instrument_id} has no usable price: {fill}")

        side = str(fill.get("side", "BUY")).upper()
        signed = quantity if side == "BUY" else -quantity

        realized = self.book.apply_fill(
            str(instrument_id), signed, price,
            stop_loss=float(fill.get("stop_loss") or 0.0),
            costs=float(fill.get("costs") or 0.0),
        )
        log.info("fill %s %s %d @ %.2f | realised %+.2f | day %+.2f | open %d",
                 side, instrument_id, quantity, price, realized,
                 self.book.net_pnl_today(), len(self.book.positions))
        self._publish_update()

    def _publish_update(self) -> None:
        """Broadcast the book. This is the event the risk manager needs to see."""
        event_bus.publish(self.client, config.STREAM_POSITION_UPDATES,
                          self.book.snapshot())

    # -- reconciliation ----------------------------------------------------
    def reconcile_now(self) -> Optional[ReconciliationResult]:
        if self.neo is None:
            return None

        # The SDK returns None (not an exception) when positions() fails, and it returns
        # error payloads as data. Left unguarded, a transient failure looked exactly like
        # "the broker holds nothing" — and reconciliation would then discard every open
        # position from the book. `safe_call` raises instead, so a failed call aborts
        # reconciliation rather than rewriting our record of what is open.
        try:
            payload = kotak_api.safe_call(self.neo, "positions", allow_empty=True)
        except kotak_api.KotakSessionExpired as exc:
            log.error("%s — halting rather than trading on an unverifiable book.", exc)
            self.kill_switch.halt(f"broker session expired: {exc}",
                                  source="position_manager")
            return None
        except kotak_api.KotakAPIError as exc:
            log.error("positions() failed (%s) — skipping reconciliation. The book is "
                      "unchanged; it is NOT assumed flat.", exc)
            return None

        broker = parse_broker_positions(payload)
        result = self.book.reconcile(broker)

        if result.clean:
            log.debug("Reconciliation %s", result.summary())
        else:
            log.error("RECONCILIATION MISMATCH — %s", result.summary())
            for detail in result.details:
                log.error("  %s", detail)
            if result.discrepancy_count >= config.RECONCILE_HALT_THRESHOLD:
                self.kill_switch.halt(
                    f"position reconciliation mismatch: {result.summary()}",
                    source="position_manager",
                )
        self._publish_update()
        return result

    # -- loop --------------------------------------------------------------
    def run(self) -> None:
        if not event_bus.ping(self.client):
            log.error("Redis not reachable — start it with `docker compose up -d`.")
            return
        log.info("Position Manager online. Reconciling every %.0fs against the broker.",
                 config.RECONCILE_INTERVAL_SECONDS)

        today = cal.now_ist().date()
        self.book.start_session(today)
        self.reconcile_now()

        while True:
            now = cal.now_ist()
            if now.date() != self.book.session_day:
                log.info("New session %s — resetting the daily book.", now.date())
                self.book.start_session(now.date())
                self.reconcile_now()

            # Reclaim anything a previous instance read but never acked, so a crash
            # mid-fill does not lose the position update.
            for entry_id, fields in self.consumer.claim_stale(
                    min_idle_ms=config.CLAIM_IDLE_MS):
                self.consumer.handle(entry_id, fields, self._handle_fill)

            for entry_id, fields in self.consumer.read(count=200, block_ms=1000):
                self.consumer.handle(entry_id, fields, self._handle_fill)

            if time.monotonic() - self._last_reconcile >= config.RECONCILE_INTERVAL_SECONDS:
                self.reconcile_now()
                self._last_reconcile = time.monotonic()


def main() -> None:
    # The SDK sets no per-request timeout; bound it before any network call.
    kotak_api.bound_network_calls()
    neo = None
    if config.needs_broker_session():
        # Reconciliation is the point of this module, and paper mode needs it as much as
        # live does — a paper session whose book was never checked against the broker
        # cannot be reconciled against a backtest afterwards.
        from src.auth import authenticate_neo
        neo = authenticate_neo()
    else:
        log.warning("Mode is %r — no broker connection, so reconciliation is DISABLED. "
                    "The book tracks fills but cannot be verified against reality. Use "
                    "TRADING_MODE=paper for anything you intend to reconcile.",
                    config.TRADING_MODE)
    PositionManager(neo).run()


if __name__ == "__main__":
    main()
