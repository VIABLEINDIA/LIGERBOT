"""Pure risk logic — no Redis, no broker, no clock reads.

Extracted from ``risk_manager.py`` so the **same code** gates trades in the backtester and
in production (DESIGN.md 2.1). A risk rule that exists only in the live path is a rule that
was never tested; a rule reimplemented for backtesting is a rule that will drift.

Implements the D2 parameter set, which is internally consistent by construction:

===========================  =======  =====================================================
Parameter                    Value    Why
===========================  =======  =====================================================
``max_daily_drawdown``        2.0%    Chosen loss limit.
``max_open_risk``             1.5%    Below the daily limit, so that every open position
                                      stopping out at once cannot breach the day.
``risk_per_trade``            0.5%    max_open_risk / max_open_positions.
``max_open_positions``          3     Secondary sanity cap only.
===========================  =======  =====================================================

The primary control is **total open risk**, not position count. Counting positions does not
bound anything: three positions with 3% stops carry ten times the risk of three with 0.3%
stops. :meth:`RiskEngine.total_open_risk` is the invariant the tests pin.

Sizing is deliberately *asymmetric* — every rounding and every cap resolves toward less
risk, never more.
"""
from __future__ import annotations

import datetime as dt
import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

log = logging.getLogger("ligerbot.risk_engine")


class Intent(Enum):
    """What a signal is asking for.

    Replaces bare BUY/SELL, which conflated "close my long" with "open a short" — the
    ambiguity behind B8 (DESIGN.md 0.1).
    """

    OPEN_LONG = "OPEN_LONG"
    CLOSE_LONG = "CLOSE_LONG"
    OPEN_SHORT = "OPEN_SHORT"
    CLOSE_SHORT = "CLOSE_SHORT"

    @property
    def is_open(self) -> bool:
        return self in (Intent.OPEN_LONG, Intent.OPEN_SHORT)

    @property
    def is_long_side(self) -> bool:
        return self in (Intent.OPEN_LONG, Intent.CLOSE_LONG)


class Side(Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass(frozen=True)
class Signal:
    """A strategy's request. Carries levels, never sizes — sizing belongs here."""

    instrument_id: str
    intent: Intent
    ref_price: float
    bar_time: dt.datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    strategy_name: str = "unknown"
    strategy_version: str = "0"
    params_hash: str = ""
    reason: str = ""


@dataclass
class Position:
    """An open position. ``quantity`` is signed: positive long, negative short."""

    instrument_id: str
    quantity: int
    entry_price: float
    stop_loss: float
    opened_at: Optional[dt.datetime] = None
    # Instruments that tend to move together (a sector, or an index and its ETF).
    # Empty means ungrouped and unconstrained.
    correlation_group: str = ""

    @property
    def is_long(self) -> bool:
        return self.quantity > 0

    @property
    def risk_amount(self) -> float:
        """Rupees at stake if the stop is hit — the quantity the open-risk cap sums."""
        return abs(self.quantity) * abs(self.entry_price - self.stop_loss)

    @property
    def notional(self) -> float:
        return abs(self.quantity) * self.entry_price


@dataclass(frozen=True)
class OrderRequest:
    """An approved, sized order, ready for the execution engine."""

    instrument_id: str
    side: Side
    quantity: int
    intent: Intent
    ref_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    order_type: str = "MARKET"
    risk_amount: float = 0.0
    strategy_name: str = "unknown"
    reason: str = ""
    correlation_group: str = ""


@dataclass(frozen=True)
class RiskDecision:
    """Outcome of evaluating one signal. Rejections always carry a reason."""

    approved: bool
    reason: str
    order: Optional[OrderRequest] = None

    @classmethod
    def reject(cls, reason: str) -> "RiskDecision":
        return cls(approved=False, reason=reason)

    @classmethod
    def approve(cls, order: OrderRequest, reason: str = "ok") -> "RiskDecision":
        return cls(approved=True, reason=reason, order=order)


@dataclass
class RiskLimits:
    """The D2 parameter set. Validated on construction so an inconsistent set can't run."""

    max_daily_drawdown: float = 0.02
    max_open_risk: float = 0.015
    risk_per_trade: float = 0.005
    max_open_positions: int = 3

    # Leverage guards, not risk guards. A very tight stop implies a huge notional for a
    # fixed rupee risk (a 0.2% stop at 0.5% risk demands 250% of equity), so these bound
    # exposure independently of the risk maths. When one binds, the trade is sized *down*,
    # taking less than the full risk budget — the safe direction.
    max_exposure_per_trade: float = 0.75
    max_gross_exposure: float = 2.0

    # A stop this close to entry is noise or a bug, not a trade. Without a floor, the
    # sizing division explodes toward infinite quantity.
    min_stop_distance_pct: float = 0.001

    # Concentration limit within a correlated group. The open-risk cap already bounds
    # the *total*, but it treats three bank stocks as three independent bets when they
    # are closer to one bet of triple the size — DESIGN.md D2 states that assumption and
    # this is what enforces it. An empty group is unconstrained; 0 disables the filter.
    max_positions_per_group: int = 1

    # Absolute backstops for live trading (DESIGN.md Phase 5). Every other limit here is
    # a *fraction of equity* — which is exactly wrong when the equity figure itself is
    # wrong. An unverified broker field mapping mis-sizes every trade by the same factor,
    # and every percentage cap scales with that error rather than catching it. These
    # bound the damage in rupees regardless. 0 disables.
    max_daily_loss_absolute: float = 0.0
    max_orders_per_session: int = 0

    allow_short: bool = False  # D3: long-only for v1

    def __post_init__(self) -> None:
        if self.max_open_risk >= self.max_daily_drawdown:
            raise ValueError(
                f"max_open_risk ({self.max_open_risk:.3%}) must be strictly below "
                f"max_daily_drawdown ({self.max_daily_drawdown:.3%}); otherwise every "
                f"open position stopping out at once breaches the day's limit before "
                f"the circuit breaker can act."
            )
        implied = self.risk_per_trade * self.max_open_positions
        if implied > self.max_open_risk + 1e-12:
            raise ValueError(
                f"risk_per_trade x max_open_positions ({implied:.3%}) exceeds "
                f"max_open_risk ({self.max_open_risk:.3%}) — the caps contradict."
            )
        if self.min_stop_distance_pct <= 0:
            raise ValueError("min_stop_distance_pct must be positive.")


class RiskEngine:
    """Stateful but pure: same inputs, same decisions, every time.

    The caller owns all I/O — it supplies equity at session start, reports fills, and
    passes in whether the session phase permits entries. Nothing here reads a clock or a
    socket, which is what lets the backtester drive it at a million bars a second.
    """

    def __init__(self, limits: Optional[RiskLimits] = None) -> None:
        self.limits = limits or RiskLimits()
        self.session_equity: float = 0.0
        self.realized_pnl_today: float = 0.0
        self.positions: Dict[str, Position] = {}
        self.halted: bool = False
        self.halt_reason: str = ""
        self.session_day: Optional[dt.date] = None
        self.orders_this_session: int = 0

    # -- session lifecycle -------------------------------------------------
    def start_session(self, day: dt.date, equity: float) -> None:
        """Snapshot equity for the day (D1).

        Sizing uses this fixed figure for the whole session. Resizing off live equity
        looks more responsive but systematically puts on the largest position right
        after the best run-up — i.e. immediately before any mean reversion.
        """
        if equity <= 0:
            raise ValueError(
                f"Session equity must be positive, got {equity}. The caller must fail "
                f"closed on a bad broker response rather than passing a placeholder."
            )
        self.session_day = day
        self.session_equity = equity
        self.realized_pnl_today = 0.0
        self.orders_this_session = 0
        self.halted = False
        self.halt_reason = ""
        log.info("Session %s opened with equity %.2f (risk/trade %.2f, open-risk cap %.2f)",
                 day, equity, self.risk_budget_per_trade, self.open_risk_cap)

    # -- derived budgets ---------------------------------------------------
    @property
    def risk_budget_per_trade(self) -> float:
        return self.session_equity * self.limits.risk_per_trade

    @property
    def open_risk_cap(self) -> float:
        return self.session_equity * self.limits.max_open_risk

    @property
    def daily_loss_cap(self) -> float:
        return self.session_equity * self.limits.max_daily_drawdown

    def total_open_risk(self) -> float:
        """Rupees at stake across all open positions. **The invariant.**"""
        return sum(p.risk_amount for p in self.positions.values())

    def total_open_risk_pct(self) -> float:
        if self.session_equity <= 0:
            return 0.0
        return self.total_open_risk() / self.session_equity

    def gross_exposure(self) -> float:
        return sum(p.notional for p in self.positions.values())

    def positions_in_group(self, group: str) -> int:
        """How many open positions share a correlation group."""
        if not group:
            return 0
        return sum(1 for p in self.positions.values() if p.correlation_group == group)

    def projected_loss(self) -> float:
        """Day's P&L if every open position stopped out right now.

        The realized-only breaker is a *trip threshold*, not a cap: it can sit at -1.9%
        realized with 1.5% of open risk still live, and a simultaneous stop-out ends the
        day at -3.4%. Entries are therefore gated on this projected figure, so committed
        risk is counted the moment it is taken on rather than when it lands.
        """
        return self.realized_pnl_today - self.total_open_risk()

    # -- sizing ------------------------------------------------------------
    def _size_position(
        self, entry: float, stop: float, lot_size: int = 1
    ) -> tuple[int, str]:
        """Quantity from the risk budget, then clamped by the leverage guards.

        Returns (quantity, note). Every step rounds down.
        """
        stop_distance = abs(entry - stop)
        quantity = int(math.floor(self.risk_budget_per_trade / stop_distance))
        note = "risk-sized"

        max_by_exposure = int(math.floor(
            (self.session_equity * self.limits.max_exposure_per_trade) / entry
        ))
        if quantity > max_by_exposure:
            quantity = max_by_exposure
            note = "clamped by per-trade exposure cap"

        headroom = self.session_equity * self.limits.max_gross_exposure - self.gross_exposure()
        max_by_gross = int(math.floor(max(0.0, headroom) / entry))
        if quantity > max_by_gross:
            quantity = max_by_gross
            note = "clamped by gross exposure cap"

        if lot_size > 1:
            quantity = (quantity // lot_size) * lot_size
        return max(0, quantity), note

    # -- the gate ----------------------------------------------------------
    def evaluate(
        self,
        signal: Signal,
        *,
        allows_entry: bool,
        allows_exit: bool,
        lot_size: int = 1,
        correlation_group: str = "",
    ) -> RiskDecision:
        """Approve or reject one signal.

        ``allows_entry`` / ``allows_exit`` come from the session phase
        (``market_calendar.Phase``). Passing them in rather than reading a clock keeps
        this replayable. ``correlation_group`` comes from the instrument master; an empty
        group is unconstrained.
        """
        if self.session_equity <= 0:
            return RiskDecision.reject("no session equity — start_session() not called")

        position = self.positions.get(signal.instrument_id)

        # -- exits first: reducing risk is permitted in states where taking it is not.
        if not signal.intent.is_open:
            if not allows_exit:
                return RiskDecision.reject("market closed — cannot exit")
            if position is None:
                return RiskDecision.reject(f"no open position in {signal.instrument_id}")
            if signal.intent.is_long_side != position.is_long:
                return RiskDecision.reject(
                    f"{signal.intent.value} does not match the open "
                    f"{'long' if position.is_long else 'short'} position"
                )
            return RiskDecision.approve(
                OrderRequest(
                    instrument_id=signal.instrument_id,
                    side=Side.SELL if position.is_long else Side.BUY,
                    quantity=abs(position.quantity),
                    intent=signal.intent,
                    ref_price=signal.ref_price,
                    strategy_name=signal.strategy_name,
                    reason=signal.reason,
                ),
                reason="exit approved",
            )

        # -- everything below is opening new risk.
        if self.halted:
            return RiskDecision.reject(f"halted: {self.halt_reason}")

        if self.check_daily_drawdown():
            return RiskDecision.reject(f"halted: {self.halt_reason}")

        if not allows_entry:
            return RiskDecision.reject("session phase does not permit new entries")

        # Absolute backstops. Checked before sizing, because their whole purpose is to
        # hold when the equity figure the percentage limits rely on is itself wrong.
        if (self.limits.max_orders_per_session > 0
                and self.orders_this_session >= self.limits.max_orders_per_session):
            return RiskDecision.reject(
                f"order cap reached ({self.orders_this_session}/"
                f"{self.limits.max_orders_per_session} this session) — bounds the blast "
                f"radius of a signal-generation bug")
        if (self.limits.max_daily_loss_absolute > 0
                and self.realized_pnl_today <= -self.limits.max_daily_loss_absolute):
            return RiskDecision.reject(
                f"absolute daily loss cap reached "
                f"({self.realized_pnl_today:,.2f} vs "
                f"-{self.limits.max_daily_loss_absolute:,.2f})")

        if signal.intent is Intent.OPEN_SHORT and not self.limits.allow_short:
            return RiskDecision.reject("short selling disabled (D3: long-only v1)")

        if position is not None:
            return RiskDecision.reject(
                f"already holding {signal.instrument_id} — pyramiding is not supported"
            )

        if len(self.positions) >= self.limits.max_open_positions:
            return RiskDecision.reject(
                f"max open positions ({self.limits.max_open_positions}) reached"
            )

        # Concentration within a correlated group. Without this, the open-risk cap counts
        # three bank stocks as three independent bets when a sector move takes all three
        # stops together — i.e. one bet of triple the size wearing a diversified label.
        # A limit of 0 disables the filter rather than forbidding every grouped trade —
        # the opposite reading would silently stop all sector trading from a config typo.
        held_in_group = self.positions_in_group(correlation_group)
        if (correlation_group and self.limits.max_positions_per_group > 0
                and held_in_group >= self.limits.max_positions_per_group):
            return RiskDecision.reject(
                f"already holding {held_in_group} position(s) in correlated group "
                f"{correlation_group!r} (limit {self.limits.max_positions_per_group}) — "
                f"they would likely stop out together"
            )

        # Stop-loss is a hard contract (fixes B7). Without it, sizing silently falls back
        # to a notional rule and the documented per-trade risk cap never actually applies.
        if signal.stop_loss is None or signal.stop_loss <= 0:
            return RiskDecision.reject("OPEN signal without a stop_loss is rejected")

        entry, stop = signal.ref_price, signal.stop_loss
        if entry <= 0:
            return RiskDecision.reject(f"invalid reference price {entry}")

        # The stop must be on the losing side of entry, or it isn't a stop.
        if signal.intent is Intent.OPEN_LONG and stop >= entry:
            return RiskDecision.reject(f"long stop {stop} must be below entry {entry}")
        if signal.intent is Intent.OPEN_SHORT and stop <= entry:
            return RiskDecision.reject(f"short stop {stop} must be above entry {entry}")

        if abs(entry - stop) / entry < self.limits.min_stop_distance_pct:
            return RiskDecision.reject(
                f"stop distance {abs(entry - stop) / entry:.4%} is below the "
                f"{self.limits.min_stop_distance_pct:.2%} floor"
            )

        quantity, note = self._size_position(entry, stop, lot_size)
        if quantity <= 0:
            return RiskDecision.reject(
                f"sized to zero ({note}) — equity {self.session_equity:.0f} is too small "
                f"to take {self.limits.risk_per_trade:.2%} risk on a {entry:.2f} instrument"
            )

        # Recompute risk from the *rounded* quantity. Rounding down means actual risk is
        # at or below budget, so the cap is checked against what we will really carry.
        new_risk = quantity * abs(entry - stop)
        projected = self.total_open_risk() + new_risk
        if projected > self.open_risk_cap + 1e-9:
            return RiskDecision.reject(
                f"total open risk would reach {projected / self.session_equity:.3%}, "
                f"over the {self.limits.max_open_risk:.3%} cap"
            )

        # Bound the *worst case*, not just what has already been realised. Without this
        # the day can end far past the stated limit: losses already banked plus risk
        # still open can exceed it together while neither breaches it alone. Blocking
        # the entry is reversible — as positions close, headroom returns — which is why
        # this is a rejection rather than a halt.
        worst_case = self.realized_pnl_today - projected
        if worst_case < -self.daily_loss_cap - 1e-9:
            return RiskDecision.reject(
                f"worst case would reach {worst_case / self.session_equity:.3%} "
                f"(realised {self.realized_pnl_today / self.session_equity:.3%} + open "
                f"risk), past the {self.limits.max_daily_drawdown:.2%} daily limit"
            )

        self.orders_this_session += 1
        return RiskDecision.approve(
            OrderRequest(
                instrument_id=signal.instrument_id,
                side=Side.BUY if signal.intent is Intent.OPEN_LONG else Side.SELL,
                quantity=quantity,
                intent=signal.intent,
                ref_price=entry,
                stop_loss=stop,
                take_profit=signal.take_profit,
                risk_amount=new_risk,
                strategy_name=signal.strategy_name,
                reason=signal.reason,
                correlation_group=correlation_group,
            ),
            reason=note,
        )

    # -- state updates -----------------------------------------------------
    def on_open_fill(
        self,
        instrument_id: str,
        quantity: int,
        fill_price: float,
        stop_loss: float,
        opened_at: Optional[dt.datetime] = None,
        correlation_group: str = "",
    ) -> None:
        """Record an opening fill. ``quantity`` is signed.

        Risk is recomputed from the **actual fill price**, not the reference price. A
        position that slipped on entry carries more risk than intended, and the open-risk
        cap must see the real figure.
        """
        self.positions[instrument_id] = Position(
            instrument_id=instrument_id,
            quantity=quantity,
            entry_price=fill_price,
            stop_loss=stop_loss,
            opened_at=opened_at,
            correlation_group=correlation_group,
        )

    def on_close_fill(self, instrument_id: str, fill_price: float) -> float:
        """Record a closing fill, bank the P&L, return the realised amount."""
        position = self.positions.pop(instrument_id, None)
        if position is None:
            log.warning("Close fill for %s with no tracked position — ignoring.", instrument_id)
            return 0.0
        pnl = (fill_price - position.entry_price) * position.quantity
        self.realized_pnl_today += pnl
        self.check_daily_drawdown()
        return pnl

    def apply_costs(self, amount: float) -> None:
        """Deduct transaction costs from the day's P&L.

        Costs count against the drawdown limit. At ~11-15% of the amount risked per round
        trip (DESIGN.md 5.2) they are far too large to leave out of the breaker's maths.
        """
        self.realized_pnl_today -= abs(amount)
        self.check_daily_drawdown()

    def check_daily_drawdown(self) -> bool:
        """Trip the breaker if the day's loss limit is breached. Returns ``halted``."""
        if self.halted:
            return True
        if self.realized_pnl_today <= -self.daily_loss_cap:
            self.halted = True
            self.halt_reason = (
                f"daily drawdown breached: {self.realized_pnl_today:.2f} <= "
                f"-{self.daily_loss_cap:.2f} ({self.limits.max_daily_drawdown:.2%} of "
                f"{self.session_equity:.2f})"
            )
            log.error("HALT %s", self.halt_reason)
        return self.halted

    def halt(self, reason: str) -> None:
        """Externally triggered halt (kill switch, reconciliation mismatch, stale feed)."""
        self.halted = True
        self.halt_reason = reason
        log.error("HALT %s", reason)

    def snapshot(self) -> dict:
        """Current risk state, for logging and dashboards."""
        return {
            "session_day": self.session_day.isoformat() if self.session_day else None,
            "session_equity": round(self.session_equity, 2),
            "realized_pnl_today": round(self.realized_pnl_today, 2),
            "open_positions": len(self.positions),
            "total_open_risk": round(self.total_open_risk(), 2),
            "total_open_risk_pct": round(self.total_open_risk_pct(), 5),
            "gross_exposure": round(self.gross_exposure(), 2),
            "halted": self.halted,
            "halt_reason": self.halt_reason,
        }
