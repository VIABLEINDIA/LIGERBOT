"""Simulated execution — the fill model (DESIGN.md 2.3).

Most retail backtests are optimistic here, and the optimism is what turns an unprofitable
strategy into a profitable-looking one. Four invariants keep this honest:

1. **Next-bar-open execution.** A signal generated on bar ``t`` (from its close) executes
   at bar ``t+1``'s open, never at bar ``t``'s close. This is the anti-look-ahead rule and
   it is asserted in tests, not merely intended.

2. **Pessimistic intrabar resolution.** When a bar's range contains both the stop and the
   target, we assume the **stop** hit first. A 1-minute bar records no path, only its
   extremes, so either assumption is a guess — and the optimistic guess is the
   second-most-common way a backtest lies.

3. **Slippage always against us.** Buys fill above the modelled price, sells below, with a
   floor of half the spread because you always cross it.

4. **Stops are not guaranteed prices.** A gap through the stop fills at the open, not at
   the stop level. Modelling stops as always-exact is how backtests understate tail risk.
"""
from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from src import market_calendar as cal
from src.backtest.costs import CostBreakdown, CostModel, Leg, SlippageModel
from src.bars import Bar
from src.risk_engine import Intent, OrderRequest, Side

log = logging.getLogger("ligerbot.sim_broker")


class FillReason(Enum):
    SIGNAL = "signal"            # ordinary entry/exit from a strategy signal
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    SQUARE_OFF = "square_off"    # forced flat at the session deadline
    GAP_THROUGH_STOP = "gap"     # opened beyond the stop; filled at the open


@dataclass(frozen=True)
class Fill:
    instrument_id: str
    side: Side
    quantity: int
    price: float
    at: dt.datetime
    intent: Intent
    reason: FillReason
    costs: CostBreakdown = field(default_factory=CostBreakdown)
    slippage_per_share: float = 0.0

    @property
    def turnover(self) -> float:
        return self.quantity * self.price


@dataclass
class OpenPosition:
    """A position as the simulated broker sees it, with excursion tracking."""

    instrument_id: str
    quantity: int
    entry_price: float
    entry_at: dt.datetime
    stop_loss: float
    take_profit: Optional[float] = None
    entry_costs: CostBreakdown = field(default_factory=CostBreakdown)
    strategy_name: str = ""
    reason: str = ""
    bars_held: int = 0
    # Maximum adverse / favourable excursion, in rupees per share.
    mae: float = 0.0
    mfe: float = 0.0

    @property
    def is_long(self) -> bool:
        return self.quantity > 0

    @property
    def risk_per_share(self) -> float:
        return abs(self.entry_price - self.stop_loss)

    def update_excursions(self, bar: Bar) -> None:
        if self.is_long:
            self.mae = max(self.mae, self.entry_price - bar.low)
            self.mfe = max(self.mfe, bar.high - self.entry_price)
        else:
            self.mae = max(self.mae, bar.high - self.entry_price)
            self.mfe = max(self.mfe, self.entry_price - bar.low)


@dataclass
class PendingOrder:
    order: OrderRequest
    submitted_at: dt.datetime


class SimBroker:
    """Executes orders against a bar stream."""

    def __init__(
        self,
        cost_model: Optional[CostModel] = None,
        slippage: Optional[SlippageModel] = None,
        *,
        max_volume_participation: float = 0.10,
        enforce_liquidity: bool = True,
    ) -> None:
        self.costs = cost_model or CostModel()
        self.slippage = slippage or SlippageModel()
        self.max_volume_participation = max_volume_participation
        self.enforce_liquidity = enforce_liquidity

        self.pending: Dict[str, PendingOrder] = {}
        self.positions: Dict[str, OpenPosition] = {}
        self.rejected: List[tuple[OrderRequest, str]] = []
        # Closing a position removes it from `positions`, but the caller still needs it
        # to build the trade record (entry price, excursions, bars held). Stashing it
        # here keeps `Fill` an immutable value object instead of a carrier for state.
        self.closed_positions: Dict[str, OpenPosition] = {}

    # -- submission --------------------------------------------------------
    def submit(self, order: OrderRequest, at: dt.datetime) -> None:
        """Queue an order. It fills at the **next** bar's open, never this one."""
        if order.instrument_id in self.pending:
            self.rejected.append((order, "an order is already pending for this instrument"))
            return
        self.pending[order.instrument_id] = PendingOrder(order=order, submitted_at=at)

    # -- per-bar processing ------------------------------------------------
    def process_bar(self, bar: Bar) -> List[Fill]:
        """Advance one bar. Returns fills, in the order they would have occurred.

        Sequencing within the bar matters and is deliberate:
          1. Pending orders fill at this bar's **open**.
          2. Stops and targets are then checked against this bar's range — so a position
             entered at the open can be stopped out inside the very same bar, exactly as
             it could in reality.
        """
        fills: List[Fill] = []

        pending = self.pending.pop(bar.instrument_id, None)
        if pending is not None:
            fill = self._execute_at_open(pending.order, bar)
            if fill is not None:
                fills.append(fill)

        position = self.positions.get(bar.instrument_id)
        if position is not None:
            position.bars_held += 1
            position.update_excursions(bar)
            exit_fill = self._check_exit_levels(position, bar)
            if exit_fill is not None:
                fills.append(exit_fill)
        return fills

    def _minutes_from_session_edge(self, moment: dt.datetime) -> float:
        """Distance in minutes to the nearer of the open and the close."""
        moment = cal.to_ist(moment)
        window = cal.session_window(moment.date())
        if window is None:
            return 999.0
        open_dt, close_dt = window
        return min(abs((moment - open_dt).total_seconds()),
                   abs((close_dt - moment).total_seconds())) / 60.0

    def _fill_price(self, raw_price: float, *, is_buy: bool, at: dt.datetime) -> float:
        return self.slippage.apply(
            raw_price, is_buy=is_buy,
            minutes_from_edge=self._minutes_from_session_edge(at),
        )

    def _liquidity_capped(self, quantity: int, bar: Bar) -> tuple[int, Optional[str]]:
        """Trim an order that would be an implausible share of the bar's volume.

        A **synthetic** bar means nothing traded in that interval, so nothing can fill in
        it — no price existed to trade at. That is different from a real bar reporting
        zero volume, which on a feed without volume data just means "unknown"; there we
        allow the fill rather than silently suppressing every trade.
        """
        if bar.synthetic:
            return 0, "synthetic bar — no trades occurred in this interval"
        if not self.enforce_liquidity or bar.volume <= 0:
            return quantity, None
        cap = int(bar.volume * self.max_volume_participation)
        if cap <= 0:
            return 0, f"bar volume {bar.volume:.0f} too thin to fill anything"
        if quantity > cap:
            return cap, (f"trimmed {quantity} -> {cap} "
                         f"({self.max_volume_participation:.0%} of bar volume)")
        return quantity, None

    def _execute_at_open(self, order: OrderRequest, bar: Bar) -> Optional[Fill]:
        is_buy = order.side is Side.BUY
        quantity, note = self._liquidity_capped(order.quantity, bar)
        if note:
            log.debug("%s: %s", order.instrument_id, note)
        if quantity <= 0:
            self.rejected.append((order, note or "zero quantity"))
            return None

        raw = bar.open
        price = self._fill_price(raw, is_buy=is_buy, at=bar.bar_start)
        leg = Leg.BUY if is_buy else Leg.SELL
        costs = self.costs.charge_leg(leg, quantity, price)

        if order.intent.is_open:
            self.positions[order.instrument_id] = OpenPosition(
                instrument_id=order.instrument_id,
                quantity=quantity if is_buy else -quantity,
                entry_price=price,
                entry_at=bar.bar_start,
                stop_loss=order.stop_loss or 0.0,
                take_profit=order.take_profit,
                entry_costs=costs,
                strategy_name=order.strategy_name,
                reason=order.reason,
            )
        else:
            closing = self.positions.pop(order.instrument_id, None)
            if closing is not None:
                self.closed_positions[order.instrument_id] = closing
                # A signalled exit closes whatever is actually open, not whatever the
                # order happened to be sized for — the two can differ if the entry was
                # liquidity-trimmed.
                quantity = abs(closing.quantity)
                costs = self.costs.charge_leg(leg, quantity, price)

        return Fill(
            instrument_id=order.instrument_id, side=order.side, quantity=quantity,
            price=price, at=bar.bar_start, intent=order.intent,
            reason=FillReason.SIGNAL, costs=costs,
            slippage_per_share=abs(price - raw),
        )

    def _check_exit_levels(self, position: OpenPosition, bar: Bar) -> Optional[Fill]:
        """Resolve stop/target against the bar's range, pessimistically."""
        stop, target = position.stop_loss, position.take_profit

        if position.is_long:
            gapped = stop > 0 and bar.open <= stop
            stop_hit = stop > 0 and bar.low <= stop
            target_hit = target is not None and bar.high >= target
        else:
            gapped = stop > 0 and bar.open >= stop
            stop_hit = stop > 0 and bar.high >= stop
            target_hit = target is not None and bar.low <= target

        if not stop_hit and not target_hit:
            return None

        if stop_hit:
            # A gap through the stop fills at the open, not at the stop level. Assuming
            # the stop always fills exactly is how backtests understate tail risk.
            if gapped:
                exit_price, reason = bar.open, FillReason.GAP_THROUGH_STOP
            else:
                exit_price, reason = stop, FillReason.STOP_LOSS
        else:
            exit_price, reason = float(target), FillReason.TAKE_PROFIT

        return self._close(position, exit_price, bar.bar_start, reason)

    def _close(
        self, position: OpenPosition, raw_price: float,
        at: dt.datetime, reason: FillReason,
    ) -> Fill:
        is_buy = not position.is_long  # closing a long means selling
        quantity = abs(position.quantity)
        price = self._fill_price(raw_price, is_buy=is_buy, at=at)
        leg = Leg.BUY if is_buy else Leg.SELL
        costs = self.costs.charge_leg(leg, quantity, price)
        self.positions.pop(position.instrument_id, None)
        self.closed_positions[position.instrument_id] = position

        return Fill(
            instrument_id=position.instrument_id,
            side=Side.BUY if is_buy else Side.SELL,
            quantity=quantity, price=price, at=at,
            intent=Intent.CLOSE_LONG if position.is_long else Intent.CLOSE_SHORT,
            reason=reason, costs=costs, slippage_per_share=abs(price - raw_price),
        )

    def force_close(self, instrument_id: str, bar: Bar) -> Optional[Fill]:
        """Flatten at the session deadline, at the bar's close.

        The bot owns this exit (DESIGN.md 1.6). Letting the broker's ~15:20 MIS
        auto-square-off do it instead means an uncontrolled fill at an adversarial moment.
        """
        position = self.positions.get(instrument_id)
        if position is None:
            return None
        self.pending.pop(instrument_id, None)
        return self._close(position, bar.close, bar.bar_end, FillReason.SQUARE_OFF)

    def cancel_pending(self, instrument_id: str) -> None:
        self.pending.pop(instrument_id, None)

    def has_position(self, instrument_id: str) -> bool:
        return instrument_id in self.positions

    @property
    def open_instruments(self) -> List[str]:
        return list(self.positions)
