"""The Indian intraday equity cost stack.

This is the single most important module in the backtester. Most retail intraday
strategies look profitable in a naive backtest and lose money live, and the entire
difference is here (DESIGN.md 2.3). Costs are not a refinement to add later — they are
the gate a strategy has to clear before anything else about it matters.

From DESIGN.md 5.2, at ~0.8% ATR stops and 0.5% risk per trade, the round trip costs
roughly **11-15% of the amount risked**, so gross expectancy must exceed about **0.12R**
just to break even. That works out to a ~45% win rate at 1.5:1 reward-to-risk, or ~37% at
2:1. Those are the pass marks.

Every component is charged **per leg** and reported separately, because the breakdown is
what tells you *why* a strategy failed. A backtest that reports only net P&L hides whether
the edge was absent or merely eaten.

.. warning::
   The default rates below are typical for a discount broker on NSE equity intraday as of
   authoring, and **they change**. STT, exchange transaction charges and stamp duty are all
   set by statute or exchange circular and have each moved in recent years. Verify against
   your contract note and your broker's live rate card before believing any backtest built
   on them (DESIGN.md 5.3 item 1). :meth:`CostModel.describe` prints exactly what was used
   so every report carries its own provenance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict


class Leg(Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass(frozen=True)
class CostBreakdown:
    """Charges for a single leg, itemised."""

    brokerage: float = 0.0
    stt: float = 0.0
    exchange_txn: float = 0.0
    gst: float = 0.0
    sebi: float = 0.0
    stamp_duty: float = 0.0

    @property
    def total(self) -> float:
        return (self.brokerage + self.stt + self.exchange_txn
                + self.gst + self.sebi + self.stamp_duty)

    def __add__(self, other: "CostBreakdown") -> "CostBreakdown":
        return CostBreakdown(
            brokerage=self.brokerage + other.brokerage,
            stt=self.stt + other.stt,
            exchange_txn=self.exchange_txn + other.exchange_txn,
            gst=self.gst + other.gst,
            sebi=self.sebi + other.sebi,
            stamp_duty=self.stamp_duty + other.stamp_duty,
        )

    def to_dict(self) -> Dict[str, float]:
        return {
            "brokerage": round(self.brokerage, 4),
            "stt": round(self.stt, 4),
            "exchange_txn": round(self.exchange_txn, 4),
            "gst": round(self.gst, 4),
            "sebi": round(self.sebi, 4),
            "stamp_duty": round(self.stamp_duty, 4),
            "total": round(self.total, 4),
        }


@dataclass(frozen=True)
class CostModel:
    """NSE cash-segment intraday (MIS) charges.

    All rates are fractions, not percentages: ``0.00025`` is 0.025%.
    """

    # Brokerage: discount brokers charge min(flat, pct of turnover) per executed order.
    brokerage_flat: float = 20.0
    brokerage_pct: float = 0.0003          # 0.03%
    brokerage_is_min_of_both: bool = True

    # Securities Transaction Tax — intraday equity is charged on the SELL leg only.
    # This asymmetry matters: it makes a long round trip cheaper than a naive
    # "both legs" model suggests, and it differs again for delivery and F&O.
    stt_sell: float = 0.00025              # 0.025%

    # Exchange transaction charges, both legs. NSE has revised this more than once.
    exchange_txn: float = 0.0000297        # 0.00297%

    # SEBI turnover fee, both legs (Rs 10 per crore).
    sebi: float = 0.000001

    # Stamp duty, BUY leg only (Rs 300 per crore for intraday equity).
    stamp_duty_buy: float = 0.00003

    # GST applies to brokerage + exchange charges + SEBI fee — not to STT or stamp duty.
    gst: float = 0.18

    def brokerage_for(self, turnover: float) -> float:
        if self.brokerage_is_min_of_both:
            return min(self.brokerage_flat, turnover * self.brokerage_pct)
        return turnover * self.brokerage_pct

    def charge_leg(self, leg: Leg, quantity: int, price: float) -> CostBreakdown:
        """Charges for one executed leg."""
        turnover = abs(quantity) * price
        if turnover <= 0:
            return CostBreakdown()

        brokerage = self.brokerage_for(turnover)
        exchange = turnover * self.exchange_txn
        sebi = turnover * self.sebi
        stt = turnover * self.stt_sell if leg is Leg.SELL else 0.0
        stamp = turnover * self.stamp_duty_buy if leg is Leg.BUY else 0.0
        gst = (brokerage + exchange + sebi) * self.gst

        return CostBreakdown(
            brokerage=brokerage, stt=stt, exchange_txn=exchange,
            gst=gst, sebi=sebi, stamp_duty=stamp,
        )

    def round_trip(
        self, quantity: int, entry_price: float, exit_price: float, *, is_long: bool = True
    ) -> CostBreakdown:
        """Charges for a complete position: entry plus exit.

        A long buys then sells; a short sells then buys. The order matters because STT
        and stamp duty attach to opposite legs.
        """
        if is_long:
            return (self.charge_leg(Leg.BUY, quantity, entry_price)
                    + self.charge_leg(Leg.SELL, quantity, exit_price))
        return (self.charge_leg(Leg.SELL, quantity, entry_price)
                + self.charge_leg(Leg.BUY, quantity, exit_price))

    # -- analysis helpers --------------------------------------------------
    def cost_as_pct_of_notional(self, notional: float) -> float:
        """Round-trip cost as a fraction of position notional, at a flat price."""
        if notional <= 0:
            return 0.0
        price = 100.0
        quantity = int(notional / price)
        if quantity <= 0:
            return 0.0
        return self.round_trip(quantity, price, price).total / notional

    def cost_as_pct_of_risk(
        self, equity: float, risk_per_trade: float, stop_distance_pct: float
    ) -> float:
        """The number from DESIGN.md 5.2: cost as a fraction of the amount risked.

        This, not cost-per-trade in rupees, is what determines whether a strategy is
        viable — it converts directly into the gross expectancy (in R) needed to break
        even.
        """
        risk_amount = equity * risk_per_trade
        if risk_amount <= 0 or stop_distance_pct <= 0:
            return 0.0
        notional = risk_amount / stop_distance_pct
        return self.round_trip(
            max(1, int(notional / 100.0)), 100.0, 100.0
        ).total / risk_amount

    def breakeven_r_multiple(
        self, equity: float, risk_per_trade: float, stop_distance_pct: float
    ) -> float:
        """Gross expectancy (in R) required just to cover costs."""
        return self.cost_as_pct_of_risk(equity, risk_per_trade, stop_distance_pct)

    def describe(self) -> str:
        """Provenance line, embedded in every backtest report."""
        brokerage = (f"min(Rs {self.brokerage_flat:.0f}, {self.brokerage_pct:.4%})"
                     if self.brokerage_is_min_of_both else f"{self.brokerage_pct:.4%}")
        return (
            f"brokerage={brokerage}/order stt_sell={self.stt_sell:.5%} "
            f"exch={self.exchange_txn:.5%} sebi={self.sebi:.6%} "
            f"stamp_buy={self.stamp_duty_buy:.5%} gst={self.gst:.0%}"
        )


@dataclass(frozen=True)
class SlippageModel:
    """Execution slippage, applied on top of the modelled fill price.

    Deliberately pessimistic. A strategy that survives only at optimistic slippage is not
    deployable, which is why DESIGN.md 2.5 rule 6 requires a doubled-slippage sensitivity
    run before any go-live decision.
    """

    slippage_bps: float = 2.5              # 0.025% per leg
    half_spread_bps: float = 1.0           # floor: you always cross the spread
    # The open and the close are materially worse than mid-session. Modelling them the
    # same flatters any strategy that concentrates trades at the edges of the day.
    open_close_multiplier: float = 2.0
    open_close_window_minutes: int = 15

    def slippage_for(self, price: float, *, minutes_from_edge: float = 999.0) -> float:
        """Absolute rupee slippage per share."""
        base_bps = max(self.slippage_bps, self.half_spread_bps)
        if minutes_from_edge < self.open_close_window_minutes:
            base_bps *= self.open_close_multiplier
        return price * base_bps / 10_000.0

    def apply(
        self, price: float, *, is_buy: bool, minutes_from_edge: float = 999.0
    ) -> float:
        """Adjust a fill price against us — buys fill higher, sells lower."""
        amount = self.slippage_for(price, minutes_from_edge=minutes_from_edge)
        return price + amount if is_buy else price - amount

    def scaled(self, factor: float) -> "SlippageModel":
        """A copy with slippage multiplied — used for the sensitivity run."""
        return SlippageModel(
            slippage_bps=self.slippage_bps * factor,
            half_spread_bps=self.half_spread_bps * factor,
            open_close_multiplier=self.open_close_multiplier,
            open_close_window_minutes=self.open_close_window_minutes,
        )

    def describe(self) -> str:
        return (f"slippage={self.slippage_bps:.2f}bps/leg "
                f"(min {self.half_spread_bps:.2f}bps, "
                f"x{self.open_close_multiplier:.1f} within "
                f"{self.open_close_window_minutes}min of the open/close)")
