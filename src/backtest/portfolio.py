"""Trade ledger and equity curve.

Records every completed round trip with enough detail to answer *why* a result happened,
not merely what it was. Gross P&L, costs and net P&L are kept strictly separate throughout
(DESIGN.md 2.3): reporting only net hides whether a strategy had no edge or had one that
costs ate, and those call for completely different responses.

The per-trade **R multiple** — net P&L divided by the rupees that were at risk — is the
unit everything else is expressed in. It makes trades comparable across instruments and
position sizes, and it is what the ~0.12R cost hurdle from DESIGN.md 5.2 is measured
against.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from src.backtest.costs import CostBreakdown
from src.backtest.sim_broker import Fill, FillReason, OpenPosition


@dataclass
class Trade:
    """One completed round trip."""

    instrument_id: str
    is_long: bool
    quantity: int
    entry_at: dt.datetime
    entry_price: float
    exit_at: dt.datetime
    exit_price: float
    exit_reason: FillReason
    costs: CostBreakdown
    risk_per_share: float
    mae: float = 0.0
    mfe: float = 0.0
    bars_held: int = 0
    strategy_name: str = ""
    reason: str = ""
    # Slippage is already baked into entry_price and exit_price, so it is invisible in
    # net P&L. Carrying it separately is what lets the report distinguish "no edge" from
    # "edge eaten by friction" — the two demand completely different responses.
    entry_slippage_per_share: float = 0.0
    exit_slippage_per_share: float = 0.0

    @property
    def gross_pnl(self) -> float:
        """Realised P&L before explicit charges — but *after* slippage."""
        direction = 1 if self.is_long else -1
        return (self.exit_price - self.entry_price) * self.quantity * direction

    @property
    def slippage_cost(self) -> float:
        """What slippage took, already deducted from :attr:`gross_pnl`."""
        return (self.entry_slippage_per_share + self.exit_slippage_per_share) * self.quantity

    @property
    def frictionless_pnl(self) -> float:
        """P&L the strategy would have made with perfect fills and no charges.

        The honest measure of whether the *signal* had any edge at all, independent of
        what execution cost to obtain.
        """
        return self.gross_pnl + self.slippage_cost

    @property
    def total_costs(self) -> float:
        return self.costs.total

    @property
    def total_friction(self) -> float:
        """Everything execution took: explicit charges plus slippage.

        This is the figure comparable to the ~0.12R hurdle in DESIGN.md 5.2.
        """
        return self.total_costs + self.slippage_cost

    @property
    def net_pnl(self) -> float:
        return self.gross_pnl - self.total_costs

    @property
    def risk_amount(self) -> float:
        return self.risk_per_share * self.quantity

    @property
    def r_multiple(self) -> float:
        """Net P&L in units of risk taken. The comparable unit across all trades."""
        if self.risk_amount <= 0:
            return 0.0
        return self.net_pnl / self.risk_amount

    @property
    def gross_r_multiple(self) -> float:
        if self.risk_amount <= 0:
            return 0.0
        return self.gross_pnl / self.risk_amount

    @property
    def cost_r_multiple(self) -> float:
        """Explicit charges expressed in R (excludes slippage)."""
        if self.risk_amount <= 0:
            return 0.0
        return self.total_costs / self.risk_amount

    @property
    def slippage_r_multiple(self) -> float:
        if self.risk_amount <= 0:
            return 0.0
        return self.slippage_cost / self.risk_amount

    @property
    def friction_r_multiple(self) -> float:
        """Total execution drag in R — the figure the 0.12R hurdle refers to."""
        if self.risk_amount <= 0:
            return 0.0
        return self.total_friction / self.risk_amount

    @property
    def frictionless_r_multiple(self) -> float:
        """What the signal was worth before execution took its cut."""
        if self.risk_amount <= 0:
            return 0.0
        return self.frictionless_pnl / self.risk_amount

    @property
    def holding_minutes(self) -> float:
        return (self.exit_at - self.entry_at).total_seconds() / 60.0

    @property
    def is_win(self) -> bool:
        return self.net_pnl > 0

    @property
    def notional(self) -> float:
        return self.quantity * self.entry_price

    def to_dict(self) -> Dict[str, object]:
        return {
            "instrument_id": self.instrument_id,
            "direction": "LONG" if self.is_long else "SHORT",
            "quantity": self.quantity,
            "entry_at": self.entry_at,
            "entry_price": round(self.entry_price, 2),
            "exit_at": self.exit_at,
            "exit_price": round(self.exit_price, 2),
            "exit_reason": self.exit_reason.value,
            "frictionless_pnl": round(self.frictionless_pnl, 2),
            "gross_pnl": round(self.gross_pnl, 2),
            "slippage": round(self.slippage_cost, 2),
            "costs": round(self.total_costs, 2),
            "friction": round(self.total_friction, 2),
            "net_pnl": round(self.net_pnl, 2),
            "risk_amount": round(self.risk_amount, 2),
            "r_multiple": round(self.r_multiple, 3),
            "gross_r": round(self.gross_r_multiple, 3),
            "frictionless_r": round(self.frictionless_r_multiple, 3),
            "cost_r": round(self.cost_r_multiple, 3),
            "slippage_r": round(self.slippage_r_multiple, 3),
            "friction_r": round(self.friction_r_multiple, 3),
            "mae": round(self.mae, 2),
            "mfe": round(self.mfe, 2),
            "bars_held": self.bars_held,
            "holding_minutes": round(self.holding_minutes, 1),
            "hour": self.entry_at.hour,
            "day": self.entry_at.date(),
            "strategy": self.strategy_name,
            "reason": self.reason,
        }


@dataclass
class DailyRecord:
    """One session's outcome, for the by-day and by-regime breakdowns."""

    day: dt.date
    starting_equity: float
    ending_equity: float
    trades: int = 0
    gross_pnl: float = 0.0
    costs: float = 0.0
    halted: bool = False

    @property
    def net_pnl(self) -> float:
        return self.ending_equity - self.starting_equity

    @property
    def return_pct(self) -> float:
        return self.net_pnl / self.starting_equity if self.starting_equity else 0.0


class Portfolio:
    """Accumulates trades, equity and per-session records over a backtest."""

    def __init__(self, starting_equity: float) -> None:
        if starting_equity <= 0:
            raise ValueError("starting_equity must be positive")
        self.starting_equity = starting_equity
        self.equity = starting_equity
        self.trades: List[Trade] = []
        self.daily: List[DailyRecord] = []
        self.equity_curve: List[tuple[dt.datetime, float]] = []
        self._day_open_equity = starting_equity
        self._current_day: Optional[dt.date] = None

    # -- session bookkeeping ----------------------------------------------
    def start_day(self, day: dt.date) -> None:
        self._current_day = day
        self._day_open_equity = self.equity

    def end_day(self, day: dt.date, *, halted: bool = False) -> DailyRecord:
        day_trades = [t for t in self.trades if t.entry_at.date() == day]
        record = DailyRecord(
            day=day,
            starting_equity=self._day_open_equity,
            ending_equity=self.equity,
            trades=len(day_trades),
            gross_pnl=sum(t.gross_pnl for t in day_trades),
            costs=sum(t.total_costs for t in day_trades),
            halted=halted,
        )
        self.daily.append(record)
        return record

    # -- trades ------------------------------------------------------------
    def record_trade(
        self,
        position: OpenPosition,
        exit_fill: Fill,
        *,
        entry_costs: CostBreakdown,
        entry_slippage: float = 0.0,
    ) -> Trade:
        """Close out a position into a :class:`Trade` and update equity."""
        trade = Trade(
            instrument_id=position.instrument_id,
            is_long=position.is_long,
            quantity=abs(position.quantity),
            entry_at=position.entry_at,
            entry_price=position.entry_price,
            exit_at=exit_fill.at,
            exit_price=exit_fill.price,
            exit_reason=exit_fill.reason,
            costs=entry_costs + exit_fill.costs,
            risk_per_share=position.risk_per_share,
            mae=position.mae,
            mfe=position.mfe,
            bars_held=position.bars_held,
            strategy_name=position.strategy_name,
            reason=position.reason,
            entry_slippage_per_share=entry_slippage,
            exit_slippage_per_share=exit_fill.slippage_per_share,
        )
        self.trades.append(trade)
        self.equity += trade.net_pnl
        self.equity_curve.append((exit_fill.at, self.equity))
        return trade

    # -- views -------------------------------------------------------------
    def trades_frame(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame([t.to_dict() for t in self.trades])

    def equity_frame(self) -> pd.DataFrame:
        if not self.equity_curve:
            return pd.DataFrame(columns=["at", "equity"])
        frame = pd.DataFrame(self.equity_curve, columns=["at", "equity"])
        frame["peak"] = frame["equity"].cummax()
        frame["drawdown"] = frame["equity"] - frame["peak"]
        frame["drawdown_pct"] = frame["drawdown"] / frame["peak"]
        return frame

    def daily_frame(self) -> pd.DataFrame:
        if not self.daily:
            return pd.DataFrame()
        return pd.DataFrame([{
            "day": d.day,
            "starting_equity": round(d.starting_equity, 2),
            "ending_equity": round(d.ending_equity, 2),
            "net_pnl": round(d.net_pnl, 2),
            "return_pct": round(d.return_pct, 5),
            "trades": d.trades,
            "gross_pnl": round(d.gross_pnl, 2),
            "costs": round(d.costs, 2),
            "halted": d.halted,
        } for d in self.daily])

    @property
    def total_return(self) -> float:
        return (self.equity - self.starting_equity) / self.starting_equity
