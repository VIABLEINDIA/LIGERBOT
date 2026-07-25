"""Backtest metrics (DESIGN.md 2.4).

Headline figures alone hide the flaws that matter, so this module also produces the
breakdowns that catch them:

  * **By hour of day** — reveals whether an "edge" is real or is one time-of-day artefact.
  * **By month** — reveals whether it is one lucky quarter wearing a trend line.
  * **By session direction** — required by D3. A long-only strategy will look good in a
    rising market for reasons that have nothing to do with skill, and this breakdown makes
    that visible instead of letting it pass as alpha.
  * **Gross vs. cost vs. net, in R** — says whether a strategy had no edge, or had one
    that costs ate. Those are different problems with different fixes.

Every ratio here is reported alongside its trade count. A profit factor computed on nine
trades is decoration, not evidence.
"""
from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from src.backtest.portfolio import Portfolio, Trade

TRADING_DAYS_PER_YEAR = 252


@dataclass
class Metrics:
    """Everything a backtest produced, in one object."""

    # Headline
    starting_equity: float = 0.0
    ending_equity: float = 0.0
    total_return: float = 0.0
    cagr: float = 0.0

    # P&L decomposition. Slippage is separated out because it hides inside gross P&L —
    # without splitting it, "no edge" and "edge eaten by friction" look identical.
    frictionless_pnl: float = 0.0    # before slippage and charges
    gross_pnl: float = 0.0           # after slippage, before charges
    total_slippage: float = 0.0
    total_costs: float = 0.0
    total_friction: float = 0.0      # slippage + charges
    net_pnl: float = 0.0
    cost_ratio: float = 0.0          # friction / |frictionless|

    # Risk
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    calmar: float = 0.0

    # Trades
    trade_count: int = 0
    win_count: int = 0
    loss_count: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    expectancy_r: float = 0.0
    gross_expectancy_r: float = 0.0
    frictionless_expectancy_r: float = 0.0
    cost_drag_r: float = 0.0
    slippage_drag_r: float = 0.0
    friction_drag_r: float = 0.0     # compared against the ~0.12R hurdle
    avg_win_r: float = 0.0
    avg_loss_r: float = 0.0
    largest_loss: float = 0.0
    max_consecutive_losses: int = 0

    # Activity
    trading_days: int = 0
    trades_per_day: float = 0.0
    avg_holding_minutes: float = 0.0
    halted_days: int = 0

    # Excursions
    avg_mae_r: float = 0.0
    avg_mfe_r: float = 0.0

    # Breakdowns
    by_hour: pd.DataFrame = field(default_factory=pd.DataFrame)
    by_month: pd.DataFrame = field(default_factory=pd.DataFrame)
    by_instrument: pd.DataFrame = field(default_factory=pd.DataFrame)
    by_session_direction: pd.DataFrame = field(default_factory=pd.DataFrame)
    by_exit_reason: pd.DataFrame = field(default_factory=pd.DataFrame)
    cost_breakdown: Dict[str, float] = field(default_factory=dict)

    @property
    def is_profitable(self) -> bool:
        return self.net_pnl > 0

    def headline(self) -> str:
        return (
            f"net {self.net_pnl:+,.0f} ({self.total_return:+.2%}) | "
            f"gross {self.gross_pnl:+,.0f} - costs {self.total_costs:,.0f} | "
            f"{self.trade_count} trades, {self.win_rate:.1%} win, "
            f"PF {self.profit_factor:.2f}, expectancy {self.expectancy_r:+.3f}R | "
            f"maxDD {self.max_drawdown_pct:.2%}, Sharpe {self.sharpe:.2f}"
        )


def _safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    return numerator / denominator if denominator else default


def _max_consecutive_losses(trades: List[Trade]) -> int:
    worst = current = 0
    for trade in trades:
        if trade.net_pnl <= 0:
            current += 1
            worst = max(worst, current)
        else:
            current = 0
    return worst


def _sharpe(daily_returns: pd.Series) -> float:
    """Annualised Sharpe from daily returns, risk-free rate assumed zero."""
    if len(daily_returns) < 2:
        return 0.0
    std = daily_returns.std(ddof=1)
    if std == 0 or math.isnan(std):
        return 0.0
    return float(daily_returns.mean() / std * math.sqrt(TRADING_DAYS_PER_YEAR))


def _sortino(daily_returns: pd.Series) -> float:
    """Like Sharpe but penalising only downside deviation."""
    if len(daily_returns) < 2:
        return 0.0
    downside = daily_returns[daily_returns < 0]
    if downside.empty:
        return 0.0
    deviation = math.sqrt((downside ** 2).mean())
    if deviation == 0:
        return 0.0
    return float(daily_returns.mean() / deviation * math.sqrt(TRADING_DAYS_PER_YEAR))


def _group_stats(frame: pd.DataFrame, key: str) -> pd.DataFrame:
    """Trade count, net P&L, win rate and mean R per group."""
    if frame.empty or key not in frame.columns:
        return pd.DataFrame()
    grouped = frame.groupby(key).agg(
        trades=("net_pnl", "size"),
        net_pnl=("net_pnl", "sum"),
        gross_pnl=("gross_pnl", "sum"),
        costs=("costs", "sum"),
        avg_r=("r_multiple", "mean"),
        wins=("net_pnl", lambda s: int((s > 0).sum())),
    ).reset_index()
    grouped["win_rate"] = grouped["wins"] / grouped["trades"]
    return grouped.round(4)


def compute(
    portfolio: Portfolio,
    *,
    session_directions: Optional[Dict[dt.date, str]] = None,
) -> Metrics:
    """Compute every metric from a completed backtest.

    ``session_directions`` maps each session to ``"up"``/``"down"``/``"flat"`` based on
    the market's own move that day, enabling the D3 bias breakdown.
    """
    trades = portfolio.trades
    metrics = Metrics(
        starting_equity=portfolio.starting_equity,
        ending_equity=portfolio.equity,
        total_return=portfolio.total_return,
        trade_count=len(trades),
    )

    daily = portfolio.daily_frame()
    metrics.trading_days = len(daily)
    metrics.halted_days = int(daily["halted"].sum()) if not daily.empty else 0

    if not trades:
        return metrics

    frame = portfolio.trades_frame()

    # -- P&L decomposition -------------------------------------------------
    # Summed from the trade objects, not the frame: `to_dict()` rounds to 2dp for
    # display, and over a thousand trades that rounding accumulates into a visible
    # discrepancy between net P&L and the equity curve.
    metrics.frictionless_pnl = sum(t.frictionless_pnl for t in trades)
    metrics.gross_pnl = sum(t.gross_pnl for t in trades)
    metrics.total_slippage = sum(t.slippage_cost for t in trades)
    metrics.total_costs = sum(t.total_costs for t in trades)
    metrics.total_friction = sum(t.total_friction for t in trades)
    metrics.net_pnl = sum(t.net_pnl for t in trades)
    metrics.cost_ratio = _safe_div(metrics.total_friction, abs(metrics.frictionless_pnl))

    combined = sum((t.costs for t in trades[1:]), trades[0].costs)
    metrics.cost_breakdown = combined.to_dict()

    # -- trade statistics --------------------------------------------------
    wins = frame[frame["net_pnl"] > 0]
    losses = frame[frame["net_pnl"] <= 0]
    metrics.win_count, metrics.loss_count = len(wins), len(losses)
    metrics.win_rate = _safe_div(len(wins), len(frame))

    gross_profit = float(wins["net_pnl"].sum())
    gross_loss = abs(float(losses["net_pnl"].sum()))
    metrics.profit_factor = _safe_div(gross_profit, gross_loss, default=float("inf")
                                      if gross_profit > 0 else 0.0)

    count = len(trades)
    metrics.expectancy_r = sum(t.r_multiple for t in trades) / count
    metrics.gross_expectancy_r = sum(t.gross_r_multiple for t in trades) / count
    metrics.frictionless_expectancy_r = sum(t.frictionless_r_multiple for t in trades) / count
    metrics.cost_drag_r = sum(t.cost_r_multiple for t in trades) / count
    metrics.slippage_drag_r = sum(t.slippage_r_multiple for t in trades) / count
    metrics.friction_drag_r = sum(t.friction_r_multiple for t in trades) / count
    metrics.avg_win_r = float(wins["r_multiple"].mean()) if len(wins) else 0.0
    metrics.avg_loss_r = float(losses["r_multiple"].mean()) if len(losses) else 0.0
    metrics.largest_loss = min(t.net_pnl for t in trades)
    metrics.max_consecutive_losses = _max_consecutive_losses(trades)
    metrics.avg_holding_minutes = float(frame["holding_minutes"].mean())
    metrics.trades_per_day = _safe_div(len(frame), max(1, metrics.trading_days))

    # Excursions in R — how far trades went against us before working out.
    risk = frame["risk_amount"].replace(0, pd.NA)
    quantity = frame["quantity"].replace(0, pd.NA)
    metrics.avg_mae_r = float((frame["mae"] * quantity / risk).mean(skipna=True) or 0.0)
    metrics.avg_mfe_r = float((frame["mfe"] * quantity / risk).mean(skipna=True) or 0.0)

    # -- risk --------------------------------------------------------------
    equity = portfolio.equity_frame()
    if not equity.empty:
        metrics.max_drawdown = float(equity["drawdown"].min())
        metrics.max_drawdown_pct = float(equity["drawdown_pct"].min())

    if not daily.empty:
        returns = daily["return_pct"].astype(float)
        metrics.sharpe = _sharpe(returns)
        metrics.sortino = _sortino(returns)
        years = max(metrics.trading_days / TRADING_DAYS_PER_YEAR, 1e-9)
        if portfolio.equity > 0 and portfolio.starting_equity > 0:
            metrics.cagr = (portfolio.equity / portfolio.starting_equity) ** (1 / years) - 1
        metrics.calmar = _safe_div(metrics.cagr, abs(metrics.max_drawdown_pct))

    # -- breakdowns --------------------------------------------------------
    metrics.by_hour = _group_stats(frame, "hour")
    metrics.by_instrument = _group_stats(frame, "instrument_id")
    metrics.by_exit_reason = _group_stats(frame, "exit_reason")

    month = frame.copy()
    # tz_localize(None) first: to_period drops timezone info and warns otherwise.
    month["month"] = (pd.to_datetime(month["entry_at"]).dt.tz_localize(None)
                      .dt.to_period("M").astype(str))
    metrics.by_month = _group_stats(month, "month")

    if session_directions:
        tagged = frame.copy()
        tagged["session_direction"] = tagged["day"].map(
            lambda d: session_directions.get(d, "unknown")
        )
        metrics.by_session_direction = _group_stats(tagged, "session_direction")

    return metrics


def format_report(metrics: Metrics, *, title: str = "Backtest") -> str:
    """Human-readable report. What actually gets read after a run."""
    rule = "=" * 78
    lines = [rule, title, rule]

    if metrics.trade_count == 0:
        lines.append("No trades were taken.")
        lines.append(rule)
        return "\n".join(lines)

    lines += [
        "",
        "P&L (friction split out — slippage otherwise hides inside gross)",
        f"  Starting equity      {metrics.starting_equity:>14,.2f}",
        f"  Ending equity        {metrics.ending_equity:>14,.2f}",
        f"  Frictionless P&L     {metrics.frictionless_pnl:>+14,.2f}   "
        f"<- what the signal was worth",
        f"    less slippage      {metrics.total_slippage:>14,.2f}",
        f"  Gross P&L            {metrics.gross_pnl:>+14,.2f}",
        f"    less charges       {metrics.total_costs:>14,.2f}",
        f"  Net P&L              {metrics.net_pnl:>+14,.2f}   "
        f"({metrics.total_return:+.2%})",
        f"  Total friction       {metrics.total_friction:>14,.2f}   "
        f"({metrics.cost_ratio:.1%} of |frictionless P&L|)",
        "",
        "Per trade, in R (risk units)",
        f"  Frictionless         {metrics.frictionless_expectancy_r:>+14.3f}R   "
        f"<- edge before execution",
        f"  Slippage drag        {metrics.slippage_drag_r:>14.3f}R",
        f"  Charges drag         {metrics.cost_drag_r:>14.3f}R",
        f"  TOTAL FRICTION       {metrics.friction_drag_r:>14.3f}R   "
        f"<- vs the ~0.12R hurdle (DESIGN.md 5.2)",
        f"  Net expectancy       {metrics.expectancy_r:>+14.3f}R",
        f"  Average win          {metrics.avg_win_r:>+14.3f}R",
        f"  Average loss         {metrics.avg_loss_r:>+14.3f}R",
        "",
        "Trades",
        f"  Count                {metrics.trade_count:>14,}   "
        f"({metrics.trades_per_day:.2f}/day over {metrics.trading_days} days)",
        f"  Win rate             {metrics.win_rate:>13.1%}    "
        f"({metrics.win_count}W / {metrics.loss_count}L)",
        f"  Profit factor        {metrics.profit_factor:>14.2f}",
        f"  Largest loss         {metrics.largest_loss:>+14,.2f}",
        f"  Max consecutive loss {metrics.max_consecutive_losses:>14,}",
        f"  Avg holding time     {metrics.avg_holding_minutes:>14.1f} min",
        f"  Avg MAE / MFE        {metrics.avg_mae_r:>14.2f}R / {metrics.avg_mfe_r:.2f}R",
        "",
        "Risk",
        f"  Max drawdown         {metrics.max_drawdown:>+14,.2f}   "
        f"({metrics.max_drawdown_pct:.2%})",
        f"  Sharpe (annualised)  {metrics.sharpe:>14.2f}",
        f"  Sortino              {metrics.sortino:>14.2f}",
        f"  Calmar               {metrics.calmar:>14.2f}",
        f"  Halted days          {metrics.halted_days:>14,}",
        "",
        "Cost breakdown",
    ]
    for key, value in metrics.cost_breakdown.items():
        if key != "total":
            lines.append(f"  {key:<20} {value:>14,.2f}")
    lines.append(f"  {'TOTAL':<20} {metrics.cost_breakdown.get('total', 0.0):>14,.2f}")

    for label, table in [
        ("By hour of day", metrics.by_hour),
        ("By month", metrics.by_month),
        ("By instrument", metrics.by_instrument),
        ("By exit reason", metrics.by_exit_reason),
        ("By session direction (D3 long-only bias check)", metrics.by_session_direction),
    ]:
        if table is not None and not table.empty:
            lines += ["", label, table.to_string(index=False)]

    lines += ["", rule]
    return "\n".join(lines)
