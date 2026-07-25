"""The backtest engine.

Drives bars through the **same** strategy and risk-engine objects the live system uses
(DESIGN.md 2.1). Only the transport and the broker are simulated. This is the property
that makes results mean anything: a risk rule tested here is literally the rule that will
run in production, not a reimplementation of it.

Ordering within each bar is the whole ballgame, so it is fixed and explicit:

1. **Fills first.** Orders queued on the previous bar execute at *this* bar's open, and
   stops/targets are then checked against this bar's range.
2. **Session control.** If the flat deadline has passed, positions are closed. No strategy
   opinion is consulted; this is not negotiable.
3. **Strategy last.** Only then does the strategy see the (closed) bar and emit signals,
   which are queued for the *next* bar.

That sequence is what enforces next-bar execution. A strategy physically cannot act on a
bar it has not been shown, and cannot be filled at a price from the bar it decided on.
"""
from __future__ import annotations

import datetime as dt
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import pandas as pd

from src import market_calendar as cal
from src.backtest import metrics as metrics_mod
from src.backtest.bar_source import BarSource, QualityReport, QualityThresholds
from src.backtest.costs import CostModel, SlippageModel
from src.backtest.metrics import Metrics
from src.backtest.portfolio import Portfolio
from src.backtest.sim_broker import Fill, FillReason, SimBroker
from src.bars import Bar
from src.risk_engine import Intent, RiskEngine, RiskLimits, Side, Signal
from src.strategy_base import Strategy, StrategyContext

log = logging.getLogger("ligerbot.backtest")

# Collapses the varying figures inside rejection messages so they tally by kind.
_NUMERIC = re.compile(r"[-+]?\d[\d,]*\.?\d*%?")


@dataclass
class BacktestConfig:
    starting_equity: float = 500_000.0
    risk_limits: RiskLimits = field(default_factory=RiskLimits)
    cost_model: CostModel = field(default_factory=CostModel)
    slippage: SlippageModel = field(default_factory=SlippageModel)
    quality_thresholds: QualityThresholds = field(default_factory=QualityThresholds)
    enforce_liquidity: bool = True
    max_volume_participation: float = 0.10
    # Compounding: each session's equity base is the running portfolio equity (D1's
    # snapshot rule applied to a backtest). Set False to size every day off the initial
    # capital, which isolates strategy behaviour from the compounding curve.
    compound: bool = True
    skip_quality_gate: bool = False


@dataclass
class BacktestResult:
    metrics: Metrics
    portfolio: Portfolio
    quality: Optional[QualityReport]
    strategy_description: str
    cost_description: str
    slippage_description: str
    start: Optional[dt.date] = None
    end: Optional[dt.date] = None
    rejections: Dict[str, int] = field(default_factory=dict)

    def report(self, title: Optional[str] = None) -> str:
        header = title or f"Backtest — {self.strategy_description}"
        body = metrics_mod.format_report(self.metrics, title=header)
        provenance = [
            "",
            "Provenance",
            f"  Strategy   {self.strategy_description}",
            f"  Costs      {self.cost_description}",
            f"  Slippage   {self.slippage_description}",
            f"  Window     {self.start} to {self.end}",
        ]
        if self.quality is not None:
            provenance.append(f"  Data       {self.quality.summary()}")
            advisory = [i for i in self.quality.issues if not i.blocking]
            for issue in advisory[:5]:
                provenance.append(f"             ! {issue.kind}: {issue.detail}")
        if self.rejections:
            provenance.append("  Signals refused")
            for reason, count in sorted(self.rejections.items(), key=lambda kv: -kv[1])[:6]:
                provenance.append(f"    {count:>7,}  {reason}")
        return body + "\n".join(provenance) + "\n" + "=" * 78


class BacktestEngine:
    def __init__(
        self,
        strategy: Strategy,
        config: Optional[BacktestConfig] = None,
    ) -> None:
        self.strategy = strategy
        self.config = config or BacktestConfig()
        self.risk = RiskEngine(self.config.risk_limits)
        self.broker = SimBroker(
            self.config.cost_model, self.config.slippage,
            max_volume_participation=self.config.max_volume_participation,
            enforce_liquidity=self.config.enforce_liquidity,
        )
        self.portfolio = Portfolio(self.config.starting_equity)
        self._entry_costs: Dict[str, object] = {}
        self._entry_slippage: Dict[str, float] = {}
        self._current_day: Optional[dt.date] = None
        self._rejections: Dict[str, int] = {}
        # For the D3 session-direction breakdown.
        self._day_first_price: Dict[dt.date, Dict[str, float]] = {}
        self._day_last_price: Dict[dt.date, Dict[str, float]] = {}

    # -- session -----------------------------------------------------------
    def _roll_session(self, day: dt.date) -> None:
        if self._current_day == day:
            return
        if self._current_day is not None:
            self._close_session(self._current_day)

        self._current_day = day
        equity = self.portfolio.equity if self.config.compound else self.config.starting_equity
        if equity <= 0:
            raise RuntimeError(
                f"Equity reached {equity:.2f} on {day} — the account is wiped out. "
                f"Halting the backtest rather than trading a negative balance."
            )
        self.portfolio.start_day(day)
        self.risk.start_session(day, equity)
        self.strategy.on_session_start(day)

    def _close_session(self, day: dt.date) -> None:
        self.portfolio.end_day(day, halted=self.risk.halted)

    # -- main loop ---------------------------------------------------------
    def run(
        self,
        source: BarSource,
        instrument_ids: Sequence[str],
        start: dt.date,
        end: dt.date,
    ) -> BacktestResult:
        quality = None
        if not self.config.skip_quality_gate:
            quality = source.assess(instrument_ids, start, end,
                                    self.config.quality_thresholds)
            # Refuse rather than warn: a warning in a log is not a decision anyone makes.
            quality.raise_if_unusable()
            log.info("Data quality: %s", quality.summary())

        for bar in source.stream(instrument_ids, start, end):
            self._process_bar(bar)

        self._finalize()

        return BacktestResult(
            metrics=metrics_mod.compute(
                self.portfolio, session_directions=self._session_directions()
            ),
            portfolio=self.portfolio,
            quality=quality,
            strategy_description=self.strategy.describe(),
            cost_description=self.config.cost_model.describe(),
            slippage_description=self.config.slippage.describe(),
            start=start, end=end,
            rejections=dict(self._rejections),
        )

    def _process_bar(self, bar: Bar) -> None:
        day = bar.bar_start.date()
        self._roll_session(day)
        self._track_session_prices(day, bar)

        # 1. Fills first — queued orders execute at this bar's open, then stops/targets
        #    are checked against this bar's range.
        for fill in self.broker.process_bar(bar):
            self._record_and_apply(fill, bar)

        phase = cal.phase(bar.bar_end)

        # 2. Session control. Not negotiable, and not a strategy decision.
        if phase.requires_flat and self.broker.has_position(bar.instrument_id):
            forced = self.broker.force_close(bar.instrument_id, bar)
            if forced is not None:
                self._record_and_apply(forced, bar)
        if phase.requires_flat or phase is cal.Phase.CLOSED:
            self.broker.cancel_pending(bar.instrument_id)
            return

        # 3. Strategy last. It sees only this closed bar and emits for the next one.
        context = StrategyContext(
            now=bar.bar_end,
            position=self.risk.positions.get(bar.instrument_id),
            seconds_to_square_off=cal.seconds_to_square_off(bar.bar_end) or 0.0,
            allows_entry=phase.allows_entry,
            session_day=day,
        )
        for signal in self.strategy.on_bar(bar, context):
            self._handle_signal(signal, phase)

    def _handle_signal(self, signal: Signal, phase) -> None:
        decision = self.risk.evaluate(
            signal, allows_entry=phase.allows_entry, allows_exit=phase.allows_exit
        )
        if not decision.approved:
            self._count_rejection(decision.reason)
            return
        self.broker.submit(decision.order, signal.bar_time)

    def _count_rejection(self, reason: str) -> None:
        """Tally rejections by *kind*.

        Reasons embed live figures ("open risk would reach 1.521%"), so counting raw
        strings produces dozens of near-identical rows that hide the actual pattern.
        Numbers are stripped so the tally answers "why were signals refused" rather than
        "which exact percentages occurred".
        """
        key = _NUMERIC.sub("#", reason).split(" (")[0].strip()[:60]
        self._rejections[key] = self._rejections.get(key, 0) + 1

    def _record_and_apply(self, fill: Fill, bar: Bar) -> None:
        """Apply a fill to risk and portfolio state."""
        if fill.intent.is_open:
            self._apply_open(fill)
        else:
            self._apply_close(fill, bar)

    def _apply_open(self, fill: Fill) -> None:
        position = self.broker.positions.get(fill.instrument_id)
        stop = position.stop_loss if position else 0.0
        self._entry_costs[fill.instrument_id] = fill.costs
        self._entry_slippage[fill.instrument_id] = fill.slippage_per_share
        # Risk is recorded from the ACTUAL fill price, not the reference price: a
        # position that slipped on entry carries more risk than was budgeted, and the
        # open-risk cap must see the real figure.
        self.risk.on_open_fill(
            fill.instrument_id,
            fill.quantity if fill.side is Side.BUY else -fill.quantity,
            fill.price, stop, opened_at=fill.at,
        )
        # Entry costs hit the day's P&L immediately, so they count toward the drawdown
        # breaker as they are incurred rather than appearing only at exit.
        self.risk.apply_costs(fill.costs.total)

    def _apply_close(self, fill: Fill, bar: Bar) -> None:
        position = self.broker.closed_positions.pop(fill.instrument_id, None)
        if position is None:
            log.warning("Closing fill for %s with no recorded position — ignoring.",
                        fill.instrument_id)
            return
        entry_costs = self._entry_costs.pop(fill.instrument_id, fill.costs)
        trade = self.portfolio.record_trade(
            position, fill, entry_costs=entry_costs,
            entry_slippage=self._entry_slippage.pop(fill.instrument_id, 0.0),
        )
        self.risk.on_close_fill(fill.instrument_id, fill.price)
        self.risk.apply_costs(fill.costs.total)
        log.debug("trade %s %s %+.2f (%.2fR) via %s", trade.instrument_id,
                  "LONG" if trade.is_long else "SHORT", trade.net_pnl,
                  trade.r_multiple, trade.exit_reason.value)

    # -- session direction (D3) -------------------------------------------
    def _track_session_prices(self, day: dt.date, bar: Bar) -> None:
        if bar.synthetic:
            return
        self._day_first_price.setdefault(day, {}).setdefault(bar.instrument_id, bar.open)
        self._day_last_price.setdefault(day, {})[bar.instrument_id] = bar.close

    def _session_directions(self, flat_threshold: float = 0.003) -> Dict[dt.date, str]:
        """Classify each session up / down / flat from the market's own move.

        Required by D3: a long-only strategy will look good in a rising market for
        reasons unrelated to skill, and this is what makes that visible.
        """
        out: Dict[dt.date, str] = {}
        for day, first in self._day_first_price.items():
            last = self._day_last_price.get(day, {})
            moves = [
                (last[i] - first[i]) / first[i]
                for i in first if i in last and first[i] > 0
            ]
            if not moves:
                continue
            average = sum(moves) / len(moves)
            out[day] = ("up" if average > flat_threshold
                        else "down" if average < -flat_threshold else "flat")
        return out

    def _finalize(self) -> None:
        if self._current_day is not None:
            self._close_session(self._current_day)
