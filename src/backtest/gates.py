"""The go-live gates (DESIGN.md 2.6).

Turns the checklist into something executable, so "did it pass?" is a computation rather
than a judgement call made while looking at an encouraging equity curve.

Two rules are structural, not cosmetic:

* **Thresholds do not move to accommodate limited data.** If the data cannot support them,
  the answer is a longer forward test, not a lower bar (D5 mitigation 4). There is no
  parameter here for relaxing a gate.
* **Every gate reports its actual value**, not just pass/fail. A profit factor of 1.29
  against a 1.30 threshold is a very different situation from 0.4, and collapsing both to
  FAIL throws away the information that tells you which.

Passing these gates authorises **paper trading**, not live capital. Live still requires the
paper period (Phase 4) and reconciliation against a backtest over the same sessions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd

from src.backtest.metrics import Metrics
from src.backtest.walk_forward import WalkForwardResult


@dataclass
class GateResult:
    name: str
    passed: bool
    actual: str
    threshold: str
    detail: str = ""
    # Advisory gates inform the decision but do not block it on their own.
    advisory: bool = False

    def __str__(self) -> str:
        mark = "PASS" if self.passed else ("WARN" if self.advisory else "FAIL")
        line = f"  [{mark}] {self.name:<44} {self.actual:>14}  (need {self.threshold})"
        return line + (f"\n         {self.detail}" if self.detail else "")


@dataclass
class GateReport:
    results: List[GateResult] = field(default_factory=list)
    context: str = ""
    title: str = "GO-LIVE GATES (DESIGN.md 2.6)"

    @property
    def blocking_failures(self) -> List[GateResult]:
        return [r for r in self.results if not r.passed and not r.advisory]

    @property
    def warnings(self) -> List[GateResult]:
        return [r for r in self.results if not r.passed and r.advisory]

    @property
    def passed(self) -> bool:
        return not self.blocking_failures

    def summary(self) -> str:
        rule = "=" * 78
        lines = [rule, self.title, rule]
        if self.context:
            lines += [f"  {self.context}", ""]
        lines += [str(r) for r in self.results]
        lines.append("")

        if self.passed and not self.warnings:
            lines.append("  VERDICT: all gates passed -> cleared for PAPER TRADING.")
            lines.append("  Not for live capital: that needs the Phase 4 paper period and")
            lines.append("  reconciliation against a backtest over those same sessions.")
        elif self.passed:
            lines.append(f"  VERDICT: gates passed with {len(self.warnings)} warning(s).")
            lines.append("  Read the warnings before treating this as evidence of anything.")
        else:
            lines.append(f"  VERDICT: BLOCKED — {len(self.blocking_failures)} gate(s) failed.")
            lines.append("  Thresholds do not move to fit the data (D5 mitigation 4). Either")
            lines.append("  improve the strategy or extend the forward test.")
        lines.append(rule)
        return "\n".join(lines)


def _fmt(value: float, suffix: str = "", places: int = 3) -> str:
    return f"{value:.{places}f}{suffix}"


def evaluate(
    metrics: Metrics,
    *,
    walk_forward: Optional[WalkForwardResult] = None,
    doubled_slippage: Optional[Metrics] = None,
    context: str = "",
) -> GateReport:
    """Run every §2.6 gate that the supplied evidence can support.

    Gates whose evidence is missing are recorded as failures with an explanatory note,
    never silently skipped — an unrun check must not read as a passed one.
    """
    report = GateReport(context=context)
    add = report.results.append

    # 1. Positive net expectancy above the cost hurdle.
    add(GateResult(
        "Net expectancy positive after costs",
        metrics.expectancy_r > 0,
        _fmt(metrics.expectancy_r, "R"), "> 0R",
        detail=(f"frictionless {metrics.frictionless_expectancy_r:+.3f}R "
                f"less friction {metrics.friction_drag_r:.3f}R"),
    ))

    # 2. Profit factor.
    add(GateResult(
        "Profit factor", metrics.profit_factor > 1.3,
        _fmt(metrics.profit_factor, "", 2), "> 1.30",
    ))

    # 3. Drawdown within appetite.
    add(GateResult(
        "Max drawdown within risk appetite",
        abs(metrics.max_drawdown_pct) < 0.20,
        f"{metrics.max_drawdown_pct:.2%}", "< 20%",
    ))

    # 4. Sample size. Below this, ratios are decoration rather than evidence.
    oos_trades = walk_forward.oos_trade_count if walk_forward else metrics.trade_count
    add(GateResult(
        "Out-of-sample trade count", oos_trades >= 200,
        f"{oos_trades:,}", ">= 200",
        detail=("" if oos_trades >= 200 else
                "Below this, the ratios above are not statistically meaningful. "
                "Widen the universe or extend the paper period — do not lower the bar."),
    ))

    # 5. Slippage sensitivity.
    if doubled_slippage is not None:
        add(GateResult(
            "Survives doubled slippage",
            doubled_slippage.expectancy_r > 0,
            _fmt(doubled_slippage.expectancy_r, "R"), "> 0R",
            detail="A strategy that survives only at optimistic slippage is not deployable.",
        ))
    else:
        add(GateResult(
            "Survives doubled slippage", False, "not run", "> 0R",
            detail="Sensitivity run not supplied — an unrun check is not a passed one.",
        ))

    # 6. Not carried by a single month or instrument.
    add(_concentration_gate(metrics))

    # 7. Long-only bias visible (D3).
    add(_bias_gate(metrics))

    # 8. Walk-forward consistency.
    if walk_forward is not None:
        add(GateResult(
            "Walk-forward OOS expectancy positive",
            walk_forward.oos_expectancy_r > 0,
            _fmt(walk_forward.oos_expectancy_r, "R"),
            "> 0R",
            detail=f"{walk_forward.profitable_folds}/{len(walk_forward.folds)} folds profitable",
        ))
        add(GateResult(
            "In-sample degradation contained",
            walk_forward.mean_degradation < 0.05,
            _fmt(walk_forward.mean_degradation, "R"), "< 0.05R",
            detail=("" if walk_forward.mean_degradation < 0.05 else
                    "Large degradation means the optimiser fit training noise."),
        ))
        add(GateResult(
            "Trial count disclosed",
            True, f"{walk_forward.total_trials:,}", "reported",
            advisory=True,
            detail=(f"{walk_forward.total_trials} configurations evaluated; discount the "
                    f"headline figures for selection bias."
                    if walk_forward.total_trials > 20 else ""),
        ))
    else:
        add(GateResult(
            "Walk-forward validation", False, "not run", "required",
            detail="A single in-sample backtest is not evidence (DESIGN.md 2.5).",
        ))

    return report


def evaluate_phase4(
    reconciliation,
    *,
    sessions_completed: int,
    sessions_required: int = 20,
    halted_sessions: int = 0,
) -> GateReport:
    """Gates for progressing from paper trading to live capital (DESIGN.md Phase 4).

    Separate from :func:`evaluate` on purpose. The §2.6 gates ask "does the strategy have
    an edge in backtest"; these ask "does the live system reproduce that edge". A strategy
    can pass the first and fail the second — that failure is the single most valuable
    signal in the project, because it means the model is wrong somewhere.
    """
    report = GateReport(
        context="Paper trading -> live capital",
        title="PHASE 4 GATES — paper trading to live capital",
    )
    add = report.results.append

    add(GateResult(
        "Paper sessions completed", sessions_completed >= sessions_required,
        f"{sessions_completed}", f">= {sessions_required}",
        detail=("" if sessions_completed >= sessions_required else
                "Sessions are calendar time, not compute. There is no way to shorten "
                "this and no substitute for it."),
    ))

    add(GateResult(
        "Paper tracks the backtest", bool(reconciliation and reconciliation.passed),
        "pass" if (reconciliation and reconciliation.passed) else "fail",
        "within tolerance",
        detail=("" if (reconciliation and reconciliation.passed)
                else "; ".join(reconciliation.failures) if reconciliation
                else "no reconciliation available"),
    ))

    if reconciliation is not None:
        add(GateResult(
            "Divergence is explained", abs(reconciliation.unexplained) <
            max(1.0, abs(reconciliation.pnl_divergence) * 0.2),
            f"{reconciliation.unexplained:+,.2f}", "attributed",
            detail=("" if abs(reconciliation.unexplained) <
                    max(1.0, abs(reconciliation.pnl_divergence) * 0.2)
                    else "An unattributed residual means the diagnosis itself cannot "
                         "be trusted, which matters more than the divergence."),
        ))

    # Advisory: halted sessions are not failures, but a high rate says the risk limits
    # and the strategy's trade frequency are not compatible.
    halt_rate = halted_sessions / sessions_completed if sessions_completed else 0.0
    add(GateResult(
        "Halt rate", halt_rate < 0.15, f"{halt_rate:.0%}", "< 15%",
        advisory=True,
        detail=("" if halt_rate < 0.15 else
                f"{halted_sessions} of {sessions_completed} sessions halted — the risk "
                f"limits and the strategy's trade frequency may be incompatible"),
    ))

    return report


def _concentration_gate(metrics: Metrics) -> GateResult:
    """Fail if one month or one instrument carries the whole result."""
    worst_share, source = 0.0, ""
    for label, table in (("month", metrics.by_month), ("instrument", metrics.by_instrument)):
        if table is None or table.empty or len(table) < 2:
            continue
        profits = table[table["net_pnl"] > 0]["net_pnl"]
        total = profits.sum()
        if total <= 0:
            continue
        share = float(profits.max() / total)
        if share > worst_share:
            worst_share, source = share, label

    if not source:
        return GateResult(
            "Not driven by one month or instrument", False, "insufficient data",
            "< 60% share",
            detail="Need at least two months and two instruments to assess concentration.")

    return GateResult(
        "Not driven by one month or instrument", worst_share < 0.60,
        f"{worst_share:.1%}", "< 60% share",
        detail=f"largest single {source} contributes {worst_share:.1%} of gross profit",
    )


def _bias_gate(metrics: Metrics) -> GateResult:
    """D3: a long-only strategy's dependence on market direction must be visible.

    Advisory rather than blocking. Losing less in down sessions than a passive long would
    is legitimate; the requirement is that the dependence is *reported*, not that it is
    absent.
    """
    table = metrics.by_session_direction
    if table is None or table.empty:
        return GateResult(
            "Long-only bias reported (D3)", False, "not computed", "required",
            advisory=True,
            detail="Session-direction breakdown missing — the bias cannot be assessed.")

    rows = {str(r["session_direction"]): r for _, r in table.iterrows()}
    up = float(rows.get("up", {}).get("avg_r", 0.0) or 0.0)
    down = float(rows.get("down", {}).get("avg_r", 0.0) or 0.0)
    return GateResult(
        "Long-only bias reported (D3)", True,
        f"up {up:+.3f}R / down {down:+.3f}R", "reported",
        advisory=True,
        detail=("Edge appears in up sessions only — largely market beta, not skill."
                if up > 0 >= down and up - down > 0.1 else ""),
    )
