"""Go-live gate tests (DESIGN.md 2.6).

The central property: **an unrun check must never read as a passed one.** A gate whose
evidence is missing fails with a note, because the alternative is a green report that
quietly means nothing.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from src.backtest.gates import evaluate
from src.backtest.metrics import Metrics
from src.backtest.walk_forward import FoldResult, WalkForwardResult, Window


def good_metrics(**overrides) -> Metrics:
    """A metrics object that passes every threshold, for one-at-a-time negation."""
    metrics = Metrics(
        starting_equity=500_000.0, ending_equity=650_000.0, total_return=0.30,
        expectancy_r=0.15, frictionless_expectancy_r=0.27, friction_drag_r=0.12,
        profit_factor=1.60, max_drawdown_pct=-0.08, trade_count=400,
        win_rate=0.45, trading_days=250,
    )
    metrics.by_month = pd.DataFrame([
        {"month": "2026-01", "net_pnl": 40_000.0, "trades": 100, "avg_r": 0.15},
        {"month": "2026-02", "net_pnl": 35_000.0, "trades": 100, "avg_r": 0.14},
        {"month": "2026-03", "net_pnl": 45_000.0, "trades": 100, "avg_r": 0.16},
        {"month": "2026-04", "net_pnl": 30_000.0, "trades": 100, "avg_r": 0.13},
    ])
    metrics.by_instrument = pd.DataFrame([
        {"instrument_id": "nse_cm:1", "net_pnl": 70_000.0, "trades": 200, "avg_r": 0.15},
        {"instrument_id": "nse_cm:2", "net_pnl": 80_000.0, "trades": 200, "avg_r": 0.15},
    ])
    metrics.by_session_direction = pd.DataFrame([
        {"session_direction": "up", "net_pnl": 90_000.0, "trades": 200, "avg_r": 0.20},
        {"session_direction": "down", "net_pnl": 30_000.0, "trades": 100, "avg_r": 0.10},
        {"session_direction": "flat", "net_pnl": 30_000.0, "trades": 100, "avg_r": 0.10},
    ])
    for key, value in overrides.items():
        setattr(metrics, key, value)
    return metrics


def good_walk_forward(**overrides) -> WalkForwardResult:
    window = Window(dt.date(2026, 1, 1), dt.date(2026, 3, 31),
                    dt.date(2026, 4, 1), dt.date(2026, 4, 30))
    folds = [
        FoldResult(window=window, best_params={"adx_min": 20},
                   in_sample=good_metrics(expectancy_r=0.16),
                   out_of_sample=good_metrics(expectancy_r=0.14, trade_count=120),
                   trials=6),
        FoldResult(window=window, best_params={"adx_min": 20},
                   in_sample=good_metrics(expectancy_r=0.15),
                   out_of_sample=good_metrics(expectancy_r=0.13, trade_count=120),
                   trials=6),
    ]
    result = WalkForwardResult(folds=folds, total_trials=12)
    for key, value in overrides.items():
        setattr(result, key, value)
    return result


def gate(report, name_fragment):
    matches = [r for r in report.results if name_fragment.lower() in r.name.lower()]
    assert matches, f"no gate matching {name_fragment!r}"
    return matches[0]


class TestFullPass:
    def test_complete_evidence_passes(self):
        report = evaluate(good_metrics(), walk_forward=good_walk_forward(),
                          doubled_slippage=good_metrics(expectancy_r=0.06))
        assert report.passed, [str(r) for r in report.blocking_failures]
        assert "PAPER TRADING" in report.summary()

    def test_summary_never_claims_live_clearance(self):
        """Passing these gates authorises paper trading, never live capital."""
        report = evaluate(good_metrics(), walk_forward=good_walk_forward(),
                          doubled_slippage=good_metrics(expectancy_r=0.06))
        summary = report.summary()
        assert "Phase 4 paper period" in summary
        assert "Not for live capital" in summary


class TestMissingEvidenceFails:
    def test_missing_walk_forward_fails(self):
        report = evaluate(good_metrics(), doubled_slippage=good_metrics())
        assert not report.passed
        assert not gate(report, "Walk-forward validation").passed
        assert "not run" in gate(report, "Walk-forward validation").actual

    def test_missing_slippage_run_fails(self):
        report = evaluate(good_metrics(), walk_forward=good_walk_forward())
        result = gate(report, "doubled slippage")
        assert not result.passed
        assert "not run" in result.actual
        assert "unrun check is not a passed one" in result.detail


class TestIndividualGates:
    def test_negative_expectancy_blocks(self):
        report = evaluate(good_metrics(expectancy_r=-0.05),
                          walk_forward=good_walk_forward(),
                          doubled_slippage=good_metrics())
        assert not gate(report, "Net expectancy").passed
        assert not report.passed

    def test_low_profit_factor_blocks(self):
        report = evaluate(good_metrics(profit_factor=1.10),
                          walk_forward=good_walk_forward(),
                          doubled_slippage=good_metrics())
        assert not gate(report, "Profit factor").passed

    def test_profit_factor_reports_its_actual_value(self):
        """A 1.29 against a 1.30 bar is a different situation from 0.4."""
        report = evaluate(good_metrics(profit_factor=1.29),
                          walk_forward=good_walk_forward(),
                          doubled_slippage=good_metrics())
        assert gate(report, "Profit factor").actual == "1.29"

    def test_excessive_drawdown_blocks(self):
        report = evaluate(good_metrics(max_drawdown_pct=-0.35),
                          walk_forward=good_walk_forward(),
                          doubled_slippage=good_metrics())
        assert not gate(report, "drawdown").passed

    def test_thin_sample_blocks(self):
        walk = good_walk_forward()
        walk.folds[0].out_of_sample.trade_count = 30
        walk.folds[1].out_of_sample.trade_count = 30
        report = evaluate(good_metrics(), walk_forward=walk,
                          doubled_slippage=good_metrics())
        result = gate(report, "trade count")
        assert not result.passed
        assert "do not lower the bar" in result.detail

    def test_failing_doubled_slippage_blocks(self):
        report = evaluate(good_metrics(), walk_forward=good_walk_forward(),
                          doubled_slippage=good_metrics(expectancy_r=-0.02))
        assert not gate(report, "doubled slippage").passed

    def test_concentration_in_one_month_blocks(self):
        metrics = good_metrics()
        metrics.by_month = pd.DataFrame([
            {"month": "2026-01", "net_pnl": 140_000.0, "trades": 100, "avg_r": 0.5},
            {"month": "2026-02", "net_pnl": 4_000.0, "trades": 100, "avg_r": 0.01},
            {"month": "2026-03", "net_pnl": 3_000.0, "trades": 100, "avg_r": 0.01},
        ])
        report = evaluate(metrics, walk_forward=good_walk_forward(),
                          doubled_slippage=good_metrics())
        result = gate(report, "one month or instrument")
        assert not result.passed
        assert "month" in result.detail

    def test_negative_walk_forward_expectancy_blocks(self):
        walk = good_walk_forward()
        for fold in walk.folds:
            fold.out_of_sample.expectancy_r = -0.05
        report = evaluate(good_metrics(), walk_forward=walk,
                          doubled_slippage=good_metrics())
        assert not gate(report, "Walk-forward OOS").passed

    def test_large_degradation_blocks(self):
        walk = good_walk_forward()
        for fold in walk.folds:
            fold.in_sample.expectancy_r = 0.40
            fold.out_of_sample.expectancy_r = 0.02
        report = evaluate(good_metrics(), walk_forward=walk,
                          doubled_slippage=good_metrics())
        result = gate(report, "degradation")
        assert not result.passed
        assert "training noise" in result.detail


class TestAdvisoryGates:
    def test_trial_count_is_advisory_not_blocking(self):
        walk = good_walk_forward(total_trials=500)
        report = evaluate(good_metrics(), walk_forward=walk,
                          doubled_slippage=good_metrics(expectancy_r=0.06))
        result = gate(report, "Trial count")
        assert result.advisory
        assert "selection bias" in result.detail
        assert report.passed  # reported, not punished

    def test_bias_gate_flags_market_beta(self):
        """D3: an edge that appears only in up sessions is beta, not skill."""
        metrics = good_metrics()
        metrics.by_session_direction = pd.DataFrame([
            {"session_direction": "up", "net_pnl": 150_000.0, "trades": 200, "avg_r": 0.40},
            {"session_direction": "down", "net_pnl": -30_000.0, "trades": 100, "avg_r": -0.15},
        ])
        report = evaluate(metrics, walk_forward=good_walk_forward(),
                          doubled_slippage=good_metrics(expectancy_r=0.06))
        result = gate(report, "Long-only bias")
        assert result.advisory
        assert "market beta" in result.detail

    def test_missing_bias_breakdown_warns(self):
        metrics = good_metrics()
        metrics.by_session_direction = pd.DataFrame()
        report = evaluate(metrics, walk_forward=good_walk_forward(),
                          doubled_slippage=good_metrics(expectancy_r=0.06))
        result = gate(report, "Long-only bias")
        assert not result.passed
        assert result.advisory


class TestReportFormatting:
    def test_failures_are_listed(self):
        report = evaluate(good_metrics(expectancy_r=-0.05))
        assert "BLOCKED" in report.summary()
        assert len(report.blocking_failures) >= 1

    def test_thresholds_are_stated_as_immovable(self):
        report = evaluate(good_metrics(expectancy_r=-0.05))
        assert "Thresholds do not move" in report.summary()
