"""Phase 4 gates — paper trading to live capital.

Deliberately separate from the §2.6 gates, and the distinction is the whole point. The
§2.6 gates ask *"does the strategy have an edge in backtest"*. These ask *"does the live
system reproduce that edge"*.

A strategy can pass the first and fail the second, and **that failure is the single most
valuable signal in the project** — it means the model is wrong somewhere, and the backtest
that justified the strategy was measuring something the live system does not do. Finding
that out with paper money is the entire reason Phase 4 exists.

So what is tested here is not arithmetic. It is that the gate **refuses by default and
refuses on each condition independently** — because a gate that passes when a required
input is simply missing is worse than no gate, and a gate whose failures mask each other
hides how far away the system actually is.
"""
from __future__ import annotations

import pytest

from src.backtest.gates import evaluate_phase4


class FakeReconciliation:
    def __init__(self, *, passed=True, failures=(), unexplained=0.0,
                 days_compared=20, pnl_divergence=0.0, match_rate=1.0):
        self.passed = passed
        self.failures = list(failures)
        self.unexplained = unexplained
        self.days_compared = days_compared
        self.pnl_divergence = pnl_divergence
        self.match_rate = match_rate


def report(**overrides):
    fields = dict(reconciliation=FakeReconciliation(), sessions_completed=20,
                  halted_sessions=0)
    fields.update(overrides)
    return evaluate_phase4(fields.pop("reconciliation"), **fields)


class TestItRefusesByDefault:
    def test_a_clean_run_passes(self):
        assert report().passed

    def test_no_sessions_blocks(self):
        """Zero paper sessions is the state every fresh clone is in. It must not pass."""
        assert not report(sessions_completed=0).passed

    def test_too_few_sessions_blocks(self):
        assert not report(sessions_completed=19).passed

    def test_exactly_the_requirement_passes(self):
        assert report(sessions_completed=20).passed

    def test_a_missing_reconciliation_blocks(self):
        """Absent evidence is not evidence. A gate that passes when the input is simply
        missing is worse than no gate, because it looks like a check happened."""
        assert not report(reconciliation=None).passed

    def test_a_failed_reconciliation_blocks(self):
        assert not report(
            reconciliation=FakeReconciliation(passed=False,
                                              failures=["pnl divergence too large"])
        ).passed

    def test_the_failure_reason_is_carried(self):
        result = report(reconciliation=FakeReconciliation(
            passed=False, failures=["match rate 0.42 below 0.80"]))
        assert any("0.42" in r.detail for r in result.blocking_failures)


class TestUnexplainedDivergence:
    """The residual after every known cause is attributed. It is the part that says the
    model is wrong in a way nobody has named yet — which is exactly the thing that must
    not be waved through."""

    def test_a_small_residual_passes(self):
        # The bar is `< max(1.0, |divergence| * 0.2)`, so 1.0 exactly is a failure.
        assert report(reconciliation=FakeReconciliation(unexplained=0.5)).passed

    def test_the_tolerance_scales_with_the_divergence(self):
        """A large divergence that is fully attributed is a diagnosis. A small one that
        is not attributed is a mystery, and the mystery is the problem."""
        assert report(reconciliation=FakeReconciliation(
            pnl_divergence=10_000.0, unexplained=500.0)).passed
        assert not report(reconciliation=FakeReconciliation(
            pnl_divergence=100.0, unexplained=500.0)).passed

    def test_a_large_residual_blocks(self):
        assert not report(
            reconciliation=FakeReconciliation(unexplained=1_000_000.0)).passed

    def test_the_sign_does_not_matter(self):
        """Paper beating backtest is just as much a modelling error as the reverse, and
        is more dangerous because it looks like good news."""
        assert not report(
            reconciliation=FakeReconciliation(unexplained=-1_000_000.0)).passed


class TestHaltRate:
    """Advisory, not blocking — and that is the right call. A halt is the system working;
    a high *rate* says the risk limits and the strategy's trade frequency may not fit each
    other, which is a design conversation rather than a disqualification."""

    def test_no_halts_produces_no_warning(self):
        assert report(halted_sessions=0).warnings == []

    def test_a_high_halt_rate_warns_without_blocking(self):
        result = report(sessions_completed=20, halted_sessions=8)
        assert result.passed, "the halt rate must not block on its own"
        assert any("incompatible" in w.detail for w in result.warnings)

    def test_an_occasional_halt_is_tolerated_silently(self):
        assert report(sessions_completed=20, halted_sessions=1).warnings == []


class TestReporting:
    def test_each_condition_fails_independently(self):
        """Masked failures hide how far away the system actually is."""
        result = report(sessions_completed=0, halted_sessions=0,
                        reconciliation=FakeReconciliation(passed=False,
                                                          failures=["divergence"]))
        names = {r.name for r in result.blocking_failures}
        assert len(names) >= 2, f"failures collapsed into one: {names}"

    def test_the_report_renders_and_names_phase_four(self):
        text = report().summary()
        assert "PHASE 4" in text.upper()

    def test_a_blocked_report_is_unambiguous(self):
        text = report(sessions_completed=0).summary().upper()
        assert "BLOCK" in text or "FAIL" in text

    def test_it_is_distinct_from_the_section_two_six_gates(self):
        assert "live capital" in report().context.lower()
