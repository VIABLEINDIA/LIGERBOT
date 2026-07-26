"""Failure attribution — *which component* is losing the money.

`metrics.py` already reports `avg_mfe_r`, `avg_mae_r` and a by-exit-reason table. Averages
were not enough. The finding that actually settled the first real-data backtest was
distributional: **no trade reached +1.0R and then finished negative**, which is what ruled
out the exit logic and pointed at the entry. An average MFE of +0.29R is compatible with
both "entries go nowhere" and "exits give winners back", and those demand opposite fixes.

So this module answers one question — *given that the strategy lost money, which part is at
fault* — and the tests are built around the failure modes being **distinguishable**. A
diagnosis that says "something is wrong" is what we already had.

The order of attribution matters and is itself under test:

1. too few trades → inconclusive, before anything else gets asserted
2. net positive → nothing to diagnose
3. gross positive, net negative → **friction**
4. winners reached target then gave it back → **exits**
5. positions never moved in our favour → **entry**
6. immediate full-loss stop-outs → **stops**
"""
from __future__ import annotations

import datetime as dt

import pytest

from src.backtest.attribution import Fault, diagnose
from src.backtest.costs import CostBreakdown
from src.backtest.portfolio import Trade
from src.backtest.sim_broker import FillReason

AT = dt.datetime(2026, 7, 21, 10, 0)


def trade(*, r: float, mfe_r: float = 0.0, mae_r: float = 0.0, bars: int = 10,
          reason: FillReason = FillReason.SIGNAL, risk: float = 10.0,
          costs: float = 0.0, quantity: int = 100,
          slippage_per_share: float = 0.0) -> Trade:
    """A trade with a chosen R outcome and excursion, expressed in R."""
    entry = 1000.0
    return Trade(
        instrument_id="nse_cm:2885", is_long=True, quantity=quantity,
        entry_at=AT, entry_price=entry,
        exit_at=AT + dt.timedelta(minutes=5 * bars),
        exit_price=entry + r * risk,
        exit_reason=reason,
        costs=CostBreakdown(brokerage=costs),
        risk_per_share=risk,
        mfe=mfe_r * risk, mae=mae_r * risk, bars_held=bars,
        entry_slippage_per_share=slippage_per_share,
        exit_slippage_per_share=slippage_per_share,
    )


def losing_book(**overrides):
    """20 trades that lose, with excursion settable to shape the failure mode."""
    defaults = dict(mfe_r=0.2, mae_r=0.5, bars=10)
    defaults.update(overrides)
    return ([trade(r=-1.0, **defaults) for _ in range(15)]
            + [trade(r=1.0, **defaults) for _ in range(5)])


# ---------------------------------------------------------------------------
class TestItRefusesToDiagnoseNoise:
    def test_no_trades_is_inconclusive(self):
        assert diagnose([]).fault is Fault.INCONCLUSIVE

    def test_a_handful_of_trades_is_inconclusive(self):
        """Eight trades produced a -0.14R headline that a wider sample revised to
        -0.27R with a different cause. Small samples get refused, not interpreted."""
        assert diagnose([trade(r=-1.0) for _ in range(5)]).fault is Fault.INCONCLUSIVE

    def test_the_threshold_is_configurable_and_enforced(self):
        book = [trade(r=-1.0, mfe_r=0.05) for _ in range(12)]
        assert diagnose(book, min_trades=20).fault is Fault.INCONCLUSIVE
        assert diagnose(book, min_trades=10).fault is not Fault.INCONCLUSIVE

    def test_inconclusive_still_reports_the_count(self):
        assert diagnose([trade(r=-1.0)]).trade_count == 1


class TestProfitableNeedsNoDiagnosis:
    def test_a_winning_book_is_not_faulted(self):
        book = [trade(r=1.0) for _ in range(15)] + [trade(r=-1.0) for _ in range(5)]
        assert diagnose(book).fault is Fault.NONE

    def test_break_even_is_not_faulted(self):
        book = [trade(r=1.0) for _ in range(10)] + [trade(r=-1.0) for _ in range(10)]
        assert diagnose(book).fault is Fault.NONE


class TestDistinguishingTheFailureModes:
    """The whole point. Each of these books loses the same money for a different reason,
    and a diagnosis that cannot tell them apart is worthless."""

    def test_friction_when_gross_is_positive_but_net_is_not(self):
        """Gross positive, eaten by charges. Fix the trade frequency, not the signal."""
        book = [trade(r=0.05, costs=200.0, risk=10.0, quantity=100)
                for _ in range(20)]
        result = diagnose(book)
        assert result.fault is Fault.FRICTION
        assert "friction" in result.detail.lower()

    def test_exits_when_winners_reach_target_then_give_it_back(self):
        """Reached +1.5R, finished negative. The signal worked; the exit did not."""
        book = [trade(r=-0.8, mfe_r=1.5, mae_r=0.9) for _ in range(20)]
        result = diagnose(book)
        assert result.fault is Fault.EXITS
        assert result.gave_back_count == 20

    def test_entry_when_positions_never_move_in_our_favour(self):
        """The real-data case: median MFE +0.29R against a stop a full R away."""
        book = [trade(r=-1.0, mfe_r=0.1, mae_r=0.6) for _ in range(20)]
        result = diagnose(book)
        assert result.fault is Fault.ENTRY
        assert result.never_moved_pct > 0.9

    def test_stops_when_trades_die_immediately_at_a_full_loss(self):
        """Stopped within a couple of bars, every time — the stop is inside the noise."""
        book = [trade(r=-1.0, mfe_r=0.02, mae_r=1.0, bars=1,
                      reason=FillReason.STOP_LOSS) for _ in range(20)]
        assert diagnose(book).fault is Fault.STOPS

    def test_entry_and_exit_faults_are_not_confused(self):
        """Both books lose. They need opposite fixes."""
        never_moved = [trade(r=-1.0, mfe_r=0.05) for _ in range(20)]
        gave_back = [trade(r=-0.8, mfe_r=1.6) for _ in range(20)]
        assert diagnose(never_moved).fault is not diagnose(gave_back).fault


class TestSlippageIsNotInvisible:
    """A defect found by probing rather than reading, in the first version of this module.

    Slippage is baked into entry and exit prices, so it is already inside `r_multiple`.
    The friction figure originally added back only `costs.total` — the explicit charges —
    which made a strategy killed by *slippage* indistinguishable from one with no edge.

    A book that is **+0.30R frictionless** and −0.30R net was reported as `MIXED` with "no
    single dominant cause", which would send someone to redesign a signal that works.
    That is a worse outcome than no diagnosis: it is a confident wrong answer.
    """

    def test_a_slippage_driven_loss_is_attributed_to_friction(self):
        book = [trade(r=-0.30, costs=1.0, slippage_per_share=3.0, mfe_r=0.9)
                for _ in range(25)]
        result = diagnose(book)
        assert result.fault is Fault.FRICTION, (
            f"got {result.fault.value}: a book profitable before execution cost was "
            f"blamed on something else")

    def test_frictionless_is_reported_separately_from_gross(self):
        """metrics.py distinguishes frictionless (before slippage AND charges) from gross
        (after slippage, before charges). Attribution must use the same vocabulary or the
        two reports contradict each other."""
        book = [trade(r=-0.30, costs=1.0, slippage_per_share=3.0) for _ in range(25)]
        result = diagnose(book)
        assert result.frictionless_r > 0
        assert result.gross_r < 0
        assert result.frictionless_r > result.gross_r

    def test_slippage_appears_in_the_detail(self):
        book = [trade(r=-0.30, costs=1.0, slippage_per_share=3.0) for _ in range(25)]
        assert "slippage" in diagnose(book).detail.lower()

    def test_a_charges_driven_loss_is_attributed_to_friction(self):
        """Profitable on price movement, turned negative by charges alone.

        Note `r` here is the *price* move, and `Trade.r_multiple` subtracts charges from
        it — so the price move has to sit **below** the charges for the net to go negative
        while frictionless stays positive. Charges of 300 on 1,000 of risk are 0.30R, so a
        +0.20R price move nets -0.10R against +0.20R frictionless.
        """
        book = [trade(r=0.20, costs=300.0, mfe_r=0.9) for _ in range(25)]
        result = diagnose(book)
        assert result.net_r < 0
        assert result.frictionless_r > 0
        assert result.fault is Fault.FRICTION

    def test_a_book_that_loses_before_costs_is_not_blamed_on_friction(self):
        """The mirror, and the reason the previous test needed rewriting. A -0.02R price
        move with heavy charges is unprofitable *before* the charges, so friction is not
        the dominant cause however large it looks next to the net figure."""
        book = [trade(r=-0.02, costs=300.0, mfe_r=0.9) for _ in range(25)]
        result = diagnose(book)
        assert result.frictionless_r < 0
        assert result.fault is not Fault.FRICTION

    def test_a_genuinely_edgeless_book_is_not_blamed_on_friction(self):
        """The guard against the fix over-firing: no friction, no edge, must still read
        as ENTRY."""
        book = [trade(r=-1.0, mfe_r=0.05, costs=0.0, slippage_per_share=0.0)
                for _ in range(25)]
        assert diagnose(book).fault is Fault.ENTRY

    def test_zero_slippage_leaves_frictionless_equal_to_gross(self):
        book = [trade(r=-1.0, mfe_r=0.05, costs=0.0) for _ in range(25)]
        result = diagnose(book)
        assert result.frictionless_r == pytest.approx(result.gross_r)


class TestTheArithmeticThatSummarisesIt:
    def test_break_even_win_rate_from_reward_to_risk(self):
        """avg win +0.707R against avg loss -0.593R needs 46%. The single sharpest
        line in the whole autopsy."""
        book = ([trade(r=0.707) for _ in range(9)]
                + [trade(r=-0.593) for _ in range(27)])
        result = diagnose(book)
        assert result.break_even_win_rate == pytest.approx(0.456, abs=0.01)
        assert result.actual_win_rate == pytest.approx(0.25, abs=0.01)

    def test_reward_to_risk_is_reported(self):
        book = [trade(r=2.0) for _ in range(5)] + [trade(r=-1.0) for _ in range(15)]
        assert diagnose(book).reward_risk == pytest.approx(2.0, abs=0.01)

    def test_the_shortfall_is_the_actionable_number(self):
        """How far the win rate is from where it needs to be."""
        book = ([trade(r=1.0) for _ in range(5)]
                + [trade(r=-1.0) for _ in range(15)])
        result = diagnose(book)
        assert result.win_rate_shortfall == pytest.approx(
            result.break_even_win_rate - result.actual_win_rate, abs=1e-9)

    def test_all_losers_does_not_divide_by_zero(self):
        assert diagnose([trade(r=-1.0) for _ in range(20)]).reward_risk == 0.0

    def test_all_winners_does_not_divide_by_zero(self):
        result = diagnose([trade(r=1.0) for _ in range(20)])
        assert result.break_even_win_rate == 0.0
        assert result.fault is Fault.NONE


class TestTheDistributionalFacts:
    def test_never_moved_counts_trades_below_the_threshold(self):
        book = ([trade(r=-1.0, mfe_r=0.05) for _ in range(15)]
                + [trade(r=1.0, mfe_r=1.2) for _ in range(5)])
        assert diagnose(book).never_moved_pct == pytest.approx(0.75)

    def test_gave_back_counts_winners_that_finished_negative(self):
        """The fact that ruled out the exit logic on real data: zero of thirty-six."""
        book = ([trade(r=-0.5, mfe_r=1.4) for _ in range(8)]
                + [trade(r=-1.0, mfe_r=0.1) for _ in range(12)])
        assert diagnose(book).gave_back_count == 8

    def test_a_trade_that_reached_target_and_kept_it_is_not_a_give_back(self):
        book = [trade(r=1.5, mfe_r=1.6) for _ in range(20)]
        assert diagnose(book).gave_back_count == 0

    def test_zero_risk_trades_are_excluded_rather_than_dividing_by_zero(self):
        book = [trade(r=-1.0, mfe_r=0.1) for _ in range(19)]
        book.append(trade(r=-1.0, risk=0.0))
        result = diagnose(book)
        assert result.trade_count == 20
        assert result.excursion_sample == 19


class TestTheReport:
    def test_it_names_the_fault_and_the_evidence(self):
        text = diagnose(losing_book(mfe_r=0.05)).report()
        assert "ENTRY" in text.upper()
        assert "break-even" in text.lower()

    def test_it_says_so_when_there_is_nothing_to_diagnose(self):
        text = diagnose([trade(r=1.0) for _ in range(20)]).report()
        assert "profitable" in text.lower()

    def test_an_inconclusive_report_does_not_pretend(self):
        text = diagnose([trade(r=-1.0) for _ in range(3)]).report()
        assert "too few" in text.lower() or "inconclusive" in text.lower()
        assert "ENTRY" not in text.upper()
