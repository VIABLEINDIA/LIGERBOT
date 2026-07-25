"""Cost and slippage model tests.

These pin the numbers DESIGN.md 5.2 relies on. If they drift, the go/no-go hurdle for
every strategy drifts with them.
"""
from __future__ import annotations

import pytest

from src.backtest.costs import CostBreakdown, CostModel, Leg, SlippageModel


@pytest.fixture
def model() -> CostModel:
    return CostModel()


class TestBrokerage:
    def test_percentage_applies_on_small_turnover(self, model):
        assert model.brokerage_for(30_000) == pytest.approx(9.0)  # 0.03%

    def test_flat_cap_applies_on_large_turnover(self, model):
        assert model.brokerage_for(200_000) == pytest.approx(20.0)

    def test_crossover_is_where_pct_reaches_the_flat_fee(self, model):
        # 0.03% of 66,667 ~= Rs 20.
        assert model.brokerage_for(66_000) < 20.0
        assert model.brokerage_for(67_000) == pytest.approx(20.0)

    def test_pure_percentage_mode(self):
        pure = CostModel(brokerage_is_min_of_both=False)
        assert pure.brokerage_for(1_000_000) == pytest.approx(300.0)


class TestLegCharges:
    def test_stt_applies_only_to_the_sell_leg(self, model):
        """The asymmetry that makes a naive 'both legs' model wrong."""
        buy = model.charge_leg(Leg.BUY, 100, 1_000.0)
        sell = model.charge_leg(Leg.SELL, 100, 1_000.0)
        assert buy.stt == 0.0
        assert sell.stt == pytest.approx(100_000 * 0.00025)

    def test_stamp_duty_applies_only_to_the_buy_leg(self, model):
        assert model.charge_leg(Leg.BUY, 100, 1_000.0).stamp_duty > 0
        assert model.charge_leg(Leg.SELL, 100, 1_000.0).stamp_duty == 0.0

    def test_gst_excludes_stt_and_stamp_duty(self, model):
        leg = model.charge_leg(Leg.SELL, 100, 1_000.0)
        expected = (leg.brokerage + leg.exchange_txn + leg.sebi) * 0.18
        assert leg.gst == pytest.approx(expected)

    def test_zero_quantity_costs_nothing(self, model):
        assert model.charge_leg(Leg.BUY, 0, 1_000.0).total == 0.0

    def test_all_components_present_on_a_round_trip(self, model):
        breakdown = model.round_trip(1_000, 100.0, 100.0)
        for component in ("brokerage", "stt", "exchange_txn", "gst", "sebi", "stamp_duty"):
            assert getattr(breakdown, component) > 0, component


class TestBreakdownArithmetic:
    def test_total_sums_components(self):
        breakdown = CostBreakdown(brokerage=20, stt=25, exchange_txn=3,
                                  gst=4, sebi=0.2, stamp_duty=3)
        assert breakdown.total == pytest.approx(55.2)

    def test_addition_combines_legs(self):
        combined = (CostBreakdown(brokerage=20, stt=25)
                    + CostBreakdown(brokerage=20, stamp_duty=3))
        assert combined.brokerage == 40
        assert combined.stt == 25
        assert combined.stamp_duty == 3


class TestDesignDocFigures:
    """The numbers DESIGN.md 5.2 quotes must actually come out of the model."""

    def test_round_trip_matches_the_documented_formula(self, model):
        # DESIGN.md 5.2: round trip ~= Rs 47 + 0.085% of notional (incl. slippage).
        notional = 312_500.0
        quantity = int(notional / 100.0)
        costs = model.round_trip(quantity, 100.0, 100.0).total
        slippage = SlippageModel().slippage_for(100.0) * quantity * 2
        documented = 47 + 0.00085 * notional
        assert costs + slippage == pytest.approx(documented, rel=0.05)

    def test_cost_drag_lands_in_the_documented_band(self, model):
        # 11-15% of the amount risked, for equity between Rs 2L and Rs 10L.
        for equity in (200_000.0, 500_000.0, 1_000_000.0):
            drag = model.cost_as_pct_of_risk(equity, 0.005, 0.008)
            slippage_drag = 0.05  # ~5% of risk, added separately
            assert 0.05 < drag < 0.16, f"equity {equity}: drag {drag:.2%}"

    def test_drag_falls_as_equity_rises(self, model):
        """The flat brokerage component dominates at small size."""
        small = model.cost_as_pct_of_risk(200_000.0, 0.005, 0.008)
        large = model.cost_as_pct_of_risk(5_000_000.0, 0.005, 0.008)
        assert small > large

    def test_small_account_drag_justifies_the_equity_floor(self, model):
        # Why MIN_EQUITY is Rs 2L: below it, costs eat any plausible edge.
        tiny = model.cost_as_pct_of_risk(100_000.0, 0.005, 0.008)
        assert tiny > 0.10

    def test_breakeven_r_is_the_cost_drag(self, model):
        equity, risk, stop = 500_000.0, 0.005, 0.008
        assert model.breakeven_r_multiple(equity, risk, stop) == pytest.approx(
            model.cost_as_pct_of_risk(equity, risk, stop))


class TestSlippage:
    def test_buys_fill_higher_and_sells_lower(self):
        slippage = SlippageModel(slippage_bps=10.0)
        assert slippage.apply(100.0, is_buy=True) > 100.0
        assert slippage.apply(100.0, is_buy=False) < 100.0

    def test_half_spread_is_a_floor(self):
        slippage = SlippageModel(slippage_bps=0.1, half_spread_bps=5.0)
        # You always cross the spread, however tight the modelled slippage.
        assert slippage.slippage_for(100.0) == pytest.approx(100.0 * 5.0 / 10_000)

    def test_session_edges_are_worse(self):
        """Modelling the open and close like mid-session flatters edge-heavy strategies."""
        slippage = SlippageModel(slippage_bps=10.0, open_close_multiplier=2.0)
        middle = slippage.slippage_for(100.0, minutes_from_edge=120)
        edge = slippage.slippage_for(100.0, minutes_from_edge=5)
        assert edge == pytest.approx(middle * 2)

    def test_scaled_doubles_for_the_sensitivity_run(self):
        # DESIGN.md 2.5 rule 6: a strategy that survives only at optimistic slippage is
        # not deployable.
        doubled = SlippageModel(slippage_bps=2.5).scaled(2.0)
        assert doubled.slippage_bps == 5.0

    def test_describe_records_provenance(self):
        assert "bps" in SlippageModel().describe()
        assert "brokerage" in CostModel().describe()
