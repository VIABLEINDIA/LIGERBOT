"""Correlation-filter tests.

Closes a gap the design carried from the start: DESIGN.md D2 *assumes* NSE large caps
move together and sizes for the worst case accordingly — but nothing stopped the bot
holding three bank stocks and calling it three independent positions. The open-risk cap
bounded the total while quietly mislabelling the concentration.
"""
from __future__ import annotations

import datetime as dt

import pytest

from src.instruments import (
    Instrument, assign_correlation_groups, correlation_group_for,
)
from src.risk_engine import Intent, RiskEngine, RiskLimits, Signal

DAY = dt.date(2026, 7, 23)
BAR_TIME = dt.datetime(2026, 7, 23, 10, 30)
EQUITY = 1_000_000.0


def engine(**overrides) -> RiskEngine:
    e = RiskEngine(RiskLimits(**overrides))
    e.start_session(DAY, EQUITY)
    return e


def open_long(instrument: str) -> Signal:
    return Signal(instrument_id=instrument, intent=Intent.OPEN_LONG,
                  ref_price=100.0, stop_loss=99.0, bar_time=BAR_TIME)


def take(e: RiskEngine, instrument: str, group: str):
    decision = e.evaluate(open_long(instrument), allows_entry=True,
                          allows_exit=True, correlation_group=group)
    if decision.approved:
        e.on_open_fill(instrument, decision.order.quantity, 100.0, 99.0,
                       correlation_group=group)
    return decision


class TestGroupMapping:
    def test_known_symbols_map_to_sectors(self):
        assert correlation_group_for("HDFCBANK") == "banking"
        assert correlation_group_for("INFY") == "it"
        assert correlation_group_for("RELIANCE") == "energy"

    def test_eq_suffix_is_stripped(self):
        """Scrip-master symbols arrive as e.g. HDFCBANK-EQ."""
        assert correlation_group_for("HDFCBANK-EQ") == "banking"

    def test_case_insensitive(self):
        assert correlation_group_for("hdfcbank") == "banking"

    def test_unknown_symbol_is_ungrouped(self):
        assert correlation_group_for("SOMETHINGELSE") == ""

    def test_index_etfs_group_together(self):
        # An index ETF alongside a basket is the same exposure twice.
        assert correlation_group_for("NIFTYBEES") == "index_etf"
        assert correlation_group_for("BANKBEES") == "index_etf"

    def test_assign_populates_instruments(self):
        instruments = [
            Instrument("nse_cm:1", "1", "HDFCBANK-EQ", "HDFCBANK", "nse_cm", "EQ", 1, 0.05),
            Instrument("nse_cm:2", "2", "INFY-EQ", "INFY", "nse_cm", "EQ", 1, 0.05),
        ]
        assigned = assign_correlation_groups(instruments)
        assert [i.correlation_group for i in assigned] == ["banking", "it"]

    def test_existing_group_is_not_overwritten(self):
        instrument = Instrument("nse_cm:1", "1", "HDFCBANK-EQ", "HDFCBANK", "nse_cm",
                                "EQ", 1, 0.05, correlation_group="custom")
        assert assign_correlation_groups([instrument])[0].correlation_group == "custom"


class TestFilterBehaviour:
    def test_second_position_in_a_group_is_refused(self):
        """Three banks is one bet of triple the size wearing a diversified label."""
        e = engine()
        assert take(e, "nse_cm:hdfc", "banking").approved
        decision = take(e, "nse_cm:icici", "banking")
        assert not decision.approved
        assert "correlated group" in decision.reason
        assert "stop out together" in decision.reason

    def test_different_groups_are_allowed(self):
        e = engine()
        assert take(e, "nse_cm:hdfc", "banking").approved
        assert take(e, "nse_cm:infy", "it").approved
        assert take(e, "nse_cm:ril", "energy").approved
        assert len(e.positions) == 3

    def test_ungrouped_instruments_are_unconstrained(self):
        """An empty group must not become a bucket that limits everything at once."""
        e = engine()
        assert take(e, "nse_cm:a", "").approved
        assert take(e, "nse_cm:b", "").approved
        assert take(e, "nse_cm:c", "").approved

    def test_limit_is_configurable(self):
        e = engine(max_positions_per_group=2)
        assert take(e, "nse_cm:hdfc", "banking").approved
        assert take(e, "nse_cm:icici", "banking").approved
        assert not take(e, "nse_cm:sbin", "banking").approved

    def test_filter_can_be_disabled(self):
        e = engine(max_positions_per_group=0)
        assert take(e, "nse_cm:hdfc", "banking").approved
        # With the limit at zero the check is skipped entirely, not made absolute.
        second = e.evaluate(open_long("nse_cm:icici"), allows_entry=True,
                            allows_exit=True, correlation_group="")
        assert second.approved

    def test_closing_frees_the_group_slot(self):
        e = engine()
        take(e, "nse_cm:hdfc", "banking")
        e.on_close_fill("nse_cm:hdfc", 99.0)
        assert take(e, "nse_cm:icici", "banking").approved

    def test_positions_in_group_counts_correctly(self):
        e = engine(max_positions_per_group=5)
        take(e, "nse_cm:hdfc", "banking")
        take(e, "nse_cm:icici", "banking")
        take(e, "nse_cm:infy", "it")
        assert e.positions_in_group("banking") == 2
        assert e.positions_in_group("it") == 1
        assert e.positions_in_group("") == 0

    def test_group_is_recorded_on_the_order(self):
        e = engine()
        decision = e.evaluate(open_long("nse_cm:hdfc"), allows_entry=True,
                              allows_exit=True, correlation_group="banking")
        assert decision.order.correlation_group == "banking"

    def test_exits_are_unaffected_by_the_filter(self):
        """The filter constrains new risk only — it must never trap a position."""
        e = engine()
        take(e, "nse_cm:hdfc", "banking")
        close = Signal(instrument_id="nse_cm:hdfc", intent=Intent.CLOSE_LONG,
                       ref_price=101.0, bar_time=BAR_TIME)
        assert e.evaluate(close, allows_entry=True, allows_exit=True,
                          correlation_group="banking").approved


class TestInteractionWithOtherCaps:
    def test_open_risk_cap_still_binds_across_groups(self):
        """The filter is additional to the risk cap, not a replacement for it."""
        e = engine()
        take(e, "nse_cm:hdfc", "banking")
        take(e, "nse_cm:infy", "it")
        take(e, "nse_cm:ril", "energy")
        assert e.total_open_risk_pct() == pytest.approx(0.015)
        # A fourth in a fresh group is refused by the position cap, not the filter.
        decision = take(e, "nse_cm:sun", "pharma")
        assert not decision.approved
        assert "correlated group" not in decision.reason

    def test_default_config_enables_the_filter(self):
        import config

        assert config.risk_limits().max_positions_per_group == 1
