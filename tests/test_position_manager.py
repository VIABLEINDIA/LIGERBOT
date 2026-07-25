"""Position book and reconciliation tests (DESIGN.md 3.2) — the fix for B2 and B3."""
from __future__ import annotations

import datetime as dt

import pytest

from src.position_manager import PositionBook, parse_broker_positions

DAY = dt.date(2026, 7, 23)


@pytest.fixture
def book() -> PositionBook:
    b = PositionBook()
    b.start_session(DAY)
    return b


class TestFillApplication:
    def test_opening_fill_creates_a_position(self, book):
        realized = book.apply_fill("nse_cm:1", 100, 1300.0, stop_loss=1274.0)
        assert realized == 0.0
        position = book.positions["nse_cm:1"]
        assert position.quantity == 100
        assert position.average_price == 1300.0
        assert position.risk_amount == pytest.approx(100 * 26.0)

    def test_closing_fill_realises_pnl(self, book):
        """B2 in one test: something must actually write realized_pnl_today."""
        book.apply_fill("nse_cm:1", 100, 1300.0)
        realized = book.apply_fill("nse_cm:1", -100, 1310.0)
        assert realized == pytest.approx(1000.0)
        assert book.realized_pnl_today == pytest.approx(1000.0)
        assert "nse_cm:1" not in book.positions

    def test_losing_trade_moves_pnl_negative(self, book):
        book.apply_fill("nse_cm:1", 100, 1300.0)
        book.apply_fill("nse_cm:1", -100, 1290.0)
        assert book.realized_pnl_today == pytest.approx(-1000.0)

    def test_adding_to_a_position_averages_the_entry(self, book):
        book.apply_fill("nse_cm:1", 100, 1300.0)
        book.apply_fill("nse_cm:1", 100, 1310.0)
        position = book.positions["nse_cm:1"]
        assert position.quantity == 200
        assert position.average_price == pytest.approx(1305.0)

    def test_partial_close_leaves_the_remainder(self, book):
        book.apply_fill("nse_cm:1", 100, 1300.0)
        realized = book.apply_fill("nse_cm:1", -40, 1310.0)
        assert realized == pytest.approx(400.0)
        assert book.positions["nse_cm:1"].quantity == 60

    def test_reversal_through_zero_opens_the_other_side(self, book):
        """The case a naive 'close the position' implementation loses."""
        book.apply_fill("nse_cm:1", 100, 1300.0)
        realized = book.apply_fill("nse_cm:1", -150, 1310.0)
        assert realized == pytest.approx(1000.0)      # only the 100 that closed
        position = book.positions["nse_cm:1"]
        assert position.quantity == -50               # the excess went short
        assert position.average_price == 1310.0

    def test_short_round_trip(self, book):
        book.apply_fill("nse_cm:1", -100, 1300.0)
        realized = book.apply_fill("nse_cm:1", 100, 1290.0)
        assert realized == pytest.approx(1000.0)

    def test_costs_accumulate_and_reduce_net(self, book):
        book.apply_fill("nse_cm:1", 100, 1300.0, costs=40.0)
        book.apply_fill("nse_cm:1", -100, 1310.0, costs=45.0)
        assert book.realized_pnl_today == pytest.approx(1000.0)
        assert book.costs_today == pytest.approx(85.0)
        # The breaker must see the figure after costs.
        assert book.net_pnl_today() == pytest.approx(915.0)

    def test_session_reset_clears_daily_totals_not_positions(self, book):
        book.apply_fill("nse_cm:1", 100, 1300.0)
        book.apply_fill("nse_cm:1", -50, 1310.0)
        book.start_session(dt.date(2026, 7, 24))
        assert book.realized_pnl_today == 0.0
        assert "nse_cm:1" in book.positions  # overnight state is not P&L


class TestRiskViews:
    def test_open_risk_sums_across_positions(self, book):
        book.apply_fill("nse_cm:1", 100, 1300.0, stop_loss=1274.0)
        book.apply_fill("nse_cm:2", 50, 1650.0, stop_loss=1617.0)
        assert book.total_open_risk() == pytest.approx(100 * 26.0 + 50 * 33.0)

    def test_position_without_a_stop_contributes_no_measurable_risk(self, book):
        # An adopted position from reconciliation has no known stop; it must not be
        # silently counted as risk-free in a way that hides it either.
        book.apply_fill("nse_cm:1", 100, 1300.0, stop_loss=0.0)
        assert book.total_open_risk() == 0.0
        assert book.gross_exposure() == pytest.approx(130_000.0)

    def test_unrealized_tracks_the_mark(self, book):
        book.apply_fill("nse_cm:1", 100, 1300.0)
        assert book.positions["nse_cm:1"].unrealized(1310.0) == pytest.approx(1000.0)

    def test_snapshot_carries_what_the_risk_manager_needs(self, book):
        book.apply_fill("nse_cm:1", 100, 1300.0, stop_loss=1274.0, costs=40.0)
        snapshot = book.snapshot()
        assert snapshot["open_positions"] == 1
        assert snapshot["costs_today"] == 40.0
        assert "net_pnl_today" in snapshot
        assert snapshot["positions"][0]["instrument_id"] == "nse_cm:1"


class TestReconciliation:
    def test_matching_state_reconciles_clean(self, book):
        book.apply_fill("nse_cm:1", 100, 1300.0)
        result = book.reconcile({"nse_cm:1": 100})
        assert result.clean
        assert result.matched == ["nse_cm:1"]

    def test_broker_wins_on_a_quantity_mismatch(self, book):
        book.apply_fill("nse_cm:1", 100, 1300.0)
        result = book.reconcile({"nse_cm:1": 60})
        assert not result.clean
        assert "nse_cm:1" in result.mismatched
        assert book.positions["nse_cm:1"].quantity == 60

    def test_position_the_broker_does_not_have_is_dropped(self, book):
        book.apply_fill("nse_cm:1", 100, 1300.0)
        result = book.reconcile({})
        assert "nse_cm:1" in result.missing_at_broker
        assert "nse_cm:1" not in book.positions

    def test_unknown_broker_position_is_adopted(self, book):
        """A position we did not know about is the most dangerous discrepancy."""
        result = book.reconcile({"nse_cm:9": 25})
        assert "nse_cm:9" in result.missing_locally
        assert book.positions["nse_cm:9"].quantity == 25
        assert book.positions["nse_cm:9"].average_price == 0.0  # unknown, not invented

    def test_broker_flat_removes_the_local_position(self, book):
        book.apply_fill("nse_cm:1", 100, 1300.0)
        book.reconcile({"nse_cm:1": 0})
        assert "nse_cm:1" not in book.positions

    def test_discrepancy_count_drives_the_halt_decision(self, book):
        book.apply_fill("nse_cm:1", 100, 1300.0)
        book.apply_fill("nse_cm:2", 50, 1650.0)
        result = book.reconcile({"nse_cm:1": 60})
        assert result.discrepancy_count == 2  # one mismatched, one missing at broker

    def test_details_explain_each_discrepancy(self, book):
        book.apply_fill("nse_cm:1", 100, 1300.0)
        result = book.reconcile({"nse_cm:1": 60})
        assert any("local 100" in d and "broker 60" in d for d in result.details)


class TestBrokerParsing:
    def test_parses_net_quantity_from_buy_and_sell(self):
        payload = {"data": [
            {"tok": "2885", "exSeg": "nse_cm", "flBuyQty": "100", "flSellQty": "40"},
        ]}
        assert parse_broker_positions(payload) == {"nse_cm:2885": 60}

    def test_handles_a_bare_list(self):
        payload = [{"tok": "1594", "exSeg": "nse_cm", "flBuyQty": "0", "flSellQty": "30"}]
        assert parse_broker_positions(payload) == {"nse_cm:1594": -30}

    def test_unparseable_rows_are_skipped_not_treated_as_flat(self, caplog):
        """A position we fail to parse must be loud, not silently absent."""
        payload = {"data": [{"garbage": True}, {"tok": "1", "flBuyQty": "x"}]}
        with caplog.at_level("ERROR"):
            assert parse_broker_positions(payload) == {}
        assert "Could not parse" in caplog.text

    def test_empty_response_is_empty(self):
        assert parse_broker_positions({}) == {}
        assert parse_broker_positions(None) == {}
