"""Broker order-status parsing (DESIGN.md 3.3) — the second half of the B3 fix.

The field names here (``nOrdNo``, ``ordSt``, ``fldQty``, ``avgPrc``, ``unFldSz``) are
corroborated by two independent working Kotak integrations found on this machine, which
agree on all of them. That is stronger evidence than the ``limits()`` mapping in
``src/account.py``, where four projects were checked and none had identified the fields.

Still not a live confirmation — both source projects flag their Kotak field names as
unverified — so the parser is written to fail visibly rather than silently: unrecognised
statuses map to None instead of being guessed at, and unparseable rows are counted.
"""
from __future__ import annotations

import pytest

from src.order_state import (
    BrokerOrderStatus, OrderStatus, parse_order_report, parse_order_status,
)


def row(**overrides) -> dict:
    payload = {
        "nOrdNo": "EX0001", "ordSt": "complete", "fldQty": "100",
        "avgPrc": "1300.50", "unFldSz": "0",
    }
    payload.update(overrides)
    return payload


class TestParseOrderStatus:
    def test_reads_the_corroborated_field_names(self):
        status = parse_order_status(row())
        assert status.broker_order_id == "EX0001"
        assert status.filled_quantity == 100
        assert status.average_price == pytest.approx(1300.50)
        assert status.pending_quantity == 0

    def test_handles_comma_formatted_prices(self):
        assert parse_order_status(row(avgPrc="1,300.50")).average_price == \
            pytest.approx(1300.50)

    def test_missing_order_id_returns_none(self):
        assert parse_order_status({"ordSt": "complete"}) is None

    def test_non_dict_returns_none(self):
        assert parse_order_status(["nope"]) is None

    def test_blank_numerics_default_to_zero(self):
        status = parse_order_status(row(fldQty="", avgPrc="", unFldSz=""))
        assert status.filled_quantity == 0
        assert status.average_price == 0.0

    def test_rejection_reason_is_captured(self):
        status = parse_order_status(row(ordSt="rejected", rejRsn="insufficient margin"))
        assert status.rejection_reason == "insufficient margin"


class TestStatusMapping:
    @pytest.mark.parametrize("raw,expected", [
        ("complete", OrderStatus.FILLED),
        ("COMPLETE", OrderStatus.FILLED),
        ("Filled", OrderStatus.FILLED),
        ("rejected", OrderStatus.REJECTED),
        ("REJECTED BY EXCHANGE", OrderStatus.REJECTED),
        ("cancelled", OrderStatus.CANCELLED),
        ("partially filled", OrderStatus.FILLED),   # "filled" matches first
        ("open", OrderStatus.ACKED),
        ("pending", OrderStatus.ACKED),
        ("trigger pending", OrderStatus.ACKED),
    ])
    def test_known_statuses_map(self, raw, expected):
        assert parse_order_status(row(ordSt=raw)).mapped_status is expected

    def test_unknown_status_maps_to_none_not_a_guess(self):
        """An unmapped status means the broker said something we don't understand.

        Treating that as FILLED or REJECTED would be worse than surfacing it.
        """
        assert parse_order_status(row(ordSt="some_new_state")).mapped_status is None

    def test_empty_status_maps_to_none(self):
        assert parse_order_status(row(ordSt="")).mapped_status is None


class TestParseOrderReport:
    def test_parses_a_data_wrapped_list(self):
        report = parse_order_report({"data": [row(), row(nOrdNo="EX0002")]})
        assert set(report) == {"EX0001", "EX0002"}

    def test_parses_a_bare_list(self):
        assert set(parse_order_report([row()])) == {"EX0001"}

    def test_empty_response_is_empty(self):
        assert parse_order_report({}) == {}
        assert parse_order_report(None) == {}
        assert parse_order_report([]) == {}

    def test_unparseable_rows_are_counted_not_silently_dropped(self, caplog):
        """An order we fail to read must be loud, not appear absent."""
        with caplog.at_level("ERROR"):
            report = parse_order_report({"data": [row(), {"garbage": True}]})
        assert set(report) == {"EX0001"}
        assert "Could not parse" in caplog.text

    def test_later_row_wins_for_a_duplicate_order_id(self):
        """order_report can carry several updates for one order; the last is current."""
        report = parse_order_report({"data": [
            row(ordSt="open", fldQty="0"),
            row(ordSt="complete", fldQty="100"),
        ]})
        assert report["EX0001"].filled_quantity == 100


class TestPartialFillProgression:
    def test_incremental_fills_are_detectable(self):
        """The engine emits one fill event per increment, not one aggregate.

        A position built from three partials is three fills — this is the arithmetic the
        execution engine relies on to compute what is newly filled.
        """
        seen = [0]
        increments = []
        for filled in (30, 70, 100):
            status = parse_order_status(row(fldQty=str(filled)))
            increments.append(status.filled_quantity - seen[0])
            seen[0] = status.filled_quantity
        assert increments == [30, 40, 30]
        assert sum(increments) == 100

    def test_pending_quantity_complements_filled(self):
        status = parse_order_status(row(fldQty="30", unFldSz="70"))
        assert status.filled_quantity + status.pending_quantity == 100
