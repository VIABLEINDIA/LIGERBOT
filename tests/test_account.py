"""Equity retrieval and session-snapshot tests (DESIGN.md D1)."""
from __future__ import annotations

import datetime as dt

import pytest

from src.account import (
    EquitySnapshot, EquityUnavailable, SessionEquity, fetch_equity, parse_limits,
    validate_snapshot,
)

DAY = dt.date(2026, 7, 23)


class FakeNeo:
    """Stand-in for the Kotak SDK client."""

    def __init__(self, payload=None, error: Exception | None = None):
        self.payload = payload
        self.error = error
        self.calls = 0

    def limits(self):
        self.calls += 1
        if self.error:
            raise self.error
        return self.payload


GOOD_LIMITS = {"CashOpenBal": "500000", "MtoMUnrealized": "2500",
               "MarginUsed": "100000", "CollateralValue": "0"}


class TestParseLimits:
    def test_equity_is_cash_plus_mtm(self):
        snapshot = parse_limits(GOOD_LIMITS)
        assert snapshot.cash == 500_000.0
        assert snapshot.unrealized_mtm == 2_500.0
        assert snapshot.equity == 502_500.0

    def test_equity_excludes_margin(self):
        """The constraint that matters most.

        MIS grants ~5x leverage, so margin far exceeds capital. Sizing off it would
        silently turn a 0.5% risk rule into a 2.5% one.
        """
        payload = {**GOOD_LIMITS, "MarginAvailable": "2500000"}
        snapshot = parse_limits(payload)
        assert snapshot.equity == 502_500.0
        assert snapshot.equity < 2_500_000.0

    def test_handles_comma_formatted_numbers(self):
        assert parse_limits({"CashOpenBal": "5,00,000"}).cash == 500_000.0

    def test_unwraps_nested_payloads(self):
        assert parse_limits({"data": GOOD_LIMITS}).equity == 502_500.0

    def test_unwraps_list_payloads(self):
        assert parse_limits({"data": [GOOD_LIMITS]}).equity == 502_500.0

    def test_missing_cash_field_raises_rather_than_defaulting(self):
        # Fail closed: a zero default sizes everything at zero, a nonzero default sizes
        # everything off a fiction.
        with pytest.raises(EquityUnavailable, match="refusing to guess"):
            parse_limits({"SomethingElse": "123"})

    def test_non_dict_response_raises(self):
        with pytest.raises(EquityUnavailable):
            parse_limits(["unexpected"])

    def test_missing_mtm_defaults_to_zero(self):
        assert parse_limits({"CashOpenBal": "100000"}).unrealized_mtm == 0.0


class TestValidation:
    def test_rejects_zero_equity(self):
        snapshot = EquitySnapshot(equity=0.0, cash=0.0, unrealized_mtm=0.0)
        with pytest.raises(EquityUnavailable, match="refusing to trade"):
            validate_snapshot(snapshot)

    def test_rejects_negative_equity(self):
        snapshot = EquitySnapshot(equity=-100.0, cash=-100.0, unrealized_mtm=0.0)
        with pytest.raises(EquityUnavailable):
            validate_snapshot(snapshot)

    def test_rejects_equity_below_the_floor(self):
        snapshot = EquitySnapshot(equity=50_000.0, cash=50_000.0, unrealized_mtm=0.0)
        with pytest.raises(EquityUnavailable, match="below the configured floor"):
            validate_snapshot(snapshot, min_equity=200_000.0)

    def test_rejects_an_implausible_overnight_jump(self):
        # More likely a parsing bug than a real account event.
        snapshot = EquitySnapshot(equity=5_000_000.0, cash=5_000_000.0, unrealized_mtm=0.0)
        with pytest.raises(EquityUnavailable, match="sanity bound"):
            validate_snapshot(snapshot, previous_equity=500_000.0)

    def test_accepts_a_normal_day_on_day_change(self):
        snapshot = EquitySnapshot(equity=510_000.0, cash=510_000.0, unrealized_mtm=0.0)
        validate_snapshot(snapshot, previous_equity=500_000.0)  # must not raise


class TestFetch:
    def test_fetches_and_parses(self):
        assert fetch_equity(FakeNeo(GOOD_LIMITS)).equity == 502_500.0

    def test_sdk_exception_becomes_equity_unavailable(self):
        # One failure mode for callers to handle, so partial results can't slip through.
        with pytest.raises(EquityUnavailable, match="limits\\(\\) call failed"):
            fetch_equity(FakeNeo(error=ConnectionError("network down")))


class TestSessionEquity:
    def test_fetches_once_and_pins(self, tmp_path):
        neo = FakeNeo(GOOD_LIMITS)
        session = SessionEquity(tmp_path / "equity.json")
        first = session.resolve(DAY, neo)
        second = session.resolve(DAY, neo)
        assert first == second
        assert neo.calls == 1  # pinned, not refetched

    def test_restart_within_a_session_reuses_the_pinned_base(self, tmp_path):
        """The reason the snapshot is persisted.

        A module restarting at 11:00 must size on the same base it used at 09:15,
        otherwise positions opened after the restart are sized differently from those
        before it.
        """
        path = tmp_path / "equity.json"
        SessionEquity(path).resolve(DAY, FakeNeo(GOOD_LIMITS))

        # Simulate a restart: fresh object, and the account has since moved.
        moved = FakeNeo({"CashOpenBal": "450000", "MtoMUnrealized": "-30000"})
        restored = SessionEquity(path).resolve(DAY, moved)
        assert restored.equity == 502_500.0
        assert moved.calls == 0  # never even asked

    def test_new_day_refetches(self, tmp_path):
        path = tmp_path / "equity.json"
        SessionEquity(path).resolve(DAY, FakeNeo(GOOD_LIMITS))
        next_day = dt.date(2026, 7, 24)
        neo = FakeNeo({"CashOpenBal": "520000", "MtoMUnrealized": "0"})
        assert SessionEquity(path).resolve(next_day, neo).equity == 520_000.0
        assert neo.calls == 1

    def test_broker_failure_propagates_rather_than_defaulting(self, tmp_path):
        session = SessionEquity(tmp_path / "equity.json")
        with pytest.raises(EquityUnavailable):
            session.resolve(DAY, FakeNeo(error=TimeoutError("no response")))
        assert session.current is None

    def test_floor_is_enforced_on_resolve(self, tmp_path):
        session = SessionEquity(tmp_path / "equity.json", min_equity=200_000.0)
        with pytest.raises(EquityUnavailable, match="below the configured floor"):
            session.resolve(DAY, FakeNeo({"CashOpenBal": "50000"}))

    def test_manual_override_for_simulation(self, tmp_path):
        session = SessionEquity(tmp_path / "equity.json")
        snapshot = session.set_manual(DAY, 1_000_000.0, reason="backtest")
        assert snapshot.equity == 1_000_000.0
        assert snapshot.source == "backtest"
        # And it pins like a fetched one.
        assert session.resolve(DAY, FakeNeo(error=RuntimeError("must not be called"))) == snapshot

    def test_last_known_equity_survives_across_instances(self, tmp_path):
        path = tmp_path / "equity.json"
        SessionEquity(path).resolve(DAY, FakeNeo(GOOD_LIMITS))
        assert SessionEquity(path).last_known_equity() == 502_500.0

    def test_unreadable_state_file_does_not_crash(self, tmp_path):
        path = tmp_path / "equity.json"
        path.write_text("{corrupt", encoding="utf-8")
        session = SessionEquity(path)
        assert session.last_known_equity() is None
        assert session.resolve(DAY, FakeNeo(GOOD_LIMITS)).equity == 502_500.0
