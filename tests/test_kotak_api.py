"""Guards around neo_api_client's actual behaviour.

Every case here corresponds to something the SDK genuinely does, found by cross-referencing
a working integration (``D:\\JEANS``). Two of them were live bugs in this codebase.

The one that mattered most: ``positions()`` returns ``None`` on failure rather than raising,
and the old code turned that into an empty mapping — which reconciliation read as *"the
broker holds nothing"* and used to discard every open position from the book.
"""
from __future__ import annotations

import datetime as dt
import socket

import pytest

from src import kotak_api
from src.kotak_api import (
    KotakAPIError, KotakSessionExpired, rows_from, safe_call, validate,
)


class FakeClient:
    def __init__(self, response=None, raises=None):
        self.response = response
        self.raises = raises
        self.calls = 0

    def positions(self, *a, **kw):
        self.calls += 1
        if self.raises:
            raise self.raises
        return self.response


class TestNoneIsFailureNotEmpty:
    """The dangerous case. None must never read as 'nothing there'."""

    def test_none_raises(self):
        with pytest.raises(KotakAPIError, match="returned None"):
            validate(None, call="positions()")

    def test_none_raises_even_when_empty_is_allowed(self):
        """allow_empty permits [] — an account really can hold nothing. Never None."""
        with pytest.raises(KotakAPIError, match="returned None"):
            validate(None, call="positions()", allow_empty=True)

    def test_error_message_explains_the_consequence(self):
        with pytest.raises(KotakAPIError, match="discard real state"):
            validate(None, call="positions()")

    def test_safe_call_converts_a_none_return(self):
        with pytest.raises(KotakAPIError):
            safe_call(FakeClient(response=None), "positions", allow_empty=True)


class TestErrorPayloads:
    """The SDK catches internally and returns errors as data."""

    @pytest.mark.parametrize("key", ["Error", "error", "errMsg", "errorMessage"])
    def test_error_keys_raise(self, key):
        with pytest.raises(KotakAPIError, match="error payload"):
            validate({key: "something broke"}, call="place_order()")

    def test_bad_stat_raises(self):
        # No errMsg here, so the stat check is what must catch it.
        with pytest.raises(KotakAPIError, match="stat="):
            validate({"stat": "Not_Ok", "nOrdNo": "EX1"}, call="place_order()")

    def test_errmsg_takes_precedence_over_stat(self):
        """When both are present the explicit message is the more useful error."""
        with pytest.raises(KotakAPIError, match="rejected"):
            validate({"stat": "Not_Ok", "errMsg": "rejected"}, call="place_order()")

    def test_ok_stat_passes(self):
        payload = {"stat": "Ok", "nOrdNo": "EX1"}
        assert validate(payload, call="place_order()") is payload

    def test_falsy_error_value_is_not_an_error(self):
        payload = {"Error": "", "data": [{"a": 1}]}
        assert validate(payload, call="positions()") is payload


class TestSessionExpiry:
    def test_the_sdk_sentinel_is_detected(self):
        with pytest.raises(KotakSessionExpired):
            validate({"message": kotak_api.SESSION_EXPIRED_MARKER}, call="positions()")

    def test_expiry_is_distinguishable_from_a_generic_error(self):
        """Retrying an expired session fails identically forever; re-login is needed."""
        with pytest.raises(KotakSessionExpired) as exc:
            validate(kotak_api.SESSION_EXPIRED_MARKER, call="positions()")
        assert isinstance(exc.value, KotakAPIError)   # still catchable as the base

    def test_detected_case_insensitively_and_when_nested(self):
        with pytest.raises(KotakSessionExpired):
            validate({"data": {"msg": kotak_api.SESSION_EXPIRED_MARKER.upper()}},
                     call="limits()")


class TestEmptyHandling:
    def test_empty_raises_unless_explicitly_allowed(self):
        with pytest.raises(KotakAPIError, match="allow_empty=True"):
            validate([], call="order_report()")

    def test_empty_allowed_when_requested(self):
        assert validate([], call="positions()", allow_empty=True) == []

    def test_the_distinction_must_be_explicit_at_the_call_site(self):
        # An account with no positions is legitimate; an empty order_report while we
        # believe orders are working is not. Forcing the flag makes that a decision.
        assert validate({}, call="positions()", allow_empty=True) == {}


class TestRowsFrom:
    def test_unwraps_the_data_envelope(self):
        assert rows_from({"data": [{"a": 1}, {"b": 2}]}, call="x()") == [{"a": 1}, {"b": 2}]

    def test_accepts_a_bare_list(self):
        assert rows_from([{"a": 1}], call="x()") == [{"a": 1}]

    def test_wraps_a_single_dict_payload(self):
        assert rows_from({"data": {"a": 1}}, call="x()") == [{"a": 1}]

    def test_empty_list_is_empty_rows(self):
        assert rows_from([], call="x()") == []

    def test_unrecognised_dict_raises_rather_than_returning_empty(self):
        """An unreadable response and an empty one demand opposite reactions."""
        with pytest.raises(KotakAPIError, match="Refusing to guess"):
            rows_from({"unexpected": "shape", "value": 1}, call="positions()")

    def test_wrong_type_raises(self):
        with pytest.raises(KotakAPIError, match="expected list or dict"):
            rows_from(42, call="x()")


class TestSafeCall:
    def test_returns_a_valid_response(self):
        client = FakeClient(response={"data": [{"a": 1}]})
        assert safe_call(client, "positions") == {"data": [{"a": 1}]}

    def test_exceptions_become_kotak_api_error(self):
        client = FakeClient(raises=ConnectionError("network down"))
        with pytest.raises(KotakAPIError, match="raised"):
            safe_call(client, "positions")

    def test_missing_method_raises(self):
        with pytest.raises(KotakAPIError, match="no callable"):
            safe_call(FakeClient(), "nonexistent_method")


class TestTimeoutBounding:
    def test_socket_timeout_is_set(self):
        original = socket.getdefaulttimeout()
        try:
            kotak_api.bound_network_calls(12.5)
            assert socket.getdefaulttimeout() == 12.5
        finally:
            socket.setdefaulttimeout(original)


class TestPositionManagerIntegration:
    """The bug in situ: a failed positions() call must not empty the book."""

    def _book_with_a_position(self):
        from src.position_manager import PositionBook

        book = PositionBook()
        book.start_session(dt.date(2026, 7, 23))
        book.apply_fill("nse_cm:2885", 100, 1300.0, stop_loss=1274.0)
        return book

    def test_reconciling_against_a_genuinely_flat_broker_clears_the_book(self):
        book = self._book_with_a_position()
        book.reconcile({})
        assert "nse_cm:2885" not in book.positions

    def test_which_is_why_a_failed_call_must_never_reach_reconcile(self):
        """Same input shape, opposite correct outcome — so the guard belongs upstream.

        `reconcile({})` cannot tell "the broker is flat" from "the call failed"; only the
        caller knows. `safe_call` raising is what keeps the failed case away from it.
        """
        with pytest.raises(KotakAPIError):
            safe_call(FakeClient(response=None), "positions", allow_empty=True)

        book = self._book_with_a_position()
        # Reconciliation was never invoked, so the book is intact.
        assert book.positions["nse_cm:2885"].quantity == 100
