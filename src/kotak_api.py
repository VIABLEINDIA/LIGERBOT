"""Guards around ``neo_api_client``'s actual behaviour.

The SDK does not behave the way exception-based code expects, and three of its habits are
actively dangerous for a trading loop. All three were found by cross-referencing a working
integration on this machine (``D:\\JEANS``), whose author had already reconstructed them —
and each one exposed a real bug here.

**1. Errors come back as data, not exceptions.** Most methods (``place_order``,
``cancel_order``, ``positions``, ``scrip_master``, …) catch internally and return an
``{"Error": ...}`` dict. A ``try/except`` around the call therefore catches nothing, and the
error flows onward as if it were a valid response.

**2. ``positions()`` can silently return ``None``.** This was the dangerous one here.
``parse_broker_positions(None)`` returned an empty mapping, which reconciliation read as
*"the broker holds nothing"* — so a transient API failure would have made the position
manager discard every open position from its book. The halt that follows is fail-safe, but
losing the record of what is actually open while halted is not.

**3. No per-request timeout.** The SDK's REST client issues ``requests`` calls without one,
so a dropped connection hangs forever. In a module whose loop must keep servicing the event
bus, that is indistinguishable from a hung process.

None of this is guaranteed by the SDK — it is observed behaviour of v2.0.0. The functions
here fail *loudly* rather than papering over it, so a changed SDK surfaces as an error
rather than as silently wrong data.
"""
from __future__ import annotations

import logging
import socket
from typing import Any, Dict, List, Optional

log = logging.getLogger("ligerbot.kotak_api")

# The SDK's own sentinel when the session is not (or no longer) authenticated. A literal
# string match is coarse, but the SDK discards HTTP status codes entirely, so there is no
# protocol-level signal left to key off.
SESSION_EXPIRED_MARKER = "Complete the 2fa process before accessing this application"

# Bounds the SDK's otherwise-unbounded network calls. Process-wide because the timeout
# cannot be passed per request.
DEFAULT_SOCKET_TIMEOUT_SECONDS = 20.0


class KotakAPIError(RuntimeError):
    """A Kotak call returned an error payload, nothing usable, or ``None``."""


class KotakSessionExpired(KotakAPIError):
    """The session is no longer authenticated and must be re-established.

    Separate from :class:`KotakAPIError` so callers can re-login rather than retry, which
    would fail identically forever.
    """


def bound_network_calls(timeout: float = DEFAULT_SOCKET_TIMEOUT_SECONDS) -> None:
    """Cap how long any SDK network call can block.

    Call once at module start-up. Without it a dropped connection stalls the calling
    module indefinitely — and a trading loop that has stopped servicing its stream is
    worse than one that has crashed, because nothing notices.
    """
    socket.setdefaulttimeout(timeout)
    log.debug("Socket timeout bounded to %.1fs (the SDK sets none itself).", timeout)


def _contains_session_expiry(payload: Any) -> bool:
    return SESSION_EXPIRED_MARKER.lower() in str(payload).lower()


def validate(response: Any, *, call: str, allow_empty: bool = False) -> Any:
    """Return ``response`` if it is usable; raise otherwise.

    ``allow_empty`` permits a legitimately empty result — an account with no positions
    genuinely returns an empty list. It does **not** permit ``None``, because that is the
    SDK's failure shape and must never be mistaken for "nothing there".
    """
    if _contains_session_expiry(response):
        raise KotakSessionExpired(
            f"{call} reports the session is not authenticated — re-login required.")

    if response is None:
        raise KotakAPIError(
            f"{call} returned None. The SDK does this on failure rather than raising, so "
            f"this is an error, not an empty result — treating it as 'nothing there' "
            f"would silently discard real state.")

    if isinstance(response, dict):
        for key in ("Error", "error", "errMsg", "errorMessage"):
            if response.get(key):
                raise KotakAPIError(f"{call} returned an error payload: {response[key]}")
        stat = str(response.get("stat", "")).lower()
        if stat and stat not in ("ok", "success"):
            raise KotakAPIError(
                f"{call} returned stat={response.get('stat')!r}: "
                f"{response.get('errMsg') or response}")

    if not allow_empty and isinstance(response, (list, dict)) and not response:
        raise KotakAPIError(
            f"{call} returned an empty {type(response).__name__}. If genuinely empty is "
            f"valid here, pass allow_empty=True at the call site so the distinction is "
            f"explicit.")
    return response


def rows_from(response: Any, *, call: str) -> List[Dict[str, Any]]:
    """Extract a list of row dicts, unwrapping the SDK's ``data`` envelope.

    Raises rather than returning ``[]`` on an unrecognised shape: an unreadable response
    and an empty one demand opposite reactions, and conflating them is how a failed call
    becomes "the account is flat".
    """
    validate(response, call=call, allow_empty=True)

    if isinstance(response, list):
        return [r for r in response if isinstance(r, dict)]
    if isinstance(response, dict):
        for key in ("data", "Data", "result", "orders", "positions"):
            value = response.get(key)
            if isinstance(value, list):
                return [r for r in value if isinstance(r, dict)]
            if isinstance(value, dict):
                return [value]
        # A dict with no recognised envelope but real content is ambiguous.
        raise KotakAPIError(
            f"{call} returned a dict with no recognised list envelope. Keys: "
            f"{sorted(response)[:15]}. Refusing to guess whether this means 'empty'.")
    raise KotakAPIError(f"{call} returned {type(response).__name__}, expected list or dict")


def safe_call(client: Any, method: str, *args, allow_empty: bool = False, **kwargs) -> Any:
    """Invoke an SDK method and validate the result.

    Converts both failure modes — a raised exception and an error-shaped return — into
    :class:`KotakAPIError`, so callers have one thing to handle.
    """
    func = getattr(client, method, None)
    if func is None or not callable(func):
        raise KotakAPIError(f"the SDK has no callable {method!r}")
    try:
        response = func(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - the SDK raises a variety of types
        raise KotakAPIError(f"{method}() raised: {exc}") from exc
    return validate(response, call=f"{method}()", allow_empty=allow_empty)
