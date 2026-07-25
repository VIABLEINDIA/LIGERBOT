"""One Kotak login per session, shared across modules (DESIGN.md 3.8).

**The problem this exists to solve.** Four modules need a broker session — ingestion,
risk manager, position manager and (in live) execution. Each previously called
:func:`src.auth.authenticate_neo` independently, and every call performs a full
``totp_login`` + ``totp_validate``. A TOTP code lives in a **30-second window and is
single-use**, and ``run_all.py`` staggers module startup by one second — so three modules
starting together generate the *same* code, and the broker will reject the second and third
as replay. Paper and live mode could not reliably start.

**Why sharing needs more than an access token.** A Kotak session is not one string: the
SDK's ``configuration`` carries ``sid``, ``bearer_token``, ``edit_token``, ``edit_sid``,
``view_token`` and ``base_url``, all populated by the two-step login. Passing only
``access_token`` to a fresh ``NeoAPI`` leaves the rest unset, and authenticated calls fail
in ways that look like permission errors. So the whole set is captured and restored.

Rather than requiring a separate service process, whichever module needs a session first
takes a **Redis lock**, logs in once, and publishes the session for the others. The losers
wait briefly and read it. Exactly one TOTP is consumed however many modules start at once,
and it works even if nothing else is running.

Sessions are keyed by trading day: a new day always means a fresh login, because a stale
token produces authentication failures mid-session rather than at startup.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import time
from typing import Any, Dict, List, Optional

import config

log = logging.getLogger("ligerbot.auth_session")

# Everything on the SDK's `configuration` that carries session state. Captured wholesale
# rather than selectively: which fields are load-bearing is undocumented, and guessing
# wrong produces authenticated calls that fail like permission errors.
SESSION_FIELDS: tuple = (
    "base_url",
    "bearer_token",
    "edit_sid",
    "edit_token",
    "sid",
    "totp_session_id",
    "view_token",
    "consumer_key",
    "neo_fin_key",
    "login_params",
)

SESSION_KEY_PREFIX = "ligerbot:kotak_session:"
LOGIN_LOCK_KEY = "ligerbot:kotak_login_lock"


class SessionUnavailable(RuntimeError):
    """No session could be obtained — neither restored nor established."""


def session_key(day: Optional[dt.date] = None) -> str:
    """Sessions are per trading day, so a new day forces a fresh login."""
    day = day or dt.date.today()
    return f"{SESSION_KEY_PREFIX}{day.isoformat()}"


# ---------------------------------------------------------------------------
# capture / restore
# ---------------------------------------------------------------------------
def capture_session(client: Any) -> Dict[str, Any]:
    """Extract the session state from an authenticated client."""
    configuration = getattr(client, "configuration", None)
    if configuration is None:
        raise SessionUnavailable("client has no configuration to capture")

    captured = {}
    for field in SESSION_FIELDS:
        value = getattr(configuration, field, None)
        # Only JSON-serialisable values survive the trip through Redis.
        if value is None or isinstance(value, (str, int, float, bool, list, dict)):
            captured[field] = value

    if not captured.get("sid") and not captured.get("bearer_token"):
        raise SessionUnavailable(
            "captured session has neither sid nor bearer_token — the login did not "
            "populate the configuration, so sharing it would hand other modules an "
            "unauthenticated client that fails later and less obviously.")
    return captured


def restore_session(session: Dict[str, Any]) -> Any:
    """Build a client from a captured session, performing **no login**.

    This is the whole point: a module that restores a session consumes no TOTP code.
    """
    from neo_api_client import NeoAPI

    client = NeoAPI(
        consumer_key=session.get("consumer_key") or config.KOTAK_CONSUMER_KEY,
        environment=config.KOTAK_ENVIRONMENT,
        neo_fin_key=session.get("neo_fin_key") or (config.KOTAK_NEO_FIN_KEY or None),
    )
    for field, value in session.items():
        try:
            setattr(client.configuration, field, value)
        except Exception:  # noqa: BLE001 - a changed SDK must not break restoration
            log.debug("Could not restore session field %r.", field)
    return client


# ---------------------------------------------------------------------------
# shared store
# ---------------------------------------------------------------------------
def read_session(redis_client, *, day: Optional[dt.date] = None) -> Optional[Dict[str, Any]]:
    try:
        raw = redis_client.get(session_key(day))
    except Exception as exc:  # noqa: BLE001
        log.warning("Cannot read the shared session (%s).", exc)
        return None
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        log.error("Shared session is not valid JSON — discarding it.")
        return None


def write_session(
    redis_client, session: Dict[str, Any], *, day: Optional[dt.date] = None,
    ttl_seconds: Optional[int] = None,
) -> None:
    ttl = ttl_seconds or config.KOTAK_SESSION_TTL_SECONDS
    try:
        redis_client.set(session_key(day), json.dumps(session), ex=ttl)
        log.info("Published the shared Kotak session (ttl %ds). Other modules will "
                 "restore it rather than logging in.", ttl)
    except Exception as exc:  # noqa: BLE001
        # Not fatal: this module has a working session, it just cannot share it. The
        # others will each log in, which is the old behaviour.
        log.error("Could not publish the shared session (%s) — other modules will each "
                  "have to log in, and may collide on the TOTP window.", exc)


def clear_session(redis_client, *, day: Optional[dt.date] = None) -> None:
    """Drop the shared session so the next caller re-establishes it.

    Called when a restored session turns out to be expired.
    """
    try:
        redis_client.delete(session_key(day))
        log.warning("Cleared the shared Kotak session; the next caller will re-login.")
    except Exception as exc:  # noqa: BLE001
        log.error("Could not clear the shared session: %s", exc)


# ---------------------------------------------------------------------------
# the coordinated entry point
# ---------------------------------------------------------------------------
def get_session(
    redis_client=None,
    *,
    day: Optional[dt.date] = None,
    allow_login: bool = True,
    wait_seconds: float = 20.0,
    login_fn=None,
) -> Any:
    """Return an authenticated client, logging in at most once across all modules.

    Order of preference:

    1. **Restore** an existing shared session — no TOTP consumed.
    2. **Win the login lock**, log in once, publish for everyone else.
    3. **Wait** for whoever holds the lock, then restore what they published.

    Falls back to a direct login if Redis is unreachable: a broken cache should not stop
    a single module from working, it should only lose the coordination.
    """
    from src.auth import authenticate_neo

    login_fn = login_fn or authenticate_neo

    if redis_client is None:
        log.warning("No Redis client — logging in directly, with no coordination. "
                    "Concurrent modules may collide on the TOTP window.")
        return login_fn()

    existing = read_session(redis_client, day=day)
    if existing:
        log.info("Restored the shared Kotak session — no login, no TOTP consumed.")
        return restore_session(existing)

    if not allow_login:
        raise SessionUnavailable(
            "no shared session available and this module is not permitted to log in.")

    # Only one winner performs the login. The lock TTL bounds a crash mid-login: it
    # expires rather than deadlocking every module for the rest of the day.
    acquired = False
    try:
        acquired = bool(redis_client.set(
            LOGIN_LOCK_KEY, "1", nx=True, ex=config.KOTAK_LOGIN_LOCK_TTL_SECONDS))
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not take the login lock (%s) — logging in uncoordinated.", exc)
        return login_fn()

    if acquired:
        try:
            log.info("Won the login lock — establishing the session for all modules.")
            client = login_fn()
            try:
                write_session(redis_client, capture_session(client), day=day)
            except SessionUnavailable as exc:
                # We are authenticated even if the capture failed; do not fail the caller.
                log.error("Session established but not shareable (%s). Other modules "
                          "will each log in.", exc)
            return client
        finally:
            try:
                redis_client.delete(LOGIN_LOCK_KEY)
            except Exception:  # noqa: BLE001
                pass

    # Someone else is logging in. Wait for their result rather than racing them.
    log.info("Another module is logging in — waiting up to %.0fs for the shared session.",
             wait_seconds)
    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline:
        time.sleep(0.5)
        published = read_session(redis_client, day=day)
        if published:
            log.info("Restored the shared session established by another module.")
            return restore_session(published)

    raise SessionUnavailable(
        f"waited {wait_seconds:.0f}s for another module to publish a session and none "
        f"appeared. Starting a competing login now would consume a second TOTP code in "
        f"the same window and likely be rejected as a replay.")


def refresh_after_expiry(redis_client, *, day: Optional[dt.date] = None) -> Any:
    """Discard the shared session and establish a new one.

    For use when a call raises :class:`src.kotak_api.KotakSessionExpired`. Nothing
    currently acts on that exception; this is what it should call.
    """
    clear_session(redis_client, day=day)
    return get_session(redis_client, day=day)
