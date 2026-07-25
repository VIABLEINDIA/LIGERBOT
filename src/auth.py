"""Kotak Neo authentication.

.. important::
   Rewritten after verifying the **installed** ``neo_api_client`` (v2.0.0) by
   introspection. The previous version of this module was wrong in three ways and would
   have raised on the very first call, with or without valid credentials:

   =========================================  ==================================================
   Previous code                              Reality in the installed SDK
   =========================================  ==================================================
   ``NeoAPI(consumer_secret=...)``            No such parameter. The signature is
                                              ``(environment, access_token, neo_fin_key,
                                              consumer_key)``; ``consumer_secret`` survives
                                              only in the docstring and commented-out code.
                                              Passing it raises ``TypeError``.
   ``client.login(mobilenumber=..., ...)``    No such method. It is
                                              ``totp_login(mobile_number=..., ucc=, totp=)``
                                              — note the underscore.
   ``client.session_2fa(OTP=...)``            No such method. It is
                                              ``totp_validate(mpin=...)``.
   =========================================  ==================================================

   The lesson is the same one the NSE holiday list taught: a plausible-looking API call
   written from recall is not an API call. Verify against the installed package.

Login is two steps. ``totp_login`` exchanges mobile + UCC + a current TOTP for a session
id; ``totp_validate`` exchanges the MPIN for the trade token. The SDK stores both on the
client internally, so callers just receive a ready client.

**A TOTP code is single-use within its window.** Every process that logs in burns one, so
several modules authenticating independently will collide — see
:func:`authenticate_neo`'s warning and DESIGN.md 3.8.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

import config

log = logging.getLogger("ligerbot.auth")


class AuthenticationError(RuntimeError):
    """Raised when login fails or returns something unusable.

    Distinct from the SDK's own exceptions so callers have one thing to catch, and so a
    half-authenticated client is never handed back.
    """


def _current_totp() -> str:
    """A TOTP code: generated from the secret if present, else the static fallback.

    The secret is strongly preferred. A static ``KOTAK_TOTP`` is only usable for a single
    manual login and will be stale by the next one.
    """
    if config.KOTAK_TOTP_SECRET:
        try:
            import pyotp

            return pyotp.TOTP(config.KOTAK_TOTP_SECRET).now()
        except ImportError:
            log.warning("pyotp is not installed; falling back to the static KOTAK_TOTP.")
    return config.KOTAK_TOTP


def _unwrap(response: Any) -> Dict[str, Any]:
    """Return the payload whether the SDK wrapped it in ``{"data": {...}}`` or not."""
    if not isinstance(response, dict):
        return {}
    inner = response.get("data")
    if isinstance(inner, dict):
        return inner
    if isinstance(inner, list) and inner and isinstance(inner[0], dict):
        return inner[0]
    return response


def _require_credentials() -> None:
    """Fail before touching the network if anything needed is absent.

    Named explicitly rather than reported as a generic auth failure: "MPIN missing" is
    actionable, "login failed" is not.
    """
    missing = [
        name for name, value in (
            ("KOTAK_CONSUMER_KEY", config.KOTAK_CONSUMER_KEY),
            ("KOTAK_MOBILE", config.KOTAK_MOBILE),
            ("KOTAK_UCC", config.KOTAK_UCC),
            ("KOTAK_MPIN", config.KOTAK_MPIN),
        )
        if not value or str(value).startswith("YOUR_")
    ]
    if not _current_totp():
        missing.append("KOTAK_TOTP_SECRET (or KOTAK_TOTP)")
    if missing:
        raise AuthenticationError(
            "Kotak Neo credentials are incomplete — missing: "
            + ", ".join(missing)
            + ". Set them in .env (generate the consumer key from the Trade API card in "
              "the Neo app/web)."
        )


def authenticate_neo():
    """Authenticate and return a ready ``NeoAPI`` client.

    .. warning::
       Each call performs a full login and therefore **consumes a TOTP code**. Several
       modules calling this independently will collide: codes are single-use within their
       window, and a new login may invalidate an earlier session. DESIGN.md 3.8 specifies
       a shared session service to hold one session and share the token; until that
       exists, run only one authenticating module at a time.
    """
    _require_credentials()

    from neo_api_client import NeoAPI  # lazy: the SDK is optional for tests

    # `consumer_secret` is deliberately NOT passed — it is not a parameter of this SDK
    # version (verified by introspection). It stays in config in case a future version
    # reinstates it.
    client = NeoAPI(
        consumer_key=config.KOTAK_CONSUMER_KEY,
        environment=config.KOTAK_ENVIRONMENT,
        neo_fin_key=config.KOTAK_NEO_FIN_KEY or None,
    )

    # Step 1: mobile + UCC + TOTP -> session id.
    try:
        login_response = client.totp_login(
            mobile_number=config.KOTAK_MOBILE,
            ucc=config.KOTAK_UCC,
            totp=_current_totp(),
        )
    except Exception as exc:  # noqa: BLE001 - the SDK raises a variety of types
        raise AuthenticationError(f"totp_login failed: {exc}") from exc

    login_data = _unwrap(login_response)
    if not login_data.get("sid"):
        # Checked rather than assumed: proceeding without a session id would fail later
        # with a far less obvious error.
        raise AuthenticationError(
            f"totp_login returned no session id. Response keys: "
            f"{sorted(login_data)[:12]}. A wrong TOTP or an already-used code is the "
            f"usual cause."
        )

    # Step 2: MPIN -> trade token.
    try:
        validate_response = client.totp_validate(mpin=config.KOTAK_MPIN)
    except Exception as exc:  # noqa: BLE001
        raise AuthenticationError(f"totp_validate failed: {exc}") from exc

    validate_data = _unwrap(validate_response)
    if not validate_data.get("token"):
        raise AuthenticationError(
            f"totp_validate returned no trade token. Response keys: "
            f"{sorted(validate_data)[:12]}. A wrong MPIN is the usual cause."
        )

    log.info("Authenticated with Kotak Neo (environment=%s).", config.KOTAK_ENVIRONMENT)
    if config.KOTAK_ENVIRONMENT.lower() != "prod":
        log.warning("Environment is %r, not 'prod' — this is the sandbox. Orders and "
                    "data will not be real.", config.KOTAK_ENVIRONMENT)
    return client
