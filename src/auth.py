"""Kotak Neo authentication helper.

Kotak Neo's live environment requires a three-step login:

  1. Initialize the ``NeoAPI`` client for the ``prod`` environment with your
     consumer key/secret.
  2. ``login`` with mobile number, UCC and a current TOTP.
  3. ``session_2fa`` with your MPIN to mint the trade token.

The ``neo_api_client`` package is only imported lazily so the rest of the codebase
(and the test suite) can be exercised without the SDK installed.
"""
from __future__ import annotations

import logging

import config

log = logging.getLogger("ligerbot.auth")


def _current_totp() -> str:
    """Return a TOTP code — generated from the secret if available, else the
    static code from config (useful for a one-shot manual login)."""
    if config.KOTAK_TOTP_SECRET:
        try:
            import pyotp

            return pyotp.TOTP(config.KOTAK_TOTP_SECRET).now()
        except ImportError:
            log.warning("pyotp not installed; falling back to static KOTAK_TOTP")
    return config.KOTAK_TOTP


def authenticate_neo():
    """Authenticate against Kotak Neo Live and return a ready-to-use client.

    Raises RuntimeError if credentials are missing so we never silently proceed
    with a half-configured client.
    """
    if not config.KOTAK_CONSUMER_KEY or config.KOTAK_CONSUMER_KEY == "YOUR_CONSUMER_KEY":
        raise RuntimeError(
            "Kotak Neo consumer key not configured. Set KOTAK_CONSUMER_KEY in .env "
            "(generate it from the Trade API card in the Neo app/web)."
        )

    from neo_api_client import NeoAPI  # lazy import

    # 1. Initialize client for the LIVE environment.
    client = NeoAPI(
        consumer_key=config.KOTAK_CONSUMER_KEY,
        consumer_secret=config.KOTAK_CONSUMER_SECRET,
        environment=config.KOTAK_ENVIRONMENT,
        access_token=None,
        neo_fin_key=None,
    )

    # 2. Login with mobile number / UCC + a current TOTP.
    client.login(
        mobilenumber=config.KOTAK_MOBILE,
        ucc=config.KOTAK_UCC,
        totp=_current_totp(),
    )

    # 3. Validate MPIN to generate the trade token.
    client.session_2fa(OTP=config.KOTAK_MPIN)

    log.info("Successfully authenticated with Kotak Neo (%s).", config.KOTAK_ENVIRONMENT)
    return client
