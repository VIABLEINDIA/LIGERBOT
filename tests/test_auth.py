"""Kotak Neo authentication tests.

These pin the *verified* SDK contract. The previous implementation was written from recall
and was wrong in three ways — no `consumer_secret` parameter exists, `login()` is actually
`totp_login()` with an underscored `mobile_number`, and `session_2fa()` is actually
`totp_validate()`. It would have raised on the first call with or without valid
credentials, and nothing in the suite would have noticed.

A fake client stands in for the SDK, shaped to the signatures confirmed by introspecting
the installed package. If a future SDK version changes them, these fail loudly rather than
the bot failing at 09:15.
"""
from __future__ import annotations

import pytest

import config
from src.auth import AuthenticationError, authenticate_neo


class FakeNeoAPI:
    """Mimics neo_api_client v2.0.0's actual surface.

    Deliberately rejects any unexpected keyword, exactly as the real
    ``NeoAPI.__init__`` would — that is the bug this guards against.
    """

    instances: list = []

    def __init__(self, environment="uat", access_token=None, neo_fin_key=None,
                 consumer_key=None):
        self.environment = environment
        self.consumer_key = consumer_key
        self.neo_fin_key = neo_fin_key
        self.login_calls = []
        self.validate_calls = []
        self.login_response = {"data": {"sid": "session-123"}}
        self.validate_response = {"data": {"token": "trade-token-456"}}
        FakeNeoAPI.instances.append(self)

    def totp_login(self, mobile_number=None, ucc=None, totp=None):
        self.login_calls.append({"mobile_number": mobile_number, "ucc": ucc, "totp": totp})
        return self.login_response

    def totp_validate(self, mpin=None):
        self.validate_calls.append({"mpin": mpin})
        return self.validate_response


@pytest.fixture(autouse=True)
def creds(monkeypatch):
    FakeNeoAPI.instances.clear()
    monkeypatch.setattr(config, "KOTAK_CONSUMER_KEY", "ck-test")
    monkeypatch.setattr(config, "KOTAK_CONSUMER_SECRET", "cs-test")
    monkeypatch.setattr(config, "KOTAK_MOBILE", "+919999999999")
    monkeypatch.setattr(config, "KOTAK_UCC", "UCC123")
    monkeypatch.setattr(config, "KOTAK_MPIN", "1234")
    monkeypatch.setattr(config, "KOTAK_TOTP_SECRET", "")
    monkeypatch.setattr(config, "KOTAK_TOTP", "654321")
    monkeypatch.setattr(config, "KOTAK_NEO_FIN_KEY", "")
    monkeypatch.setattr(config, "KOTAK_ENVIRONMENT", "prod")


@pytest.fixture
def fake_sdk(monkeypatch):
    import sys
    import types

    module = types.ModuleType("neo_api_client")
    module.NeoAPI = FakeNeoAPI
    monkeypatch.setitem(sys.modules, "neo_api_client", module)
    return module


class TestVerifiedSdkContract:
    def test_consumer_secret_is_not_passed(self, fake_sdk):
        """The bug that would have raised TypeError on the first call.

        ``consumer_secret`` is not a parameter of this SDK version — it survives only in
        a docstring and commented-out code.
        """
        authenticate_neo()
        # The fake rejects unexpected kwargs the same way the real class would, so simply
        # completing proves consumer_secret was not passed.
        assert len(FakeNeoAPI.instances) == 1
        assert FakeNeoAPI.instances[0].consumer_key == "ck-test"

    def test_uses_totp_login_with_underscored_mobile_number(self, fake_sdk):
        authenticate_neo()
        call = FakeNeoAPI.instances[0].login_calls[0]
        assert call["mobile_number"] == "+919999999999"   # not "mobilenumber"
        assert call["ucc"] == "UCC123"
        assert call["totp"] == "654321"

    def test_uses_totp_validate_with_mpin(self, fake_sdk):
        authenticate_neo()
        assert FakeNeoAPI.instances[0].validate_calls == [{"mpin": "1234"}]

    def test_environment_is_forwarded(self, fake_sdk):
        authenticate_neo()
        assert FakeNeoAPI.instances[0].environment == "prod"

    def test_returns_the_client(self, fake_sdk):
        assert isinstance(authenticate_neo(), FakeNeoAPI)


class TestResponseValidation:
    def test_missing_session_id_raises(self, fake_sdk, monkeypatch):
        """Proceeding without a session id would fail later, far less obviously."""
        original = FakeNeoAPI.__init__

        def patched(self, **kw):
            original(self, **kw)
            self.login_response = {"data": {}}

        monkeypatch.setattr(FakeNeoAPI, "__init__", patched)
        with pytest.raises(AuthenticationError, match="no session id"):
            authenticate_neo()

    def test_missing_token_raises(self, fake_sdk, monkeypatch):
        original = FakeNeoAPI.__init__

        def patched(self, **kw):
            original(self, **kw)
            self.validate_response = {"data": {}}

        monkeypatch.setattr(FakeNeoAPI, "__init__", patched)
        with pytest.raises(AuthenticationError, match="no trade token"):
            authenticate_neo()

    def test_unwraps_flat_responses(self, fake_sdk, monkeypatch):
        """Some SDK versions return the payload flat rather than under "data"."""
        original = FakeNeoAPI.__init__

        def patched(self, **kw):
            original(self, **kw)
            self.login_response = {"sid": "flat-session"}
            self.validate_response = {"token": "flat-token"}

        monkeypatch.setattr(FakeNeoAPI, "__init__", patched)
        assert authenticate_neo() is not None

    def test_sdk_exception_becomes_authentication_error(self, fake_sdk, monkeypatch):
        def boom(self, **kw):
            raise ConnectionError("network down")

        monkeypatch.setattr(FakeNeoAPI, "totp_login", boom)
        with pytest.raises(AuthenticationError, match="totp_login failed"):
            authenticate_neo()


class TestCredentialGuard:
    @pytest.mark.parametrize("field", [
        "KOTAK_CONSUMER_KEY", "KOTAK_MOBILE", "KOTAK_UCC", "KOTAK_MPIN",
    ])
    def test_each_missing_credential_is_named(self, fake_sdk, monkeypatch, field):
        """"MPIN missing" is actionable; "login failed" is not."""
        monkeypatch.setattr(config, field, "")
        with pytest.raises(AuthenticationError, match=field):
            authenticate_neo()

    def test_missing_totp_is_named(self, fake_sdk, monkeypatch):
        monkeypatch.setattr(config, "KOTAK_TOTP", "")
        monkeypatch.setattr(config, "KOTAK_TOTP_SECRET", "")
        with pytest.raises(AuthenticationError, match="KOTAK_TOTP"):
            authenticate_neo()

    def test_placeholder_values_are_rejected(self, fake_sdk, monkeypatch):
        monkeypatch.setattr(config, "KOTAK_CONSUMER_KEY", "YOUR_CONSUMER_KEY")
        with pytest.raises(AuthenticationError, match="KOTAK_CONSUMER_KEY"):
            authenticate_neo()

    def test_guard_runs_before_any_network_call(self, fake_sdk, monkeypatch):
        monkeypatch.setattr(config, "KOTAK_MPIN", "")
        with pytest.raises(AuthenticationError):
            authenticate_neo()
        assert FakeNeoAPI.instances == []   # never even constructed


class TestTotpGeneration:
    def test_secret_is_preferred_over_static_code(self, fake_sdk, monkeypatch):
        pytest.importorskip("pyotp")
        import pyotp

        secret = pyotp.random_base32()
        monkeypatch.setattr(config, "KOTAK_TOTP_SECRET", secret)
        monkeypatch.setattr(config, "KOTAK_TOTP", "000000")
        authenticate_neo()
        used = FakeNeoAPI.instances[0].login_calls[0]["totp"]
        assert used != "000000"
        assert used == pyotp.TOTP(secret).now()
