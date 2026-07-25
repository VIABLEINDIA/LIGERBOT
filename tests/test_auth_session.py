"""Shared broker session tests (DESIGN.md 3.8).

The bug this closes: four modules each called ``authenticate_neo()`` independently, and
every call performs a full ``totp_login`` + ``totp_validate``. A TOTP code is **single-use
within a 30-second window**, and ``run_all.py`` staggers startup by one second — so modules
starting together generate the *same* code and all but one are rejected as replays.

Fixing paper mode made this acute rather than theoretical: it took paper from one
authenticating module to three.

The invariant under test throughout: **however many modules ask, exactly one login
happens.**
"""
from __future__ import annotations

import datetime as dt

import fakeredis
import pytest

import config
from src.auth_session import (
    LOGIN_LOCK_KEY, SessionUnavailable, capture_session, clear_session, get_session,
    read_session, refresh_after_expiry, session_key, write_session,
)

DAY = dt.date(2026, 7, 23)


class FakeConfiguration:
    def __init__(self, authenticated: bool = True):
        self.base_url = "https://gw.kotak.com" if authenticated else None
        self.bearer_token = "bearer-abc" if authenticated else None
        self.sid = "sid-123" if authenticated else None
        self.edit_sid = "edit-sid" if authenticated else None
        self.edit_token = "edit-token" if authenticated else None
        self.view_token = "view-token" if authenticated else None
        self.totp_session_id = "totp-sid" if authenticated else None
        self.consumer_key = "ck-test"
        self.neo_fin_key = None
        self.login_params = None


class FakeClient:
    def __init__(self, authenticated: bool = True):
        self.configuration = FakeConfiguration(authenticated)


@pytest.fixture
def redis_client():
    return fakeredis.FakeStrictRedis(server=fakeredis.FakeServer(), decode_responses=True)


@pytest.fixture
def counting_login():
    """A login function that records how many times it was called."""
    calls = []

    def login():
        calls.append(dt.datetime.now())
        return FakeClient()

    login.calls = calls
    return login


class TestExactlyOneLogin:
    """The invariant. However many modules ask, one TOTP is consumed."""

    def test_three_modules_produce_one_login(self, redis_client, counting_login,
                                             monkeypatch):
        monkeypatch.setattr("src.auth_session.restore_session",
                           lambda s: FakeClient())
        for _ in range(3):
            get_session(redis_client, day=DAY, login_fn=counting_login)
        assert len(counting_login.calls) == 1, (
            f"{len(counting_login.calls)} logins — each burns a TOTP code, and codes are "
            f"single-use within their 30-second window")

    def test_the_second_caller_restores_rather_than_logs_in(self, redis_client,
                                                            counting_login, monkeypatch):
        restored = []
        monkeypatch.setattr("src.auth_session.restore_session",
                           lambda s: restored.append(s) or FakeClient())
        get_session(redis_client, day=DAY, login_fn=counting_login)
        get_session(redis_client, day=DAY, login_fn=counting_login)
        assert len(counting_login.calls) == 1
        assert len(restored) == 1

    def test_lock_is_released_after_login(self, redis_client, counting_login, monkeypatch):
        monkeypatch.setattr("src.auth_session.restore_session", lambda s: FakeClient())
        get_session(redis_client, day=DAY, login_fn=counting_login)
        assert not redis_client.exists(LOGIN_LOCK_KEY)

    def test_lock_is_released_even_if_login_raises(self, redis_client):
        def failing_login():
            raise RuntimeError("totp rejected")

        with pytest.raises(RuntimeError):
            get_session(redis_client, day=DAY, login_fn=failing_login)
        # Otherwise a single bad login would deadlock every module for the whole day.
        assert not redis_client.exists(LOGIN_LOCK_KEY)

    def test_a_waiting_caller_gives_up_rather_than_racing(self, redis_client,
                                                          counting_login):
        """Starting a competing login would burn a second code in the same window."""
        redis_client.set(LOGIN_LOCK_KEY, "1", ex=60)
        with pytest.raises(SessionUnavailable, match="replay"):
            get_session(redis_client, day=DAY, wait_seconds=1.0,
                        login_fn=counting_login)
        assert counting_login.calls == []


class TestCaptureRestore:
    def test_captures_every_session_field(self):
        captured = capture_session(FakeClient())
        for field in ("sid", "bearer_token", "edit_token", "edit_sid", "view_token",
                      "base_url"):
            assert captured.get(field), f"{field} not captured"

    def test_captured_session_is_json_serialisable(self, redis_client):
        write_session(redis_client, capture_session(FakeClient()), day=DAY)
        assert read_session(redis_client, day=DAY) is not None

    def test_unauthenticated_client_refuses_to_be_captured(self):
        """Sharing an unauthenticated session hands other modules a client that fails
        later and less obviously than it would have failed here."""
        with pytest.raises(SessionUnavailable, match="neither sid nor bearer_token"):
            capture_session(FakeClient(authenticated=False))

    def test_client_without_configuration_raises(self):
        class Bare:
            pass

        with pytest.raises(SessionUnavailable, match="no configuration"):
            capture_session(Bare())


class TestDayScoping:
    def test_key_is_per_day(self):
        assert session_key(DAY) != session_key(DAY + dt.timedelta(days=1))

    def test_a_new_day_forces_a_fresh_login(self, redis_client, counting_login,
                                            monkeypatch):
        """A stale token fails mid-session rather than at startup, which is worse."""
        monkeypatch.setattr("src.auth_session.restore_session", lambda s: FakeClient())
        get_session(redis_client, day=DAY, login_fn=counting_login)
        get_session(redis_client, day=DAY + dt.timedelta(days=1),
                    login_fn=counting_login)
        assert len(counting_login.calls) == 2

    def test_session_is_written_with_a_ttl(self, redis_client):
        write_session(redis_client, capture_session(FakeClient()), day=DAY)
        assert redis_client.ttl(session_key(DAY)) > 0


class TestExpiryRecovery:
    def test_clear_then_relogin(self, redis_client, counting_login, monkeypatch):
        monkeypatch.setattr("src.auth_session.restore_session", lambda s: FakeClient())
        get_session(redis_client, day=DAY, login_fn=counting_login)
        assert read_session(redis_client, day=DAY) is not None

        clear_session(redis_client, day=DAY)
        assert read_session(redis_client, day=DAY) is None

        get_session(redis_client, day=DAY, login_fn=counting_login)
        assert len(counting_login.calls) == 2

    def test_refresh_after_expiry_discards_and_relogs(self, redis_client, counting_login,
                                                       monkeypatch):
        monkeypatch.setattr("src.auth_session.restore_session", lambda s: FakeClient())
        monkeypatch.setattr("src.auth.authenticate_neo", counting_login)
        get_session(redis_client, day=DAY, login_fn=counting_login)

        refresh_after_expiry(redis_client, day=DAY)
        assert len(counting_login.calls) == 2


class TestDegradedModes:
    def test_no_redis_falls_back_to_a_direct_login(self, counting_login):
        """A broken cache should lose coordination, not stop a module working."""
        get_session(None, day=DAY, login_fn=counting_login)
        assert len(counting_login.calls) == 1

    def test_unreadable_session_is_discarded_not_used(self, redis_client, counting_login,
                                                       monkeypatch):
        monkeypatch.setattr("src.auth_session.restore_session", lambda s: FakeClient())
        redis_client.set(session_key(DAY), "{not json", ex=600)
        get_session(redis_client, day=DAY, login_fn=counting_login)
        assert len(counting_login.calls) == 1

    def test_allow_login_false_refuses_rather_than_logging_in(self, redis_client,
                                                              counting_login):
        with pytest.raises(SessionUnavailable, match="not permitted"):
            get_session(redis_client, day=DAY, allow_login=False,
                        login_fn=counting_login)
        assert counting_login.calls == []

    def test_login_still_returns_a_client_when_publishing_fails(self, counting_login,
                                                                monkeypatch):
        """We are authenticated even if sharing failed; do not fail the caller."""
        class WriteFails(fakeredis.FakeStrictRedis):
            def set(self, *a, **kw):
                if kw.get("nx"):
                    return True          # lock acquired
                raise ConnectionError("write failed")

        client = WriteFails(decode_responses=True)
        assert get_session(client, day=DAY, login_fn=counting_login) is not None
        assert len(counting_login.calls) == 1


class TestModulesUseTheSharedSession:
    """No module may call authenticate_neo() directly any more."""

    @pytest.mark.parametrize("module", [
        "src/risk_manager.py", "src/position_manager.py",
        "src/data_ingestion.py", "src/execution_engine.py",
    ])
    def test_module_obtains_a_shared_session(self, module):
        from pathlib import Path

        text = (Path(__file__).resolve().parent.parent / module).read_text(
            encoding="utf-8")
        assert "get_session(" in text, f"{module} does not use the shared session"
        assert "authenticate_neo()" not in text, (
            f"{module} still logs in directly — that consumes a TOTP code and will "
            f"collide with the other modules")
