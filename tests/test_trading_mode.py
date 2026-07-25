"""Trading-mode consistency tests.

``TRADING_MODE`` and ``DRY_RUN`` were previously independent settings, which let them
disagree. With ``TRADING_MODE=paper`` and ``DRY_RUN`` at its default, every module that
branched on ``DRY_RUN`` skipped the broker entirely — so paper mode ran on a *configured*
equity figure with reconciliation **disabled**, and reported nothing unusual.

Since Phase 4's whole purpose is reconciling paper results against a backtest, that
silently invalidated the very sessions it was accumulating. These tests pin the invariant
that makes it impossible: ``TRADING_MODE`` is primary and ``DRY_RUN`` is derived from it.
"""
from __future__ import annotations

import importlib
import os

import pytest

import config


def reload_config(monkeypatch, **env):
    """Re-import config with a given environment.

    Both keys are set explicitly — to "" when a test wants them absent — because
    ``load_dotenv()`` fills in anything missing from the developer's own ``.env``, which
    would otherwise leak local settings into these assertions.
    """
    monkeypatch.setenv("TRADING_MODE", env.pop("TRADING_MODE", ""))
    monkeypatch.setenv("DRY_RUN", env.pop("DRY_RUN", "true"))
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return importlib.reload(config)


@pytest.fixture(autouse=True)
def restore_config():
    yield
    importlib.reload(config)


class TestModeIsAuthoritative:
    def test_dry_run_is_derived_not_independent(self, monkeypatch):
        """The invariant. They cannot disagree, because one is computed from the other."""
        fresh = reload_config(monkeypatch, TRADING_MODE="paper", DRY_RUN="true")
        assert fresh.TRADING_MODE == "paper"
        assert fresh.DRY_RUN is False, (
            "DRY_RUN must follow TRADING_MODE. Previously this stayed True and paper mode "
            "silently ran without a broker session.")

    @pytest.mark.parametrize("mode,expect_dry", [
        ("dry_run", True), ("paper", False), ("live", False),
    ])
    def test_derivation_for_each_mode(self, monkeypatch, mode, expect_dry):
        fresh = reload_config(monkeypatch, TRADING_MODE=mode)
        assert fresh.DRY_RUN is expect_dry

    def test_legacy_dry_run_false_still_means_live(self, monkeypatch):
        """Existing configuration that only set DRY_RUN must keep working."""
        fresh = reload_config(monkeypatch, DRY_RUN="false")
        assert fresh.TRADING_MODE == "live"

    def test_legacy_dry_run_true_means_dry_run(self, monkeypatch):
        fresh = reload_config(monkeypatch, DRY_RUN="true")
        assert fresh.TRADING_MODE == "dry_run"

    def test_explicit_mode_wins_over_legacy_flag(self, monkeypatch):
        fresh = reload_config(monkeypatch, TRADING_MODE="live", DRY_RUN="true")
        assert fresh.TRADING_MODE == "live"
        assert fresh.DRY_RUN is False

    def test_unknown_mode_fails_at_import(self, monkeypatch):
        """A typo must not fall through to a default that decides whether real orders go out."""
        with pytest.raises(ValueError, match="not one of"):
            reload_config(monkeypatch, TRADING_MODE="papper")

    def test_mode_is_case_insensitive(self, monkeypatch):
        assert reload_config(monkeypatch, TRADING_MODE="PAPER").TRADING_MODE == "paper"


class TestPredicates:
    def test_paper_needs_a_broker_session(self, monkeypatch):
        """The bug in one assertion: equity and reconciliation are real in paper mode."""
        fresh = reload_config(monkeypatch, TRADING_MODE="paper")
        assert fresh.needs_broker_session() is True

    def test_live_needs_a_broker_session(self, monkeypatch):
        assert reload_config(monkeypatch, TRADING_MODE="live").needs_broker_session()

    def test_dry_run_does_not(self, monkeypatch):
        fresh = reload_config(monkeypatch, TRADING_MODE="dry_run")
        assert fresh.needs_broker_session() is False

    @pytest.mark.parametrize("mode,expected", [
        ("dry_run", False), ("paper", False), ("live", True),
    ])
    def test_only_live_sends_real_orders(self, monkeypatch, mode, expected):
        """The single predicate gating real money."""
        assert reload_config(monkeypatch, TRADING_MODE=mode).sends_real_orders() is expected

    @pytest.mark.parametrize("mode,expected", [
        ("dry_run", False), ("paper", True), ("live", False),
    ])
    def test_only_paper_simulates_fills(self, monkeypatch, mode, expected):
        assert reload_config(monkeypatch, TRADING_MODE=mode).simulates_fills() is expected


class TestExactlyOneFiller:
    """Two fillers on the same stream would double-count every trade.

    Both the execution engine and the paper broker consume ``approved_orders`` from
    separate consumer groups, so each would receive every order independently — and the
    result would look entirely healthy.
    """

    @pytest.mark.parametrize("mode", ["dry_run", "paper", "live"])
    def test_exactly_one_filler_claims_each_mode(self, monkeypatch, mode):
        fresh = reload_config(monkeypatch, TRADING_MODE=mode)
        execution_claims = not fresh.simulates_fills()
        paper_claims = fresh.simulates_fills()
        assert execution_claims != paper_claims, f"{mode}: exactly one must fill"

    def test_execution_engine_refuses_paper_mode(self, monkeypatch):
        reload_config(monkeypatch, TRADING_MODE="paper")
        import fakeredis

        from src import event_bus, execution_engine

        monkeypatch.setattr(
            event_bus, "get_client",
            lambda *a, **k: fakeredis.FakeStrictRedis(decode_responses=True))
        monkeypatch.setattr(event_bus, "ping", lambda *a, **k: True)
        importlib.reload(execution_engine)

        engine = execution_engine.ExecutionEngine(None)
        assert engine._check_mode() is False

    @pytest.mark.parametrize("mode", ["dry_run", "live"])
    def test_execution_engine_accepts_its_own_modes(self, monkeypatch, mode):
        reload_config(monkeypatch, TRADING_MODE=mode)
        import fakeredis

        from src import event_bus, execution_engine

        monkeypatch.setattr(
            event_bus, "get_client",
            lambda *a, **k: fakeredis.FakeStrictRedis(decode_responses=True))
        monkeypatch.setattr(event_bus, "ping", lambda *a, **k: True)
        importlib.reload(execution_engine)

        assert execution_engine.ExecutionEngine(None)._check_mode() is True


class TestTemplateAndRunner:
    def test_env_template_documents_the_three_modes(self):
        from pathlib import Path

        text = (Path(__file__).resolve().parent.parent / ".env.example").read_text(
            encoding="utf-8")
        for mode in ("dry_run", "paper", "live"):
            assert mode in text

    def test_runner_starts_one_executor_per_mode(self):
        import run_all

        for mode in ("dry_run", "paper", "live"):
            names = [n for n, _ in run_all.modules_for_mode(mode)]
            fillers = [n for n in names if n in ("execution", "paper")]
            assert len(fillers) == 1, f"{mode} starts {fillers}"

    def test_paper_mode_starts_the_paper_broker(self):
        import run_all

        names = [n for n, _ in run_all.modules_for_mode("paper")]
        assert "paper" in names and "execution" not in names
