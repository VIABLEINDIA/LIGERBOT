"""Risk manager — the daemon layer around the already-tested engine.

`RiskEngine` is pure and sits at 96%: it provably cannot exceed the open-risk cap. The
adapter that *runs* it in production sat at 67%, and the gap was not evenly spread — it was
concentrated in `_resolve_equity`, `run()` and `main()`, none of which had ever executed.

That gap matters more than the percentage suggests, because of what `_resolve_equity` does.
**Equity is the denominator of every sizing decision the engine makes.** The engine is
proven correct given an equity figure; nothing was checking how that figure is obtained, or
what happens when it cannot be. A silently wrong equity does not trip any cap — it
mis-sizes every trade of the day by the same factor, and every downstream percentage check
still passes because they are all relative to the same wrong base.

So the property under test is not "does it size correctly" — that is the engine's job and
is tested there. It is **"does it refuse to trade when it does not know the number"**.
D1 requires failing closed, and this is where that is decided.
"""
from __future__ import annotations

import datetime as dt

import fakeredis
import pytest

import config
from src import event_bus
from src import market_calendar as cal
from src.kill_switch import KillSwitch
from src.risk_manager import RiskManager

DAY = dt.date(2026, 3, 2)


@pytest.fixture
def client():
    return fakeredis.FakeStrictRedis(server=fakeredis.FakeServer(), decode_responses=True)


@pytest.fixture(autouse=True)
def _bus(monkeypatch, client, tmp_path):
    monkeypatch.setattr(event_bus, "get_client", lambda *a, **k: client)
    monkeypatch.setattr(event_bus, "ping", lambda *a, **k: True)
    # Never touch the developer's real equity state file.
    monkeypatch.setattr(config, "EQUITY_STATE_PATH", str(tmp_path / "equity.json"))


class FakeNeo:
    pass


def with_equity(monkeypatch, value=500_000.0, raises=None):
    """Stub SessionEquity.resolve, which is tested in its own right in test_account.py."""
    class Resolved:
        equity = value

    def resolve(self, day, neo):
        if raises:
            raise raises
        return Resolved()

    monkeypatch.setattr("src.account.SessionEquity.resolve", resolve)


# ---------------------------------------------------------------------------
class TestEquityResolution:
    """D1: the day's sizing base, and the decision to fail closed when it is unknown."""

    def test_the_brokers_figure_is_used(self, monkeypatch):
        with_equity(monkeypatch, 750_000.0)
        manager = RiskManager(FakeNeo())
        assert manager._resolve_equity(DAY) == 750_000.0

    def test_a_broker_failure_returns_nothing(self, monkeypatch, caplog):
        """A wrong figure would mis-size every trade today by the same factor, and no
        percentage-based cap would notice, because they all share the wrong base."""
        with_equity(monkeypatch, raises=RuntimeError("limits() unavailable"))
        manager = RiskManager(FakeNeo())
        with caplog.at_level("ERROR"):
            assert manager._resolve_equity(DAY) is None
        assert "mis-size" in caplog.text

    def test_a_broker_failure_halts(self, monkeypatch, client):
        with_equity(monkeypatch, raises=RuntimeError("limits() unavailable"))
        RiskManager(FakeNeo())._resolve_equity(DAY)
        assert KillSwitch(client).state().halted

    def test_no_broker_falls_back_and_says_it_is_not_live_valid(self, monkeypatch,
                                                                caplog):
        monkeypatch.setattr(config, "TOTAL_EQUITY", 250_000.0)
        manager = RiskManager(None)
        with caplog.at_level("WARNING"):
            assert manager._resolve_equity(DAY) == 250_000.0
        assert "NOT valid for live trading" in caplog.text

    def test_start_session_fails_closed(self, monkeypatch):
        with_equity(monkeypatch, raises=RuntimeError("no limits"))
        assert RiskManager(FakeNeo()).start_session(DAY) is False

    def test_start_session_arms_the_engine_on_success(self, monkeypatch):
        with_equity(monkeypatch, 600_000.0)
        manager = RiskManager(FakeNeo())
        assert manager.start_session(DAY) is True
        assert manager.engine.session_equity == 600_000.0
        assert manager.engine.session_day == DAY


class TestSignalParsing:
    def test_a_datetime_bar_time_is_accepted_directly(self):
        moment = cal.at(DAY, dt.time(10, 0))
        assert RiskManager._parse_bar_time(moment) == cal.to_ist(moment)

    def test_an_unknown_intent_is_rejected_loudly(self):
        """Coercing an unrecognised intent to a default would turn a typo in an upstream
        payload into a real order in an unintended direction."""
        manager = RiskManager(None)
        with pytest.raises(ValueError, match="unknown intent"):
            manager._handle_signal({"instrument_id": "nse_cm:2885",
                                    "intent": "OPEN_SIDEWAYS", "ref_price": "1300"})


class TestCorrelationGroupResolution:
    """Unknown must mean unconstrained: the filter must never block a trade because a
    lookup failed. Blocking on a missing lookup would silently shrink the universe."""

    def test_it_falls_back_to_the_symbol_when_there_is_no_master(self):
        manager = RiskManager(None)
        assert isinstance(manager._correlation_group("nse_cm:HDFCBANK"), str)

    def test_an_unknown_instrument_is_ungrouped_not_blocked(self):
        manager = RiskManager(None)
        assert manager._correlation_group("nse_cm:ZZZZZZ") == ""

    def test_the_master_is_preferred_when_present(self):
        from src.instruments import InstrumentMaster, parse_scrip_master

        master = InstrumentMaster(parse_scrip_master([
            {"pSymbol": "1333", "pTrdSymbol": "HDFCBANK-EQ", "pSymbolName": "HDFCBANK",
             "pGroup": "EQ", "lLotSize": "1", "dTickSize": "5"},
        ]))
        manager = RiskManager(None, master)
        assert manager._correlation_group("nse_cm:1333") == "banking"

    def test_a_master_miss_falls_through_to_the_symbol(self):
        from src.instruments import InstrumentMaster

        manager = RiskManager(None, InstrumentMaster([]))
        assert isinstance(manager._correlation_group("nse_cm:2885"), str)


class TestTheMainLoop:
    def test_it_stops_when_redis_is_unreachable(self, monkeypatch, caplog):
        monkeypatch.setattr(event_bus, "ping", lambda *a, **k: False)
        with caplog.at_level("ERROR"):
            RiskManager(None).run()
        assert "Redis not reachable" in caplog.text

    def test_it_refuses_to_run_without_a_session(self, monkeypatch, caplog):
        """No equity means no sizing. Guessing one would be worse than not trading."""
        with_equity(monkeypatch, raises=RuntimeError("no limits"))
        manager = RiskManager(FakeNeo())
        must_not_run = lambda **k: (_ for _ in ()).throw(
            AssertionError("entered the consume loop without a session"))
        monkeypatch.setattr(manager.signals, "read", must_not_run)
        with caplog.at_level("ERROR"):
            manager.run()
        assert "halting rather than guessing" in caplog.text

    def test_one_pass_wires_every_stage_in_order(self, monkeypatch):
        """Position updates must be read BEFORE signals: sizing and the drawdown breaker
        have to act on the freshest P&L available before any new signal is judged."""
        with_equity(monkeypatch)
        manager = RiskManager(FakeNeo())
        monkeypatch.setattr(cal, "now_ist", lambda: cal.at(DAY, dt.time(10, 0)))

        calls = []

        class Stop(RuntimeError):
            pass

        monkeypatch.setattr(manager.positions, "read",
                            lambda **k: calls.append("positions") or [])
        monkeypatch.setattr(manager.signals, "claim_stale",
                            lambda **k: calls.append("claim") or [])
        monkeypatch.setattr(manager.signals, "read",
                            lambda **k: calls.append("signals") or [])

        def backlog(*a, **k):
            calls.append("backlog")
            raise Stop()

        monkeypatch.setattr(manager.signals, "check_backlog", backlog)
        with pytest.raises(Stop):
            manager.run()
        assert calls == ["positions", "claim", "signals", "backlog"]

    def test_a_new_day_re_establishes_the_session(self, monkeypatch):
        with_equity(monkeypatch)
        manager = RiskManager(FakeNeo())

        class Stop(RuntimeError):
            pass

        days = iter([DAY, DAY, dt.date(2026, 3, 4)])
        monkeypatch.setattr(cal, "now_ist",
                            lambda: cal.at(next(days), dt.time(9, 20)))
        monkeypatch.setattr(manager.positions, "read", lambda **k: [])
        monkeypatch.setattr(manager.signals, "claim_stale", lambda **k: [])
        monkeypatch.setattr(manager.signals, "read", lambda **k: [])

        rounds = {"n": 0}

        def backlog(*a, **k):
            rounds["n"] += 1
            if rounds["n"] >= 2:
                raise Stop()

        monkeypatch.setattr(manager.signals, "check_backlog", backlog)
        with pytest.raises(Stop):
            manager.run()
        assert manager.engine.session_day == dt.date(2026, 3, 4)

    def test_it_stops_if_the_new_day_cannot_be_established(self, monkeypatch):
        """A broker that fails at the day roll must not leave the loop running with
        yesterday's equity."""
        with_equity(monkeypatch, 500_000.0)
        manager = RiskManager(FakeNeo())
        assert manager.start_session(DAY)

        days = iter([dt.date(2026, 3, 4)])
        monkeypatch.setattr(cal, "now_ist",
                            lambda: cal.at(next(days), dt.time(9, 20)))
        with_equity(monkeypatch, raises=RuntimeError("broker down"))
        must_not_read = lambda **k: (_ for _ in ()).throw(
            AssertionError("kept consuming after the session failed"))
        monkeypatch.setattr(manager.positions, "read", must_not_read)
        monkeypatch.setattr(event_bus, "ping", lambda *a, **k: True)
        manager.run()   # returns rather than looping


class TestStartup:
    def test_paper_mode_obtains_a_broker_session(self, monkeypatch):
        """Paper needs the broker too: equity must come from it (D1). Branching on
        DRY_RUN here used to skip this, so sizing silently fell back to a configured
        figure and every recorded paper session was invalid."""
        import src.risk_manager as mod

        monkeypatch.setattr(config, "TRADING_MODE", "paper")
        neo = FakeNeo()
        monkeypatch.setattr("src.auth_session.get_session", lambda *a, **k: neo)
        monkeypatch.setattr("src.instruments.load_or_download_master",
                            lambda *a, **k: None)
        monkeypatch.setattr(mod.kotak_api, "bound_network_calls", lambda *a, **k: None)
        started = []
        monkeypatch.setattr(mod.RiskManager, "run", lambda self: started.append(self))
        mod.main()
        assert started[0].neo is neo

    def test_dry_run_needs_no_broker(self, monkeypatch):
        import src.risk_manager as mod

        monkeypatch.setattr(config, "TRADING_MODE", "dry_run")
        monkeypatch.setattr("src.instruments.load_or_download_master",
                            lambda *a, **k: None)
        monkeypatch.setattr(mod.kotak_api, "bound_network_calls", lambda *a, **k: None)
        started = []
        monkeypatch.setattr(mod.RiskManager, "run", lambda self: started.append(self))
        mod.main()
        assert started[0].neo is None

    def test_a_missing_instrument_master_degrades_gracefully(self, monkeypatch, caplog):
        """The correlation filter is a refinement, not a safety control. Losing it must
        not stop the bot — but it must be said out loud, because concentration limits
        quietly stop applying."""
        import src.risk_manager as mod

        monkeypatch.setattr(config, "TRADING_MODE", "dry_run")
        monkeypatch.setattr(mod.kotak_api, "bound_network_calls", lambda *a, **k: None)
        monkeypatch.setattr(
            "src.instruments.load_or_download_master",
            lambda *a, **k: (_ for _ in ()).throw(OSError("no scrip master")))
        started = []
        monkeypatch.setattr(mod.RiskManager, "run", lambda self: started.append(self))
        with caplog.at_level("WARNING"):
            mod.main()
        assert started and started[0].instruments is None
        assert "ungrouped" in caplog.text

    def test_network_calls_are_bounded_first(self, monkeypatch):
        import src.risk_manager as mod

        monkeypatch.setattr(config, "TRADING_MODE", "dry_run")
        monkeypatch.setattr("src.instruments.load_or_download_master",
                            lambda *a, **k: None)
        bounded = []
        monkeypatch.setattr(mod.kotak_api, "bound_network_calls",
                            lambda *a, **k: bounded.append(1))
        monkeypatch.setattr(mod.RiskManager, "run", lambda self: None)
        mod.main()
        assert bounded == [1]
