"""Operator-facing entry points: the kill-switch CLI and the paper broker's guard.

These are CLIs, which is why they were skipped. That was the wrong instinct for at least
one of them.

**`python -m src.kill_switch halt "reason"` is the emergency stop.** It is the thing a
human reaches for when something is going wrong and the automated guards have not caught
it. Its whole value is being reliable on the worst day of the year, and it had never been
executed by a test. A halt command that silently fails is worse than not having one,
because the operator believes the bot is stopped.

The paper broker's `main()` guard is the mirror of the execution engine's: exactly one
module may fill orders in any given mode, and both refuse rather than warn. Two fillers
running at once double-count every trade while looking entirely healthy.
"""
from __future__ import annotations

import fakeredis
import pytest

import config
from src import event_bus
from src.kill_switch import KillSwitch


@pytest.fixture
def client():
    return fakeredis.FakeStrictRedis(server=fakeredis.FakeServer(), decode_responses=True)


@pytest.fixture(autouse=True)
def _bus(monkeypatch, client):
    monkeypatch.setattr(event_bus, "get_client", lambda *a, **k: client)
    monkeypatch.setattr(event_bus, "ping", lambda *a, **k: True)


class TestKillSwitchCLI:
    def test_halt_engages_the_switch(self, client, monkeypatch, capsys):
        import src.kill_switch as mod

        monkeypatch.setattr("sys.argv", ["kill_switch", "halt", "manual stop"])
        mod.main()
        assert KillSwitch(client).state().halted
        assert "manual stop" in capsys.readouterr().out

    def test_halt_tells_the_operator_exits_still_work(self, monkeypatch, capsys):
        """The one thing someone hitting the emergency stop most needs to know."""
        import src.kill_switch as mod

        monkeypatch.setattr("sys.argv", ["kill_switch", "halt", "manual stop"])
        mod.main()
        assert "Only new entries are blocked" in capsys.readouterr().out

    def test_status_reports_a_running_bot(self, monkeypatch, capsys):
        import src.kill_switch as mod

        monkeypatch.setattr("sys.argv", ["kill_switch", "status"])
        mod.main()
        assert capsys.readouterr().out.strip()

    def test_status_reports_the_halt_reason(self, client, monkeypatch, capsys):
        import src.kill_switch as mod

        KillSwitch(client).halt("drawdown breached")
        monkeypatch.setattr("sys.argv", ["kill_switch", "status"])
        mod.main()
        assert "drawdown breached" in capsys.readouterr().out

    def test_clear_lifts_the_halt(self, client, monkeypatch, capsys):
        import src.kill_switch as mod

        KillSwitch(client).halt("test")
        monkeypatch.setattr("sys.argv", ["kill_switch", "clear"])
        mod.main()
        assert not KillSwitch(client).state().halted
        assert "cleared" in capsys.readouterr().out.lower()

    def test_it_exits_nonzero_when_redis_is_unreachable(self, monkeypatch, capsys):
        """And says the modules are already refusing entries — otherwise the operator
        cannot tell whether an unreachable kill switch means they are exposed."""
        import src.kill_switch as mod

        monkeypatch.setattr("sys.argv", ["kill_switch", "status"])
        monkeypatch.setattr(event_bus, "ping", lambda *a, **k: False)
        with pytest.raises(SystemExit) as exit_info:
            mod.main()
        assert exit_info.value.code == 1
        assert "already" in capsys.readouterr().out

    def test_a_corrupt_payload_still_reads_as_halted(self, client):
        """Fail closed on garbage. Treating an unparseable halt record as 'not halted'
        would lift a halt because of a serialisation bug."""
        client.set(config.HALT_KEY, "not-json-at-all")
        state = KillSwitch(client).state()
        assert state.halted
        assert "not-json-at-all" in state.reason

    def test_is_halted_matches_state(self, client):
        switch = KillSwitch(client)
        assert switch.is_halted() is False
        switch.halt("x")
        assert switch.is_halted() is True


class TestPaperBrokerGuard:
    @pytest.mark.parametrize("mode", ["dry_run", "live"])
    def test_it_refuses_to_run_outside_paper_mode(self, monkeypatch, mode, caplog):
        """The mirror of the execution engine's guard. Exactly one module fills."""
        import src.paper_broker as mod

        monkeypatch.setattr(config, "TRADING_MODE", mode)
        started = []
        monkeypatch.setattr(mod.PaperBroker, "run", lambda self: started.append(self))
        with caplog.at_level("ERROR"):
            mod.main()
        assert started == []
        assert "twice" in caplog.text

    def test_it_runs_in_paper_mode(self, monkeypatch):
        import src.paper_broker as mod

        monkeypatch.setattr(config, "TRADING_MODE", "paper")
        started = []
        monkeypatch.setattr(mod.PaperBroker, "run", lambda self: started.append(self))
        mod.main()
        assert started

    def test_it_stops_when_redis_is_unreachable(self, monkeypatch, caplog):
        import src.paper_broker as mod

        monkeypatch.setattr(config, "TRADING_MODE", "paper")
        monkeypatch.setattr(event_bus, "ping", lambda *a, **k: False)
        broker = mod.PaperBroker()
        with caplog.at_level("ERROR"):
            broker.run()
        assert "Redis not reachable" in caplog.text

    def test_one_loop_pass_consumes_orders_and_bars(self, monkeypatch):
        import src.paper_broker as mod

        monkeypatch.setattr(config, "TRADING_MODE", "paper")
        broker = mod.PaperBroker()
        calls = []

        class Stop(RuntimeError):
            pass

        monkeypatch.setattr(broker.orders, "read",
                            lambda **k: calls.append("orders") or [])
        monkeypatch.setattr(broker.bars, "claim_stale",
                            lambda **k: calls.append("claim") or [])

        def stop(**k):
            calls.append("bars")
            raise Stop()

        monkeypatch.setattr(broker.bars, "read", stop)
        with pytest.raises(Stop):
            broker.run()
        assert calls == ["orders", "claim", "bars"]


class TestWiringCheckConsumer:
    """`src/consumer.py` is scaffolding — the documented "smallest possible reader",
    superseded by the strategy engine. Tested because it ships, not because it matters.
    It is a reasonable candidate for deletion."""

    def test_it_stops_when_redis_is_unreachable(self, monkeypatch, caplog):
        import src.consumer as mod

        monkeypatch.setattr(event_bus, "ping", lambda *a, **k: False)
        with caplog.at_level("ERROR"):
            mod.main()
        assert "Redis not reachable" in caplog.text

    def test_it_prints_ticks_it_reads(self, monkeypatch, caplog):
        import src.consumer as mod

        class Stop(RuntimeError):
            pass

        calls = {"n": 0}

        def read_new(*a, **k):
            calls["n"] += 1
            if calls["n"] > 1:
                raise Stop()
            return [("1-1", {"instrument": "RELIANCE", "ltp": 1300.5})], "1-1"

        monkeypatch.setattr(event_bus, "read_new", read_new)
        with caplog.at_level("INFO"), pytest.raises(Stop):
            mod.main()
        assert "RELIANCE" in caplog.text
