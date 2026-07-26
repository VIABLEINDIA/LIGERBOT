"""Position manager — the daemon layer around the already-tested book.

`PositionBook` is pure and well covered by `test_position_manager.py`. The class that
*runs* it was not: lines 304-431 had never been executed. That split is the same one the
coverage audit found everywhere — the logic is proven, the process that carries it is not.

What lives only in the daemon, and therefore only here:

* **B3 enforcement at the consumer boundary.** The book must move on genuine fills and
  nothing else. An acceptance-only event reaching `apply_fill` is precisely the defect that
  made the bot believe it owned things it did not.
* **Reconciliation refusing to guess.** The documented near-miss: the SDK returns `None`
  rather than raising when `positions()` fails, so a transient error looked exactly like
  *"the broker holds nothing"* — and reconciliation would then discard every open position
  from the book. A bot that has just been told it is flat while holding three positions
  will happily open three more.
* **Halting on an unverifiable book.** A session expiry means we cannot check what we own.
  Trading on an unverifiable book is how a bookkeeping bug becomes a large position.
"""
from __future__ import annotations

import fakeredis
import pytest

import config
from src import event_bus, kotak_api
from src.kill_switch import KillSwitch
from src.position_manager import PositionManager


@pytest.fixture
def client():
    return fakeredis.FakeStrictRedis(server=fakeredis.FakeServer(), decode_responses=True)


@pytest.fixture(autouse=True)
def _bus(monkeypatch, client):
    monkeypatch.setattr(event_bus, "get_client", lambda *a, **k: client)
    monkeypatch.setattr(event_bus, "ping", lambda *a, **k: True)


@pytest.fixture(autouse=True)
def _isolate_alerter():
    from src.alerting import reset_alerter

    reset_alerter()
    yield
    reset_alerter()


class FakeNeo:
    def __init__(self, payload=None):
        self.payload = payload if payload is not None else {"data": []}
        self.calls = 0

    def positions(self):
        self.calls += 1
        return self.payload


def fill(**overrides) -> dict:
    payload = {
        "instrument_id": "nse_cm:2885",
        "status": "FILLED",
        "filled_quantity": "10",
        "average_fill_price": "1300.0",
        "side": "BUY",
    }
    payload.update(overrides)
    return payload


def broker_row(token="2885", buy=10, sell=0, segment="nse_cm") -> dict:
    return {"tok": token, "exSeg": segment, "flBuyQty": str(buy), "flSellQty": str(sell)}


def updates(client) -> list[dict]:
    return [row[1] for row in client.xrange(config.STREAM_POSITION_UPDATES)]


# ---------------------------------------------------------------------------
class TestFillConsumption:
    def test_a_filled_event_moves_the_book(self):
        manager = PositionManager()
        manager._handle_fill(fill())
        assert manager.book.positions["nse_cm:2885"].quantity == 10

    def test_a_partial_moves_the_book_too(self):
        manager = PositionManager()
        manager._handle_fill(fill(status="PARTIAL", filled_quantity="4"))
        assert manager.book.positions["nse_cm:2885"].quantity == 4

    @pytest.mark.parametrize("status", ["ACKED", "SENT", "REJECTED", "CANCELLED",
                                        "EXPIRED", ""])
    def test_non_fill_statuses_are_ignored(self, status):
        """B3 at the consumer boundary. Acceptance is not execution, and treating it as
        one is how the bot came to believe it owned things it did not."""
        manager = PositionManager()
        manager._handle_fill(fill(status=status))
        assert manager.book.positions == {}

    def test_a_sell_is_signed_negative(self):
        manager = PositionManager()
        manager._handle_fill(fill(side="SELL"))
        assert manager.book.positions["nse_cm:2885"].quantity == -10

    def test_a_round_trip_realises_pnl(self):
        manager = PositionManager()
        manager._handle_fill(fill())
        manager._handle_fill(fill(side="SELL", average_fill_price="1310.0"))
        assert manager.book.realized_pnl_today == pytest.approx(100.0)
        assert manager.book.positions == {}

    def test_a_fill_without_an_instrument_is_an_error(self):
        manager = PositionManager()
        payload = fill()
        del payload["instrument_id"]
        with pytest.raises(ValueError, match="instrument"):
            manager._handle_fill(payload)

    def test_a_fill_without_a_usable_price_is_an_error(self):
        """Booking a position at zero would corrupt every subsequent average and every
        P&L figure derived from it. Better to fail the entry and let it dead-letter."""
        manager = PositionManager()
        with pytest.raises(ValueError, match="price"):
            manager._handle_fill(fill(average_fill_price="0"))

    def test_a_zero_quantity_fill_is_ignored(self):
        manager = PositionManager()
        manager._handle_fill(fill(filled_quantity="0"))
        assert manager.book.positions == {}

    def test_every_fill_publishes_the_book(self, client):
        """The risk manager's drawdown breaker is driven by this stream. A fill that
        does not publish is a fill the breaker never sees (B2)."""
        manager = PositionManager()
        manager._handle_fill(fill())
        rows = updates(client)
        assert rows and rows[-1]["open_positions"] == "1"

    def test_costs_are_carried_into_the_book(self, client):
        manager = PositionManager()
        manager._handle_fill(fill(costs="42.5"))
        assert manager.book.costs_today == pytest.approx(42.5)


class TestReconciliationRefusesToGuess:
    """The documented near-miss, tested from every side.

    `positions()` returning nothing must never be read as "flat". If it were, one
    transient network blip would empty the book, and the bot would size its next trades
    as though it held nothing while holding a full set.
    """

    def _with_position(self, neo=None):
        manager = PositionManager(neo)
        manager._handle_fill(fill())
        return manager

    def test_a_failed_call_leaves_the_book_intact(self, monkeypatch, caplog):
        manager = self._with_position(FakeNeo())

        def boom(*a, **k):
            raise kotak_api.KotakAPIError("positions() unavailable")

        monkeypatch.setattr(kotak_api, "safe_call", boom)
        with caplog.at_level("ERROR"):
            assert manager.reconcile_now() is None
        assert manager.book.positions["nse_cm:2885"].quantity == 10
        assert "NOT assumed flat" in caplog.text

    def test_a_session_expiry_halts_rather_than_reconciling(self, client, monkeypatch):
        """We cannot verify what we own. Continuing to trade on an unverifiable book is
        the failure this prevents."""
        manager = self._with_position(FakeNeo())

        def expired(*a, **k):
            raise kotak_api.KotakSessionExpired("2fa required")

        monkeypatch.setattr(kotak_api, "safe_call", expired)
        manager.reconcile_now()
        assert KillSwitch(client).state().halted
        assert manager.book.positions["nse_cm:2885"].quantity == 10

    def test_no_broker_means_no_reconciliation(self):
        manager = self._with_position(None)
        assert manager.reconcile_now() is None
        assert manager.book.positions["nse_cm:2885"].quantity == 10

    def test_a_matching_broker_view_reconciles_clean(self, client):
        manager = self._with_position(FakeNeo({"data": [broker_row()]}))
        result = manager.reconcile_now()
        assert result.clean
        assert not KillSwitch(client).state().halted

    def test_a_genuine_empty_response_still_drops_the_position(self, client):
        """The counterpart. A *successful* call reporting nothing is real information and
        must be acted on — the guard is against failures, not against empty books."""
        manager = self._with_position(FakeNeo({"data": []}))
        result = manager.reconcile_now()
        assert manager.book.positions == {}
        assert result.missing_at_broker == ["nse_cm:2885"]

    def test_a_mismatch_raises_a_critical_alert(self, client):
        manager = self._with_position(FakeNeo({"data": [broker_row(buy=25)]}))
        manager.reconcile_now()
        alerts = client.xrange(config.STREAM_ALERTS)
        assert any(row[1]["severity"] == "critical" for row in alerts)

    def test_a_mismatch_beyond_the_threshold_halts(self, client, monkeypatch):
        monkeypatch.setattr(config, "RECONCILE_HALT_THRESHOLD", 1)
        manager = self._with_position(FakeNeo({"data": [broker_row(buy=25)]}))
        manager.reconcile_now()
        assert KillSwitch(client).state().halted

    def test_a_mismatch_below_the_threshold_does_not_halt(self, client, monkeypatch):
        monkeypatch.setattr(config, "RECONCILE_HALT_THRESHOLD", 99)
        manager = self._with_position(FakeNeo({"data": [broker_row(buy=25)]}))
        manager.reconcile_now()
        assert not KillSwitch(client).state().halted

    def test_the_broker_wins_on_a_mismatch(self):
        manager = self._with_position(FakeNeo({"data": [broker_row(buy=25)]}))
        manager.reconcile_now()
        assert manager.book.positions["nse_cm:2885"].quantity == 25

    def test_a_position_only_at_the_broker_is_adopted(self):
        manager = PositionManager(FakeNeo({"data": [broker_row(token="1333", buy=7)]}))
        manager.reconcile_now()
        assert manager.book.positions["nse_cm:1333"].quantity == 7

    def test_a_broken_alerter_does_not_stop_the_halt(self, client, monkeypatch):
        """Alerting is observability; it must never be able to suppress a safety action."""
        monkeypatch.setattr(config, "RECONCILE_HALT_THRESHOLD", 1)
        monkeypatch.setattr("src.alerting.get_alerter",
                            lambda *a, **k: (_ for _ in ()).throw(ConnectionError("down")))
        manager = self._with_position(FakeNeo({"data": [broker_row(buy=25)]}))
        manager.reconcile_now()
        assert KillSwitch(client).state().halted

    def test_reconciliation_publishes_the_book(self, client):
        manager = self._with_position(FakeNeo({"data": [broker_row()]}))
        before = len(updates(client))
        manager.reconcile_now()
        assert len(updates(client)) > before


class TestTheMainLoop:
    def test_it_stops_when_redis_is_unreachable(self, monkeypatch, caplog):
        monkeypatch.setattr(event_bus, "ping", lambda *a, **k: False)
        manager = PositionManager()
        with caplog.at_level("ERROR"):
            manager.run()
        assert "Redis not reachable" in caplog.text

    def test_one_pass_reads_fills_and_reconciles(self, monkeypatch, client):
        manager = PositionManager(FakeNeo({"data": [broker_row()]}))

        class Stop(RuntimeError):
            pass

        monkeypatch.setattr(manager.consumer, "claim_stale", lambda **k: [])
        monkeypatch.setattr(manager.consumer, "read", lambda **k: [("1-1", fill())])

        reconciles = {"n": 0}
        original = manager.reconcile_now

        def counted():
            reconciles["n"] += 1
            if reconciles["n"] > 1:
                raise Stop()
            return original()

        monkeypatch.setattr(manager, "reconcile_now", counted)
        monkeypatch.setattr(config, "RECONCILE_INTERVAL_SECONDS", 0)
        with pytest.raises(Stop):
            manager.run()
        assert manager.book.positions

    def test_a_new_day_resets_the_daily_book(self, monkeypatch):
        import datetime as dt

        from src import market_calendar as cal

        manager = PositionManager()
        manager.book.start_session(dt.date(2026, 3, 2))
        manager.book.realized_pnl_today = 5_000.0

        class Stop(RuntimeError):
            pass

        days = iter([dt.date(2026, 3, 2), dt.date(2026, 3, 4)])
        monkeypatch.setattr(cal, "now_ist",
                            lambda: cal.at(next(days), dt.time(9, 20)))
        monkeypatch.setattr(manager.consumer, "claim_stale", lambda **k: [])

        def stop(**k):
            raise Stop()

        monkeypatch.setattr(manager.consumer, "read", stop)
        with pytest.raises(Stop):
            manager.run()
        assert manager.book.session_day == dt.date(2026, 3, 4)
        assert manager.book.realized_pnl_today == 0.0


class TestStartup:
    def test_a_broker_session_is_obtained_when_the_mode_needs_one(self, monkeypatch):
        import src.position_manager as mod

        monkeypatch.setattr(config, "TRADING_MODE", "paper")
        monkeypatch.setattr("src.auth_session.get_session", lambda *a, **k: FakeNeo())
        monkeypatch.setattr(mod.kotak_api, "bound_network_calls", lambda *a, **k: None)
        started = []
        monkeypatch.setattr(mod.PositionManager, "run", lambda self: started.append(self))
        mod.main()
        assert isinstance(started[0].neo, FakeNeo)

    def test_dry_run_disables_reconciliation_loudly(self, monkeypatch, caplog):
        """Silently running without reconciliation is how a paper session gets recorded
        that cannot afterwards be compared against a backtest."""
        import src.position_manager as mod

        monkeypatch.setattr(config, "TRADING_MODE", "dry_run")
        monkeypatch.setattr(mod.kotak_api, "bound_network_calls", lambda *a, **k: None)
        started = []
        monkeypatch.setattr(mod.PositionManager, "run", lambda self: started.append(self))
        with caplog.at_level("WARNING"):
            mod.main()
        assert started[0].neo is None
        assert "DISABLED" in caplog.text

    def test_network_calls_are_bounded_first(self, monkeypatch):
        import src.position_manager as mod

        monkeypatch.setattr(config, "TRADING_MODE", "dry_run")
        bounded = []
        monkeypatch.setattr(mod.kotak_api, "bound_network_calls",
                            lambda *a, **k: bounded.append(1))
        monkeypatch.setattr(mod.PositionManager, "run", lambda self: None)
        mod.main()
        assert bounded == [1]
