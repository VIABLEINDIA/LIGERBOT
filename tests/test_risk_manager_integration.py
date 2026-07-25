"""Risk-manager adapter tests — the Redis-facing layer around the pure engine.

The regression that motivated these: the adapter judged the session phase against
**wall-clock now** while the backtester judges it at the signal's own ``bar_time``. In
live trading those usually coincide, so it looked fine — but any queueing, restart or
replay made the live path disagree with the backtested one, which is the exact class of
divergence DESIGN.md 2.1 exists to prevent.
"""
from __future__ import annotations

import datetime as dt

import fakeredis
import pytest

import config
from src import event_bus
from src import market_calendar as cal

TRADING_DAY = dt.date(2026, 7, 24)   # a Friday


@pytest.fixture
def client(monkeypatch):
    server = fakeredis.FakeServer()

    def factory(*_a, **_k):
        return fakeredis.FakeStrictRedis(server=server, decode_responses=True)

    monkeypatch.setattr(event_bus, "get_client", factory)
    monkeypatch.setattr(event_bus, "ping", lambda *a, **k: True)
    return factory()


@pytest.fixture
def manager(client, monkeypatch):
    from src import feed_health
    from src.risk_manager import RiskManager

    monkeypatch.setattr(config, "TOTAL_EQUITY", 1_000_000.0)
    rm = RiskManager(None)
    rm.start_session(TRADING_DAY)
    # The feed gate fails closed, so a test that does not mark the instrument live is
    # testing the stale path whether it means to or not.
    feed_health.publish_liveness(client, "nse_cm:2885", ttl_seconds=3600)
    return rm


def signal_payload(hour: int, minute: int, **overrides) -> dict:
    payload = {
        "instrument_id": "nse_cm:2885",
        "intent": "OPEN_LONG",
        "ref_price": 1300.0,
        "stop_loss": 1274.0,
        "bar_time": cal.at(TRADING_DAY, dt.time(hour, minute)).isoformat(),
        "strategy_name": "test",
    }
    payload.update(overrides)
    return payload


def approved(client) -> list:
    return client.xrange(config.STREAM_APPROVED_ORDERS)


class TestPhaseUsesSignalTime:
    def test_signal_inside_the_entry_window_is_approved(self, manager, client, monkeypatch):
        """The regression. Judged at bar_time, 10:31 on a Friday is ENTRY —
        regardless of what the wall clock happens to say when this test runs."""
        monkeypatch.setattr(config, "MAX_SIGNAL_AGE_SECONDS", 1e9)
        manager._handle_signal(signal_payload(10, 31))
        assert len(approved(client)) == 1

    def test_signal_in_the_opening_range_is_refused(self, manager, client, monkeypatch):
        monkeypatch.setattr(config, "MAX_SIGNAL_AGE_SECONDS", 1e9)
        manager._handle_signal(signal_payload(9, 20))   # before 09:30
        assert approved(client) == []

    def test_signal_after_the_entry_cutoff_is_refused(self, manager, client, monkeypatch):
        monkeypatch.setattr(config, "MAX_SIGNAL_AGE_SECONDS", 1e9)
        manager._handle_signal(signal_payload(15, 0))   # after 14:45
        assert approved(client) == []

    def test_exit_is_permitted_after_the_entry_cutoff(self, manager, client, monkeypatch):
        """Exits stay available where entries do not — the standing asymmetry."""
        monkeypatch.setattr(config, "MAX_SIGNAL_AGE_SECONDS", 1e9)
        manager._handle_signal(signal_payload(10, 31))
        manager.engine.on_open_fill("nse_cm:2885", 100, 1300.0, 1274.0)

        manager._handle_signal(signal_payload(
            15, 0, intent="CLOSE_LONG", stop_loss=0))
        orders = approved(client)
        assert len(orders) == 2
        assert orders[-1][1]["intent"] == "CLOSE_LONG"


class TestSignalStaleness:
    def test_an_old_signal_is_refused_for_entry(self, manager, client, monkeypatch):
        """Judging the phase at bar_time is what makes this check necessary: without
        it, an hours-old signal would pass on the strength of when it was generated."""
        monkeypatch.setattr(config, "MAX_SIGNAL_AGE_SECONDS", 60.0)
        manager._handle_signal(signal_payload(10, 31))
        assert approved(client) == []

    def test_a_fresh_signal_passes(self, manager, client, monkeypatch):
        monkeypatch.setattr(config, "MAX_SIGNAL_AGE_SECONDS", 1e9)
        manager._handle_signal(signal_payload(10, 31))
        assert len(approved(client)) == 1

    def test_future_bar_times_are_not_treated_as_stale(self, manager):
        """Replay and clock skew produce negative ages; those are not staleness."""
        from src.risk_manager import RiskManager

        future = cal.now_ist() + dt.timedelta(hours=1)
        assert RiskManager._signal_age_seconds(future) is None

    def test_unparseable_bar_time_defaults_to_now(self):
        """A missing time should read as fresh, not as ancient and silently dropped."""
        from src.risk_manager import RiskManager

        parsed = RiskManager._parse_bar_time("not-a-timestamp")
        assert abs((cal.now_ist() - parsed).total_seconds()) < 5


class TestOrderPayload:
    @pytest.fixture(autouse=True)
    def _fresh(self, monkeypatch):
        monkeypatch.setattr(config, "MAX_SIGNAL_AGE_SECONDS", 1e9)

    def test_approved_order_carries_a_client_order_id(self, manager, client):
        manager._handle_signal(signal_payload(10, 31))
        fields = approved(client)[0][1]
        assert fields["client_order_id"]
        assert fields["correlation_id"] == fields["client_order_id"]

    def test_client_order_id_is_deterministic_for_the_same_signal(self, manager, client):
        payload = signal_payload(10, 31)
        manager._handle_signal(payload)
        first = approved(client)[0][1]["client_order_id"]

        manager.engine.positions.clear()   # allow a second evaluation
        manager._handle_signal(payload)
        assert approved(client)[1][1]["client_order_id"] == first

    def test_order_carries_the_stop_and_risk(self, manager, client):
        manager._handle_signal(signal_payload(10, 31))
        fields = approved(client)[0][1]
        assert float(fields["stop_loss"]) == 1274.0
        assert float(fields["risk_amount"]) > 0


class TestGates:
    @pytest.fixture(autouse=True)
    def _fresh(self, monkeypatch):
        monkeypatch.setattr(config, "MAX_SIGNAL_AGE_SECONDS", 1e9)

    def test_kill_switch_blocks_entries(self, manager, client):
        manager.kill_switch.halt("test halt")
        manager._handle_signal(signal_payload(10, 31))
        assert approved(client) == []

    def test_kill_switch_does_not_block_exits(self, manager, client):
        manager._handle_signal(signal_payload(10, 31))
        manager.engine.on_open_fill("nse_cm:2885", 100, 1300.0, 1274.0)
        manager.kill_switch.halt("test halt")

        manager._handle_signal(signal_payload(11, 0, intent="CLOSE_LONG", stop_loss=0))
        assert len(approved(client)) == 2

    def test_stale_feed_blocks_entries(self, manager, client, monkeypatch):
        from src import feed_health

        monkeypatch.setattr(feed_health, "is_feed_live", lambda *a, **k: False)
        manager._handle_signal(signal_payload(10, 31))
        assert approved(client) == []

    def test_signal_without_a_stop_is_refused(self, manager, client):
        manager._handle_signal(signal_payload(10, 31, stop_loss=0))
        assert approved(client) == []

    def test_missing_instrument_raises(self, manager):
        with pytest.raises(ValueError, match="without an instrument"):
            manager._handle_signal({"intent": "OPEN_LONG", "ref_price": 100.0})


class TestPositionUpdates:
    def test_pnl_from_the_position_manager_drives_the_breaker(self, manager, client):
        """B2's fix, end to end: the breaker acts on the position manager's figures."""
        equity = manager.engine.session_equity
        manager._handle_position_update({"net_pnl_today": -0.025 * equity})
        assert manager.engine.halted
        assert manager.kill_switch.is_halted()

    def test_small_loss_does_not_halt(self, manager):
        manager._handle_position_update(
            {"net_pnl_today": -0.005 * manager.engine.session_equity})
        assert not manager.engine.halted

    def test_update_without_pnl_is_ignored(self, manager):
        manager._handle_position_update({"open_positions": 2})
        assert manager.engine.realized_pnl_today == 0.0
