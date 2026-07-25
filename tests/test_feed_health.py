"""Feed staleness, reconnection and kill-switch tests (DESIGN.md 3.5, 3.7)."""
from __future__ import annotations

import datetime as dt

import fakeredis
import pytest

import config
from src import market_calendar as cal
from src.feed_health import FeedMonitor, FeedState, Heartbeat, ReconnectPolicy
from src.kill_switch import KillSwitch

DAY = dt.date(2026, 7, 23)
MIDDAY = cal.at(DAY, dt.time(12, 0))
OVERNIGHT = cal.at(DAY, dt.time(22, 0))


class TestStaleness:
    """B10: a dead socket left the bot computing indicators on a frozen price."""

    def test_recent_tick_is_live(self):
        monitor = FeedMonitor(stale_after_seconds=30)
        monitor.record_tick("nse_cm:1", 1300.0, now=1000.0)
        monitor.evaluate(now=1010.0, moment=MIDDAY)
        assert monitor.state_of("nse_cm:1") is FeedState.LIVE

    def test_silence_beyond_the_threshold_is_stale(self):
        monitor = FeedMonitor(stale_after_seconds=30)
        monitor.record_tick("nse_cm:1", 1300.0, now=1000.0)
        monitor.evaluate(now=1100.0, moment=MIDDAY)
        assert monitor.state_of("nse_cm:1") is FeedState.STALE

    def test_stale_blocks_entries_but_never_exits(self):
        """The asymmetry: not seeing a price is a reason to stop taking risk,
        never a reason to be trapped in it."""
        monitor = FeedMonitor(stale_after_seconds=30)
        monitor.record_tick("nse_cm:1", 1300.0, now=1000.0)
        monitor.evaluate(now=1100.0, moment=MIDDAY)
        assert not monitor.allows_entry("nse_cm:1")
        assert monitor.state_of("nse_cm:1").allows_exit

    def test_staleness_is_per_instrument(self):
        """Feeds do not fail uniformly — one symbol can die while others continue."""
        monitor = FeedMonitor(stale_after_seconds=30)
        monitor.record_tick("nse_cm:1", 1300.0, now=1000.0)
        monitor.record_tick("nse_cm:2", 1650.0, now=1090.0)
        monitor.evaluate(now=1100.0, moment=MIDDAY)
        assert monitor.state_of("nse_cm:1") is FeedState.STALE
        assert monitor.state_of("nse_cm:2") is FeedState.LIVE

    def test_recovery_returns_to_live(self):
        monitor = FeedMonitor(stale_after_seconds=30)
        monitor.record_tick("nse_cm:1", 1300.0, now=1000.0)
        monitor.evaluate(now=1100.0, moment=MIDDAY)
        monitor.record_tick("nse_cm:1", 1301.0, now=1105.0)
        monitor.evaluate(now=1110.0, moment=MIDDAY)
        assert monitor.state_of("nse_cm:1") is FeedState.LIVE

    def test_nothing_is_stale_outside_market_hours(self):
        """Silence overnight is correct; alerting on it trains people to ignore alerts."""
        monitor = FeedMonitor(stale_after_seconds=30)
        monitor.record_tick("nse_cm:1", 1300.0, now=1000.0)
        monitor.evaluate(now=99_000.0, moment=OVERNIGHT)
        assert monitor.state_of("nse_cm:1") is not FeedState.STALE

    def test_tracked_but_never_seen_is_unknown_not_live(self):
        monitor = FeedMonitor()
        monitor.track(["nse_cm:1"])
        monitor.evaluate(now=1000.0, moment=MIDDAY)
        assert monitor.state_of("nse_cm:1") is FeedState.UNKNOWN
        assert not monitor.allows_entry("nse_cm:1")

    def test_disconnect_marks_everything_disconnected(self):
        monitor = FeedMonitor()
        monitor.record_tick("nse_cm:1", 1300.0, now=1000.0)
        monitor.mark_disconnected()
        monitor.evaluate(now=1001.0, moment=MIDDAY)
        assert monitor.state_of("nse_cm:1") is FeedState.DISCONNECTED
        assert not monitor.allows_entry("nse_cm:1")

    def test_all_stale_detects_a_whole_feed_outage(self):
        monitor = FeedMonitor(stale_after_seconds=30)
        monitor.record_tick("nse_cm:1", 1300.0, now=1000.0)
        monitor.record_tick("nse_cm:2", 1650.0, now=1000.0)
        monitor.evaluate(now=1010.0, moment=MIDDAY)
        assert not monitor.all_stale()
        monitor.evaluate(now=1200.0, moment=MIDDAY)
        assert monitor.all_stale()

    def test_state_change_callback_fires(self):
        seen = []
        monitor = FeedMonitor(
            stale_after_seconds=30,
            on_state_change=lambda i, old, new: seen.append((i, old, new)))
        monitor.record_tick("nse_cm:1", 1300.0, now=1000.0)
        monitor.evaluate(now=1100.0, moment=MIDDAY)
        assert any(new is FeedState.STALE for _, _, new in seen)

    def test_snapshot_reports_ages(self):
        monitor = FeedMonitor()
        monitor.record_tick("nse_cm:1", 1300.0)
        snapshot = monitor.snapshot()
        assert snapshot["tracked"] == 1
        assert snapshot["instruments"]["nse_cm:1"]["ticks"] == 1


class TestReconnectPolicy:
    def test_delay_grows_exponentially(self):
        policy = ReconnectPolicy(base_seconds=1.0, max_seconds=100.0,
                                 max_attempts=5, jitter=0.0)
        assert [policy.next_delay() for _ in range(4)] == [1.0, 2.0, 4.0, 8.0]

    def test_delay_is_capped(self):
        policy = ReconnectPolicy(base_seconds=1.0, max_seconds=5.0,
                                 max_attempts=10, jitter=0.0)
        delays = [policy.next_delay() for _ in range(8)]
        assert max(delays) <= 5.0

    def test_jitter_desynchronises_retries(self):
        """Without jitter, parallel reconnects hammer the broker in lockstep."""
        policy = ReconnectPolicy(base_seconds=10.0, max_attempts=50, jitter=0.5)
        delays = {round(policy.next_delay(), 4) for _ in range(10)}
        assert len(delays) > 1

    def test_exhaustion_returns_none(self):
        """Reconnecting forever looks resilient and is not — the process stays alive,
        passes liveness checks, and never trades."""
        policy = ReconnectPolicy(base_seconds=0.1, max_attempts=3, jitter=0.0)
        for _ in range(3):
            assert policy.next_delay() is not None
        assert policy.next_delay() is None
        assert policy.exhausted

    def test_reset_after_a_successful_connect(self):
        policy = ReconnectPolicy(base_seconds=1.0, max_attempts=3, jitter=0.0)
        policy.next_delay()
        policy.next_delay()
        policy.reset()
        assert policy.next_delay() == 1.0


class TestKillSwitch:
    @pytest.fixture
    def client(self):
        return fakeredis.FakeStrictRedis(decode_responses=True)

    def test_starts_clear(self, client):
        assert not KillSwitch(client).is_halted()

    def test_halt_and_clear(self, client):
        switch = KillSwitch(client)
        switch.halt("investigating fills")
        assert switch.is_halted()
        assert "investigating fills" in switch.state().reason
        switch.clear()
        assert not switch.is_halted()

    def test_halt_records_provenance(self, client):
        state = KillSwitch(client).halt("drawdown", source="risk_engine")
        assert state.source == "risk_engine"
        assert state.at and state.by

    def test_visible_to_a_separate_instance(self, client):
        """Every module must see the same switch without a restart."""
        KillSwitch(client).halt("stop")
        assert KillSwitch(client).is_halted()

    def test_fails_closed_when_redis_is_unreachable(self):
        """A bot that cannot check whether it was told to stop must not keep trading."""
        class BrokenClient:
            def get(self, _key):
                raise ConnectionError("redis down")

        state = KillSwitch(BrokenClient()).state()
        assert state.halted
        assert "unreadable" in state.reason

    def test_fail_open_is_opt_in(self):
        class BrokenClient:
            def get(self, _key):
                raise ConnectionError("redis down")

        with pytest.raises(ConnectionError):
            KillSwitch(BrokenClient(), fail_closed=False).state()

    def test_describe_is_human_readable(self, client):
        switch = KillSwitch(client)
        assert "RUNNING" in switch.state().describe()
        switch.halt("because")
        assert "HALTED" in switch.state().describe()


class TestHeartbeat:
    def test_beats_are_rate_limited(self):
        client = fakeredis.FakeStrictRedis(decode_responses=True)
        heartbeat = Heartbeat(client, "test", interval=999.0)
        assert heartbeat.beat(force=True)
        assert not heartbeat.beat()   # too soon
        assert client.xlen(config.STREAM_HEARTBEAT) == 1

    def test_payload_identifies_the_module(self):
        client = fakeredis.FakeStrictRedis(decode_responses=True)
        Heartbeat(client, "ingestion").beat(force=True)
        record = client.xrange(config.STREAM_HEARTBEAT)[-1][1]
        assert record["module"] == "ingestion"
