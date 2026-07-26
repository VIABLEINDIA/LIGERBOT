"""Daily briefings — the pre-open go/no-go and the post-close report.

The morning briefing is the last thing between a configuration mistake and a trading day
run on it. Its value is entirely in **refusing to say "ready" when it should not**, so
that is what is tested: every blocking condition individually, and the distinction between
a blocker and an advisory. An advisory that blocks trains people to ignore it; a blocker
that only warns is worse than absent, because it looks like a check was performed.

The evening briefing's job is narrower — make the day legible enough to decide whether to
trade the next one. The case that matters there is the quiet one: **no trades**. That reads
as "nothing happened", when it may equally mean the strategy found no setups or that every
signal was refused. Those call for opposite responses, so the briefing must not let them
look the same.
"""
from __future__ import annotations

import datetime as dt

import fakeredis
import pytest

import config
from src import event_bus
from src.briefing import (
    Check, build_evening, build_morning, read_alerts,
)

DAY = dt.date(2026, 3, 2)
HOLIDAY = dt.date(2026, 3, 3)


@pytest.fixture
def client():
    return fakeredis.FakeStrictRedis(server=fakeredis.FakeServer(), decode_responses=True)


@pytest.fixture(autouse=True)
def _bus(monkeypatch, client):
    monkeypatch.setattr(event_bus, "get_client", lambda *a, **k: client)
    monkeypatch.setattr(event_bus, "ping", lambda *a, **k: True)


def healthy(**overrides):
    fields = dict(day=DAY, equity=500_000.0, open_positions=0, strategy="trend_pullback")
    fields.update(overrides)
    day = fields.pop("day")
    return build_morning(day, **fields)


class TestMorningGoNoGo:
    def test_a_healthy_setup_is_ready(self):
        assert healthy().ready is True

    def test_a_non_trading_day_blocks(self):
        briefing = healthy(day=HOLIDAY)
        assert briefing.ready is False
        assert any("not an NSE trading day" in c.detail for c in briefing.blockers)

    def test_unresolved_equity_blocks(self):
        """Every trade would be mis-sized by the same factor, and no percentage cap
        would notice."""
        briefing = healthy(equity=None)
        assert briefing.ready is False
        assert any("mis-sized" in c.detail for c in briefing.blockers)

    def test_equity_below_the_floor_blocks(self, monkeypatch):
        monkeypatch.setattr(config, "MIN_EQUITY", 100_000.0)
        briefing = healthy(equity=50_000.0)
        assert briefing.ready is False
        assert any("floor" in c.detail for c in briefing.blockers)

    def test_a_carried_position_blocks(self):
        """Intraday means intraday. A position at the open is a failure of yesterday's
        square-off and must be understood before adding to it."""
        briefing = healthy(open_positions=2)
        assert briefing.ready is False
        assert any("overnight" in c.detail for c in briefing.blockers)

    def test_an_outstanding_halt_blocks(self, client):
        from src.kill_switch import KillSwitch

        KillSwitch(client).halt("yesterday's drawdown")
        briefing = build_morning(DAY, client=client, equity=500_000.0)
        assert briefing.ready is False
        assert any("drawdown" in c.detail for c in briefing.blockers)

    def test_a_cleared_halt_does_not_block(self, client):
        from src.kill_switch import KillSwitch

        switch = KillSwitch(client)
        switch.halt("x")
        switch.clear()
        assert build_morning(DAY, client=client, equity=500_000.0).ready is True


class TestBlockersVersusAdvisories:
    def test_an_unverified_holiday_year_warns_without_blocking(self, monkeypatch):
        """Advisory by design — but it must still appear, because wrong session
        boundaries silently change what every phase check means."""
        from src import market_calendar as cal

        monkeypatch.setattr(cal, "covers_year", lambda year: False)
        briefing = healthy()
        assert briefing.ready is True
        assert any("holiday list" in c.detail for c in briefing.warnings)

    def test_a_dead_feed_pre_open_is_advisory_only(self, client):
        """Nothing is expected to be ticking before 09:15. Blocking on it would make the
        briefing useless at the exact moment it is run."""
        briefing = build_morning(DAY, client=client, equity=500_000.0,
                                 feed_instruments=["nse_cm:2885", "nse_cm:1333"])
        assert briefing.ready is True
        assert any("not yet ticking" in c.detail for c in briefing.warnings)

    def test_a_live_feed_produces_no_warning(self, client):
        from src import feed_health

        feed_health.publish_liveness(client, "nse_cm:2885")
        briefing = build_morning(DAY, client=client, equity=500_000.0,
                                 feed_instruments=["nse_cm:2885"])
        assert not any("not yet ticking" in c.detail for c in briefing.warnings)

    def test_warnings_and_blockers_do_not_overlap(self):
        briefing = healthy(equity=None)
        assert not set(c.name for c in briefing.blockers) & \
            set(c.name for c in briefing.warnings)


class TestRendering:
    def test_the_morning_report_names_what_is_wrong(self):
        text = healthy(equity=None, open_positions=3).render()
        assert "mis-sized" in text
        assert "overnight" in text

    def test_a_ready_report_says_so(self):
        assert "READY" in healthy().render().upper()

    def test_a_blocked_report_does_not_say_ready(self):
        text = healthy(equity=None).render().upper()
        assert "NOT READY" in text or "BLOCKED" in text

    def test_a_check_line_renders_both_states(self):
        assert Check("X", True, "").line()
        assert "why" in Check("X", False, "why").line()


class TestEveningBriefing:
    def test_it_survives_a_missing_session(self, client):
        """Running the evening briefing on a day nothing ran must report that, not crash."""
        briefing = build_evening(DAY, store=None, client=client)
        assert briefing.session is None
        assert briefing.render()

    def test_alerts_are_collected(self, client):
        from src.alerting import Alerter, redis_sink

        Alerter([redis_sink(client)], cooldown_seconds=0).warning(
            "Order rejected", "insufficient margin", source="execution_engine")
        assert build_evening(DAY, client=client).alerts

    def test_alerts_are_rendered_when_a_session_exists(self, client, tmp_path):
        """`render()` returns early when there is no session, so the alert section only
        appears on a day that actually ran — which is the only day it could matter."""
        from src.alerting import Alerter, redis_sink

        Alerter([redis_sink(client)], cooldown_seconds=0).warning(
            "Order rejected", "insufficient margin", source="execution_engine")
        store = self._store_with(tmp_path, trade_count=2, net_pnl=500.0,
                                 total_costs=80.0)
        assert "margin" in build_evening(DAY, store=store, client=client).render()

    def test_the_reconciliation_summary_is_rendered(self, tmp_path):
        store = self._store_with(tmp_path, trade_count=2, net_pnl=500.0,
                                 total_costs=80.0)
        text = build_evening(DAY, store=store,
                             reconciliation_summary="3 session(s): BLOCKED").render()
        assert "BLOCKED" in text

    def test_a_broken_alert_stream_does_not_break_the_report(self, monkeypatch):
        class Broken:
            def xrange(self, *a, **k):
                raise ConnectionError("redis gone")

        assert read_alerts(Broken()) == []

    def test_no_client_means_no_alerts(self):
        assert read_alerts(None) == []

    def test_a_zero_trade_day_is_called_out(self, tmp_path):
        """"No trades" reads as "nothing happened", but it may equally mean every signal
        was refused. Those call for opposite responses."""
        store = self._store_with(tmp_path, trade_count=0, net_pnl=0.0, total_costs=0.0)
        briefing = build_evening(DAY, store=store)
        assert any("no setups" in note or "refused" in note for note in briefing.notes)

    def test_costs_exceeding_pnl_is_called_out(self, tmp_path):
        store = self._store_with(tmp_path, trade_count=12, net_pnl=200.0,
                                 total_costs=900.0)
        briefing = build_evening(DAY, store=store)
        assert any("over-trading" in note for note in briefing.notes)

    def test_a_healthy_day_gets_no_scolding(self, tmp_path):
        store = self._store_with(tmp_path, trade_count=4, net_pnl=5_000.0,
                                 total_costs=400.0)
        assert build_evening(DAY, store=store).notes == []

    def _store_with(self, tmp_path, *, trade_count=0, net_pnl=0.0, total_costs=0.0):
        """A SessionStore holding one real SessionRecord, not a stub.

        Using the real record matters here: `render()` reads a dozen derived properties,
        and a stub that happens to satisfy today's code would stop exercising it the
        moment another field is added.
        """
        from src.session_recorder import RecordedTrade, SessionRecord, SessionStore

        trades = [
            RecordedTrade(
                instrument_id="nse_cm:2885", direction="long", quantity=10,
                entry_at=f"{DAY}T10:00:00", entry_price=1300.0,
                exit_at=f"{DAY}T11:00:00", exit_price=1310.0,
                exit_reason="signal",
                gross_pnl=net_pnl / max(1, trade_count),
                costs=total_costs / max(1, trade_count), slippage=1.0,
                net_pnl=net_pnl / max(1, trade_count),
                risk_amount=2_500.0, r_multiple=0.5,
                strategy_name="trend_pullback",
            )
            for _ in range(trade_count)
        ]
        record = SessionRecord(
            day=DAY.isoformat(), source="paper",
            starting_equity=500_000.0, ending_equity=500_000.0 + net_pnl,
            trades=trades, signals_generated=trade_count, signals_rejected=0,
        )
        store = SessionStore(tmp_path)
        store.save(record)
        return store


class TestCLI:
    def test_morning_prints_a_report(self, monkeypatch, tmp_path, capsys):
        import src.briefing as mod

        monkeypatch.setattr("sys.argv", ["briefing", "morning",
                                         "--date", DAY.isoformat(),
                                         "--store", str(tmp_path)])
        mod.main()
        assert capsys.readouterr().out.strip()

    def test_evening_prints_a_report(self, monkeypatch, tmp_path, capsys):
        import src.briefing as mod

        monkeypatch.setattr("sys.argv", ["briefing", "evening",
                                         "--date", DAY.isoformat(),
                                         "--store", str(tmp_path)])
        mod.main()
        assert capsys.readouterr().out.strip()

    def test_evening_survives_reconciliation_being_unavailable(self, monkeypatch,
                                                               tmp_path, capsys):
        """A reconciliation failure must degrade the report, not suppress it — the rest
        of the day's numbers are still what someone needs to see."""
        import src.briefing as mod

        monkeypatch.setattr("sys.argv", ["briefing", "evening",
                                         "--date", DAY.isoformat(),
                                         "--store", str(tmp_path)])
        monkeypatch.setattr("src.reconciliation.reconcile",
                            lambda *a, **k: (_ for _ in ()).throw(OSError("no data")))
        mod.main()
        # The report still prints. The summary itself is only rendered on a day that
        # recorded a session, which is checked directly in TestEveningBriefing.
        assert capsys.readouterr().out.strip()

    def test_it_works_without_redis(self, monkeypatch, tmp_path, capsys):
        import src.briefing as mod

        monkeypatch.setattr("sys.argv", ["briefing", "morning", "--store", str(tmp_path)])
        monkeypatch.setattr(event_bus, "ping", lambda *a, **k: False)
        mod.main()
        assert capsys.readouterr().out.strip()
