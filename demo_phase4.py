"""Phase 4 verification — paper trading machinery.

Phase 4 is a **time-based** phase: its exit criterion needs 20-40 real trading sessions.
That cannot be simulated away, and this demo does not pretend otherwise. What it verifies
is that the machinery is correct and ready to run:

  1. Paper fills use the backtester's model, not the old optimistic DRY_RUN path.
  2. Session recording captures what reconciliation needs — including rejections.
  3. Reconciliation **attributes** divergence rather than only reporting it.
  4. The morning briefing blocks on problems and leads with them.
  5. The evening briefing surfaces divergence while it is still small.
  6. The Phase 4 gate correctly refuses to pass on a partial session count.

    python demo_phase4.py
"""
from __future__ import annotations

import datetime as dt
import logging
import shutil
import tempfile
from pathlib import Path

import fakeredis

import config
from src import event_bus, market_calendar as cal

logging.getLogger().setLevel(logging.CRITICAL)

_server = fakeredis.FakeServer()
event_bus.get_client = lambda *a, **k: fakeredis.FakeStrictRedis(
    server=_server, decode_responses=True)
event_bus.ping = lambda *a, **k: True

from src.backtest.costs import CostModel, SlippageModel  # noqa: E402
from src.backtest.gates import evaluate_phase4  # noqa: E402
from src.bars import Bar  # noqa: E402
from src.briefing import build_evening, build_morning  # noqa: E402
from src.paper_broker import PaperBroker  # noqa: E402
from src.reconciliation import reconcile  # noqa: E402
from src.session_recorder import RecordedTrade, SessionRecord, SessionStore  # noqa: E402

RULE = "=" * 78
DAY = dt.date(2026, 7, 23)


def heading(number: int, text: str) -> None:
    print(f"\n{RULE}\n{number}. {text}\n{RULE}")


def demo_fills() -> None:
    heading(1, "Paper fills use the backtester's model, not DRY_RUN's")
    client = event_bus.get_client()
    broker = PaperBroker(CostModel(), SlippageModel(slippage_bps=2.5))

    signal_price = 100.0
    broker._handle_order({
        "client_order_id": "lbdemo01", "instrument_id": "nse_cm:1",
        "side": "BUY", "intent": "OPEN_LONG", "quantity": 100,
        "price": signal_price, "stop_loss": 99.0, "strategy_name": "demo",
    })
    print(f"  Signal fired at {signal_price:.2f}. Order queued — nothing filled yet.")

    # A modest 15bps gap from signal close to next open — typical, not dramatic.
    start = cal.at(DAY, dt.time(11, 0))
    next_bar = Bar("nse_cm:1", start, start + dt.timedelta(minutes=1),
                   open=100.15, high=100.60, low=100.05, close=100.40,
                   volume=80_000.0, vwap=100.3, tick_count=400)
    broker._handle_bar(next_bar.to_event())

    fills = [f for _, f in client.xrange(config.STREAM_FILLED_ORDERS)]
    fill = fills[-1]
    price = float(fill["average_fill_price"])
    quantity = int(fill["filled_quantity"])
    costs = float(fill["costs"])
    print(f"  Next bar opened at {next_bar.open:.2f} — a 15bps gap, entirely typical.")
    print(f"  Filled at {price:.4f} (slippage {float(fill['slippage_per_share']):.4f}, "
          f"cost {costs:.2f})\n")

    print(f"  Old DRY_RUN filled at {signal_price:.2f} — the signal price, instantly,")
    print("  with no slippage and no cost.")

    risk_amount = quantity * 1.00      # 100 shares, 1.00 stop distance
    overstatement = (price - signal_price) * quantity + costs
    print(f"\n  Per trade that overstates results by {overstatement:,.2f} on "
          f"{risk_amount:,.0f} of risk")
    print(f"  = {overstatement / risk_amount:.0%} of the amount risked, on ONE trade,")
    print("  in the favourable direction every time.")
    print(f"\n  For scale: total friction is ~0.12R (DESIGN.md 5.2). An error of this")
    print("  size, always favourable, is enough to turn a losing strategy into a")
    print("  winning-looking one — and it would have made the entire paper-vs-backtest")
    print("  comparison measure the bug rather than the strategy.")


def demo_recording_and_reconciliation(store: SessionStore) -> None:
    heading(2, "Session recording and reconciliation")

    def trade(instrument, entry, exit_, net, costs=20.0, r=0.9, hour=10,
              reason="signal", slip=5.0):
        return RecordedTrade(
            instrument_id=instrument, direction="LONG", quantity=100,
            entry_at=f"2026-07-23T{hour:02d}:00:00+05:30", entry_price=entry,
            exit_at=f"2026-07-23T{hour + 1:02d}:00:00+05:30", exit_price=exit_,
            exit_reason=reason, gross_pnl=net + costs, costs=costs, slippage=slip,
            net_pnl=net, risk_amount=200.0, r_multiple=r)

    # The backtest took three trades.
    backtest_trades = [
        trade("nse_cm:1", 100.00, 102.00, 180.0, hour=10),
        trade("nse_cm:2", 200.00, 203.00, 280.0, hour=11),
        trade("nse_cm:3", 150.00, 148.00, -220.0, r=-1.1, hour=13, reason="stop_loss"),
    ]
    # Paper took two of them, at slightly worse fills, and missed one entirely.
    paper_trades = [
        trade("nse_cm:1", 100.06, 101.94, 156.0, costs=22.0, hour=10),
        trade("nse_cm:3", 150.05, 147.95, -230.0, costs=22.0, r=-1.15, hour=13,
              reason="stop_loss"),
    ]

    store.save(SessionRecord(
        day=DAY.isoformat(), source="backtest", starting_equity=500_000.0,
        ending_equity=500_000.0 + sum(t.net_pnl for t in backtest_trades),
        trades=backtest_trades))
    store.save(SessionRecord(
        day=DAY.isoformat(), source="paper", starting_equity=500_000.0,
        ending_equity=500_000.0 + sum(t.net_pnl for t in paper_trades),
        trades=paper_trades, signals_generated=6, signals_rejected=4,
        rejection_reasons={"feed stale for nse_cm:2 — no tick within 30s": 3,
                           "max open positions (3) reached": 1}))

    print(f"  Backtest: {len(backtest_trades)} trades, "
          f"{sum(t.net_pnl for t in backtest_trades):+,.2f}")
    print(f"  Paper   : {len(paper_trades)} trades, "
          f"{sum(t.net_pnl for t in paper_trades):+,.2f}")

    result = reconcile(store)
    print()
    print(result.report())

    print("\n  Note what the attribution does that a total cannot: it separates the")
    print("  missed trade (a feed problem) from the worse fills (a slippage-model")
    print("  problem). Those have different fixes, and a single divergence number")
    print("  would have sent someone after the wrong one.")
    return result


def demo_briefings(store: SessionStore, reconciliation) -> None:
    heading(3, "Daily briefings")
    client = event_bus.get_client()

    print("Morning — healthy state:\n")
    print(build_morning(DAY, client=client, store=store, equity=500_000.0,
                        open_positions=0, strategy="trend_pullback v1").render())

    print("\n\nMorning — with problems:\n")
    print(build_morning(DAY, client=client, store=store, equity=None,
                        open_positions=2, strategy="trend_pullback v1").render())

    summary = (f"{reconciliation.days_compared} session(s): divergence "
               f"{reconciliation.pnl_divergence:+,.2f}, match rate "
               f"{reconciliation.match_rate:.0%}, "
               f"{'PASS' if reconciliation.passed else 'BLOCKED'}")
    print("\n\nEvening:\n")
    print(build_evening(DAY, store=store, reconciliation_summary=summary).render())


def demo_gate(store: SessionStore, reconciliation) -> None:
    heading(4, "Phase 4 gate — paper to live")
    report = evaluate_phase4(
        reconciliation,
        sessions_completed=len(store.days("paper")),
        sessions_required=config.PAPER_SESSIONS_REQUIRED,
        halted_sessions=0,
    )
    print(report.summary())


def main() -> None:
    workdir = Path(tempfile.mkdtemp(prefix="ligerbot_phase4_"))
    try:
        print(RULE)
        print("LIGERBOT — Phase 4 verification (paper trading machinery)")
        print(RULE)

        store = SessionStore(workdir / "sessions")
        demo_fills()
        reconciliation = demo_recording_and_reconciliation(store)
        demo_briefings(store, reconciliation)
        demo_gate(store, reconciliation)

        print(f"\n{RULE}\nPhase 4 status\n{RULE}")
        print("  DELIVERED")
        print("    [x] Paper broker reusing the backtester's fill model")
        print("    [x] Three trading modes (dry_run / paper / live)")
        print("    [x] Per-session recording, including rejections and halts")
        print("    [x] Reconciliation with divergence ATTRIBUTION")
        print("    [x] Morning go/no-go and evening briefings")
        print("    [x] Phase 4 gate (sessions + reconciliation)")
        print()
        print("  NOT DELIVERED — and not deliverable by writing code")
        print(f"    [ ] {config.PAPER_SESSIONS_REQUIRED}+ paper sessions. This is calendar")
        print("        time against live market data. There is no way to shorten it and")
        print("        no substitute for it.")
        print(f"\n{RULE}")
        print("  To start accumulating sessions:")
        print("      TRADING_MODE=paper python run_all.py")
        print("      python -m src.briefing morning   # each pre-open")
        print("      python -m src.briefing evening   # each post-close")
        print("\n  Which still requires the Kotak probe first — live data, and the")
        print("  equity field names in src/account.py are still unverified guesses.")
        print(RULE)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
