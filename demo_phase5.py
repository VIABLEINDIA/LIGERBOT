"""Phase 5 verification — live-trading safeguards.

Phase 5 is "live with the smallest tradable size and a hard daily loss cap", then "scale
only after a sustained period matching paper behaviour".

**This demo shows the system refusing to trade live, which is the correct outcome today.**
Every prerequisite currently fails: no validated strategy, no paper sessions, no broker
probe. A Phase 5 demo that showed live trading being enabled would be demonstrating a
defect, not a feature.

What is verified:

  1. ``TRADING_MODE=live`` alone is not sufficient — the guard blocks it.
  2. Each prerequisite blocks independently, with a remedy attached.
  3. Authorisation is a deliberate human act, expires, and must match the capital.
  4. The scaling ladder starts at minimum size and promotes only on evidence.
  5. Demotion is immediate and skips straight to the floor.
  6. Absolute rupee backstops hold even when percentage limits are computed from a
     wrong equity figure.

    python demo_phase5.py
"""
from __future__ import annotations

import datetime as dt
import logging
import shutil
import tempfile
from pathlib import Path

import config
from src.live_guard import evaluate, write_authorisation
from src.live_scaling import DEFAULT_LADDER, ScalingLadder
from src.risk_engine import Intent, RiskEngine, RiskLimits, Signal

logging.getLogger().setLevel(logging.CRITICAL)

RULE = "=" * 78
DAY = dt.date(2026, 7, 23)
SIGNAL_TIME = dt.datetime(2026, 7, 23, 10, 30)


def heading(number: int, text: str) -> None:
    print(f"\n{RULE}\n{number}. {text}\n{RULE}")


def demo_guard_blocks() -> None:
    heading(1, "TRADING_MODE=live is not sufficient")
    print("  Simulating an operator setting TRADING_MODE=live and starting the bot.\n")
    print(evaluate(day=DAY).render())
    print("\n  Every check defaults to NOT PASSED. An unrun check must never read as a")
    print("  cleared one — the same rule the §2.6 gates follow. Unknown is not safe.")


def demo_individual_blockers() -> None:
    heading(2, "Each prerequisite blocks independently")
    workdir = Path(tempfile.mkdtemp(prefix="ligerbot_p5_"))
    try:
        config.LIVE_AUTH_PATH = str(workdir / "live_authorisation.json")
        write_authorisation(400_000.0, "demo operator", day=DAY)

        base = dict(day=DAY, backtest_gates_passed=True, phase4_gates_passed=True,
                    paper_sessions=25, probe_completed=True, equity=400_000.0,
                    instrument_master_loaded=True)

        print(f"  {'scenario':<38} {'cleared':>8}   first blocker")
        print(f"  {'-' * 38} {'-' * 8}   {'-' * 30}")
        for label, override in [
            ("everything in order", {}),
            ("backtest gates never passed", {"backtest_gates_passed": False}),
            ("no paper sessions", {"phase4_gates_passed": False, "paper_sessions": 0}),
            ("broker probe never run", {"probe_completed": False}),
            ("instrument master missing", {"instrument_master_loaded": False}),
            ("equity below the floor", {"equity": 50_000.0}),
            ("equity unresolved", {"equity": None}),
        ]:
            report = evaluate(**{**base, **override})
            first = report.blockers[0].name if report.blockers else "-"
            print(f"  {label:<38} {'YES' if report.cleared else 'no':>8}   {first}")

        print("\n  Authorisation is a FILE a human had to write, not a flag:")
        print(f"    {config.LIVE_AUTH_PATH}")
        print("  An environment variable is one typo, one copied .env, one careless")
        print("  export away from committing real capital.\n")

        stale = write_authorisation(400_000.0, "demo operator",
                                    day=DAY - dt.timedelta(days=30))
        report = evaluate(**base)
        blocker = next((c for c in report.blockers if "authorisation" in c.name), None)
        print(f"  Authorisation from {stale.authorised_on} (30 days old):")
        print(f"    -> {blocker.detail if blocker else 'accepted'}")

        write_authorisation(50_000.0, "demo operator", day=DAY)
        report = evaluate(**base)
        blocker = next((c for c in report.blockers if "authorised capital" in c.name), None)
        print(f"\n  Authorised for 50,000 but the account holds 400,000:")
        print(f"    -> {blocker.detail if blocker else 'accepted'}")
        print("  Authorising for a small account and running against a large one is")
        print("  how a test becomes a position.")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def demo_scaling_ladder() -> None:
    heading(3, "The scaling ladder — minimum size, evidence-gated promotion")
    workdir = Path(tempfile.mkdtemp(prefix="ligerbot_p5s_"))
    try:
        print("  Ladder:")
        for rung in DEFAULT_LADDER:
            print(f"    {rung.describe()}")

        ladder = ScalingLadder(state_path=workdir / "scaling.json")
        equity = 1_000_000.0
        print(f"\n  Account equity {equity:,.0f}")
        print(f"  Starting rung: {ladder.rung.name} -> risk sized off "
              f"{ladder.scaled_equity(equity):,.0f}")
        print("\n  Scaling multiplies the equity BASE, never the risk rules. 0.5% of a")
        print("  10% base is 0.05% of the account, so every proportional guarantee from")
        print("  D2 survives the ramp unchanged.\n")

        print(f"  {'session':>8}  {'rung':>14}  {'size':>6}  event")
        print(f"  {'-' * 8}  {'-' * 14}  {'-' * 6}  {'-' * 40}")
        session = 0
        for _ in range(46):
            session += 1
            rung = ladder.rung
            trades = max(2, rung.min_trades // max(1, rung.min_sessions) + 1)
            note = ladder.record_session(
                DAY, trades=trades, net_pnl=400.0, expectancy_r_sum=0.3 * trades)
            if "PROMOTED" in note or session == 1:
                print(f"  {session:>8}  {ladder.rung.name:>14}  "
                      f"{ladder.size_multiplier:>5.0%}  {note}")
            if ladder.at_full_size:
                break
        print(f"\n  Reached full size after {session} profitable sessions.")

        print("\n  Now three consecutive losing sessions at full size:")
        for i in range(3):
            note = ladder.record_session(DAY, trades=3, net_pnl=-800.0,
                                         expectancy_r_sum=-1.2)
            print(f"    loss {i + 1}: {ladder.rung.name:>14} "
                  f"({ladder.size_multiplier:.0%})  {note}")

        print("\n  Demotion skips straight to the floor rather than stepping down.")
        print("  The costs are asymmetric: promoting too slowly loses a little upside,")
        print(f"  promoting too quickly loses capital. It took {session} sessions to")
        print("  earn full size and three to lose it — deliberately.")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def demo_absolute_backstops() -> None:
    heading(4, "Absolute backstops — when the percentage limits are computed wrong")
    print("  Every risk limit so far is a FRACTION OF EQUITY. That is exactly wrong if")
    print("  the equity figure itself is wrong — and src/account.py's field names are")
    print("  still unverified guesses. A mis-read equity mis-sizes every trade by the")
    print("  same factor, and every percentage cap scales with the error rather than")
    print("  catching it.\n")

    true_equity = 400_000.0
    misread = true_equity * 10          # a plausible parsing error: paise vs rupees
    engine = RiskEngine(RiskLimits(max_daily_loss_absolute=5_000.0,
                                   max_orders_per_session=20))
    engine.start_session(DAY, misread)

    print(f"  True equity        {true_equity:>12,.0f}")
    print(f"  Mis-read as        {misread:>12,.0f}   (10x — e.g. a units error)")
    print(f"  2% 'daily limit'   {engine.daily_loss_cap:>12,.0f}   "
          f"= {engine.daily_loss_cap / true_equity:.0%} of the real account")
    print(f"  Absolute cap       {engine.limits.max_daily_loss_absolute:>12,.0f}   "
          f"= {engine.limits.max_daily_loss_absolute / true_equity:.1%} of the real "
          f"account\n")

    engine.realized_pnl_today = -6_000.0
    decision = engine.evaluate(
        Signal(instrument_id="nse_cm:1", intent=Intent.OPEN_LONG, ref_price=100.0,
               stop_loss=99.0, bar_time=SIGNAL_TIME),
        allows_entry=True, allows_exit=True)
    print(f"  After losing 6,000 — percentage limit says keep trading "
          f"(6,000 < {engine.daily_loss_cap:,.0f}).")
    print(f"  Absolute cap: {'REFUSED' if not decision.approved else 'allowed'} — "
          f"{decision.reason}")
    print("\n  The percentage limit would have permitted a 20% loss of the real account")
    print("  before tripping. The absolute cap bounds the damage regardless of what the")
    print("  equity figure says. Live mode only — in backtest and paper these would cap")
    print("  activity for reasons unrelated to the strategy.")


def main() -> None:
    print(RULE)
    print("LIGERBOT — Phase 5 verification (live-trading safeguards)")
    print(RULE)
    print("\n  This demo shows the system REFUSING to trade live.")
    print("  That is the correct outcome: every prerequisite currently fails.")

    demo_guard_blocks()
    demo_individual_blockers()
    demo_scaling_ladder()
    demo_absolute_backstops()

    print(f"\n{RULE}\nPhase 5 status\n{RULE}")
    print("  DELIVERED")
    print("    [x] Live guard — TRADING_MODE=live is not sufficient on its own")
    print("    [x] File-based authorisation, expiring, capital-matched")
    print("    [x] Scaling ladder — minimum size, evidence-gated promotion")
    print("    [x] Immediate demotion to the floor on divergence")
    print("    [x] Absolute rupee backstops independent of the equity figure")
    print("    [x] Execution engine refuses to arm live when blocked")
    print()
    print("  NOT DELIVERED — and must not be")
    print("    [ ] Live trading. The guard blocks it, correctly:")
    print("          - no validated strategy (Phase 2 gates fail)")
    print("          - zero paper sessions (Phase 4 gate: 0/20)")
    print("          - broker probe never run (equity mapping unverified)")
    print()
    print(f"{RULE}")
    print("  The order is not negotiable, and each step gates the next:")
    print("      1. python -m tools.probe_kotak_history     <- still the blocker")
    print("      2. backfill + walk-forward until §2.6 gates pass")
    print("      3. TRADING_MODE=paper, 20+ sessions, reconcile daily")
    print("      4. python -m src.live_guard authorise ...")
    print("      5. TRADING_MODE=live at minimum size")
    print(RULE)


if __name__ == "__main__":
    main()
