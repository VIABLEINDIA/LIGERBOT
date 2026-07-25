"""Daily briefing reports (DESIGN.md 4, Phase 4 item 25).

A paper period nobody reads is not a test. Phase 4's exit criterion requires investigating
*every* divergence between paper and backtest — and that only happens if something puts
the state in front of a human twice a day, before the open and after the close.

Two reports, with different jobs:

**Morning (pre-open)** — a go/no-go. Is the bot fit to trade today? Equity resolved, feed
alive, no halt outstanding, universe valid, yesterday's positions actually flat. The point
is to catch a problem at 09:00 rather than discover it at 09:20 with money committed.

**Evening (post-close)** — what happened, and does it match what the backtest said would
happen. This is where reconciliation divergence surfaces day by day, while it is still
small enough to diagnose.

The morning report deliberately leads with **blockers**, not with P&L. Someone skimming it
at 08:55 should see the reason not to trade before they see anything encouraging.

    python -m src.briefing morning
    python -m src.briefing evening
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import config
from src import market_calendar as cal
from src.session_recorder import SessionRecord, SessionStore

log = logging.getLogger("ligerbot.briefing")

RULE = "=" * 70


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""
    blocking: bool = True

    def line(self) -> str:
        mark = "OK  " if self.ok else ("BLOCK" if self.blocking else "warn ")
        return f"  [{mark}] {self.name}" + (f" — {self.detail}" if self.detail else "")


@dataclass
class MorningBriefing:
    day: dt.date
    checks: List[Check] = field(default_factory=list)
    equity: Optional[float] = None
    open_positions: int = 0
    strategy: str = ""
    sessions_completed: int = 0
    sessions_required: int = 20

    @property
    def blockers(self) -> List[Check]:
        return [c for c in self.checks if not c.ok and c.blocking]

    @property
    def warnings(self) -> List[Check]:
        return [c for c in self.checks if not c.ok and not c.blocking]

    @property
    def ready(self) -> bool:
        return not self.blockers

    def render(self) -> str:
        lines = [RULE,
                 f"MORNING BRIEFING — {self.day} ({self.day.strftime('%A')})",
                 RULE]

        # Blockers first: someone skimming at 08:55 should see the reason not to trade
        # before they see anything reassuring.
        if self.blockers:
            lines += ["", f"  NOT READY TO TRADE — {len(self.blockers)} blocker(s):"]
            lines += [f"    - {c.name}: {c.detail}" for c in self.blockers]
        else:
            lines += ["", "  READY TO TRADE"]

        lines += ["", "Pre-flight checks"]
        lines += [c.line() for c in self.checks]

        lines += ["", "State"]
        lines.append(f"  Equity              "
                     f"{f'{self.equity:,.2f}' if self.equity else 'UNRESOLVED':>16}")
        lines.append(f"  Open positions      {self.open_positions:>16}")
        lines.append(f"  Strategy            {self.strategy or '(none)':>16}")
        lines.append(f"  Mode                {config.TRADING_MODE:>16}")

        remaining = max(0, self.sessions_required - self.sessions_completed)
        lines += ["", "Phase 4 progress"]
        lines.append(f"  Paper sessions      {self.sessions_completed}/"
                     f"{self.sessions_required}  ({remaining} remaining)")
        lines.append(RULE)
        return "\n".join(lines)


@dataclass
class EveningBriefing:
    day: dt.date
    session: Optional[SessionRecord] = None
    reconciliation_summary: str = ""
    notes: List[str] = field(default_factory=list)

    def render(self) -> str:
        lines = [RULE,
                 f"EVENING BRIEFING — {self.day} ({self.day.strftime('%A')})",
                 RULE]

        if self.session is None:
            lines += ["", "  No session recorded for this day.",
                      "  Either the bot did not run, or the recorder failed — both are",
                      "  worth checking before tomorrow.", RULE]
            return "\n".join(lines)

        s = self.session
        lines += [
            "",
            "Result",
            f"  Net P&L             {s.net_pnl:>+16,.2f}   ({s.return_pct:+.3%})",
            f"  Trades              {s.trade_count:>16}   "
            f"({s.win_count}W / {s.trade_count - s.win_count}L, {s.win_rate:.0%})",
            f"  Expectancy          {s.expectancy_r:>+16.3f}R",
            f"  Costs               {s.total_costs:>16,.2f}",
            f"  Slippage            {s.total_slippage:>16,.2f}",
            "",
            "Signal flow",
            f"  Generated           {s.signals_generated:>16}",
            f"  Rejected            {s.signals_rejected:>16}",
        ]
        if s.rejection_reasons:
            for reason, count in sorted(
                    s.rejection_reasons.items(), key=lambda kv: -kv[1])[:5]:
                lines.append(f"      {count:>5}  {reason}")
            lines.append("")
            lines.append("  Rejections matter as much as fills: a day where twenty")
            lines.append("  signals became three trades looks identical in P&L to a day")
            lines.append("  with three signals, and diverges from the backtest for a")
            lines.append("  completely different reason.")

        if s.halted:
            lines += ["", f"  HALTED: {s.halt_reason}",
                      "  This session is excluded from reconciliation — it stopped",
                      "  partway through and the backtest of the same day did not."]

        if s.trades:
            lines += ["", "Trades"]
            for t in s.trades[:10]:
                lines.append(
                    f"  {t.instrument_id:<16} {t.direction:<5} {t.quantity:>5} "
                    f"@ {t.entry_price:>9,.2f} -> {t.exit_price:>9,.2f}  "
                    f"{t.net_pnl:>+10,.2f}  {t.r_multiple:>+6.2f}R  {t.exit_reason}")
            if len(s.trades) > 10:
                lines.append(f"  ... and {len(s.trades) - 10} more")

        if self.reconciliation_summary:
            lines += ["", "Reconciliation vs backtest", f"  {self.reconciliation_summary}"]

        if self.notes:
            lines += ["", "Notes"] + [f"  - {n}" for n in self.notes]

        lines.append(RULE)
        return "\n".join(lines)


def build_morning(
    day: Optional[dt.date] = None,
    *,
    client=None,
    store: Optional[SessionStore] = None,
    equity: Optional[float] = None,
    open_positions: int = 0,
    strategy: str = "",
    feed_instruments: Optional[List[str]] = None,
) -> MorningBriefing:
    """Assemble the pre-open go/no-go."""
    day = day or cal.now_ist().date()
    briefing = MorningBriefing(day=day, equity=equity,
                               open_positions=open_positions, strategy=strategy)
    add = briefing.checks.append

    add(Check("Trading day", cal.is_trading_day(day),
              "" if cal.is_trading_day(day) else f"{day} is not an NSE trading day"))

    add(Check("Holiday calendar verified", cal.covers_year(day.year),
              "" if cal.covers_year(day.year)
              else f"no verified NSE holiday list for {day.year} — set NSE_HOLIDAYS_FILE",
              blocking=False))

    add(Check("Equity resolved", equity is not None and equity > 0,
              "" if equity else "could not read equity from the broker — every trade "
                                "today would be mis-sized"))

    if equity:
        add(Check("Equity above the floor", equity >= config.MIN_EQUITY,
                  "" if equity >= config.MIN_EQUITY
                  else f"{equity:,.0f} is below the {config.MIN_EQUITY:,.0f} floor; "
                       f"costs would exceed any plausible edge"))

    add(Check("Flat at the open", open_positions == 0,
              "" if open_positions == 0
              else f"{open_positions} position(s) carried overnight — intraday means "
                   f"intraday; investigate before trading"))

    if client is not None:
        from src.kill_switch import KillSwitch

        state = KillSwitch(client).state()
        add(Check("No halt outstanding", not state.halted,
                  state.reason if state.halted else ""))

        if feed_instruments:
            from src import feed_health

            dead = [i for i in feed_instruments
                    if not feed_health.is_feed_live(client, i)]
            # Pre-open, no feed is expected to be live — this is advisory until 09:15.
            add(Check("Feed liveness", not dead,
                      f"{len(dead)} instrument(s) not yet ticking: "
                      f"{', '.join(dead[:5])}", blocking=False))

    add(Check("Mode", config.TRADING_MODE in ("dry_run", "paper", "live"),
              f"TRADING_MODE={config.TRADING_MODE!r}"))

    if store is not None:
        briefing.sessions_completed = len(store.days("paper"))

    return briefing


def build_evening(
    day: Optional[dt.date] = None,
    *,
    store: Optional[SessionStore] = None,
    reconciliation_summary: str = "",
) -> EveningBriefing:
    """Assemble the post-close report."""
    day = day or cal.now_ist().date()
    session = store.load("paper", day) if store is not None else None
    briefing = EveningBriefing(day=day, session=session,
                               reconciliation_summary=reconciliation_summary)

    if session is not None:
        if session.trade_count == 0:
            briefing.notes.append(
                "No trades. Check whether the strategy found no setups or whether "
                "signals were refused — the evening flow section distinguishes them.")
        if session.total_costs > abs(session.net_pnl) and session.trade_count:
            briefing.notes.append(
                "Costs exceeded net P&L. At ~0.12R of friction per trade (DESIGN.md "
                "5.2), that is the expected outcome of over-trading.")
    return briefing


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    parser = argparse.ArgumentParser(description="LIGERBOT daily briefing")
    parser.add_argument("when", choices=["morning", "evening"])
    parser.add_argument("--date", default=None, help="YYYY-MM-DD (default: today)")
    parser.add_argument("--store", default=config.SESSION_STORE_ROOT)
    args = parser.parse_args()

    day = dt.date.fromisoformat(args.date) if args.date else cal.now_ist().date()
    store = SessionStore(args.store)

    if args.when == "morning":
        from src import event_bus

        client = event_bus.get_client() if event_bus.ping() else None
        print(build_morning(day, client=client, store=store).render())
    else:
        summary = ""
        try:
            from src.reconciliation import reconcile

            result = reconcile(store)
            if result.days_compared:
                summary = (f"{result.days_compared} session(s): divergence "
                           f"{result.pnl_divergence:+,.2f}, match rate "
                           f"{result.match_rate:.0%}, "
                           f"{'PASS' if result.passed else 'BLOCKED'}")
        except Exception as exc:  # noqa: BLE001
            summary = f"reconciliation unavailable: {exc}"
        print(build_evening(day, store=store, reconciliation_summary=summary).render())


if __name__ == "__main__":
    main()
