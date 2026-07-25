"""The live-trading guard (DESIGN.md Phase 5).

**Setting ``TRADING_MODE=live`` is not sufficient to trade live.** That is the entire point
of this module. An environment variable is one typo, one copied ``.env``, one careless
shell export away from committing real capital — so live mode additionally requires every
prior phase to have demonstrably passed, plus an explicit authorisation file that a human
had to create deliberately.

The checks are ordered by what they protect against:

1. **Evidence** — the §2.6 backtest gates and the Phase 4 paper gate. Without these there
   is no reason to believe the strategy makes money.
2. **Correctness** — the broker probe, so the equity field names in :mod:`src.account` are
   verified rather than guessed. An unverified mapping mis-sizes *every* trade by the same
   factor, silently.
3. **Intent** — an authorisation file naming the date, the capital and the person. Not a
   flag: something someone had to write on purpose.

Every failure is **blocking**. There is no override flag and no ``--force``, deliberately:
the moment a bypass exists, it becomes the thing people reach for at 09:14.

    python -m src.live_guard check
    python -m src.live_guard authorise --capital 300000 --by "your name"
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import socket
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import config
from src import market_calendar as cal

log = logging.getLogger("ligerbot.live_guard")

RULE = "=" * 74


class LiveTradingBlocked(RuntimeError):
    """Raised when live trading is attempted without every prerequisite met."""


@dataclass
class GuardCheck:
    name: str
    passed: bool
    detail: str = ""
    remedy: str = ""

    def render(self) -> str:
        mark = "OK   " if self.passed else "BLOCK"
        lines = [f"  [{mark}] {self.name}"]
        if self.detail:
            lines.append(f"          {self.detail}")
        if not self.passed and self.remedy:
            lines.append(f"          -> {self.remedy}")
        return "\n".join(lines)


@dataclass
class GuardReport:
    checks: List[GuardCheck] = field(default_factory=list)
    requested_mode: str = ""

    @property
    def blockers(self) -> List[GuardCheck]:
        return [c for c in self.checks if not c.passed]

    @property
    def cleared(self) -> bool:
        return not self.blockers

    def render(self) -> str:
        lines = [RULE, "LIVE TRADING GUARD (DESIGN.md Phase 5)", RULE, ""]
        lines += [c.render() for c in self.checks]
        lines.append("")
        if self.cleared:
            lines += [
                "  CLEARED — live trading is permitted.",
                "",
                "  Start at the smallest tradable size. The scaling ladder in",
                "  src/live_scaling.py promotes only on accumulated evidence.",
            ]
        else:
            lines += [
                f"  BLOCKED — {len(self.blockers)} prerequisite(s) not met.",
                "",
                "  There is no override. A bypass would become the thing someone",
                "  reaches for at 09:14 on a morning they are in a hurry.",
            ]
        lines.append(RULE)
        return "\n".join(lines)


@dataclass
class LiveAuthorisation:
    """A deliberate, human-written record that live trading was intended."""

    authorised_on: str
    authorised_by: str
    capital: float
    host: str
    strategy: str = ""
    notes: str = ""

    @property
    def date(self) -> Optional[dt.date]:
        try:
            return dt.date.fromisoformat(self.authorised_on)
        except (TypeError, ValueError):
            return None

    def is_valid_for(self, day: dt.date, *, max_age_days: int = 7) -> tuple[bool, str]:
        """Authorisation expires. A file written months ago is not today's decision."""
        authorised = self.date
        if authorised is None:
            return False, f"unparseable authorisation date {self.authorised_on!r}"
        if authorised > day:
            return False, f"authorised for {authorised}, which is in the future"
        age = (day - authorised).days
        if age > max_age_days:
            return False, (f"authorisation is {age} days old (limit {max_age_days}) — "
                           f"re-authorise so the decision is current")
        return True, ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "authorised_on": self.authorised_on,
            "authorised_by": self.authorised_by,
            "capital": self.capital,
            "host": self.host,
            "strategy": self.strategy,
            "notes": self.notes,
        }


def authorisation_path() -> Path:
    return Path(config.LIVE_AUTH_PATH)


def write_authorisation(
    capital: float, by: str, *, strategy: str = "", notes: str = "",
    day: Optional[dt.date] = None,
) -> LiveAuthorisation:
    auth = LiveAuthorisation(
        authorised_on=(day or cal.now_ist().date()).isoformat(),
        authorised_by=by,
        capital=capital,
        host=socket.gethostname(),
        strategy=strategy,
        notes=notes,
    )
    path = authorisation_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(auth.to_dict(), indent=2), encoding="utf-8")
    log.warning("Live trading authorised by %s for capital %.2f on %s",
                by, capital, auth.authorised_on)
    return auth


def read_authorisation() -> Optional[LiveAuthorisation]:
    path = authorisation_path()
    if not path.exists():
        return None
    try:
        return LiveAuthorisation(**json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError) as exc:
        log.error("Live authorisation at %s is unreadable (%s).", path, exc)
        return None


# ---------------------------------------------------------------------------
def evaluate(
    *,
    day: Optional[dt.date] = None,
    backtest_gates_passed: Optional[bool] = None,
    phase4_gates_passed: Optional[bool] = None,
    paper_sessions: int = 0,
    probe_completed: Optional[bool] = None,
    equity: Optional[float] = None,
    instrument_master_loaded: bool = False,
) -> GuardReport:
    """Decide whether live trading may proceed.

    Arguments default to *unknown*, and unknown is treated as **not passed**. An unrun
    check must never read as a cleared one — the same rule the §2.6 gates follow.
    """
    day = day or cal.now_ist().date()
    report = GuardReport(requested_mode=config.TRADING_MODE)
    add = report.checks.append

    # 1. Evidence that the strategy works.
    add(GuardCheck(
        "Backtest gates passed (DESIGN.md 2.6)",
        bool(backtest_gates_passed),
        "" if backtest_gates_passed else "not demonstrated",
        "Run the walk-forward validation and clear every §2.6 gate out-of-sample.",
    ))

    add(GuardCheck(
        "Paper trading gate passed (Phase 4)",
        bool(phase4_gates_passed),
        f"{paper_sessions} paper session(s) recorded",
        f"Accumulate {config.PAPER_SESSIONS_REQUIRED}+ sessions and reconcile them "
        f"against a backtest of the same days.",
    ))

    # 2. Correctness of the plumbing that sizes trades.
    add(GuardCheck(
        "Broker probe completed",
        bool(probe_completed),
        "" if probe_completed else "src/account.py field names are unverified guesses",
        "Run `python -m tools.probe_kotak_history` and confirm parse_limits() returns "
        "the equity you see in the app. A wrong mapping mis-sizes every trade.",
    ))

    add(GuardCheck(
        "Instrument master loaded",
        instrument_master_loaded,
        "" if instrument_master_loaded else "trading symbols cannot be resolved (B4)",
        "Download the scrip master so orders carry real trading symbols.",
    ))

    # 3. Capital.
    has_equity = equity is not None and equity > 0
    add(GuardCheck(
        "Equity resolved from the broker",
        has_equity,
        f"{equity:,.2f}" if has_equity else "unresolved",
        "Live sizing must never fall back to a configured figure.",
    ))
    if has_equity:
        add(GuardCheck(
            "Equity above the viability floor",
            equity >= config.MIN_EQUITY,
            f"{equity:,.2f} vs floor {config.MIN_EQUITY:,.2f}",
            "Below the floor, round-trip costs exceed ~15% of the amount risked "
            "(DESIGN.md 5.2) and no plausible intraday edge survives.",
        ))

    # 4. The absolute backstop must be a decision, not an inherited default.
    add(GuardCheck(
        "Absolute daily loss cap set",
        config.LIVE_MAX_DAILY_LOSS > 0,
        (f"Rs {config.LIVE_MAX_DAILY_LOSS:,.0f}" if config.LIVE_MAX_DAILY_LOSS > 0
         else "unset"),
        "Set LIVE_MAX_DAILY_LOSS. It ships unset on purpose — the right figure is a "
        "statement about how much you are willing to lose in a day, and every "
        "percentage limit becomes wrong together if the equity figure is misread.",
    ))

    # 5. Deliberate human intent.
    auth = read_authorisation()
    if auth is None:
        add(GuardCheck(
            "Live authorisation on file", False,
            f"no authorisation at {authorisation_path()}",
            "Run `python -m src.live_guard authorise --capital N --by 'name'`. "
            "Deliberately a file, not a flag — an env var is one typo from live.",
        ))
    else:
        valid, reason = auth.is_valid_for(day, max_age_days=config.LIVE_AUTH_MAX_AGE_DAYS)
        add(GuardCheck(
            "Live authorisation on file", valid,
            f"authorised {auth.authorised_on} by {auth.authorised_by} "
            f"for {auth.capital:,.2f}" if valid else reason,
            "Re-authorise so the decision reflects today.",
        ))
        if valid and has_equity and auth.capital > 0:
            # Guards against authorising for a small account and pointing the bot at a
            # much larger one.
            ratio = equity / auth.capital
            add(GuardCheck(
                "Equity matches the authorised capital",
                ratio <= config.LIVE_AUTH_CAPITAL_TOLERANCE,
                f"account holds {equity:,.2f}, authorised for {auth.capital:,.2f} "
                f"({ratio:.1f}x)",
                "Re-authorise for the actual capital. Authorising for a small account "
                "and running against a large one is how a test becomes a position.",
            ))

    return report


def require_live_clearance(**kwargs) -> GuardReport:
    """Raise unless live trading is fully cleared. Called on the live execution path."""
    report = evaluate(**kwargs)
    if not report.cleared:
        raise LiveTradingBlocked(
            "Live trading is blocked:\n" +
            "\n".join(f"  - {c.name}: {c.detail or 'not met'}" for c in report.blockers)
        )
    return report


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="LIGERBOT live-trading guard")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check", help="show whether live trading is permitted")
    auth = sub.add_parser("authorise", help="record deliberate authorisation")
    auth.add_argument("--capital", type=float, required=True)
    auth.add_argument("--by", required=True, help="who is authorising this")
    auth.add_argument("--strategy", default="")
    auth.add_argument("--notes", default="")
    sub.add_parser("revoke", help="remove the authorisation file")
    args = parser.parse_args()

    if args.command == "check":
        from src.session_recorder import SessionStore

        store = SessionStore(config.SESSION_STORE_ROOT)
        print(evaluate(paper_sessions=len(store.days("paper"))).render())
    elif args.command == "authorise":
        auth_record = write_authorisation(
            args.capital, args.by, strategy=args.strategy, notes=args.notes)
        print(f"Authorisation written to {authorisation_path()}")
        print(json.dumps(auth_record.to_dict(), indent=2))
        print("\nThis authorises intent only. Every other prerequisite still applies —")
        print("run `python -m src.live_guard check` to see what remains.")
    elif args.command == "revoke":
        path = authorisation_path()
        if path.exists():
            path.unlink()
            print(f"Revoked. Removed {path}")
        else:
            print("No authorisation on file.")


if __name__ == "__main__":
    main()
