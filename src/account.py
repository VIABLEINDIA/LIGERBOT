"""Account equity retrieval and the session-equity snapshot (DESIGN.md D1).

``TOTAL_EQUITY`` as a static config value is gone. Equity is read from Kotak each session,
consistent with the standing rule that the broker is authoritative.

Three constraints, each non-obvious and each load-bearing:

1. **Size off own capital, not buying power.** MIS grants roughly 5x intraday leverage, so
   the margin figure the broker reports is several times the actual capital. Sizing off it
   would inflate every position by that multiple and silently turn a 0.5% risk rule into a
   2.5% one. Equity here is ``cash + unrealised MTM``, and margin is tracked separately as
   a *placement* constraint.

2. **Snapshot once at the open, hold it all day.** Sizing off continuously updating equity
   is intuitive and wrong: a profitable morning inflates position sizes, so the largest
   position of the day gets taken right before any mean reversion. Intraday P&L moves the
   drawdown counter; it never moves the sizing base. The snapshot is persisted so a module
   restarting at 11:00 resumes the same base it started with, rather than re-deriving a
   different one mid-session.

3. **Fail closed.** A wrong equity figure mis-sizes every trade in the session. On a bad or
   implausible broker response we refuse to open positions and alert, rather than falling
   back to a config default. Exits stay permitted.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

log = logging.getLogger("ligerbot.account")


class EquityUnavailable(RuntimeError):
    """Raised when equity cannot be established. Callers must fail closed, never guess."""


# Kotak's ``limits()`` field names, as candidate lists because the response shape varies
# by account type and SDK version.
#
# VERIFY BEFORE LIVE USE. These are best-effort guesses at the field names, and an
# unverified mapping is exactly the kind of thing that silently sizes every trade wrongly.
# Phase 1's broker probe must dump a real limits() response and pin these down.
# The parser raises rather than defaulting when nothing matches, so an unverified mapping
# fails loudly at startup instead of quietly at 09:15.
CASH_FIELDS = ("CashOpenBal", "Net", "MarginAvailable", "cash", "AvailableCash")
MTM_FIELDS = ("MtoMUnrealized", "UnrealizedMTM", "mtm", "Mtm")
REALIZED_FIELDS = ("MtoMRealized", "RealizedMTM", "realized")
MARGIN_USED_FIELDS = ("MarginUsed", "marginUsed", "UtilizedMargin")
COLLATERAL_FIELDS = ("CollateralValue", "collateral", "AuxColl")


@dataclass(frozen=True)
class EquitySnapshot:
    """Account state at a point in time."""

    equity: float           # cash + unrealised MTM — the sizing base
    cash: float
    unrealized_mtm: float
    margin_used: float = 0.0
    margin_available: float = 0.0
    collateral: float = 0.0
    captured_at: Optional[str] = None
    source: str = "broker"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _pick(payload: Dict[str, Any], keys: Sequence[str]) -> Optional[float]:
    """First parseable numeric value among ``keys``. None if nothing matched."""
    for key in keys:
        if key not in payload:
            continue
        try:
            value = payload[key]
            if value in (None, ""):
                continue
            return float(str(value).replace(",", ""))
        except (TypeError, ValueError):
            continue
    return None


def parse_limits(payload: Dict[str, Any]) -> EquitySnapshot:
    """Turn a Kotak ``limits()`` response into an :class:`EquitySnapshot`.

    Raises :class:`EquityUnavailable` if the cash figure cannot be located — deliberately,
    because a zero default here would size every position at zero, and a nonzero default
    would size them off a fiction.
    """
    if not isinstance(payload, dict):
        raise EquityUnavailable(f"limits() returned {type(payload).__name__}, expected dict")

    # Some SDK versions nest the useful part one level down.
    for wrapper in ("data", "Data", "result"):
        inner = payload.get(wrapper)
        if isinstance(inner, dict):
            payload = {**payload, **inner}
        elif isinstance(inner, list) and inner and isinstance(inner[0], dict):
            payload = {**payload, **inner[0]}

    cash = _pick(payload, CASH_FIELDS)
    if cash is None:
        raise EquityUnavailable(
            f"Could not find a cash balance in the limits() response. Looked for "
            f"{CASH_FIELDS}; response had keys {sorted(payload)[:25]}. Update "
            f"account.CASH_FIELDS — refusing to guess an equity figure."
        )

    mtm = _pick(payload, MTM_FIELDS) or 0.0
    return EquitySnapshot(
        equity=cash + mtm,
        cash=cash,
        unrealized_mtm=mtm,
        margin_used=_pick(payload, MARGIN_USED_FIELDS) or 0.0,
        collateral=_pick(payload, COLLATERAL_FIELDS) or 0.0,
        captured_at=dt.datetime.now().isoformat(timespec="seconds"),
    )


def validate_snapshot(
    snapshot: EquitySnapshot,
    *,
    previous_equity: Optional[float] = None,
    min_equity: float = 0.0,
    max_jump_ratio: float = 0.5,
) -> None:
    """Sanity-check a snapshot before it is allowed to size anything.

    The jump check catches the failure mode that matters most: a malformed response that
    parses cleanly into a plausible-looking but wrong number. Equity moving more than 50%
    overnight is far more likely to be a parsing bug than a real account event, so it
    stops the bot and asks a human.
    """
    if snapshot.equity <= 0:
        raise EquityUnavailable(
            f"Equity resolved to {snapshot.equity:.2f} (cash {snapshot.cash:.2f}, "
            f"MTM {snapshot.unrealized_mtm:.2f}) — refusing to trade."
        )
    if min_equity > 0 and snapshot.equity < min_equity:
        raise EquityUnavailable(
            f"Equity {snapshot.equity:.2f} is below the configured floor {min_equity:.2f}. "
            f"Below roughly Rs 2L, round-trip costs exceed 15% of the amount risked per "
            f"trade (DESIGN.md 5.2) and no plausible intraday edge survives that."
        )
    if previous_equity and previous_equity > 0:
        change = abs(snapshot.equity - previous_equity) / previous_equity
        if change > max_jump_ratio:
            raise EquityUnavailable(
                f"Equity moved {change:.1%} since the last session "
                f"({previous_equity:.2f} -> {snapshot.equity:.2f}), beyond the "
                f"{max_jump_ratio:.0%} sanity bound. This is more likely a parsing error "
                f"than a real change — verify before trading."
            )


def fetch_equity(neo_client: Any) -> EquitySnapshot:
    """Read live equity from the broker.

    Any SDK exception becomes :class:`EquityUnavailable` so callers have exactly one
    failure mode to handle, and cannot accidentally proceed on a partial result.
    """
    try:
        payload = neo_client.limits()
    except Exception as exc:  # broker SDK raises a wide variety of types
        raise EquityUnavailable(f"limits() call failed: {exc}") from exc
    return parse_limits(payload)


class SessionEquity:
    """Holds the day's equity base, persisted across restarts.

    Persistence is the point. Without it, a module restarting mid-session would refetch
    and get a *different* figure (the account has moved since the open), so positions
    opened after the restart would be sized on a different base than those before it.
    """

    def __init__(self, state_path: str | Path, *, min_equity: float = 0.0) -> None:
        self.state_path = Path(state_path)
        self.min_equity = min_equity
        self._day: Optional[dt.date] = None
        self._snapshot: Optional[EquitySnapshot] = None

    # -- persistence -------------------------------------------------------
    def _load_state(self) -> Dict[str, Any]:
        if not self.state_path.exists():
            return {}
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            log.warning("Session-equity state at %s unreadable (%s).", self.state_path, exc)
            return {}

    def _save_state(self, day: dt.date, snapshot: EquitySnapshot) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"day": day.isoformat(), "snapshot": snapshot.to_dict()}
        self.state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def last_known_equity(self) -> Optional[float]:
        state = self._load_state()
        snapshot = state.get("snapshot") or {}
        value = snapshot.get("equity")
        return float(value) if value else None

    # -- the session base --------------------------------------------------
    def resolve(self, day: dt.date, neo_client: Any) -> EquitySnapshot:
        """Return the equity base for ``day``, fetching only if not already established.

        Idempotent within a session: the first call fetches and pins, every later call
        returns the pinned value regardless of what the account has done since.
        """
        if self._snapshot is not None and self._day == day:
            return self._snapshot

        state = self._load_state()
        if state.get("day") == day.isoformat() and state.get("snapshot"):
            # Restart within the same session: reuse the pinned base rather than
            # refetching a figure that has drifted with the day's P&L.
            snapshot = EquitySnapshot(**state["snapshot"])
            self._day, self._snapshot = day, snapshot
            log.info("Restored session equity %.2f for %s from %s",
                     snapshot.equity, day, self.state_path)
            return snapshot

        snapshot = fetch_equity(neo_client)
        validate_snapshot(
            snapshot,
            previous_equity=self.last_known_equity(),
            min_equity=self.min_equity,
        )
        self._day, self._snapshot = day, snapshot
        self._save_state(day, snapshot)
        log.info("Session equity for %s: %.2f (cash %.2f + MTM %.2f). "
                 "Fixed for the session.", day, snapshot.equity, snapshot.cash,
                 snapshot.unrealized_mtm)
        return snapshot

    def set_manual(self, day: dt.date, equity: float, *, reason: str = "manual") -> EquitySnapshot:
        """Pin equity explicitly. For backtests and simulation, not for live trading."""
        snapshot = EquitySnapshot(
            equity=equity, cash=equity, unrealized_mtm=0.0,
            captured_at=dt.datetime.now().isoformat(timespec="seconds"), source=reason,
        )
        self._day, self._snapshot = day, snapshot
        self._save_state(day, snapshot)
        return snapshot

    @property
    def current(self) -> Optional[EquitySnapshot]:
        return self._snapshot
