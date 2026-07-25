"""Position-size ladder for live trading (DESIGN.md Phase 5, items 25-26).

"Live with the smallest tradable size" and "scale only after a sustained period matching
paper behaviour" are the two Phase 5 requirements. This implements them as a ladder rather
than a switch.

The asymmetry is the design:

* **Promotion is slow and evidence-gated.** A rung is earned by trading a minimum number
  of sessions *and* a minimum number of trades at that rung, with results that still track
  expectation. Sessions alone are not enough — five quiet days prove nothing.
* **Demotion is immediate and skips rungs.** One bad enough day drops you straight to the
  floor. The costs are asymmetric: promoting too slowly loses a little upside, promoting
  too quickly loses capital.

Why a ladder at all, rather than going straight to full size once paper passes: paper
trading cannot reproduce the two things that most often break a live system — real fills
against real liquidity, and the operator's own reaction to a live drawdown. The ladder buys
information about both while the stake is small.

Scaling multiplies the **risk budget**, never the risk *rules*. The 0.5%-per-trade and
1.5%-open-risk caps are fractions of a scaled equity base, so every proportional guarantee
from D2 survives the ramp unchanged.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import config

log = logging.getLogger("ligerbot.live_scaling")


@dataclass(frozen=True)
class Rung:
    """One step on the ladder."""

    name: str
    size_multiplier: float          # fraction of full risk budget
    min_sessions: int               # sessions at this rung before promotion
    min_trades: int                 # trades at this rung before promotion

    def describe(self) -> str:
        return (f"{self.name} ({self.size_multiplier:.0%} size, needs "
                f"{self.min_sessions} sessions / {self.min_trades} trades)")


# The default ladder. Starts at 10% of the normal risk budget — small enough that a
# systemic error is affordable, large enough that fills are representative. Going below
# ~10% stops being informative: fixed costs dominate and the results say more about
# brokerage than about the strategy.
DEFAULT_LADDER: List[Rung] = [
    Rung("minimum", 0.10, min_sessions=10, min_trades=20),
    Rung("quarter", 0.25, min_sessions=10, min_trades=30),
    Rung("half", 0.50, min_sessions=15, min_trades=50),
    Rung("three-quarter", 0.75, min_sessions=15, min_trades=50),
    Rung("full", 1.00, min_sessions=0, min_trades=0),
]


@dataclass
class RungProgress:
    sessions: int = 0
    trades: int = 0
    net_pnl: float = 0.0
    expectancy_r_sum: float = 0.0

    @property
    def expectancy_r(self) -> float:
        return self.expectancy_r_sum / self.trades if self.trades else 0.0

    def reset(self) -> None:
        self.sessions = 0
        self.trades = 0
        self.net_pnl = 0.0
        self.expectancy_r_sum = 0.0


@dataclass
class ScalingState:
    """Persisted ladder position. Survives restarts — progress is hard-won."""

    rung_index: int = 0
    progress: RungProgress = field(default_factory=RungProgress)
    history: List[str] = field(default_factory=list)
    consecutive_losing_sessions: int = 0
    demotions: int = 0

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["progress"] = asdict(self.progress)
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ScalingState":
        progress = RungProgress(**payload.get("progress", {}))
        known = {f for f in cls.__dataclass_fields__} - {"progress"}
        fields_ = {k: v for k, v in payload.items() if k in known}
        return cls(progress=progress, **fields_)


class ScalingLadder:
    """Tracks position on the ladder and decides promotion and demotion."""

    def __init__(
        self,
        ladder: Optional[List[Rung]] = None,
        state_path: Optional[str | Path] = None,
        *,
        expectancy_floor_r: float = -0.05,
    ) -> None:
        self.ladder = ladder or DEFAULT_LADDER
        self.state_path = Path(state_path or config.LIVE_SCALING_STATE_PATH)
        # Below this, live results are not tracking expectation closely enough to earn
        # more size — even if P&L happens to be positive.
        self.expectancy_floor_r = expectancy_floor_r
        self.state = self._load()

    # -- persistence -------------------------------------------------------
    def _load(self) -> ScalingState:
        if not self.state_path.exists():
            return ScalingState()
        try:
            return ScalingState.from_dict(
                json.loads(self.state_path.read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError) as exc:
            # Fail to the FLOOR, not to the last known rung. If we cannot read how much
            # size was earned, the safe assumption is none.
            log.error("Scaling state at %s unreadable (%s) — restarting at the floor.",
                      self.state_path, exc)
            return ScalingState()

    def save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(
            json.dumps(self.state.to_dict(), indent=2), encoding="utf-8")

    # -- current position --------------------------------------------------
    @property
    def rung(self) -> Rung:
        index = max(0, min(self.state.rung_index, len(self.ladder) - 1))
        return self.ladder[index]

    @property
    def size_multiplier(self) -> float:
        return self.rung.size_multiplier

    @property
    def at_full_size(self) -> bool:
        return self.state.rung_index >= len(self.ladder) - 1

    def scaled_equity(self, equity: float) -> float:
        """The equity base risk rules apply to.

        Scaling the *base* rather than the rules keeps every proportional guarantee from
        D2 intact: 0.5% of a 10% base is 0.05% of the account, and the open-risk and
        drawdown caps stay in the same ratio to each other.
        """
        return equity * self.size_multiplier

    # -- session outcomes --------------------------------------------------
    def record_session(
        self,
        day: dt.date,
        *,
        trades: int,
        net_pnl: float,
        expectancy_r_sum: float,
        halted: bool = False,
    ) -> str:
        """Record a completed live session and re-evaluate the rung.

        Returns a human-readable note about what changed.
        """
        progress = self.state.progress
        progress.sessions += 1
        progress.trades += trades
        progress.net_pnl += net_pnl
        progress.expectancy_r_sum += expectancy_r_sum

        if net_pnl < 0:
            self.state.consecutive_losing_sessions += 1
        else:
            self.state.consecutive_losing_sessions = 0

        note = self._evaluate_demotion(day, halted=halted)
        if note is None:
            note = self._evaluate_promotion(day)
        if note is None:
            note = (f"holding at {self.rung.name}: "
                    f"{progress.sessions}/{self.rung.min_sessions} sessions, "
                    f"{progress.trades}/{self.rung.min_trades} trades, "
                    f"{progress.expectancy_r:+.3f}R")

        self.state.history.append(f"{day.isoformat()} {note}")
        self.save()
        return note

    def _evaluate_demotion(self, day: dt.date, *, halted: bool) -> Optional[str]:
        """Demote hard and fast. Promoting too slowly costs upside; too fast costs capital."""
        reasons: List[str] = []
        if halted:
            reasons.append("session halted")
        if self.state.consecutive_losing_sessions >= config.LIVE_MAX_LOSING_SESSIONS:
            reasons.append(
                f"{self.state.consecutive_losing_sessions} consecutive losing sessions")
        progress = self.state.progress
        if progress.trades >= 10 and progress.expectancy_r < self.expectancy_floor_r:
            reasons.append(
                f"expectancy {progress.expectancy_r:+.3f}R below the "
                f"{self.expectancy_floor_r:+.3f}R floor")

        if not reasons or self.state.rung_index == 0:
            return None

        self.state.rung_index = 0      # straight to the floor, no gradual step down
        self.state.demotions += 1
        progress.reset()
        self.state.consecutive_losing_sessions = 0
        note = f"DEMOTED to {self.rung.name} — {'; '.join(reasons)}"
        log.error("%s on %s", note, day)
        return note

    def _evaluate_promotion(self, day: dt.date) -> Optional[str]:
        if self.at_full_size:
            return None
        rung, progress = self.rung, self.state.progress

        # Sessions AND trades: five quiet days prove nothing about fills.
        if progress.sessions < rung.min_sessions or progress.trades < rung.min_trades:
            return None
        if progress.expectancy_r < self.expectancy_floor_r:
            return None

        self.state.rung_index += 1
        progress.reset()
        note = (f"PROMOTED to {self.rung.name} "
                f"({self.rung.size_multiplier:.0%} size)")
        log.warning("%s on %s", note, day)
        return note

    def force_floor(self, reason: str) -> None:
        """Drop to the smallest size immediately. For incidents, not for routine use."""
        self.state.rung_index = 0
        self.state.progress.reset()
        self.state.demotions += 1
        self.state.history.append(f"forced to floor: {reason}")
        log.error("Scaling forced to the floor: %s", reason)
        self.save()

    def summary(self) -> str:
        progress = self.state.progress
        lines = [
            f"  Rung                 {self.rung.name} ({self.size_multiplier:.0%} size)",
            f"  Sessions at rung     {progress.sessions}"
            + (f"/{self.rung.min_sessions}" if not self.at_full_size else ""),
            f"  Trades at rung       {progress.trades}"
            + (f"/{self.rung.min_trades}" if not self.at_full_size else ""),
            f"  Expectancy at rung   {progress.expectancy_r:+.3f}R",
            f"  Net P&L at rung      {progress.net_pnl:+,.2f}",
            f"  Demotions to date    {self.state.demotions}",
        ]
        return "\n".join(lines)
