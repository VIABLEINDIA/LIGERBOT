"""Per-session result recording (DESIGN.md 4, Phase 4).

Phase 4 runs for 20-40 sessions and its exit criterion compares paper results against a
backtest over *those same sessions*. That comparison needs both sides on disk in a
comparable shape — so every paper session is written out as it completes, and the
backtester writes the same structure.

Two things this deliberately records that are easy to omit and painful to reconstruct:

* **Rejections.** A session where the strategy wanted twenty trades and the risk engine
  allowed three is a very different session from one with three signals, and the P&L looks
  identical. Divergence from the backtest is often explained entirely by which signals
  were refused live but allowed in replay.
* **Halts and feed gaps.** A day that halted at 11:00 cannot be compared like-for-like
  against a backtest that traded until 15:10. Recording it means the reconciliation can
  exclude the day rather than report a divergence it cannot explain.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger("ligerbot.session_recorder")


@dataclass
class RecordedTrade:
    """One completed round trip, in the shape both paper and backtest produce."""

    instrument_id: str
    direction: str
    quantity: int
    entry_at: str
    entry_price: float
    exit_at: str
    exit_price: float
    exit_reason: str
    gross_pnl: float
    costs: float
    slippage: float
    net_pnl: float
    risk_amount: float
    r_multiple: float
    strategy_name: str = ""
    reason: str = ""


@dataclass
class SessionRecord:
    """Everything one trading session produced."""

    day: str
    source: str                        # "paper" | "backtest" | "live"
    starting_equity: float
    ending_equity: float
    trades: List[RecordedTrade] = field(default_factory=list)
    signals_generated: int = 0
    signals_rejected: int = 0
    rejection_reasons: Dict[str, int] = field(default_factory=dict)
    halted: bool = False
    halt_reason: str = ""
    stale_feed_seconds: float = 0.0
    strategy_description: str = ""
    notes: List[str] = field(default_factory=list)

    # -- derived -----------------------------------------------------------
    @property
    def net_pnl(self) -> float:
        return self.ending_equity - self.starting_equity

    @property
    def return_pct(self) -> float:
        return self.net_pnl / self.starting_equity if self.starting_equity else 0.0

    @property
    def trade_count(self) -> int:
        return len(self.trades)

    @property
    def win_count(self) -> int:
        return sum(1 for t in self.trades if t.net_pnl > 0)

    @property
    def win_rate(self) -> float:
        return self.win_count / self.trade_count if self.trades else 0.0

    @property
    def total_costs(self) -> float:
        return sum(t.costs for t in self.trades)

    @property
    def total_slippage(self) -> float:
        return sum(t.slippage for t in self.trades)

    @property
    def expectancy_r(self) -> float:
        if not self.trades:
            return 0.0
        return sum(t.r_multiple for t in self.trades) / len(self.trades)

    @property
    def comparable(self) -> bool:
        """Whether this session can be compared like-for-like against a backtest.

        A halted day stopped trading partway through; a backtest of the same day did
        not. Comparing them produces a divergence with a known, uninteresting cause.
        """
        return not self.halted

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["net_pnl"] = round(self.net_pnl, 2)
        payload["return_pct"] = round(self.return_pct, 6)
        payload["trade_count"] = self.trade_count
        payload["win_rate"] = round(self.win_rate, 4)
        payload["expectancy_r"] = round(self.expectancy_r, 4)
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "SessionRecord":
        trades = [RecordedTrade(**t) for t in payload.get("trades", [])]
        known = {f for f in cls.__dataclass_fields__}
        fields_ = {k: v for k, v in payload.items() if k in known and k != "trades"}
        return cls(trades=trades, **fields_)


class SessionStore:
    """Reads and writes session records, one JSON file per (source, day)."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, source: str, day: dt.date | str) -> Path:
        day_str = day.isoformat() if isinstance(day, dt.date) else str(day)
        folder = self.root / source
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f"{day_str}.json"

    def save(self, record: SessionRecord) -> Path:
        path = self.path_for(record.source, record.day)
        path.write_text(json.dumps(record.to_dict(), indent=2, default=str),
                        encoding="utf-8")
        log.info("Recorded %s session %s: %d trade(s), net %+.2f",
                 record.source, record.day, record.trade_count, record.net_pnl)
        return path

    def load(self, source: str, day: dt.date | str) -> Optional[SessionRecord]:
        path = self.path_for(source, day)
        if not path.exists():
            return None
        try:
            return SessionRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError) as exc:
            log.error("Session record at %s is unreadable (%s).", path, exc)
            return None

    def days(self, source: str) -> List[str]:
        folder = self.root / source
        if not folder.exists():
            return []
        return sorted(p.stem for p in folder.glob("*.json"))

    def load_all(self, source: str) -> List[SessionRecord]:
        records = [self.load(source, day) for day in self.days(source)]
        return [r for r in records if r is not None]

    def common_days(self, left: str, right: str) -> List[str]:
        """Days both sources recorded — the only ones that can be compared."""
        return sorted(set(self.days(left)) & set(self.days(right)))


def record_from_backtest(result: Any, day: dt.date, starting_equity: float) -> SessionRecord:
    """Build a session record from a backtest result, for one day.

    Kept alongside the paper recorder so both sides are constructed from the same
    definitions — a reconciliation comparing differently-derived numbers would measure
    the derivation, not the strategy.
    """
    trades = [
        RecordedTrade(
            instrument_id=t.instrument_id,
            direction="LONG" if t.is_long else "SHORT",
            quantity=t.quantity,
            entry_at=t.entry_at.isoformat(),
            entry_price=round(t.entry_price, 4),
            exit_at=t.exit_at.isoformat(),
            exit_price=round(t.exit_price, 4),
            exit_reason=t.exit_reason.value,
            gross_pnl=round(t.gross_pnl, 2),
            costs=round(t.total_costs, 2),
            slippage=round(t.slippage_cost, 2),
            net_pnl=round(t.net_pnl, 2),
            risk_amount=round(t.risk_amount, 2),
            r_multiple=round(t.r_multiple, 4),
            strategy_name=t.strategy_name,
            reason=t.reason,
        )
        for t in result.portfolio.trades if t.entry_at.date() == day
    ]
    ending = starting_equity + sum(t.net_pnl for t in trades)
    return SessionRecord(
        day=day.isoformat(),
        source="backtest",
        starting_equity=round(starting_equity, 2),
        ending_equity=round(ending, 2),
        trades=trades,
        rejection_reasons=dict(getattr(result, "rejections", {}) or {}),
        signals_rejected=sum((getattr(result, "rejections", {}) or {}).values()),
        strategy_description=result.strategy_description,
    )
