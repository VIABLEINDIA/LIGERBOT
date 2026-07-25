"""A deterministic, human-readable record of what the pipeline decided.

Built for the golden-file test (DESIGN.md 3.10), but useful on its own: when a paper
session and a backtest of the same day disagree, this is the artefact you diff to find
*where* they diverged rather than *that* they diverged.

Three properties are load-bearing, and all three are about the file being diffable by a
person:

**Byte-stability.** Every float is formatted to a fixed number of decimals at the moment it
is recorded, never at render time and never by `repr`. Rounding at the boundary means the
same run produces the same bytes on any platform or Python version — otherwise the golden
file fails on trivia and people learn to regenerate it without reading it, which destroys
the only thing it was for.

**No wall-clock, no paths, no object ids.** Anything that varies between runs would have to
be filtered out later, and filters are where golden tests go to die.

**Fixed-width columns.** A unified diff of a ragged file highlights whole lines; a diff of
an aligned one highlights the field that moved.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import List, Optional

# Money is recorded in paise-precision rupees. Two decimals is what the exchange quotes,
# and more would record float noise as though it were information.
MONEY = 2
PERCENT = 3


def money(value: Optional[float]) -> str:
    return "-" if value is None else f"{value:.{MONEY}f}"


def when(moment: dt.datetime) -> str:
    return moment.strftime("%Y-%m-%d %H:%M")


@dataclass(frozen=True)
class TraceEvent:
    seq: int
    at: str
    kind: str
    instrument_id: str
    detail: str

    def render(self) -> str:
        return f"{self.at}  {self.kind:<8}{self.instrument_id:<13}{self.detail}".rstrip()


@dataclass
class Trace:
    """Append-only event log for one backtest run."""

    header: List[str] = field(default_factory=list)
    events: List[TraceEvent] = field(default_factory=list)

    # -- header ------------------------------------------------------------
    def describe(self, **fields: object) -> None:
        """Record run provenance. A trace without it cannot be interpreted later."""
        for key, value in fields.items():
            self.header.append(f"# {key:<12}{value}")

    # -- events ------------------------------------------------------------
    def _add(self, kind: str, at: dt.datetime, instrument_id: str, detail: str) -> None:
        self.events.append(TraceEvent(len(self.events), when(at), kind,
                                      instrument_id, detail))

    def session_start(self, at: dt.datetime, day: dt.date, equity: float) -> None:
        # Labelled `sizing_equity`, not `equity`: under compound=False this is the fixed
        # base positions are sized off, which is NOT the portfolio's current value. Giving
        # both the same name in a file meant for human review invites exactly the
        # misreading where a flat sizing base looks like a day of zero P&L.
        self._add("SESSION", at, "-", f"start {day}  sizing_equity={money(equity)}")

    def session_end(self, at: dt.datetime, day: dt.date, equity: float,
                    trades: int, halted: bool) -> None:
        state = "  HALTED" if halted else ""
        self._add("SESSION", at, "-",
                  f"end   {day}  equity={money(equity)}  trades={trades}{state}")

    def signal(self, signal) -> None:
        self._add("SIGNAL", signal.bar_time, signal.instrument_id,
                  f"{signal.intent.name:<12} ref={money(signal.ref_price)} "
                  f"stop={money(signal.stop_loss)}")

    def submit(self, at: dt.datetime, order) -> None:
        self._add("SUBMIT", at, order.instrument_id,
                  f"{order.side.name:<4} {order.quantity:>6}  "
                  f"risk={money(order.risk_amount)}")

    def reject(self, at: dt.datetime, instrument_id: str, reason: str) -> None:
        self._add("REJECT", at, instrument_id, reason)

    def fill(self, fill) -> None:
        self._add("FILL", fill.at, fill.instrument_id,
                  f"{fill.side.name:<4} {fill.quantity:>6} @ {money(fill.price)}  "
                  f"{fill.reason.value:<10} costs={money(fill.costs.total)} "
                  f"slip={money(fill.slippage_per_share)}")

    # -- output ------------------------------------------------------------
    def lines(self) -> List[str]:
        return [event.render() for event in self.events]

    def render(self) -> str:
        body = "\n".join(self.lines())
        head = "\n".join([
            "# LIGERBOT pipeline trace",
            *self.header,
            "#",
            # Without this note a reviewer cannot tell next-bar execution from same-bar
            # execution, because the two timestamps are the same clock value. That is the
            # single most important invariant in the engine, so it is spelled out rather
            # than left to be inferred.
            "# SIGNAL/SUBMIT are stamped at the CLOSE of the bar the strategy saw;",
            "# FILL is stamped at the OPEN of the FOLLOWING bar. Equal times on adjacent",
            "# lines are therefore correct next-bar execution, not a same-bar fill.",
        ])
        return f"{head}\n\n{body}\n"

    def __len__(self) -> int:
        return len(self.events)
