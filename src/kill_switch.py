"""The kill switch (DESIGN.md 3.7).

Three independent layers, because one is not enough — each fails in a different way and
covers the others' blind spots:

1. **``DRY_RUN``** (config, `config.py`). No order ever leaves the machine. Requires a
   restart to change, which makes it the safest and the least useful in an emergency.
2. **The halt key** (this module). A Redis flag checked by the risk manager and execution
   engine on every event. Takes effect within one event loop, no restart. This is the one
   a human reaches for.
3. **Automatic halts** (risk engine, position manager, feed health). Drawdown breach,
   reconciliation mismatch, stale feed, repeated rejections.

**Halting stops new risk; it never stops exits.** A switch that also blocked exits would
strand open positions in exactly the situation where someone reached for it — the
asymmetry runs through the whole system (`market_calendar.Phase.allows_exit`,
`RiskEngine.evaluate`).

Usage::

    python -m src.kill_switch status
    python -m src.kill_switch halt "manual — investigating fills"
    python -m src.kill_switch clear
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import socket
from dataclasses import dataclass
from typing import Any, Dict, Optional

import config

log = logging.getLogger("ligerbot.kill_switch")


@dataclass(frozen=True)
class HaltState:
    halted: bool
    reason: str = ""
    at: str = ""
    by: str = ""
    source: str = ""

    def describe(self) -> str:
        if not self.halted:
            return "RUNNING — no halt in effect"
        return (f"HALTED — {self.reason}\n"
                f"  set at {self.at} by {self.by} ({self.source})")


class KillSwitch:
    """Redis-backed halt flag, shared by every module.

    Deliberately fails **closed**: if Redis is unreachable the switch reports halted.
    A bot that cannot check whether it has been told to stop must not keep trading —
    the whole point is that this remains reachable when other things are going wrong.
    """

    def __init__(self, client, key: Optional[str] = None, *, fail_closed: bool = True) -> None:
        self.client = client
        self.key = key or config.HALT_KEY
        self.fail_closed = fail_closed

    def halt(self, reason: str, *, source: str = "manual", by: Optional[str] = None) -> HaltState:
        state = HaltState(
            halted=True,
            reason=reason,
            at=dt.datetime.now().isoformat(timespec="seconds"),
            by=by or socket.gethostname(),
            source=source,
        )
        self.client.set(self.key, json.dumps({
            "reason": state.reason, "at": state.at, "by": state.by, "source": state.source,
        }))
        log.error("HALT ENGAGED (%s): %s", source, reason)
        return state

    def clear(self) -> None:
        """Lift the halt. Deliberately manual — nothing clears this automatically.

        An automatic clear would let a transient fault resolve itself into resumed
        trading without anyone having looked at why it tripped.
        """
        self.client.delete(self.key)
        log.warning("Halt cleared — new entries permitted again.")

    def state(self) -> HaltState:
        try:
            raw = self.client.get(self.key)
        except Exception as exc:  # noqa: BLE001 - any Redis failure counts
            if self.fail_closed:
                log.error("Cannot read the halt key (%s) — failing closed.", exc)
                return HaltState(halted=True, reason=f"halt key unreadable: {exc}",
                                 source="fail-closed")
            raise
        if not raw:
            return HaltState(halted=False)
        try:
            payload: Dict[str, Any] = json.loads(raw)
        except (ValueError, TypeError):
            payload = {"reason": str(raw)}
        return HaltState(
            halted=True,
            reason=payload.get("reason", "unspecified"),
            at=payload.get("at", ""),
            by=payload.get("by", ""),
            source=payload.get("source", "unknown"),
        )

    def is_halted(self) -> bool:
        return self.state().halted


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="LIGERBOT kill switch")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="show the current halt state")
    halt = sub.add_parser("halt", help="block all new entries immediately")
    halt.add_argument("reason", help="why — recorded and shown to every module")
    sub.add_parser("clear", help="lift the halt")
    args = parser.parse_args()

    from src import event_bus

    client = event_bus.get_client()
    if not event_bus.ping(client):
        print("Redis is not reachable — cannot reach the kill switch.")
        print("Note: modules fail closed when they cannot read it, so they are already")
        print("refusing new entries.")
        raise SystemExit(1)

    switch = KillSwitch(client)
    if args.command == "status":
        print(switch.state().describe())
    elif args.command == "halt":
        print(switch.halt(args.reason).describe())
        print("\nOpen positions can still be exited. Only new entries are blocked.")
    elif args.command == "clear":
        switch.clear()
        print("Halt cleared.")


if __name__ == "__main__":
    main()
