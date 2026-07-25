"""Module 4 — Risk Management System (the firewall).

Consumes ``trade_signals``, applies the capital-protection rules, sizes the position, and
only then emits ``approved_orders``. Nothing reaches the broker without passing through
here.

Rewritten in Phase 3 to be a thin adapter around :class:`src.risk_engine.RiskEngine`. The
decision logic lives there, pure and I/O-free, so the **same code** gates trades in the
backtester and in production (DESIGN.md 2.1). A risk rule that exists only in the live path
is a rule that was never tested.

What this module adds on top of the pure engine — all of it I/O:

* **Real P&L.** Consumes ``position_updates`` from the position manager, which is what
  finally makes the drawdown breaker load-bearing rather than dead code (defect B2).
* **Durable consumption.** Consumer groups with explicit acks, so a restart does not
  discard queued signals (defect B6).
* **Kill switch and feed staleness.** Both block new entries and neither blocks exits.
* **Client order ids.** Assigned here so the execution engine can recognise a redelivered
  order as a duplicate (§3.3).

    python -m src.risk_manager
"""
from __future__ import annotations

import datetime as dt
import logging
from typing import Any, Dict, Optional

import config
from src import event_bus
from src import kotak_api
from src import feed_health
from src import market_calendar as cal
from src.account import SessionEquity
from src.kill_switch import KillSwitch
from src.order_state import client_order_id
from src.risk_engine import Intent, RiskEngine, Signal

logging.basicConfig(level=logging.INFO, format="%(asctime)s [risk] %(message)s")
log = logging.getLogger("ligerbot.risk")


class RiskManager:
    """Redis adapter around the pure risk engine."""

    def __init__(self, neo_client: Any = None, instrument_master: Any = None) -> None:
        self.client = event_bus.get_client()
        self.engine = RiskEngine(config.risk_limits())
        self.kill_switch = KillSwitch(self.client)
        self.neo = neo_client
        self.instruments = instrument_master
        self.session_equity = SessionEquity(
            config.EQUITY_STATE_PATH, min_equity=config.MIN_EQUITY)

        group = f"{config.CONSUMER_GROUP_PREFIX}.risk"
        self.signals = event_bus.StreamConsumer(
            self.client, config.STREAM_TRADE_SIGNALS, group, "risk-1",
            max_deliveries=config.MAX_DELIVERIES)
        self.positions = event_bus.StreamConsumer(
            self.client, config.STREAM_POSITION_UPDATES, group, "risk-1",
            max_deliveries=config.MAX_DELIVERIES)

    # -- session -----------------------------------------------------------
    def _resolve_equity(self, day: dt.date) -> Optional[float]:
        """Establish the day's sizing base (D1). Fails closed on any doubt."""
        if self.neo is not None:
            try:
                return self.session_equity.resolve(day, self.neo).equity
            except Exception as exc:  # noqa: BLE001
                log.error("Cannot establish session equity (%s). Refusing new entries — "
                          "a wrong figure would mis-size every trade today.", exc)
                self.kill_switch.halt(f"session equity unavailable: {exc}",
                                      source="risk_manager")
                return None
        log.warning("No broker connection — using TOTAL_EQUITY=%.0f as a simulation "
                    "fallback. This is NOT valid for live trading.", config.TOTAL_EQUITY)
        return config.TOTAL_EQUITY

    def start_session(self, day: dt.date) -> bool:
        equity = self._resolve_equity(day)
        if equity is None:
            return False
        self.engine.start_session(day, equity)
        return True

    # -- position updates (fixes B2) --------------------------------------
    def _handle_position_update(self, update: Dict[str, Any]) -> None:
        """Adopt the position manager's P&L as truth.

        The position manager reconciles against the broker, so its figures are closer to
        reality than anything this module could derive from the signals it happens to
        have seen.
        """
        net = update.get("net_pnl_today")
        if net is None:
            return
        previous = self.engine.realized_pnl_today
        self.engine.realized_pnl_today = float(net)
        if self.engine.check_daily_drawdown():
            self.kill_switch.halt(self.engine.halt_reason, source="risk_engine")
        if abs(float(net) - previous) > 1e-9:
            log.info("P&L now %+.2f (%.3f%% of equity) | open risk %.3f%%",
                     float(net),
                     100.0 * float(net) / max(1.0, self.engine.session_equity),
                     100.0 * self.engine.total_open_risk_pct())

    # -- signals -----------------------------------------------------------
    @staticmethod
    def _parse_bar_time(raw: Any) -> dt.datetime:
        """Recover the signal's bar time, defaulting to now if it is unusable.

        Defaulting to *now* rather than to the epoch is deliberate: a missing bar time
        should make the signal look fresh and be judged against the current window,
        not be silently rejected as ancient.
        """
        if isinstance(raw, dt.datetime):
            return cal.to_ist(raw)
        if isinstance(raw, str) and raw:
            try:
                return cal.to_ist(dt.datetime.fromisoformat(raw))
            except ValueError:
                pass
        return cal.now_ist()

    @staticmethod
    def _signal_age_seconds(bar_time: dt.datetime) -> Optional[float]:
        """Seconds since the signal's bar closed. Negative values (replay) return None."""
        age = (cal.now_ist() - cal.to_ist(bar_time)).total_seconds()
        return age if age >= 0 else None

    def _correlation_group(self, instrument_id: str) -> str:
        """Sector group for the concentration filter, or "" if unknown.

        Resolved from the instrument master where available; otherwise from the id's
        trailing symbol. Unknown means unconstrained — the filter should never block a
        trade because a lookup failed.
        """
        from src.instruments import correlation_group_for

        if self.instruments is not None:
            instrument = self.instruments.by_id(instrument_id)
            if instrument is not None:
                return (instrument.correlation_group
                        or correlation_group_for(instrument.trading_symbol))
        return correlation_group_for(instrument_id.split(":")[-1])

    def _entry_permission(self, signal: Signal) -> tuple[bool, bool, str]:
        """Decide what this signal is allowed to do, and say precisely why not.

        Returns ``(allows_entry, allows_exit, block_reason)``. Every gate here blocks new
        risk and none blocks exits — the asymmetry that runs through the whole system.
        """
        # The phase is evaluated at the signal's OWN bar time, not at wall-clock now.
        # The backtester does exactly this (`cal.phase(bar.bar_end)`), and a live path
        # using `now` would judge a queued or replayed signal against a different window
        # than the one it was generated in — the live/backtest divergence DESIGN.md 2.1
        # exists to prevent.
        phase = cal.phase(signal.bar_time)
        allows_exit = phase.allows_exit
        if not phase.allows_entry:
            return False, allows_exit, f"session phase {phase.value} at {signal.bar_time}"

        # Judging the phase at bar_time is exactly why staleness needs its own check:
        # otherwise an hours-old signal passes on the strength of when it was generated.
        age = self._signal_age_seconds(signal.bar_time)
        if age is not None and age > config.MAX_SIGNAL_AGE_SECONDS:
            return False, allows_exit, (
                f"signal is {age:.0f}s old (limit {config.MAX_SIGNAL_AGE_SECONDS:.0f}s) — "
                f"the price it was based on is no longer current")

        halt = self.kill_switch.state()
        if halt.halted:
            return False, allows_exit, f"halted: {halt.reason}"

        if not feed_health.is_feed_live(self.client, signal.instrument_id):
            return False, allows_exit, (
                f"feed stale for {signal.instrument_id} — no tick within "
                f"{config.FEED_STALE_SECONDS:.0f}s")

        return True, allows_exit, ""

    def _handle_signal(self, payload: Dict[str, Any]) -> None:
        instrument_id = payload.get("instrument_id") or payload.get("instrument")
        if not instrument_id:
            raise ValueError(f"signal without an instrument: {payload}")

        try:
            intent = Intent(payload.get("intent", "OPEN_LONG"))
        except ValueError as exc:
            raise ValueError(f"unknown intent in {payload}") from exc

        bar_time = self._parse_bar_time(payload.get("bar_time"))
        stop_loss = payload.get("stop_loss")
        signal = Signal(
            instrument_id=str(instrument_id),
            intent=intent,
            ref_price=float(payload["ref_price"] if "ref_price" in payload
                            else payload.get("price", 0.0)),
            bar_time=bar_time,
            stop_loss=float(stop_loss) if stop_loss not in (None, "", "None", 0) else None,
            take_profit=(float(payload["take_profit"])
                         if payload.get("take_profit") else None),
            strategy_name=str(payload.get("strategy_name", "unknown")),
            strategy_version=str(payload.get("strategy_version", "0")),
            params_hash=str(payload.get("params_hash", "")),
            reason=str(payload.get("reason", "")),
        )

        allows_entry, allows_exit, block_reason = self._entry_permission(signal)

        decision = self.engine.evaluate(
            signal, allows_entry=allows_entry, allows_exit=allows_exit,
            correlation_group=self._correlation_group(signal.instrument_id))
        if not decision.approved:
            # Prefer our specific cause over the engine's generic "phase does not
            # permit" text. During an incident, a rejection reason that names the wrong
            # subsystem sends whoever is on the keyboard down the wrong path.
            reason = block_reason if (block_reason and intent.is_open) else decision.reason
            log.info("REJECTED %s %s — %s", intent.value, instrument_id, reason)
            return

        order = decision.order
        coid = client_order_id(
            order.instrument_id, order.intent, signal.bar_time, order.quantity)
        event_bus.publish(self.client, config.STREAM_APPROVED_ORDERS, {
            "client_order_id": coid,
            "correlation_id": coid,
            "instrument_id": order.instrument_id,
            "side": order.side.value,
            "intent": order.intent.value,
            "quantity": order.quantity,
            "order_type": order.order_type,
            "price": order.ref_price,
            "stop_loss": order.stop_loss or 0.0,
            "take_profit": order.take_profit or 0.0,
            "risk_amount": order.risk_amount,
            "strategy_name": order.strategy_name,
            "reason": order.reason,
            "timestamp": dt.datetime.now().timestamp(),
        })
        log.info("APPROVED %s %d %s @ %.2f | risk %.0f | open risk %.3f%% | %s",
                 order.side.value, order.quantity, order.instrument_id,
                 order.ref_price, order.risk_amount,
                 100.0 * self.engine.total_open_risk_pct(), coid)

    # -- loop --------------------------------------------------------------
    def run(self) -> None:
        if not event_bus.ping(self.client):
            log.error("Redis not reachable — start it with `docker compose up -d`.")
            return

        today = cal.now_ist().date()
        if not self.start_session(today):
            log.error("Could not start the session — halting rather than guessing.")
            return

        limits = self.engine.limits
        log.info("Risk Manager online. equity=%.0f risk/trade=%.2f%% open-risk<=%.2f%% "
                 "max_dd=%.2f%% max_pos=%d short=%s",
                 self.engine.session_equity, limits.risk_per_trade * 100,
                 limits.max_open_risk * 100, limits.max_daily_drawdown * 100,
                 limits.max_open_positions, limits.allow_short)

        while True:
            now = cal.now_ist()
            if now.date() != self.engine.session_day:
                log.info("New trading day %s — re-establishing the session.", now.date())
                if not self.start_session(now.date()):
                    return

            # Position updates first: sizing and the breaker must act on the freshest
            # P&L available before any new signal is judged.
            for entry_id, fields in self.positions.read(count=50, block_ms=10):
                self.positions.handle(entry_id, fields, self._handle_position_update)

            for entry_id, fields in self.signals.claim_stale(
                    min_idle_ms=config.CLAIM_IDLE_MS):
                self.signals.handle(entry_id, fields, self._handle_signal)

            for entry_id, fields in self.signals.read(count=100, block_ms=1000):
                self.signals.handle(entry_id, fields, self._handle_signal)


def main() -> None:
    # The SDK sets no per-request timeout; bound it before any network call.
    kotak_api.bound_network_calls()

    neo = None
    if config.needs_broker_session():
        # Paper mode needs the broker too: equity must come from it (D1). Branching on
        # DRY_RUN here used to skip this in paper mode, so sizing silently fell back to
        # a configured figure and every recorded session was invalid.
        from src.auth import authenticate_neo
        neo = authenticate_neo()

    master = None
    try:
        from src.instruments import load_or_download_master
        master = load_or_download_master(neo)
    except Exception as exc:  # noqa: BLE001 - the correlation filter degrades gracefully
        log.warning("No instrument master (%s) — the correlation filter will treat every "
                    "instrument as ungrouped.", exc)

    RiskManager(neo, master).run()


if __name__ == "__main__":
    main()
