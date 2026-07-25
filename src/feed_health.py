"""Feed health: staleness detection and reconnection (DESIGN.md 3.5).

Fixes **B10**. The ingestion module's ``_on_close`` logged a warning and did nothing. The
WebSocket dies, the process stays alive, and the strategy keeps computing indicators from
a price that stopped updating — trading a frozen market with real money. It is a quiet
failure: no crash, no error, just decisions made on stale data.

Three mechanisms:

* **Per-instrument staleness.** Feeds do not fail uniformly. One symbol can stop updating
  while others continue, so liveness is tracked per instrument rather than per connection.
* **Asymmetric response.** A stale instrument blocks new *entries* but never *exits*. Being
  unable to see a price is a reason to stop taking risk, not a reason to be trapped in it.
* **Backoff reconnection.** Exponential with a cap, and a hard attempt limit that halts the
  bot rather than reconnecting forever in a tight loop.

Staleness is only meaningful during market hours. Outside them, silence is correct, and a
watchdog that alerted overnight would train everyone to ignore it.
"""
from __future__ import annotations

import datetime as dt
import logging
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import config
from src import market_calendar as cal

log = logging.getLogger("ligerbot.feed_health")


class FeedState(Enum):
    UNKNOWN = "unknown"      # no tick seen yet this session
    LIVE = "live"
    STALE = "stale"
    DISCONNECTED = "disconnected"

    @property
    def allows_entry(self) -> bool:
        """Only a live feed permits new risk."""
        return self is FeedState.LIVE

    @property
    def allows_exit(self) -> bool:
        """Always. Never trap a position because the data went quiet."""
        return True


@dataclass
class InstrumentHealth:
    instrument_id: str
    last_tick_at: Optional[float] = None
    last_price: Optional[float] = None
    tick_count: int = 0
    stale_since: Optional[float] = None

    def age_seconds(self, now: float) -> Optional[float]:
        if self.last_tick_at is None:
            return None
        return now - self.last_tick_at


class FeedMonitor:
    """Tracks per-instrument liveness. Pure — the clock is injected.

    Injecting the clock is not fussiness: a watchdog that reads wall time internally
    cannot be tested for the exact behaviour that matters, which is what it does after
    thirty seconds of silence.
    """

    def __init__(
        self,
        stale_after_seconds: Optional[float] = None,
        *,
        on_state_change: Optional[Callable[[str, FeedState, FeedState], None]] = None,
        alerter=None,
    ) -> None:
        self.stale_after = stale_after_seconds or config.FEED_STALE_SECONDS
        self.instruments: Dict[str, InstrumentHealth] = {}
        self.states: Dict[str, FeedState] = {}
        self.connected: bool = True
        self.on_state_change = on_state_change
        # A stale feed silently blocks entries. Without an alert the bot looks healthy
        # while quietly declining to trade — the most expensive kind of quiet.
        self.alerter = alerter

    # -- ingestion ---------------------------------------------------------
    def record_tick(self, instrument_id: str, price: float, *, now: Optional[float] = None) -> None:
        now = now if now is not None else time.time()
        health = self.instruments.setdefault(
            instrument_id, InstrumentHealth(instrument_id))
        health.last_tick_at = now
        health.last_price = price
        health.tick_count += 1
        health.stale_since = None
        self._set_state(instrument_id, FeedState.LIVE)

    def track(self, instrument_ids: List[str]) -> None:
        """Register instruments we expect ticks for, before any arrive."""
        for instrument_id in instrument_ids:
            self.instruments.setdefault(instrument_id, InstrumentHealth(instrument_id))
            self.states.setdefault(instrument_id, FeedState.UNKNOWN)

    def mark_disconnected(self) -> None:
        self.connected = False
        for instrument_id in self.instruments:
            self._set_state(instrument_id, FeedState.DISCONNECTED)

    def mark_connected(self) -> None:
        self.connected = True

    # -- evaluation --------------------------------------------------------
    def evaluate(self, *, now: Optional[float] = None,
                 moment: Optional[dt.datetime] = None) -> Dict[str, FeedState]:
        """Recompute every instrument's state.

        Outside market hours nothing is stale — silence is the correct state then, and
        alerting on it would make the alert worthless during the hours that matter.
        """
        now = now if now is not None else time.time()
        moment = moment or cal.now_ist()

        if not cal.is_market_open(moment):
            for instrument_id in self.instruments:
                self._set_state(instrument_id, FeedState.UNKNOWN, quiet=True)
            return dict(self.states)

        if not self.connected:
            for instrument_id in self.instruments:
                self._set_state(instrument_id, FeedState.DISCONNECTED)
            return dict(self.states)

        for instrument_id, health in self.instruments.items():
            age = health.age_seconds(now)
            if age is None:
                self._set_state(instrument_id, FeedState.UNKNOWN)
            elif age > self.stale_after:
                if health.stale_since is None:
                    health.stale_since = now
                self._set_state(instrument_id, FeedState.STALE)
            else:
                health.stale_since = None
                self._set_state(instrument_id, FeedState.LIVE)
        return dict(self.states)

    def _set_state(self, instrument_id: str, state: FeedState, *, quiet: bool = False) -> None:
        previous = self.states.get(instrument_id, FeedState.UNKNOWN)
        if previous is state:
            return
        self.states[instrument_id] = state
        if quiet:
            return
        if state is FeedState.STALE:
            log.error("FEED STALE %s — no tick for over %.0fs. Blocking new entries on "
                      "it; exits remain permitted.", instrument_id, self.stale_after)
            self._alert(instrument_id, state, previous)
        elif state is FeedState.DISCONNECTED:
            self._alert(instrument_id, state, previous)
        elif state is FeedState.LIVE and previous in (FeedState.STALE, FeedState.DISCONNECTED):
            log.warning("Feed recovered for %s.", instrument_id)
        if self.on_state_change:
            self.on_state_change(instrument_id, previous, state)

    def _alert(self, instrument_id: str, state: FeedState, previous: FeedState) -> None:
        """Raise feed loss where a human sees it.

        Deduplicated per instrument, because ``evaluate()`` runs on every loop iteration
        and would otherwise emit an alert per second for as long as the feed stayed down.
        """
        if self.alerter is None:
            return
        try:
            whole_feed = self.all_stale()
            self.alerter.critical(
                "Market feed down" if whole_feed else "Feed stale",
                (f"Every tracked instrument has stopped ticking ({state.value}). "
                 f"No new entries anywhere; exits remain permitted."
                 if whole_feed else
                 f"{instrument_id} has no tick within {self.stale_after:.0f}s "
                 f"({state.value}). Entries on it are blocked; exits remain permitted."),
                source="feed_health",
                dedup_key=("feed:all" if whole_feed else f"feed:{instrument_id}"),
                context={"previous_state": previous.value},
            )
        except Exception as exc:  # noqa: BLE001 - alerting must not break the watchdog
            log.error("Could not raise the feed alert: %s", exc)

    # -- queries -----------------------------------------------------------
    def state_of(self, instrument_id: str) -> FeedState:
        return self.states.get(instrument_id, FeedState.UNKNOWN)

    def allows_entry(self, instrument_id: str) -> bool:
        return self.state_of(instrument_id).allows_entry

    def stale_instruments(self) -> List[str]:
        return [i for i, s in self.states.items()
                if s in (FeedState.STALE, FeedState.DISCONNECTED)]

    def all_stale(self) -> bool:
        """True if nothing at all is live — a whole-feed outage, not one bad symbol."""
        if not self.states:
            return False
        return all(s is not FeedState.LIVE for s in self.states.values())

    def snapshot(self) -> Dict[str, Any]:
        now = time.time()
        return {
            "connected": self.connected,
            "tracked": len(self.instruments),
            "live": sum(1 for s in self.states.values() if s is FeedState.LIVE),
            "stale": len(self.stale_instruments()),
            "instruments": {
                i: {
                    "state": self.states.get(i, FeedState.UNKNOWN).value,
                    "age_seconds": round(h.age_seconds(now) or -1.0, 1),
                    "ticks": h.tick_count,
                    "last_price": h.last_price,
                }
                for i, h in self.instruments.items()
            },
        }


FEED_KEY_PREFIX = "ligerbot:feed:"


def publish_liveness(client, instrument_id: str, *, ttl_seconds: Optional[float] = None) -> None:
    """Mark an instrument live, with a TTL that expires it automatically.

    Using key expiry rather than a stored timestamp means staleness needs no reader-side
    clock comparison and no writer to mark things dead: the key simply stops existing.
    A consumer that crashes cannot leave a stale "live" flag behind.
    """
    ttl = int(ttl_seconds or config.FEED_STALE_SECONDS)
    client.set(f"{FEED_KEY_PREFIX}{instrument_id}", time.time(), ex=max(1, ttl))


def is_feed_live(client, instrument_id: str, *, fail_closed: bool = True) -> bool:
    """True if a tick arrived recently enough for the instrument to be tradable.

    Fails closed: if Redis cannot be reached, the feed is treated as dead. Trading on an
    unverifiable price is the exact failure B10 describes.
    """
    try:
        return bool(client.exists(f"{FEED_KEY_PREFIX}{instrument_id}"))
    except Exception as exc:  # noqa: BLE001
        log.error("Cannot read feed liveness for %s (%s) — treating as stale.",
                  instrument_id, exc)
        return not fail_closed


class ReconnectPolicy:
    """Exponential backoff with jitter and a hard attempt cap.

    Jitter matters if several instruments or processes reconnect at once — without it
    they retry in lockstep and hammer the broker in synchronised waves.

    The attempt cap matters more. Reconnecting forever looks resilient and is not: the
    process stays alive, appears healthy to any liveness check, and never trades. Failing
    loudly after a bounded number of attempts gets a human involved.
    """

    def __init__(
        self,
        base_seconds: Optional[float] = None,
        max_seconds: Optional[float] = None,
        max_attempts: Optional[int] = None,
        *,
        jitter: float = 0.25,
    ) -> None:
        self.base = base_seconds or config.RECONNECT_BASE_SECONDS
        self.cap = max_seconds or config.RECONNECT_MAX_SECONDS
        self.max_attempts = max_attempts or config.RECONNECT_MAX_ATTEMPTS
        self.jitter = jitter
        self.attempts = 0

    def next_delay(self) -> Optional[float]:
        """Seconds to wait before the next attempt, or None once exhausted."""
        if self.attempts >= self.max_attempts:
            return None
        delay = min(self.cap, self.base * (2 ** self.attempts))
        self.attempts += 1
        spread = delay * self.jitter
        return max(0.1, delay + random.uniform(-spread, spread))

    def reset(self) -> None:
        self.attempts = 0

    @property
    def exhausted(self) -> bool:
        return self.attempts >= self.max_attempts


class Heartbeat:
    """Periodic liveness beat, so other modules can tell 'quiet' from 'dead'.

    A module that stops publishing is indistinguishable from one with nothing to say
    unless it says so explicitly.
    """

    def __init__(self, client, module: str, interval: Optional[float] = None) -> None:
        self.client = client
        self.module = module
        self.interval = interval or config.HEARTBEAT_INTERVAL_SECONDS
        self._last = 0.0

    def beat(self, extra: Optional[Dict[str, Any]] = None, *, force: bool = False) -> bool:
        now = time.monotonic()
        if not force and now - self._last < self.interval:
            return False
        from src import event_bus

        payload = {"module": self.module, "timestamp": time.time()}
        if extra:
            payload.update(extra)
        event_bus.publish(self.client, config.STREAM_HEARTBEAT, payload, maxlen=1000)
        self._last = now
        return True
