"""Alert routing (DESIGN.md 3.9).

The gap this closes: when the drawdown breaker fires, the bot logs ``ERROR`` to stdout and
carries on. If nobody is watching that terminal — and during a 6-hour session nobody is —
the halt is invisible until someone next looks. The same is true of a stale feed, a
reconciliation mismatch, and a dead-lettered order.

Three properties matter more than the transport:

**Alerting must never break trading.** Every failure here is swallowed and logged. A
webhook timing out is not a reason to stop managing positions, so no sink is allowed to
raise into the caller.

**Repetition must not become noise.** A stale feed evaluated every second would emit an
alert every second, and an alert stream nobody can read is the same as no alerts. Each
alert carries a dedup key and is suppressed for a cooldown window; the suppressed count is
reported when it finally re-fires, so the volume is visible without the flood.

**Alerts persist.** They go to a Redis stream as well as the log, so the evening briefing
can report what happened during a session nobody watched — which is the actual use case.

The webhook sink is deliberately generic (a JSON POST) rather than Telegram- or
Slack-specific: the payload shape is the caller's business, and vendor lock-in in an alert
path is a poor trade for a few lines saved.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import config

log = logging.getLogger("ligerbot.alerting")


class Severity(Enum):
    INFO = "info"            # notable, no action needed
    WARNING = "warning"      # degraded; trading continues
    CRITICAL = "critical"    # trading stopped or money is at risk

    @property
    def rank(self) -> int:
        return {"info": 0, "warning": 1, "critical": 2}[self.value]


@dataclass
class Alert:
    """One notable event."""

    title: str
    message: str
    severity: Severity = Severity.WARNING
    source: str = ""
    # Alerts sharing a dedup key are treated as repeats of the same condition. Defaults
    # to the title, which is usually right; pass one explicitly when the message varies
    # but the condition does not (e.g. per-instrument staleness).
    dedup_key: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    at: str = field(default_factory=lambda: dt.datetime.now().isoformat(timespec="seconds"))
    suppressed_since_last: int = 0

    def __post_init__(self) -> None:
        if not self.dedup_key:
            self.dedup_key = self.title

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["severity"] = self.severity.value
        payload["context"] = json.dumps(self.context, default=str)[:2000]
        return payload

    def render(self) -> str:
        lines = [f"[{self.severity.value.upper()}] {self.title}"]
        if self.source:
            lines[0] += f"  ({self.source})"
        lines.append(f"  {self.message}")
        if self.suppressed_since_last:
            lines.append(f"  ({self.suppressed_since_last} repeat(s) suppressed since "
                         f"the last alert)")
        for key, value in self.context.items():
            lines.append(f"  {key}: {value}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Sinks
# ---------------------------------------------------------------------------
def log_sink(alert: Alert) -> None:
    """Always-on sink. Uses a level matching severity so filters behave sensibly."""
    level = {Severity.INFO: logging.INFO,
             Severity.WARNING: logging.WARNING,
             Severity.CRITICAL: logging.ERROR}[alert.severity]
    log.log(level, "ALERT %s", alert.render())


def redis_sink(client) -> Callable[[Alert], None]:
    """Persist to a Redis stream so the evening briefing can report the session."""
    from src import event_bus

    def sink(alert: Alert) -> None:
        event_bus.publish(client, config.STREAM_ALERTS, alert.to_dict(), maxlen=5000)

    return sink


def webhook_sink(url: str, timeout: float = 5.0) -> Callable[[Alert], None]:
    """POST the alert as JSON. Generic on purpose — works with anything that accepts one."""
    def sink(alert: Alert) -> None:
        import requests

        requests.post(url, json=alert.to_dict(), timeout=timeout)

    return sink


# ---------------------------------------------------------------------------
class Alerter:
    """Routes alerts to sinks, with deduplication and failure isolation."""

    def __init__(
        self,
        sinks: Optional[List[Callable[[Alert], None]]] = None,
        *,
        cooldown_seconds: Optional[float] = None,
        min_severity: Severity = Severity.INFO,
    ) -> None:
        self.sinks: List[Callable[[Alert], None]] = list(sinks or [log_sink])
        self.cooldown = (cooldown_seconds if cooldown_seconds is not None
                         else config.ALERT_COOLDOWN_SECONDS)
        self.min_severity = min_severity
        self._last_sent: Dict[str, float] = {}
        self._suppressed: Dict[str, int] = {}
        self.sent = 0
        self.suppressed_total = 0

    def add_sink(self, sink: Callable[[Alert], None]) -> None:
        self.sinks.append(sink)

    def _should_send(self, alert: Alert, now: float) -> bool:
        if alert.severity.rank < self.min_severity.rank:
            return False
        last = self._last_sent.get(alert.dedup_key)
        if last is None:
            return True
        # CRITICAL bypasses nothing — a condition that keeps tripping is still one
        # condition, and paging every second helps no one.
        return (now - last) >= self.cooldown

    def send(self, alert: Alert) -> bool:
        """Route one alert. Returns whether it was actually emitted.

        Never raises. A failing sink is logged and the remaining sinks still run — losing
        an alert is bad, but losing the trading loop because an alert failed is worse.
        """
        now = time.monotonic()
        if not self._should_send(alert, now):
            self._suppressed[alert.dedup_key] = self._suppressed.get(alert.dedup_key, 0) + 1
            self.suppressed_total += 1
            return False

        alert.suppressed_since_last = self._suppressed.pop(alert.dedup_key, 0)
        self._last_sent[alert.dedup_key] = now
        self.sent += 1

        for sink in self.sinks:
            try:
                sink(alert)
            except Exception as exc:  # noqa: BLE001 - a sink must never break the caller
                log.error("Alert sink %s failed: %s", getattr(sink, "__name__", sink), exc)
        return True

    # -- convenience -------------------------------------------------------
    def info(self, title: str, message: str, **kw) -> bool:
        return self.send(Alert(title, message, Severity.INFO, **kw))

    def warning(self, title: str, message: str, **kw) -> bool:
        return self.send(Alert(title, message, Severity.WARNING, **kw))

    def critical(self, title: str, message: str, **kw) -> bool:
        return self.send(Alert(title, message, Severity.CRITICAL, **kw))

    def snapshot(self) -> Dict[str, Any]:
        return {
            "sent": self.sent,
            "suppressed": self.suppressed_total,
            "sinks": len(self.sinks),
            "cooldown_seconds": self.cooldown,
        }


# ---------------------------------------------------------------------------
_default: Optional[Alerter] = None


def build_alerter(redis_client=None, *, source: str = "") -> Alerter:
    """Assemble an alerter from configuration.

    Always includes the log sink, so a misconfigured webhook cannot leave a module with
    no alerting at all.
    """
    sinks: List[Callable[[Alert], None]] = [log_sink]
    if redis_client is not None:
        sinks.append(redis_sink(redis_client))
    if config.ALERT_WEBHOOK_URL:
        sinks.append(webhook_sink(config.ALERT_WEBHOOK_URL))
        log.info("Alert webhook configured.")
    return Alerter(sinks)


def get_alerter(redis_client=None) -> Alerter:
    """Process-wide alerter, built on first use."""
    global _default
    if _default is None:
        _default = build_alerter(redis_client)
    return _default


def reset_alerter() -> None:
    """Drop the process-wide alerter. For tests."""
    global _default
    _default = None
