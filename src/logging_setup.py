"""Structured logging with correlation-ID threading (DESIGN.md 3.9).

Five processes participate in every trade. A signal is born in the strategy engine, judged
in the risk manager, sized into an order, placed by the execution engine, filled, and
booked by the position manager — each in its own process, each writing to its own log. When
something goes wrong at 11:04, reconstructing *which* signal became *which* order means
reading five files and matching on timestamps and instrument names. Timestamps collide and
instrument names repeat.

**The correlation id is the fix, and threading it by hand is not.** Passing it as an
argument through every function that might log would mean touching every call site and
would be forgotten exactly once — in the error path, where it matters. So it lives in a
:class:`~contextvars.ContextVar`, and :meth:`src.event_bus.StreamConsumer.handle` adopts it
from the message being handled. Any log line emitted while handling a message carries the
id automatically, across every module, without a single call site changing.

Three properties are deliberate:

**Text is still the default.** A human watching a terminal during a live session needs to
read it. JSON is opt-in via ``LOG_FORMAT=json``, for when something is shipping these
somewhere that parses them.

**Logging cannot break trading.** A formatter that raises on an unserialisable object would
turn a log line into an outage. Every failure path here degrades to a plainer line instead.

**Secrets never reach the log.** Structured logging invites passing rich context objects,
and a broker session is a dict containing a bearer token. Sensitive keys are redacted on
the way out, at the formatter, so no call site has to remember.
"""
from __future__ import annotations

import contextlib
import contextvars
import datetime as dt
import json
import logging
import sys
from typing import Any, Dict, Iterator, Optional

import config

# The id of whatever unit of work is in flight on this thread/task.
_correlation_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "ligerbot_correlation_id", default=None)

# Redacted wherever they appear as a key, at any depth. Matched case-insensitively as a
# substring, so `KOTAK_MPIN`, `bearer_token` and `view_token` are all caught without
# enumerating every spelling.
SENSITIVE_KEYS = (
    "token", "secret", "password", "passwd", "mpin", "totp", "sid",
    "consumer_key", "auth", "credential", "api_key",
)

REDACTED = "***redacted***"

# Attributes the stdlib puts on every record. Anything else was passed by the caller as
# `extra=` and is therefore worth carrying into the structured output.
_STANDARD_ATTRS = frozenset((
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename", "module",
    "exc_info", "exc_text", "stack_info", "lineno", "funcName", "created", "msecs",
    "relativeCreated", "thread", "threadName", "processName", "process", "taskName",
    "message", "asctime",
))


# ---------------------------------------------------------------------------
# Correlation id
# ---------------------------------------------------------------------------
def current_correlation_id() -> Optional[str]:
    return _correlation_id.get()


def set_correlation_id(value: Optional[str]) -> contextvars.Token:
    return _correlation_id.set(value)


@contextlib.contextmanager
def correlation(value: Optional[str]) -> Iterator[Optional[str]]:
    """Bind a correlation id for the duration of the block.

    Restores the previous value on exit rather than clearing it, so nesting works and a
    handler cannot leak its id into whatever runs next — the bug that would make the whole
    mechanism worse than useless, because it would attribute one order's failure to
    another order.
    """
    token = _correlation_id.set(value)
    try:
        yield value
    finally:
        _correlation_id.reset(token)


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------
def _is_sensitive(key: str) -> bool:
    lowered = str(key).lower()
    return any(needle in lowered for needle in SENSITIVE_KEYS)


def redact(value: Any, _depth: int = 0) -> Any:
    """Recursively replace sensitive values. Depth-bounded against cyclic structures."""
    if _depth > 6:
        return "***depth-limit***"
    if isinstance(value, dict):
        return {
            key: (REDACTED if _is_sensitive(key) else redact(item, _depth + 1))
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact(item, _depth + 1) for item in value]
    return value


def _jsonable(value: Any) -> Any:
    """Best-effort conversion. Never raises — a log line is not worth an exception."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    try:
        return str(value)
    except Exception:  # noqa: BLE001 - a __str__ that raises must not take the log down
        return "***unrepresentable***"


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------
class JsonFormatter(logging.Formatter):
    """One JSON object per line."""

    def __init__(self, component: str = "") -> None:
        super().__init__()
        self.component = component

    def format(self, record: logging.LogRecord) -> str:
        try:
            payload: Dict[str, Any] = {
                "ts": dt.datetime.fromtimestamp(record.created).isoformat(
                    timespec="milliseconds"),
                "level": record.levelname,
                "component": self.component or record.name,
                "logger": record.name,
                "msg": record.getMessage(),
            }
            cid = current_correlation_id()
            if cid:
                payload["correlation_id"] = cid
            if record.exc_info:
                payload["exception"] = self.formatException(record.exc_info)

            extras = {
                key: value for key, value in record.__dict__.items()
                if key not in _STANDARD_ATTRS and not key.startswith("_")
            }
            if extras:
                payload["context"] = _jsonable(redact(extras))

            return json.dumps(redact(payload), default=str, ensure_ascii=False)
        except Exception as exc:  # noqa: BLE001
            # Degrade rather than raise. A logging failure must never become an outage,
            # and a line that says why it is degraded is still worth more than nothing.
            return json.dumps({
                "level": "ERROR",
                "component": self.component,
                "msg": "log formatting failed",
                "error": str(exc)[:200],
            })


class TextFormatter(logging.Formatter):
    """The human-readable default, with the correlation id appended when present."""

    def __init__(self, component: str = "") -> None:
        super().__init__(fmt=f"%(asctime)s [{component or '%(name)s'}] %(message)s")

    def format(self, record: logging.LogRecord) -> str:
        line = super().format(record)
        cid = current_correlation_id()
        return f"{line}  <{cid}>" if cid else line


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
#: Marks handlers this module installed, so re-configuring replaces only its own.
_OWNED = "_ligerbot_handler"


def configure(component: str, *, stream=None) -> logging.Logger:
    """Install the configured formatter on the root logger.

    Replaces the ``logging.basicConfig`` call each module used to make.

    Idempotent, and deliberately **only removes handlers it installed itself**. The
    obvious implementation clears every root handler, which is wrong twice over: it would
    silently discard a handler an operator attached deliberately, and under pytest it
    would rip out ``caplog``'s handler the first time a module is imported inside a test —
    turning "the log line was not emitted" and "the log line went somewhere else" into the
    same observation.
    """
    root = logging.getLogger()
    formatter = (JsonFormatter(component) if config.LOG_FORMAT == "json"
                 else TextFormatter(component))

    for existing in [h for h in root.handlers if getattr(h, _OWNED, False)]:
        root.removeHandler(existing)

    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(formatter)
    setattr(handler, _OWNED, True)
    root.addHandler(handler)

    level = getattr(logging, config.LOG_LEVEL, logging.INFO)
    if not isinstance(level, int):
        level = logging.INFO
    root.setLevel(level)
    return logging.getLogger(f"ligerbot.{component}") if component else root
