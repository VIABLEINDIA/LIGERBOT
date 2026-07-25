"""Central configuration for LIGERBOT.

All tunables are read from environment variables (loaded from a local ``.env``
file if present). Import ``config`` anywhere and read the typed constants —
never scatter ``os.getenv`` calls across the modules.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # dotenv is optional; env vars can be set another way
    pass


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Instrument:
    """A tradable instrument the bot tracks."""

    token: str
    name: str
    segment: str


def _parse_instruments() -> List[Instrument]:
    """Parse INSTRUMENTS="token:name:segment,token:name:segment"."""
    raw = os.getenv("INSTRUMENTS", "11536:Nifty 50:nse_cm")
    out: List[Instrument] = []
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        parts = entry.split(":")
        if len(parts) < 3:
            continue
        token, name, segment = parts[0], ":".join(parts[1:-1]), parts[-1]
        out.append(Instrument(token=token.strip(), name=name.strip(), segment=segment.strip()))
    return out


# --------------------------------------------------------------------------
# Safety
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# Trading mode — the single source of truth (DESIGN.md Phase 4)
# --------------------------------------------------------------------------
#   "dry_run" — nothing fills; orders are logged and discarded. Smoke-testing only.
#   "paper"   — realistic simulated fills via the backtester's own model, against a REAL
#               broker session for equity and reconciliation. What Phase 4 runs.
#   "live"    — real orders to the broker.
#
# TRADING_MODE is primary and DRY_RUN is derived from it below. They were previously
# independent, which let them disagree: with TRADING_MODE=paper and DRY_RUN left at its
# default, modules branching on DRY_RUN skipped the broker entirely — so paper mode ran on
# a configured equity figure with reconciliation disabled, and said nothing about it. Since
# Phase 4's whole purpose is reconciling paper against backtest, that silently invalidated
# the very sessions it was accumulating.
MODE_DRY_RUN = "dry_run"
MODE_PAPER = "paper"
MODE_LIVE = "live"
VALID_TRADING_MODES = (MODE_DRY_RUN, MODE_PAPER, MODE_LIVE)

_raw_mode = os.getenv("TRADING_MODE", "").strip().lower()
if not _raw_mode:
    # No explicit mode: honour a legacy DRY_RUN=false as "live" so existing
    # configuration keeps working.
    _raw_mode = MODE_DRY_RUN if _bool("DRY_RUN", True) else MODE_LIVE
if _raw_mode not in VALID_TRADING_MODES:
    raise ValueError(
        f"TRADING_MODE={_raw_mode!r} is not one of {VALID_TRADING_MODES}. Failing at "
        f"import rather than falling through to a default — a typo here would otherwise "
        f"decide whether real orders are sent."
    )
TRADING_MODE: str = _raw_mode

# Derived, never configured independently. Kept because it reads naturally at call sites
# that only care whether anything reaches the broker.
DRY_RUN: bool = TRADING_MODE == MODE_DRY_RUN


def needs_broker_session() -> bool:
    """True when a real broker session is required.

    Paper mode needs one: equity must come from the broker (D1) and the position manager
    must reconcile against it. Only *order placement* is simulated in paper — everything
    else is real, which is the entire point of the phase.
    """
    return TRADING_MODE in (MODE_PAPER, MODE_LIVE)


def sends_real_orders() -> bool:
    """True only in live mode. The single predicate that gates real money."""
    return TRADING_MODE == MODE_LIVE


def simulates_fills() -> bool:
    """True in paper mode, where ``src.paper_broker`` fills instead of the exchange."""
    return TRADING_MODE == MODE_PAPER

# Share of a bar's volume a single order may take, in paper and backtest alike.
MAX_VOLUME_PARTICIPATION: float = _float("MAX_VOLUME_PARTICIPATION", 0.10)

# --------------------------------------------------------------------------
# Kotak Neo
# --------------------------------------------------------------------------
# "prod" or "uat". The SDK itself defaults to uat — a genuine sandbox, useful for
# exercising the full login and order path without real money.
KOTAK_ENVIRONMENT: str = os.getenv("KOTAK_ENVIRONMENT", "prod")
KOTAK_CONSUMER_KEY: str = os.getenv("KOTAK_CONSUMER_KEY", "")
# Retained but NOT passed to the SDK: neo_api_client v2.0.0 has no consumer_secret
# parameter (verified by introspection — it survives only in a docstring and commented-out
# code). Kept in case a future version reinstates it. See src/auth.py.
KOTAK_CONSUMER_SECRET: str = os.getenv("KOTAK_CONSUMER_SECRET", "")
# Optional tracking key the SDK accepts as neo_fin_key.
KOTAK_NEO_FIN_KEY: str = os.getenv("KOTAK_NEO_FIN_KEY", "")

# Shared broker session (DESIGN.md 3.8). A TOTP code is single-use within its 30-second
# window, so several modules logging in at once collide and all but one are rejected as
# replays. One module logs in and shares the session; the rest restore it.
# TTL spans a trading day with margin — the key is also dated, so a new day forces a
# fresh login regardless.
KOTAK_SESSION_TTL_SECONDS: int = _int("KOTAK_SESSION_TTL_SECONDS", 10 * 3600)
# Bounds a crash mid-login: the lock expires rather than deadlocking every module.
KOTAK_LOGIN_LOCK_TTL_SECONDS: int = _int("KOTAK_LOGIN_LOCK_TTL_SECONDS", 60)
KOTAK_MOBILE: str = os.getenv("KOTAK_MOBILE", "")
KOTAK_UCC: str = os.getenv("KOTAK_UCC", "")
KOTAK_MPIN: str = os.getenv("KOTAK_MPIN", "")
KOTAK_TOTP_SECRET: str = os.getenv("KOTAK_TOTP_SECRET", "")
KOTAK_TOTP: str = os.getenv("KOTAK_TOTP", "")

# --------------------------------------------------------------------------
# Redis event bus
# --------------------------------------------------------------------------
REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT: int = _int("REDIS_PORT", 6379)
REDIS_DB: int = _int("REDIS_DB", 0)

# Stream names (the "topics" on the event bus)
STREAM_MARKET_TICKS = "market_ticks"
STREAM_MARKET_BARS = "market_bars"
STREAM_TRADE_SIGNALS = "trade_signals"
STREAM_APPROVED_ORDERS = "approved_orders"
STREAM_FILLED_ORDERS = "filled_orders"
STREAM_POSITION_UPDATES = "position_updates"
STREAM_ORDER_EVENTS = "order_events"
STREAM_DEAD_LETTER = "dead_letter"
STREAM_HEARTBEAT = "heartbeat"

# Approximate cap on every stream. Without this, tick volume grows Redis without bound
# until it refuses writes — which stops the market feed mid-session.
STREAM_MAXLEN: int = _int("STREAM_MAXLEN", 100_000)

# Consumer groups (DESIGN.md 3.1). Each module owns one; the name must be stable across
# restarts or the module rejoins as a new group and re-reads the whole backlog.
CONSUMER_GROUP_PREFIX: str = os.getenv("CONSUMER_GROUP_PREFIX", "ligerbot")
# Deliveries before a message is treated as poison and dead-lettered.
MAX_DELIVERIES: int = _int("MAX_DELIVERIES", 5)
# Idle time after which another consumer may reclaim unacked work.
CLAIM_IDLE_MS: int = _int("CLAIM_IDLE_MS", 60_000)

# --------------------------------------------------------------------------
# Operational safety (DESIGN.md 3.5, 3.7)
# --------------------------------------------------------------------------
# Redis key that halts new entries across every module without a restart.
HALT_KEY: str = os.getenv("HALT_KEY", "ligerbot:halt")
# An instrument with no tick for this long during market hours is stale: entries are
# blocked on it, exits still permitted.
FEED_STALE_SECONDS: float = _float("FEED_STALE_SECONDS", 30.0)
HEARTBEAT_INTERVAL_SECONDS: float = _float("HEARTBEAT_INTERVAL_SECONDS", 5.0)
# Reconnect backoff for the market-data WebSocket.
RECONNECT_BASE_SECONDS: float = _float("RECONNECT_BASE_SECONDS", 2.0)
RECONNECT_MAX_SECONDS: float = _float("RECONNECT_MAX_SECONDS", 60.0)
RECONNECT_MAX_ATTEMPTS: int = _int("RECONNECT_MAX_ATTEMPTS", 10)

# Order lifecycle (DESIGN.md 3.3)
# A signal older than this is refused for entries. The phase is judged at the signal's
# own bar time (to match the backtester), so this is what stops an hours-old signal
# being acted on because the window was open when it was generated.
MAX_SIGNAL_AGE_SECONDS: float = _float("MAX_SIGNAL_AGE_SECONDS", 120.0)

ORDER_ACK_TIMEOUT_SECONDS: float = _float("ORDER_ACK_TIMEOUT_SECONDS", 15.0)
ORDER_POLL_INTERVAL_SECONDS: float = _float("ORDER_POLL_INTERVAL_SECONDS", 2.0)
# How long a client order id stays in the dedupe set.
ORDER_DEDUPE_TTL_SECONDS: int = _int("ORDER_DEDUPE_TTL_SECONDS", 86_400)

# Reconciliation (DESIGN.md 3.2)
RECONCILE_INTERVAL_SECONDS: float = _float("RECONCILE_INTERVAL_SECONDS", 60.0)
# Quantity mismatch against the broker that triggers a halt rather than a correction.
RECONCILE_HALT_THRESHOLD: int = _int("RECONCILE_HALT_THRESHOLD", 1)

# --------------------------------------------------------------------------
# InfluxDB
# --------------------------------------------------------------------------
INFLUX_URL: str = os.getenv("INFLUX_URL", "http://localhost:8086")
INFLUX_TOKEN: str = os.getenv("INFLUX_TOKEN", "")
INFLUX_ORG: str = os.getenv("INFLUX_ORG", "ligerbot")
INFLUX_BUCKET: str = os.getenv("INFLUX_BUCKET", "trading_logs")

# --------------------------------------------------------------------------
# Bars (DESIGN.md 1.2, D4)
# --------------------------------------------------------------------------
# Store 1-minute, trade 5-minute. These are two separate decisions: 1-min aggregates up
# to any coarser interval but never the reverse, so the *store* stays fine-grained
# regardless of what the strategy consumes.
BAR_INTERVAL_SECONDS: int = _int("BAR_INTERVAL_SECONDS", 60)
STRATEGY_BAR_SECONDS: int = _int("STRATEGY_BAR_SECONDS", 300)
# "cumulative" matches Kotak (day-running total, which we difference per bar);
# "incremental" for feeds that send per-tick quantity; "none" for feeds without volume.
BAR_VOLUME_MODE: str = os.getenv("BAR_VOLUME_MODE", "cumulative")
BAR_FILL_GAPS: bool = _bool("BAR_FILL_GAPS", True)
BAR_STORE_ROOT: str = os.getenv("BAR_STORE_ROOT", "bar_data")
# Per-session results, for the Phase 4 paper-vs-backtest reconciliation.
SESSION_STORE_ROOT: str = os.getenv("SESSION_STORE_ROOT", "sessions")
# Paper sessions required before the Phase 4 gate can pass. DESIGN.md D5 mitigation 4
# raises this to 40 when the backtest evidence was thin.
PAPER_SESSIONS_REQUIRED: int = _int("PAPER_SESSIONS_REQUIRED", 20)

# --------------------------------------------------------------------------
# Live trading (DESIGN.md Phase 5)
# --------------------------------------------------------------------------
# Deliberate human authorisation. A FILE, not a flag — an environment variable is one
# typo or one copied .env away from committing real capital.
LIVE_AUTH_PATH: str = os.getenv("LIVE_AUTH_PATH", "state/live_authorisation.json")
# Authorisation expires: a decision made weeks ago is not today's decision.
LIVE_AUTH_MAX_AGE_DAYS: int = _int("LIVE_AUTH_MAX_AGE_DAYS", 7)
# Refuse if the account holds far more than what was authorised — authorising for a
# small account and pointing the bot at a large one is how a test becomes a position.
LIVE_AUTH_CAPITAL_TOLERANCE: float = _float("LIVE_AUTH_CAPITAL_TOLERANCE", 1.5)

LIVE_SCALING_STATE_PATH: str = os.getenv(
    "LIVE_SCALING_STATE_PATH", "state/live_scaling.json")
# Consecutive losing sessions before dropping to the smallest size.
LIVE_MAX_LOSING_SESSIONS: int = _int("LIVE_MAX_LOSING_SESSIONS", 3)
# Absolute rupee loss cap for the day, independent of the percentage limit. A percentage
# of a mis-read equity figure is still wrong; an absolute cap bounds the damage either way.
#
# Deliberately UNSET (0 = disabled). There is no sensible default: the right figure is a
# statement about how much money this operator is willing to lose in a day, which no
# library can guess. The live guard refuses to clear live trading until it is set, so the
# effect of leaving it blank is a refusal rather than an inherited stranger's number.
LIVE_MAX_DAILY_LOSS: float = _float("LIVE_MAX_DAILY_LOSS", 0.0)
# Hard cap on orders per session — bounds the blast radius of a signal-generation bug.
LIVE_MAX_ORDERS_PER_DAY: int = _int("LIVE_MAX_ORDERS_PER_DAY", 20)
BAR_PERSIST_INTERVAL_SECONDS: int = _int("BAR_PERSIST_INTERVAL_SECONDS", 30)


def bar_interval_label() -> str:
    """Human/path-friendly label for the store interval, e.g. ``1m``, ``5m``."""
    seconds = BAR_INTERVAL_SECONDS
    if seconds % 3600 == 0:
        return f"{seconds // 3600}h"
    if seconds % 60 == 0:
        return f"{seconds // 60}m"
    return f"{seconds}s"


# --------------------------------------------------------------------------
# Strategy
# --------------------------------------------------------------------------
# Which registered strategy the live engine runs. "sma_crossover" is the negative
# control and loses money by design — never point this at it for real trading.
STRATEGY_NAME: str = os.getenv("STRATEGY_NAME", "trend_pullback")

# Reference SMA-crossover parameters. Retained only as the backtester's negative control
# (DESIGN.md 2.5 rule 5) — a correct cost model must show this losing money.
SMA_SHORT: int = _int("SMA_SHORT", 10)
SMA_LONG: int = _int("SMA_LONG", 50)
WINDOW_SIZE: int = _int("WINDOW_SIZE", max(SMA_LONG, 50))

# --------------------------------------------------------------------------
# Risk management (DESIGN.md D2 — internally consistent by construction)
# --------------------------------------------------------------------------
# Equity is no longer configured: it is read from the broker each session and pinned
# (D1). TOTAL_EQUITY survives only as a simulation/backtest fallback for code paths that
# have no broker to ask — a round illustrative figure, not an account size.
TOTAL_EQUITY: float = _float("TOTAL_EQUITY", 1_000_000.0)

# Floor below which trading is refused. Derived rather than chosen: it is the level at
# which round-trip costs reach ~15% of the amount risked per trade (DESIGN.md 5.2), which
# follows from the broker's fee structure, not from any particular account. Recompute it
# if your brokerage plan differs.
MIN_EQUITY: float = _float("MIN_EQUITY", 200_000.0)
EQUITY_STATE_PATH: str = os.getenv("EQUITY_STATE_PATH", "state/session_equity.json")

MAX_DAILY_DRAWDOWN: float = _float("MAX_DAILY_DRAWDOWN", 0.02)   # 2.0% of session equity
MAX_OPEN_RISK: float = _float("MAX_OPEN_RISK", 0.015)            # 1.5% — the real cap
RISK_PER_TRADE: float = _float("RISK_PER_TRADE", 0.005)          # 0.5% = 1.5% / 3
MAX_OPEN_POSITIONS: int = _int("MAX_OPEN_POSITIONS", 3)          # secondary sanity cap

# Leverage guards, not risk guards. A tight stop implies a large notional for a fixed
# rupee risk; these bound exposure independently. When one binds, the trade is sized
# down — taking less than the full risk budget, which is the safe direction.
MAX_EXPOSURE_PER_TRADE: float = _float("MAX_EXPOSURE_PER_TRADE", 0.75)
MAX_GROSS_EXPOSURE: float = _float("MAX_GROSS_EXPOSURE", 2.0)
MIN_STOP_DISTANCE_PCT: float = _float("MIN_STOP_DISTANCE_PCT", 0.001)

# Concentration limit within a correlated group (sector, or index ETFs). The open-risk
# cap bounds the total but treats three bank stocks as three independent bets when they
# are closer to one bet of triple the size. Set to 0 to disable the filter.
MAX_POSITIONS_PER_GROUP: int = _int("MAX_POSITIONS_PER_GROUP", 1)

ALLOW_SHORT: bool = _bool("ALLOW_SHORT", False)                  # D3: long-only for v1

# --------------------------------------------------------------------------
# Instrument master & universe screen (DESIGN.md D6)
# --------------------------------------------------------------------------
SCRIP_MASTER_DIR: str = os.getenv("SCRIP_MASTER_DIR", "state/scrip_master")
SCREEN_MIN_AVG_DAILY_VALUE: float = _float("SCREEN_MIN_AVG_DAILY_VALUE", 200_00_00_000.0)
SCREEN_MAX_PRICE: float = _float("SCREEN_MAX_PRICE", 5_000.0)
SCREEN_MIN_PRICE: float = _float("SCREEN_MIN_PRICE", 50.0)
SCREEN_REQUIRE_FNO: bool = _bool("SCREEN_REQUIRE_FNO", True)
SCREEN_TARGET_SIZE: int = _int("SCREEN_TARGET_SIZE", 12)

# --------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------
MAX_ORDERS_PER_SECOND: int = _int("MAX_ORDERS_PER_SECOND", 8)
DEFAULT_PRODUCT: str = os.getenv("DEFAULT_PRODUCT", "MIS")
DEFAULT_EXCHANGE_SEGMENT: str = os.getenv("DEFAULT_EXCHANGE_SEGMENT", "nse_cm")

# --------------------------------------------------------------------------
# Instruments
# --------------------------------------------------------------------------
INSTRUMENTS: List[Instrument] = _parse_instruments()


def summary() -> str:
    """Human-readable snapshot of the active config (secrets redacted)."""
    return (
        f"LIGERBOT config | DRY_RUN={DRY_RUN} env={KOTAK_ENVIRONMENT} "
        f"redis={REDIS_HOST}:{REDIS_PORT} bars={bar_interval_label()}/"
        f"{STRATEGY_BAR_SECONDS // 60}m "
        f"risk/trade={RISK_PER_TRADE:.2%} open_risk<={MAX_OPEN_RISK:.2%} "
        f"max_dd={MAX_DAILY_DRAWDOWN:.2%} max_pos={MAX_OPEN_POSITIONS} "
        f"short={ALLOW_SHORT} instruments={len(INSTRUMENTS)}"
    )


def risk_limits():
    """Build a :class:`src.risk_engine.RiskLimits` from the active config.

    Imported lazily so ``config`` stays dependency-free and importable from anywhere.
    Construction validates the D2 consistency rules, so a contradictory ``.env`` fails
    at startup rather than at the first trade.
    """
    from src.risk_engine import RiskLimits

    return RiskLimits(
        max_daily_drawdown=MAX_DAILY_DRAWDOWN,
        max_open_risk=MAX_OPEN_RISK,
        risk_per_trade=RISK_PER_TRADE,
        max_open_positions=MAX_OPEN_POSITIONS,
        max_exposure_per_trade=MAX_EXPOSURE_PER_TRADE,
        max_gross_exposure=MAX_GROSS_EXPOSURE,
        min_stop_distance_pct=MIN_STOP_DISTANCE_PCT,
        max_positions_per_group=MAX_POSITIONS_PER_GROUP,
        # Absolute backstops apply in live mode only. In backtest and paper they would
        # distort results by capping activity for reasons unrelated to the strategy.
        max_daily_loss_absolute=(LIVE_MAX_DAILY_LOSS
                                 if TRADING_MODE == "live" else 0.0),
        max_orders_per_session=(LIVE_MAX_ORDERS_PER_DAY
                                if TRADING_MODE == "live" else 0),
        allow_short=ALLOW_SHORT,
    )
