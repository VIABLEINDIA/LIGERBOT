"""In-process demo of the full LIGERBOT pipeline.

Runs the **real** module classes wired over a single shared in-memory Redis (fakeredis),
so no Docker, no Redis server, no broker and no market hours are needed::

    ticks -> bar builder -> bars -> strategy -> signals -> risk -> approved
          -> execution -> fills -> position manager -> position updates -> risk

Every module uses its production code path: consumer groups with explicit acks, the
order state machine, idempotency, and the real risk engine. The only substitutions are
the Redis server and the broker.

For hardening behaviour specifically — crash recovery, the drawdown breaker, feed
staleness, the kill switch — see ``demo_phase3.py``.
"""
from __future__ import annotations

import datetime as dt
import logging
import threading
import time

import fakeredis

# --- Shared in-memory Redis, patched in before any module builds a client -----
_server = fakeredis.FakeServer()


def _fake_client(*_a, **_k):
    return fakeredis.FakeStrictRedis(server=_server, decode_responses=True)


import config  # noqa: E402
from src import event_bus  # noqa: E402

event_bus.get_client = _fake_client
event_bus.ping = lambda *a, **k: True

from src import feed_health  # noqa: E402
from src import market_calendar as cal  # noqa: E402

_feed_client = _fake_client()

# Replay against the most recent trading day: the session logic is real and will
# correctly refuse to build bars on a weekend or holiday.
_DAY = cal.now_ist().date()
while not cal.is_trading_day(_DAY):
    _DAY -= dt.timedelta(days=1)

# Demo tuning: fast bars and a short strategy interval so signals appear in seconds.
config.BAR_INTERVAL_SECONDS = 60
config.STRATEGY_BAR_SECONDS = 60
config.STRATEGY_NAME = "trend_pullback"
config.TOTAL_EQUITY = 5_000_000.0

from src.bar_builder import BarBuilder  # noqa: E402
from src.bar_store import ParquetBarStore  # noqa: E402
from src.execution_engine import ExecutionEngine  # noqa: E402
from src.position_manager import PositionManager  # noqa: E402
from src.risk_manager import RiskManager  # noqa: E402
from src.storage_logger import StorageLogger  # noqa: E402
from src.strategy_engine import StrategyEngine  # noqa: E402
from src.strategies.trend_pullback import TrendPullback  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
for quiet in ("ligerbot.storage", "ligerbot.event_bus", "ligerbot.bar_builder",
              "ligerbot.bars", "ligerbot.account"):
    logging.getLogger(quiet).setLevel(logging.WARNING)

RULE = "=" * 74


def _thread(target, name):
    t = threading.Thread(target=target, name=name, daemon=True)
    t.start()
    return t


def main(run_seconds: float = 12.0) -> None:
    import tempfile
    from pathlib import Path

    workdir = Path(tempfile.mkdtemp(prefix="ligerbot_sim_"))
    print(RULE)
    print("LIGERBOT — in-process pipeline demo (no Docker required)")
    print(config.summary())
    print(f"Replaying session {_DAY} ({_DAY.strftime('%A')})")
    print(RULE)

    # Consumers first, so they are subscribed before anything is published.
    _thread(StorageLogger().run, "storage")
    _thread(PositionManager(None).run, "positions")
    _thread(ExecutionEngine(None).run, "execution")
    _thread(RiskManager(None).run, "risk")
    _thread(StrategyEngine(TrendPullback(
        ema_fast=5, ema_slow=10, adx_period=5, adx_min=10.0, atr_period=5,
        atr_mult=3.0, min_stop_pct=0.002, rvol_min=0.0, rsi_max=100.0,
    )).run, "strategy")

    builder = BarBuilder(store=ParquetBarStore(workdir / "bars", "1m"))
    builder.open_session(_DAY)
    time.sleep(0.5)

    # Feed a compressed trending session so the strategy has something to react to.
    session_open = cal.at(_DAY, cal.SESSION_OPEN)
    price = 1_300.0
    instrument = "nse_cm:2885"
    cumulative_volume = 0.0
    deadline = time.time() + run_seconds

    for step in range(0, 300 * 60, 20):
        if time.time() > deadline:
            break
        moment = session_open + dt.timedelta(seconds=step)
        if moment.time() >= cal.SESSION_CLOSE:
            break
        # A trend with pullbacks — the structure the strategy looks for.
        minute = step // 60
        price += (1.4 if (minute % 25) < 18 else -2.2)
        price = round(max(1.0, price), 2)
        cumulative_volume += 900
        builder.feed({
            "instrument_id": instrument,
            "ltp": price,
            "timestamp": moment.timestamp(),
            "volume": cumulative_volume,
        }, now=moment)
        # This demo feeds the bar builder directly, bypassing ingestion — so it must
        # publish the liveness key ingestion normally would. The feed gate fails closed,
        # so without this the risk manager correctly refuses every entry as stale.
        feed_health.publish_liveness(_feed_client, instrument, ttl_seconds=3600)
        time.sleep(0.001)

    builder.shutdown()
    time.sleep(1.5)

    client = _fake_client()
    print(f"\n{RULE}\nStream totals\n{RULE}")
    for stream in (config.STREAM_MARKET_TICKS, config.STREAM_MARKET_BARS,
                   config.STREAM_TRADE_SIGNALS, config.STREAM_APPROVED_ORDERS,
                   config.STREAM_FILLED_ORDERS, config.STREAM_POSITION_UPDATES,
                   config.STREAM_DEAD_LETTER):
        print(f"  {stream:<18} : {client.xlen(stream):>6} events")
    print(RULE)
    print("Every module ran its production path: consumer groups with acks, the order")
    print("state machine, idempotency and the real risk engine. Only Redis and the")
    print("broker were substituted.")
    print(RULE)


if __name__ == "__main__":
    main()
