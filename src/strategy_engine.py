"""Module 3 — Strategy Engine (the brain).

Consumes ``market_bars`` and emits ``trade_signals``. Rewritten in Phase 3 to run the
registered :class:`src.strategy_base.Strategy` implementations rather than carrying its own
inline logic.

The original computed SMAs over raw **ticks** (defect B1), so "50 periods" might be two
seconds or two hours depending on how busy the tape was — a strategy whose horizon changes
with liquidity has no defined behaviour. It also emitted signals with no stop-loss (B7),
which meant the risk manager silently fell back to notional sizing and the documented
per-trade risk cap never applied.

Both are gone. This module now:

* consumes closed bars only, so a strategy cannot see a forming bar (no look-ahead);
* runs the *same* strategy objects the backtester runs, so results transfer;
* resamples to the configured trading interval (D4: store 1-minute, trade 5-minute);
* emits the full signal schema including the mandatory stop.

    python -m src.strategy_engine
    python -m src.strategy_engine --strategy trend_pullback
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
from typing import Any, Dict, List, Optional  # noqa: F401  (List used by flush())

import config
from src import event_bus
from src import health as health_mod
from src import logging_setup
from src import market_calendar as cal
from src.bars import Bar
from src.risk_engine import Position
from src.strategy_base import Strategy, StrategyContext, create

logging_setup.configure("strategy")
log = logging.getLogger("ligerbot.strategy")


class BarResampler:
    """Aggregates incoming 1-minute bars into the strategy's trading interval.

    Lives here rather than in the bar builder so every strategy can choose its own
    interval from one shared stream, and so the live path uses the same
    session-anchored bucketing the backtester's resampler does.
    """

    def __init__(self, interval_seconds: int) -> None:
        self.interval = interval_seconds
        self._buckets: Dict[str, List[Bar]] = {}
        self._bucket_start: Dict[str, dt.datetime] = {}

    def _bucket_start_of(self, instrument: str) -> Optional[dt.datetime]:
        return self._bucket_start.get(instrument)

    def _bucket_for(self, bar: Bar) -> Optional[dt.datetime]:
        window = cal.session_window(bar.bar_start.date())
        if window is None:
            return None
        session_open = window[0]
        elapsed = (bar.bar_start - session_open).total_seconds()
        if elapsed < 0:
            return None
        return session_open + dt.timedelta(
            seconds=int(elapsed // self.interval) * self.interval)

    def add(self, bar: Bar) -> Optional[Bar]:
        """Returns a completed coarse bar when one closes, else None."""
        if self.interval <= 60:
            return bar

        start = self._bucket_for(bar)
        if start is None:
            return None

        instrument = bar.instrument_id
        current = self._bucket_start.get(instrument)
        completed: Optional[Bar] = None

        if current is not None and start > current:
            completed = self._merge(instrument, current)

        if self._bucket_start.get(instrument) != start:
            self._bucket_start[instrument] = start
            self._buckets[instrument] = []
        self._buckets[instrument].append(bar)
        return completed

    def held_session_day(self) -> Optional[dt.date]:
        """Session the currently-held buckets belong to, if any.

        Lets the caller detect a session rollover *before* feeding the new day's bar,
        which matters because a bucket must be closed out inside its own session.
        """
        for start in self._bucket_start.values():
            if start is not None:
                return start.date()
        return None

    def flush(self) -> List[Bar]:
        """Close out every held bucket and return the resulting bars.

        Needed because :meth:`add` only completes a bucket when the *next* one opens —
        so the final bucket of a session has nothing to close it. Without this the
        strategy never sees the last bar of the day live while the backtester does, and
        the stale bucket would instead surface at the next session's open.
        """
        out: List[Bar] = []
        for instrument, start in list(self._bucket_start.items()):
            if start is None:
                continue
            merged = self._merge(instrument, start)
            if merged is not None:
                out.append(merged)
            self._bucket_start[instrument] = None
            self._buckets[instrument] = []
        out.sort(key=lambda b: (b.bar_start, b.instrument_id))
        return out

    def _merge(self, instrument: str, start: dt.datetime) -> Optional[Bar]:
        parts = self._buckets.get(instrument) or []
        if not parts:
            return None
        volume = sum(b.volume for b in parts)
        vwap = (sum(b.vwap * b.volume for b in parts) / volume) if volume > 0 else parts[-1].close
        return Bar(
            instrument_id=instrument,
            bar_start=start,
            bar_end=start + dt.timedelta(seconds=self.interval),
            open=parts[0].open,
            high=max(b.high for b in parts),
            low=min(b.low for b in parts),
            close=parts[-1].close,
            volume=volume,
            vwap=round(vwap, 4),
            tick_count=sum(b.tick_count for b in parts),
            # Only a wholly-synthetic bucket is synthetic: one real trade in the window
            # means something genuinely happened in it.
            synthetic=all(b.synthetic for b in parts),
        )


class StrategyEngine:
    def __init__(self, strategy: Optional[Strategy] = None) -> None:
        self.client = event_bus.get_client()
        if strategy is None:
            # Importing the package is what registers the implementations. Done here
            # rather than at module import so a caller that supplies its own strategy
            # never pays for it, and so the registry is populated however we are entered.
            import src.strategies  # noqa: F401
            strategy = create(config.STRATEGY_NAME)
        self.strategy = strategy
        self.resampler = BarResampler(config.STRATEGY_BAR_SECONDS)
        self.session_day: Optional[dt.date] = None
        # Mirrors what the risk manager holds open, so the strategy's context is
        # accurate. Sourced from position_updates rather than assumed.
        self.positions: Dict[str, Position] = {}

        group = f"{config.CONSUMER_GROUP_PREFIX}.strategy"
        self.bars = event_bus.StreamConsumer(
            self.client, config.STREAM_MARKET_BARS, group, "strategy-1",
            max_deliveries=config.MAX_DELIVERIES)
        self.position_updates = event_bus.StreamConsumer(
            self.client, config.STREAM_POSITION_UPDATES, group, "strategy-1",
            max_deliveries=config.MAX_DELIVERIES)

    # -- position mirror ---------------------------------------------------
    def _handle_position_update(self, update: Dict[str, Any]) -> None:
        rows = update.get("positions") or []
        if isinstance(rows, str):
            return
        self.positions = {
            row["instrument_id"]: Position(
                instrument_id=row["instrument_id"],
                quantity=int(row["quantity"]),
                entry_price=float(row["average_price"]),
                stop_loss=float(row.get("stop_loss") or 0.0),
            )
            for row in rows if isinstance(row, dict) and row.get("instrument_id")
        }

    # -- bars --------------------------------------------------------------
    def _handle_bar(self, fields: Dict[str, Any]) -> None:
        bar = Bar.from_event(fields)
        day = bar.bar_start.date()

        # Close out the previous session's final bucket BEFORE rolling the session.
        # `add()` only completes a bucket when the next one opens, so without this the
        # last bucket of the day would surface after `on_session_start` had already
        # reset the session-anchored indicators — anchoring the new day's VWAP on
        # yesterday's close, and handing the strategy a bar stamped a session earlier.
        held = self.resampler.held_session_day()
        if held is not None and held != day:
            for leftover in self.resampler.flush():
                self._dispatch(leftover, leftover.bar_start.date())

        if day != self.session_day:
            self.session_day = day
            self.strategy.on_session_start(day)
            log.info("Session %s — %s", day, self.strategy.describe())

        coarse = self.resampler.add(bar)
        if coarse is None:
            return
        self._dispatch(coarse, day)

    def _dispatch(self, coarse: Bar, day: dt.date) -> None:
        """Show one closed coarse bar to the strategy and publish any signals."""
        phase = cal.phase(coarse.bar_end)
        context = StrategyContext(
            now=coarse.bar_end,
            position=self.positions.get(coarse.instrument_id),
            seconds_to_square_off=cal.seconds_to_square_off(coarse.bar_end) or 0.0,
            allows_entry=phase.allows_entry,
            session_day=day,
        )

        for signal in self.strategy.on_bar(coarse, context):
            event_bus.publish(self.client, config.STREAM_TRADE_SIGNALS, {
                "instrument_id": signal.instrument_id,
                "intent": signal.intent.value,
                "ref_price": signal.ref_price,
                "bar_time": signal.bar_time.isoformat(),
                "stop_loss": signal.stop_loss or 0.0,
                "take_profit": signal.take_profit or 0.0,
                "strategy_name": signal.strategy_name,
                "strategy_version": signal.strategy_version,
                "params_hash": signal.params_hash,
                "reason": signal.reason,
            })
            log.info("SIGNAL %s %s @ %.2f stop %.2f — %s",
                     signal.intent.value, signal.instrument_id, signal.ref_price,
                     signal.stop_loss or 0.0, signal.reason)

    def run(self) -> None:
        if not event_bus.ping(self.client):
            log.error("Redis not reachable — start it with `docker compose up -d`.")
            return
        # Refuse rather than run uselessly. A strategy whose warmup exceeds a session
        # cannot trade at this interval, and would otherwise log normally, pass every
        # health check, and take no trades all day.
        feasible, note = self.strategy.check_interval(config.STRATEGY_BAR_SECONDS)
        if not feasible:
            log.error("STRATEGY CANNOT TRADE AT THIS BAR INTERVAL — refusing to start.")
            log.error("  %s", note)
            return
        if note:
            log.warning("%s", note)

        log.info("Strategy Engine online: %s on %ds bars",
                 self.strategy.describe(), config.STRATEGY_BAR_SECONDS)

        health, _server = health_mod.serve("strategy")

        while True:
            for entry_id, fields in self.position_updates.read(count=20, block_ms=10):
                self.position_updates.handle(entry_id, fields, self._handle_position_update)

            for entry_id, fields in self.bars.claim_stale(min_idle_ms=config.CLAIM_IDLE_MS):
                self.bars.handle(entry_id, fields, self._handle_bar)
                health.event_processed()

            for entry_id, fields in self.bars.read(count=200, block_ms=1000):
                self.bars.handle(entry_id, fields, self._handle_bar)
                health.event_processed()

            health.set_backlog(self.bars.check_backlog())
            health.set(session_day=str(self.session_day),
                       strategy=self.strategy.describe(),
                       mirrored_positions=len(self.positions))
            health.loop_tick()


def main() -> None:
    parser = argparse.ArgumentParser(description="LIGERBOT strategy engine")
    parser.add_argument("--strategy", default=None, help="registered strategy name")
    args = parser.parse_args()

    import src.strategies  # noqa: F401  (registers the implementations)

    name = args.strategy or config.STRATEGY_NAME
    StrategyEngine(create(name)).run()


if __name__ == "__main__":
    main()
