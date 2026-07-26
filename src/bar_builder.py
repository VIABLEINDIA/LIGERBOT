"""Module 2b — Bar Builder.

Sits between ingestion and strategy, converting the raw tick firehose into the time bars
everything downstream actually reasons about::

    market_ticks --> [ Bar Builder ] --> market_bars --> [ Strategy Engine ]
                            |
                            +--> Parquet store (the historical dataset)

Why a separate process rather than folding this into the strategy engine: multiple
strategies and the backtester all need *identical* bars, and the archived stream is what
becomes our historical dataset (DESIGN.md D5 mitigation 2). Building bars per-strategy
would give each one a subtly different view of the same market.

Bars are emitted only when **closed**. A forming bar is never published — that is the most
common accidental source of look-ahead, because a strategy that sees a partial bar is
seeing the future relative to where a backtest would place it.

    python -m src.bar_builder
    python -m src.bar_builder --replay-date 2026-07-23
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import signal
import time
from typing import Optional

import config
from src import event_bus
from src import health as health_mod
from src import logging_setup
from src import market_calendar as cal
from src.bar_store import ParquetBarStore
from src.bars import Bar, MultiInstrumentAggregator, VolumeMode

logging_setup.configure("bars")
log = logging.getLogger("ligerbot.bar_builder")

_running = True


class BarBuilder:
    def __init__(self, store: Optional[ParquetBarStore] = None) -> None:
        self.client = event_bus.get_client()
        self.last_id = "$"
        self.interval = config.BAR_INTERVAL_SECONDS
        self.volume_mode = VolumeMode(config.BAR_VOLUME_MODE)
        self.store = store if store is not None else ParquetBarStore(
            config.BAR_STORE_ROOT, config.bar_interval_label()
        )
        self.aggregator: Optional[MultiInstrumentAggregator] = None
        self.session_day: Optional[dt.date] = None
        self._last_persist = time.monotonic()
        self._bars_built = 0

    # -- session -----------------------------------------------------------
    def _ensure_session(self, day: dt.date) -> bool:
        """Roll the aggregator over to ``day``. False if it isn't a trading day."""
        if self.session_day == day and self.aggregator is not None:
            return True
        if not cal.is_trading_day(day):
            return False

        # Flush the outgoing session before abandoning its aggregator, or the final
        # bars of the day are lost. Bounded by the last tick we actually saw, never
        # padded to the session close: a synthetic bar must mean "market open, nothing
        # traded", never "the bot wasn't running". Padding would overstate coverage and
        # mislead every backtest built on this data.
        if self.aggregator is not None:
            self._flush_to_last_tick()
            self.store.flush()

        self.aggregator = MultiInstrumentAggregator(
            self.interval, day, volume_mode=self.volume_mode, fill_gaps=config.BAR_FILL_GAPS
        )
        self.session_day = day
        if not cal.covers_year(day.year):
            log.warning("No verified NSE holiday list for %d — session boundaries for %s "
                        "may be wrong. Set NSE_HOLIDAYS_FILE.", day.year, day)
        log.info("Bar session opened for %s (%s bars, volume=%s)",
                 day, config.bar_interval_label(), self.volume_mode.value)
        return True

    # -- output ------------------------------------------------------------
    def _handle_bars(self, bars: list[Bar]) -> None:
        if not bars:
            return
        for bar in bars:
            event_bus.publish(self.client, config.STREAM_MARKET_BARS, bar.to_event())
        self.store.buffer(bars)
        self._bars_built += len(bars)

        real = [b for b in bars if not b.synthetic]
        if real:
            latest = real[-1]
            log.info("bar %s %s O%.2f H%.2f L%.2f C%.2f V%.0f (%d ticks)%s",
                     latest.instrument_id, latest.bar_start.strftime("%H:%M"),
                     latest.open, latest.high, latest.low, latest.close,
                     latest.volume, latest.tick_count,
                     f" +{len(bars) - len(real)} synthetic" if len(bars) > len(real) else "")

    def _persist_if_due(self, *, force: bool = False) -> None:
        elapsed = time.monotonic() - self._last_persist
        if force or elapsed >= config.BAR_PERSIST_INTERVAL_SECONDS:
            written = self.store.flush()
            self._last_persist = time.monotonic()
            if written:
                log.debug("Persisted %d bars to %s", written, self.store.root)

    # -- tick handling -----------------------------------------------------
    def _instrument_id(self, tick: dict) -> Optional[str]:
        """Prefer the canonical id; fall back to the legacy display name.

        The fallback is transitional. Once ingestion emits ``instrument_id`` everywhere
        (fixing B4 end to end), the legacy branch can go.
        """
        value = tick.get("instrument_id") or tick.get("instrument")
        return str(value) if value else None

    def _handle_tick(self, tick: dict) -> None:
        if self.aggregator is None:
            return
        instrument_id = self._instrument_id(tick)
        if not instrument_id:
            return
        try:
            price = float(tick["ltp"])
        except (KeyError, TypeError, ValueError):
            return

        raw_ts = tick.get("timestamp")
        try:
            timestamp = dt.datetime.fromtimestamp(float(raw_ts), tz=cal.IST)
        except (TypeError, ValueError):
            timestamp = cal.now_ist()

        volume = None
        for key in ("volume", "cum_volume", "v"):
            if tick.get(key) is not None:
                try:
                    volume = float(tick[key])
                    break
                except (TypeError, ValueError):
                    pass

        self._handle_bars(self.aggregator.add_tick(instrument_id, timestamp, price, volume))

    def feed(self, tick: dict, now: Optional[dt.datetime] = None) -> None:
        """Process one tick and close any elapsed bars.

        The public entry point used by replay and tests, so a deterministic driver can
        exercise exactly the code the live loop runs rather than a parallel imitation.
        """
        if self.aggregator is None:
            raise RuntimeError("call _ensure_session() before feeding ticks")
        self._handle_tick(tick)
        if now is not None:
            self._handle_bars(self.aggregator.flush_until(now))

    def open_session(self, day: dt.date) -> bool:
        """Public wrapper around session rollover, for replay drivers."""
        return self._ensure_session(day)

    # -- main loop ---------------------------------------------------------
    def run(self) -> None:
        if not event_bus.ping(self.client):
            log.error("Redis not reachable — start it with `docker compose up -d`.")
            return
        log.info("Bar Builder online: %s bars -> stream %r + %s",
                 config.bar_interval_label(), config.STREAM_MARKET_BARS, self.store.root)

        health, _server = health_mod.serve("bars")

        while _running:
            now = cal.now_ist()
            if not self._ensure_session(now.date()):
                # Non-trading day: idle cheaply rather than spinning on the stream.
                # The heartbeat still ticks — idle on a holiday is correct behaviour,
                # and reporting it as wedged would page someone every Sunday.
                health.set(session="closed")
                health.loop_tick()
                time.sleep(5)
                continue

            entries, self.last_id = event_bus.read_new(
                self.client, config.STREAM_MARKET_TICKS, self.last_id,
                count=1000, block_ms=1000,
            )
            for _entry_id, tick in entries:
                self._handle_tick(tick)
            if entries:
                health.event_processed(len(entries))

            # Time-driven closure. Without this an instrument that stops trading also
            # stops producing bars, and downstream cannot distinguish quiet from dead.
            if self.aggregator is not None:
                self._handle_bars(self.aggregator.flush_until(cal.now_ist()))
            self._persist_if_due()

            health.set(session=str(self.session_day), bars_built=self._bars_built)
            health.loop_tick()

        self.shutdown()

    def _flush_to_last_tick(self) -> None:
        """Close bars up to the last tick observed, and no further."""
        if self.aggregator is None:
            return
        boundary = self.aggregator.last_tick_time
        if boundary is not None:
            self._handle_bars(self.aggregator.flush_until(boundary))

    def shutdown(self) -> None:
        """Close out cleanly — unpersisted bars are data we can never recover."""
        self._flush_to_last_tick()
        self._persist_if_due(force=True)
        log.info("Bar Builder stopped. %d bars built this run.", self._bars_built)


def _handle_signal(signum, _frame) -> None:
    global _running
    log.info("Received signal %s — shutting down.", signum)
    _running = False


def main() -> None:
    parser = argparse.ArgumentParser(description="LIGERBOT bar builder")
    parser.add_argument("--interval", type=int, default=None,
                        help="bar interval in seconds (default: config.BAR_INTERVAL_SECONDS)")
    args = parser.parse_args()

    if args.interval:
        config.BAR_INTERVAL_SECONDS = args.interval

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    builder = BarBuilder()
    try:
        builder.run()
    except KeyboardInterrupt:
        builder.shutdown()


if __name__ == "__main__":
    main()
