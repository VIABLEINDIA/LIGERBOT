"""Turn the scrip master into a tradable universe (DESIGN.md D6).

The hole this fills: D6 specified "a screen, not a list", and both the screening logic and
the criteria were built — but nothing fetched the two inputs the screen filters on. The
scrip master carries **no price and no volume**, so `apply_screen` correctly excluded
everything and returned an empty universe. Even with the broker probe run and the master
downloaded, there would have been nothing to subscribe to and nothing to backfill.

Two inputs, two very different difficulties:

``last_price``
    Straightforward: ``quotes()`` returns it live for a batch of tokens.

``avg_daily_value``
    The hard one, and the reason this module has a fallback chain. Average daily traded
    value needs *history* — precisely what the probe is still measuring. Three sources are
    tried in order of trustworthiness, and **which one was used is recorded on the
    universe**, because a screen run on a one-day proxy is not the same claim as one run
    on sixty days of history and should not be reported as though it were.

The screen deliberately fails **closed**: an instrument whose liquidity cannot be
established is excluded rather than assumed adequate. Trading an illiquid name because its
data was missing is a worse outcome than trading a shorter list.
"""
from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

import config
from src import kotak_api
from src import market_calendar as cal
from src.instruments import (
    Instrument, InstrumentMaster, ScreenCriteria, Universe, apply_screen,
    assign_correlation_groups,
)

log = logging.getLogger("ligerbot.universe")

# Field names in a Kotak quote payload. Best-effort, like every other Kotak mapping that
# has not yet met a live response — but unlike the equity mapping, a wrong guess here
# excludes an instrument rather than mis-sizing a trade, so it fails safe.
LTP_FIELDS = ("ltp", "last_traded_price", "lp", "lastPrice", "c")
VOLUME_FIELDS = ("v", "volume", "cum_volume", "vol", "totalTradedVolume")
TOKEN_FIELDS = ("tk", "instrument_token", "token", "pSymbol")


class LiquiditySource(Enum):
    """Where average daily value came from, in descending order of trust."""

    HISTORY = "history"          # real daily bars — the only fully trustworthy source
    RECORDED_BARS = "recorded"   # our own Parquet store, as it accumulates
    QUOTE_PROXY = "quote_proxy"  # today's traded value so far — a proxy, not an average
    UNAVAILABLE = "unavailable"

    @property
    def trustworthy(self) -> bool:
        return self in (LiquiditySource.HISTORY, LiquiditySource.RECORDED_BARS)


@dataclass
class LiquidityReading:
    instrument_id: str
    last_price: Optional[float] = None
    avg_daily_value: Optional[float] = None
    source: LiquiditySource = LiquiditySource.UNAVAILABLE
    days_observed: int = 0

    @property
    def usable(self) -> bool:
        return (self.last_price is not None and self.last_price > 0
                and self.avg_daily_value is not None and self.avg_daily_value > 0)


def _pick(payload: Dict[str, Any], keys: Sequence[str]) -> Optional[float]:
    for key in keys:
        raw = payload.get(key)
        if raw in (None, ""):
            continue
        try:
            return float(str(raw).replace(",", ""))
        except (TypeError, ValueError):
            continue
    return None


# ---------------------------------------------------------------------------
def fetch_quotes(
    neo_client: Any, instruments: Sequence[Instrument], *, batch_size: int = 50
) -> Dict[str, Dict[str, Any]]:
    """Live quotes keyed by instrument id.

    Batched because a universe screen runs across hundreds of scrips and a single call
    with all of them is the kind of request brokers rate-limit or truncate.
    """
    by_token = {i.token: i for i in instruments}
    out: Dict[str, Dict[str, Any]] = {}

    tokens = [{"instrument_token": i.token, "exchange_segment": i.exchange_segment}
              for i in instruments]
    for start in range(0, len(tokens), batch_size):
        chunk = tokens[start:start + batch_size]
        try:
            response = kotak_api.safe_call(
                neo_client, "quotes", instrument_tokens=chunk, quote_type="ltp",
                allow_empty=True)
        except kotak_api.KotakAPIError as exc:
            log.warning("quotes() failed for a batch of %d (%s) — those instruments "
                        "will be excluded rather than assumed liquid.", len(chunk), exc)
            continue

        for row in kotak_api.rows_from(response, call="quotes()"):
            token = None
            for field in TOKEN_FIELDS:
                if row.get(field):
                    token = str(row[field])
                    break
            instrument = by_token.get(token)
            if instrument is not None:
                out[instrument.instrument_id] = row
    return out


def liquidity_from_recorded_bars(
    instrument_id: str, *, lookback_days: int = 20, store=None
) -> Optional[LiquidityReading]:
    """Average daily traded value from our own Parquet store.

    Becomes the best available source as self-recording accumulates (D5 mitigation 2),
    and needs no broker call at all.
    """
    from src.bar_store import ParquetBarStore

    store = store or ParquetBarStore(config.BAR_STORE_ROOT, config.bar_interval_label())
    days = store.available_days(instrument_id)
    if not days:
        return None

    recent = days[-lookback_days:]
    totals: List[float] = []
    last_close: Optional[float] = None
    for day in recent:
        frame = store.read_day(instrument_id, day)
        if frame.empty:
            continue
        real = frame[~frame["synthetic"].astype(bool)]
        if real.empty:
            continue
        totals.append(float((real["close"] * real["volume"]).sum()))
        last_close = float(real["close"].iloc[-1])

    if not totals or last_close is None:
        return None
    return LiquidityReading(
        instrument_id=instrument_id,
        last_price=last_close,
        avg_daily_value=sum(totals) / len(totals),
        source=LiquiditySource.RECORDED_BARS,
        days_observed=len(totals),
    )


def liquidity_from_quote(instrument_id: str, quote: Dict[str, Any]) -> LiquidityReading:
    """Today's traded value so far, as a **proxy** for average daily value.

    Explicitly the weakest source. Early in a session it understates liquidity badly, and
    it reflects one day rather than an average — so a universe built from it is a
    provisional universe, and the recorded source replaces it as bars accumulate.
    """
    price = _pick(quote, LTP_FIELDS)
    volume = _pick(quote, VOLUME_FIELDS)
    if price is None or price <= 0:
        return LiquidityReading(instrument_id, source=LiquiditySource.UNAVAILABLE)
    if volume is None or volume <= 0:
        return LiquidityReading(instrument_id, last_price=price,
                                source=LiquiditySource.UNAVAILABLE)
    return LiquidityReading(
        instrument_id=instrument_id,
        last_price=price,
        avg_daily_value=price * volume,
        source=LiquiditySource.QUOTE_PROXY,
        days_observed=1,
    )


def measure_liquidity(
    instruments: Sequence[Instrument],
    *,
    neo_client: Any = None,
    lookback_days: int = 20,
    store=None,
) -> Dict[str, LiquidityReading]:
    """Best available liquidity reading per instrument, preferring real history."""
    readings: Dict[str, LiquidityReading] = {}

    # 1. Our own recorded bars — no broker call, and the most trustworthy source we can
    #    obtain without knowing what the history API offers.
    for instrument in instruments:
        reading = liquidity_from_recorded_bars(
            instrument.instrument_id, lookback_days=lookback_days, store=store)
        if reading is not None and reading.usable:
            readings[instrument.instrument_id] = reading

    # 2. Live quotes for whatever is left.
    missing = [i for i in instruments if i.instrument_id not in readings]
    if missing and neo_client is not None:
        quotes = fetch_quotes(neo_client, missing)
        for instrument in missing:
            quote = quotes.get(instrument.instrument_id)
            if quote:
                readings[instrument.instrument_id] = liquidity_from_quote(
                    instrument.instrument_id, quote)

    for instrument in instruments:
        readings.setdefault(
            instrument.instrument_id,
            LiquidityReading(instrument.instrument_id,
                             source=LiquiditySource.UNAVAILABLE))
    return readings


def enrich(
    instruments: Sequence[Instrument], readings: Dict[str, LiquidityReading]
) -> List[Instrument]:
    """Attach price and liquidity so the screen has something to filter on."""
    from dataclasses import replace

    enriched = []
    for instrument in instruments:
        reading = readings.get(instrument.instrument_id)
        if reading is None:
            enriched.append(instrument)
            continue
        enriched.append(replace(
            instrument,
            last_price=reading.last_price,
            avg_daily_value=reading.avg_daily_value,
        ))
    return enriched


def build_universe(
    master: InstrumentMaster,
    *,
    neo_client: Any = None,
    criteria: Optional[ScreenCriteria] = None,
    candidates: Optional[Sequence[str]] = None,
    lookback_days: int = 20,
    store=None,
    day: Optional[dt.date] = None,
) -> tuple[Universe, Dict[str, Any]]:
    """Screen the instrument master into a tradable universe.

    Returns ``(universe, provenance)``. The provenance matters as much as the universe:
    a screen run on a one-day quote proxy is a weaker claim than one run on sixty days of
    recorded bars, and a backtest should record which it was rather than presenting both
    as "the universe".
    """
    criteria = criteria or ScreenCriteria(
        min_avg_daily_value=config.SCREEN_MIN_AVG_DAILY_VALUE,
        max_price=config.SCREEN_MAX_PRICE,
        min_price=config.SCREEN_MIN_PRICE,
        require_fno=config.SCREEN_REQUIRE_FNO,
        target_size=config.SCREEN_TARGET_SIZE,
    )

    pool: List[Instrument] = []
    for instrument_id in (candidates or []):
        found = master.resolve(instrument_id)
        if found is not None:
            pool.append(found)
    if not pool:
        pool = [master.require(i) for i in master._by_id]  # noqa: SLF001

    tradable = [i for i in pool if i.is_tradable_cash]
    log.info("Screening %d tradable instrument(s) from a master of %d.",
             len(tradable), len(master))

    readings = measure_liquidity(tradable, neo_client=neo_client,
                                 lookback_days=lookback_days, store=store)
    enriched = assign_correlation_groups(enrich(tradable, readings))
    universe = apply_screen(enriched, criteria, screened_on=day or dt.date.today())

    counts: Dict[str, int] = {}
    for reading in readings.values():
        counts[reading.source.value] = counts.get(reading.source.value, 0) + 1

    selected_sources = {
        readings[i.instrument_id].source
        for i in universe.instruments if i.instrument_id in readings
    }
    weakest = (LiquiditySource.QUOTE_PROXY
               if LiquiditySource.QUOTE_PROXY in selected_sources
               else (LiquiditySource.RECORDED_BARS if selected_sources
                     else LiquiditySource.UNAVAILABLE))

    provenance = {
        "screened_on": (day or dt.date.today()).isoformat(),
        "candidates": len(tradable),
        "selected": len(universe),
        "criteria": criteria.describe(),
        "liquidity_sources": counts,
        "weakest_source_used": weakest.value,
        "trustworthy": weakest.trustworthy,
    }

    if not universe.instruments:
        log.error("The screen selected nothing. Liquidity sources: %s. Without price and "
                  "traded-value data every candidate is excluded — the screen fails "
                  "closed rather than assuming liquidity it cannot see.", counts)
    elif not weakest.trustworthy:
        log.warning("Universe %s rests on a one-day quote proxy, not historical average "
                    "traded value. Treat it as provisional: it will firm up as the "
                    "Parquet store accumulates (D5 mitigation 2).", universe.version)

    return universe, provenance
