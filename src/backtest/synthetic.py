"""Synthetic bar generation, for testing the harness itself.

Used for the negative control (DESIGN.md 2.5 rule 5). The generator produces a driftless
**geometric random walk** — by construction there is no exploitable pattern in it. Any
strategy that appears profitable here at scale is revealing a bug in the harness, not an
edge in the market.

This is not a substitute for historical data and cannot validate a strategy. It validates
the *backtester*: that costs bite, that fills are pessimistic, that look-ahead is
impossible. Real data arrives via the Kotak probe and the self-recorded Parquet store.
"""
from __future__ import annotations

import datetime as dt
import math
import random
from typing import Dict, List, Optional, Sequence

import pandas as pd

from src import market_calendar as cal


def generate_session_bars(
    instrument_id: str,
    day: dt.date,
    *,
    open_price: float,
    interval_seconds: int = 60,
    annual_volatility: float = 0.28,
    drift: float = 0.0,
    momentum: float = 0.0,
    rng: Optional[random.Random] = None,
    base_volume: float = 8_000.0,
) -> pd.DataFrame:
    """One session of bars following a geometric random walk.

    ``drift`` is an annualised rate and defaults to zero.

    ``momentum`` is the AR(1) coefficient on returns and is what makes a **positive
    control** possible. At zero, returns are independent and no strategy can beat costs.
    Above zero, returns are serially correlated — trends genuinely persist — so a
    trend-following strategy *should* extract something. That is not a planted answer: it
    is a real statistical property that the strategy either detects or does not, which is
    exactly what makes it a fair test of the strategy rather than of the generator.
    Values above ~0.3 are far stronger than any real market.
    """
    rng = rng or random.Random()
    window = cal.session_window(day)
    if window is None:
        return pd.DataFrame()
    session_open, session_close = window

    bar_count = int((session_close - session_open).total_seconds() // interval_seconds)
    # Convert an annual vol into per-bar vol: 252 trading days, this many bars per day.
    bars_per_year = 252 * bar_count
    sigma = annual_volatility / math.sqrt(bars_per_year)
    mu = drift / bars_per_year

    rows: List[dict] = []
    price = open_price
    previous_return = 0.0
    for index in range(bar_count):
        bar_start = session_open + dt.timedelta(seconds=index * interval_seconds)
        bar_open = price
        # Four sub-steps per bar give a plausible high/low spread rather than the
        # degenerate one a single step produces.
        highs, lows = [bar_open], [bar_open]
        for _ in range(4):
            shock = (sigma / 2) * rng.gauss(0, 1)
            step = momentum * previous_return + shock
            previous_return = step
            price *= math.exp(mu / 4 + step)
            highs.append(price)
            lows.append(price)
        bar_close = price

        # Volume is U-shaped across the session, as it is in reality.
        progress = index / max(1, bar_count - 1)
        shape = 1.0 + 1.8 * (math.exp(-6 * progress) + math.exp(-6 * (1 - progress)))
        volume = round(base_volume * shape * rng.uniform(0.6, 1.4))

        rows.append({
            "instrument_id": instrument_id,
            "bar_start": bar_start,
            "bar_end": bar_start + dt.timedelta(seconds=interval_seconds),
            "open": round(bar_open, 2),
            "high": round(max(highs), 2),
            "low": round(min(lows), 2),
            "close": round(bar_close, 2),
            "volume": float(volume),
            "vwap": round((max(highs) + min(lows) + bar_close) / 3, 2),
            "tick_count": int(volume / 40),
            "synthetic": False,
        })

    frame = pd.DataFrame(rows)
    frame["bar_start"] = pd.to_datetime(frame["bar_start"])
    frame["bar_end"] = pd.to_datetime(frame["bar_end"])
    return frame


def generate_history(
    instrument_ids: Sequence[str],
    start: dt.date,
    end: dt.date,
    *,
    start_prices: Optional[Dict[str, float]] = None,
    interval_seconds: int = 60,
    annual_volatility: float = 0.28,
    drift: float = 0.0,
    momentum: float = 0.0,
    overnight_gap_vol: float = 0.008,
    seed: int = 42,
) -> Dict[str, pd.DataFrame]:
    """Multi-day, multi-instrument history across real NSE trading days.

    With ``momentum=0`` (the default) this is the **negative control** dataset: no edge
    exists, so any strategy showing a profit indicates a broken harness. With
    ``momentum>0`` it becomes a **positive control**: a real, detectable trend effect that
    a working trend strategy should find. Running both is what distinguishes "the strategy
    is blind" from "the market has nothing to give".
    """
    rng = random.Random(seed)
    days = cal.trading_days_between(start, end)
    prices = dict(start_prices or {})
    frames: Dict[str, List[pd.DataFrame]] = {i: [] for i in instrument_ids}

    for instrument_id in instrument_ids:
        prices.setdefault(instrument_id, 1_000.0)

    for day in days:
        for instrument_id in instrument_ids:
            session = generate_session_bars(
                instrument_id, day,
                open_price=prices[instrument_id],
                interval_seconds=interval_seconds,
                annual_volatility=annual_volatility,
                drift=drift,
                momentum=momentum,
                rng=rng,
            )
            if session.empty:
                continue
            frames[instrument_id].append(session)
            # Carry the close forward with an overnight gap, so consecutive sessions are
            # linked but not continuous — as real equities behave.
            prices[instrument_id] = float(session["close"].iloc[-1]) * math.exp(
                overnight_gap_vol * rng.gauss(0, 1)
            )

    return {
        instrument_id: (pd.concat(parts, ignore_index=True) if parts else pd.DataFrame())
        for instrument_id, parts in frames.items()
    }
