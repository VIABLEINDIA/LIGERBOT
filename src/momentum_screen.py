"""Momentum ranking — turning a liquid universe into a *ranked* one.

The existing screen (D6, :mod:`src.instruments`) answers *"can this be traded?"* — enough
value changing hands, a sane price, tradable intraday. It says nothing about *"is this
worth trading today?"*, and it targets twelve names because that is a size a
correlation-limited book can use.

This module answers the second question. It ranks a screened universe so a strategy that
trades three positions can pick them from the names actually trending, rather than from
whatever the liquidity screen happened to return in alphabetical order.

Four decisions here are load-bearing, and each exists because the obvious version is wrong.

**Raw return is the wrong ranking.** Sorting by "biggest move" selects for volatility, and
those names then blow through ATR stops at a rate the backtest's cost model already says is
unaffordable. The ranking is *risk-adjusted*: return divided by the volatility that produced
it. A 6% move on 1% daily vol beats a 10% move on 4%.

**The most recent bars are skipped.** Short-term reversal contaminates raw momentum — a
stock that ran hard *yesterday* is as likely to give it back as continue. Classic momentum
research uses 12-month returns skipping the most recent month for exactly this reason. The
same logic applies at any horizon, so the lookback ends `skip_bars` before today.

**Trend quality is scored separately from trend size.** A stock that gapped 20% on an
earnings surprise and then went sideways has enormous "momentum" by any return measure and
is not trending at all — there is nothing for a pullback strategy to pull back *to*. The R²
of log-price against time separates a clean advance from a single event, and it is the one
that actually predicts whether the next pullback resolves upward.

**It fails closed.** An instrument without enough history to score is *excluded*, never
ranked at the bottom and never assumed neutral. Trading a name because its data was missing
is the same mistake the liquidity screen already refuses to make.

.. note::
   This ranks. It does not decide. The output is a **watchlist** — the set of instruments
   worth subscribing to and evaluating — and the risk engine still gates every individual
   entry. Screening 200 names to trade three is the intended shape, not an inconsistency.
"""
from __future__ import annotations

import datetime as dt
import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence

log = logging.getLogger("ligerbot.momentum")

# Below this many usable bars a score is noise dressed as a number.
MIN_BARS = 30


class Direction(Enum):
    UP = "up"
    DOWN = "down"
    FLAT = "flat"


@dataclass(frozen=True)
class MomentumScore:
    """One instrument's ranking inputs, kept separate so a rank can be explained.

    Every component is retained rather than collapsed into a single number at computation
    time. A ranking nobody can interrogate is one nobody can debug when it starts choosing
    badly — and "why is this at the top today" is the first question anyone asks.
    """

    instrument_id: str
    lookback_return: float          # raw, over the lookback window
    volatility: float               # stdev of per-bar log returns, same window
    risk_adjusted: float            # the ranking key
    trend_quality: float            # R² of log-price vs time, 0..1
    rvol: float                     # recent volume vs the window's average
    bars_used: int
    direction: Direction

    @property
    def usable(self) -> bool:
        return self.bars_used >= MIN_BARS and self.volatility > 0.0

    def describe(self) -> str:
        return (f"{self.instrument_id}: ret={self.lookback_return:+.2%} "
                f"vol={self.volatility:.2%} risk_adj={self.risk_adjusted:+.2f} "
                f"R²={self.trend_quality:.2f} rvol={self.rvol:.2f}")


@dataclass
class MomentumCriteria:
    """Thresholds for the ranking. Defaults are deliberately permissive.

    This is a *ranking* stage, not a second risk gate: the risk engine already refuses
    trades on its own terms, and filtering aggressively here would silently shrink the
    universe in ways no rejection tally would explain.
    """

    lookback_bars: int = 60
    # Skip the most recent bars: short-term reversal contaminates raw momentum.
    skip_bars: int = 1
    top_n: int = 200
    min_trend_quality: float = 0.0      # 0 disables; ~0.3 demands a visibly clean trend
    min_rvol: float = 0.0               # 0 disables
    require_direction: Optional[Direction] = Direction.UP   # long-only v1 (D3)
    # A move this small is not momentum, whatever its risk-adjusted figure says.
    min_abs_return: float = 0.0

    def describe(self) -> str:
        return (f"lookback={self.lookback_bars} skip={self.skip_bars} top={self.top_n} "
                f"minR²={self.min_trend_quality:.2f} minRVOL={self.min_rvol:.2f} "
                f"dir={self.require_direction.value if self.require_direction else 'any'}")


# ---------------------------------------------------------------------------
def _log_returns(closes: Sequence[float]) -> List[float]:
    out: List[float] = []
    for previous, current in zip(closes, closes[1:]):
        if previous > 0 and current > 0:
            out.append(math.log(current / previous))
    return out


def _stdev(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(variance)


def trend_quality(closes: Sequence[float]) -> float:
    """R² of log(price) against time — how *cleanly* the move happened.

    Distinguishes a steady advance from a single gap. A stock that jumped 20% on earnings
    and then went sideways scores near zero here while scoring enormously on return, and
    ranking it highly would hand a pullback strategy an instrument with no pullbacks in it.

    Returns 0.0 rather than raising on degenerate input: a flat series has no trend to
    measure, which is information, not an error.
    """
    usable = [c for c in closes if c > 0]
    n = len(usable)
    if n < 3:
        return 0.0

    ys = [math.log(c) for c in usable]
    xs = list(range(n))
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n

    sxx = sum((x - mean_x) ** 2 for x in xs)
    syy = sum((y - mean_y) ** 2 for y in ys)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))

    if sxx <= 0 or syy <= 0:
        return 0.0                      # a perfectly flat series: no trend, not an error
    return min(1.0, max(0.0, (sxy * sxy) / (sxx * syy)))


def score(instrument_id: str, closes: Sequence[float],
          volumes: Optional[Sequence[float]] = None,
          *, criteria: Optional[MomentumCriteria] = None) -> MomentumScore:
    """Score one instrument. Never raises — an unscoreable series returns ``usable=False``."""
    criteria = criteria or MomentumCriteria()

    # Drop the most recent bars *before* anything else, so every component below is
    # measured over the same window.
    series = list(closes)
    if criteria.skip_bars > 0:
        series = series[:-criteria.skip_bars] if len(series) > criteria.skip_bars else []
    series = series[-criteria.lookback_bars:]

    if len(series) < MIN_BARS or series[0] <= 0:
        return MomentumScore(instrument_id, 0.0, 0.0, 0.0, 0.0, 0.0,
                             len(series), Direction.FLAT)

    total_return = (series[-1] - series[0]) / series[0]
    returns = _log_returns(series)
    volatility = _stdev(returns)
    quality = trend_quality(series)

    # Risk-adjusted: the ranking key. Zero volatility cannot be divided by, and a series
    # that never moved has no momentum regardless of its endpoints.
    risk_adjusted = (total_return / volatility) if volatility > 0 else 0.0

    rvol = 1.0
    if volumes:
        window = list(volumes)
        if criteria.skip_bars > 0 and len(window) > criteria.skip_bars:
            window = window[:-criteria.skip_bars]
        window = window[-criteria.lookback_bars:]
        recent = window[-5:]
        if window and recent:
            average = sum(window) / len(window)
            if average > 0:
                rvol = (sum(recent) / len(recent)) / average

    if total_return > 0.005:
        direction = Direction.UP
    elif total_return < -0.005:
        direction = Direction.DOWN
    else:
        direction = Direction.FLAT

    return MomentumScore(
        instrument_id=instrument_id,
        lookback_return=total_return,
        volatility=volatility,
        risk_adjusted=risk_adjusted,
        trend_quality=quality,
        rvol=rvol,
        bars_used=len(series),
        direction=direction,
    )


@dataclass
class RankedUniverse:
    """The ranked watchlist, with everything needed to explain it."""

    scores: List[MomentumScore] = field(default_factory=list)
    excluded: Dict[str, str] = field(default_factory=dict)
    criteria: MomentumCriteria = field(default_factory=MomentumCriteria)
    ranked_on: Optional[dt.date] = None

    @property
    def instrument_ids(self) -> List[str]:
        return [s.instrument_id for s in self.scores]

    def __len__(self) -> int:
        return len(self.scores)

    def report(self, limit: int = 15) -> str:
        lines = [
            "=" * 78,
            f"MOMENTUM RANKING — {self.ranked_on or 'unspecified'}",
            "=" * 78,
            f"  {self.criteria.describe()}",
            f"  ranked {len(self.scores)} of {len(self.scores) + len(self.excluded)} "
            f"candidates ({len(self.excluded)} excluded)",
            "",
        ]
        for i, s in enumerate(self.scores[:limit], 1):
            lines.append(f"  {i:>3}. {s.describe()}")
        if len(self.scores) > limit:
            lines.append(f"       ... and {len(self.scores) - limit} more")

        if self.excluded:
            tally: Dict[str, int] = {}
            for reason in self.excluded.values():
                tally[reason] = tally.get(reason, 0) + 1
            lines += ["", "  Excluded"]
            for reason, count in sorted(tally.items(), key=lambda kv: -kv[1]):
                lines.append(f"    {count:>5}  {reason}")
        lines.append("=" * 78)
        return "\n".join(lines)


def rank(
    closes_by_instrument: Dict[str, Sequence[float]],
    volumes_by_instrument: Optional[Dict[str, Sequence[float]]] = None,
    *,
    criteria: Optional[MomentumCriteria] = None,
    day: Optional[dt.date] = None,
) -> RankedUniverse:
    """Rank a universe, returning the top ``criteria.top_n`` and why the rest were dropped.

    Exclusions are *counted and reported*, never silent. A screen that quietly returns
    forty names when it was asked for two hundred is indistinguishable from a broken one,
    and this project has already been bitten once by a screen that returned nothing at all
    without saying why (§5.7).
    """
    criteria = criteria or MomentumCriteria()
    volumes_by_instrument = volumes_by_instrument or {}

    scored: List[MomentumScore] = []
    excluded: Dict[str, str] = {}

    for instrument_id, closes in closes_by_instrument.items():
        s = score(instrument_id, closes, volumes_by_instrument.get(instrument_id),
                  criteria=criteria)

        if not s.usable:
            excluded[instrument_id] = f"insufficient history (<{MIN_BARS} bars)"
            continue
        if criteria.require_direction and s.direction is not criteria.require_direction:
            excluded[instrument_id] = f"direction {s.direction.value}"
            continue
        if abs(s.lookback_return) < criteria.min_abs_return:
            excluded[instrument_id] = "move too small to be momentum"
            continue
        if s.trend_quality < criteria.min_trend_quality:
            excluded[instrument_id] = "trend too noisy (low R²)"
            continue
        if s.rvol < criteria.min_rvol:
            excluded[instrument_id] = "insufficient relative volume"
            continue
        scored.append(s)

    # Sort by the risk-adjusted figure, tie-broken by trend quality — between two names
    # with the same risk-adjusted move, prefer the one that got there cleanly.
    scored.sort(key=lambda s: (s.risk_adjusted, s.trend_quality), reverse=True)

    if excluded:
        log.info("Momentum ranking: %d ranked, %d excluded.", len(scored), len(excluded))

    return RankedUniverse(scores=scored[:criteria.top_n], excluded=excluded,
                          criteria=criteria, ranked_on=day)


def rank_from_store(
    instrument_ids: Sequence[str],
    store,
    *,
    criteria: Optional[MomentumCriteria] = None,
    day: Optional[dt.date] = None,
    lookback_days: int = 90,
) -> RankedUniverse:
    """Rank from the Parquet store — our own recorded bars, no broker call.

    Deliberately source-agnostic: it takes daily closes from wherever they came from. The
    store is the default because it is the one source that improves every day the bot runs
    and costs nothing to query (D5 mitigation 2).
    """
    from src import market_calendar as cal

    criteria = criteria or MomentumCriteria()
    day = day or cal.now_ist().date()
    start = day - dt.timedelta(days=lookback_days * 2)   # calendar days ⊃ trading days

    closes: Dict[str, List[float]] = {}
    volumes: Dict[str, List[float]] = {}
    for instrument_id in instrument_ids:
        try:
            frame = store.read_range(instrument_id, start, day)
        except Exception as exc:  # noqa: BLE001 - a missing partition is not fatal
            log.debug("No stored bars for %s: %s", instrument_id, exc)
            continue
        if frame is None or frame.empty:
            continue
        # Collapse intraday bars to one close and one volume per session, so the ranking
        # measures day-over-day momentum rather than the shape of the last few minutes.
        frame = frame.copy()
        frame["session"] = frame["bar_start"].dt.date
        daily = frame.groupby("session").agg(close=("close", "last"),
                                             volume=("volume", "sum"))
        closes[instrument_id] = daily["close"].tolist()
        volumes[instrument_id] = daily["volume"].tolist()

    return rank(closes, volumes, criteria=criteria, day=day)
