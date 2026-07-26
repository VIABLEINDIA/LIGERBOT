"""Nifty 500 momentum shortlist from the TradingView scanner.

Solves a bootstrapping problem the rest of the project cannot solve on its own. The
momentum ranker in :mod:`src.momentum_screen` needs sixty-odd sessions of price history per
name, and this bot has recorded **none** — the Parquet store starts empty and fills at one
session per day (D5 mitigation 2). Waiting three months to know which stocks are trending is
not a plan.

One HTTP request returns roughly three thousand NSE stocks with performance, volatility,
relative volume and sector attached, which is enough to shortlist a watchlist on day one.

## What this is, precisely

**The universe is a proxy, not the index.** TradingView's scanner does not expose Nifty 500
membership under any filter tried, so the universe is *the top N NSE primary common stocks
by market capitalisation*. That is close to Nifty 500 by construction — the index is
essentially the top 500 by full market cap — but it is **not** the official constituent
list, will drift from it between rebalances, and is labelled a proxy everywhere it appears.
Anyone who needs exact membership should load NSE's published list instead.

**The endpoint is undocumented.** `scanner.tradingview.com` is TradingView's internal
screener API. It has no stability guarantee, and its terms are for interactive research
rather than redistribution. Two consequences are designed in: the result is **cached to
disk**, so a session does not depend on the endpoint answering at 09:00 on the day; and
every failure path returns *nothing* rather than raising, because a screener being down is
not a reason to stop trading a watchlist that already exists.

## The ranking

The same reasoning as :mod:`src.momentum_screen`, applied to the fields available here:
performance divided by volatility rather than raw performance, because ranking on the
biggest move selects for the names that then blow through ATR stops. TradingView supplies
`Perf.3M` and `Volatility.M` directly, so the risk adjustment costs nothing.

What is *not* available here is trend quality — the R² that separates a clean advance from
a single earnings gap. That needs the price series, which this endpoint does not return. It
is the main reason this module is a **bootstrap** rather than a replacement: once the
Parquet store has sixty sessions, `momentum_screen.rank_from_store` is the better ranking
and this one becomes a cross-check.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import config

log = logging.getLogger("ligerbot.tv_screener")

SCANNER_URL = "https://scanner.tradingview.com/india/scan"

# Order matters: rows come back as a positional list, and this is the key to it.
COLUMNS: Sequence[str] = (
    "name", "close", "volume", "Perf.1M", "Perf.3M", "Volatility.M",
    "relative_volume_10d_calc", "average_volume_10d_calc", "sector",
    "market_cap_basic",
)

HEADERS = {
    "Content-Type": "application/json",
    "User-Agent": "Mozilla/5.0",
    "Origin": "https://in.tradingview.com",
    "Referer": "https://in.tradingview.com/",
}


@dataclass(frozen=True)
class ScreenedStock:
    """One row, with the components kept separate so a rank can be explained."""

    symbol: str                 # "NSE:RELIANCE"
    name: str                   # "RELIANCE"
    close: float
    volume: float
    perf_1m: float              # percent
    perf_3m: float              # percent
    volatility_m: float         # percent, monthly
    rvol: float
    avg_volume_10d: float
    sector: str
    market_cap: float

    @property
    def turnover(self) -> float:
        """Average daily traded value — the liquidity figure D6 screens on."""
        return self.close * self.avg_volume_10d

    @property
    def risk_adjusted(self) -> float:
        """3-month performance per unit of monthly volatility.

        The ranking key, and the same choice `momentum_screen` makes: raw performance
        ranks the most volatile names first, and those are precisely the ones whose stops
        get taken out at a rate the cost model says is unaffordable.
        """
        if self.volatility_m <= 0:
            return 0.0
        return self.perf_3m / self.volatility_m

    def describe(self) -> str:
        return (f"{self.name:<14} ₹{self.close:>9,.1f}  3M={self.perf_3m:>+7.2f}% "
                f"vol={self.volatility_m:>5.2f}  ra={self.risk_adjusted:>+6.2f}  "
                f"rvol={self.rvol:>4.2f}  {self.sector[:22]}")


@dataclass
class ScreenResult:
    stocks: List[ScreenedStock] = field(default_factory=list)
    universe_size: int = 0
    fetched_at: Optional[dt.datetime] = None
    source: str = "tradingview_scanner"
    from_cache: bool = False
    excluded: Dict[str, int] = field(default_factory=dict)

    @property
    def symbols(self) -> List[str]:
        return [s.symbol for s in self.stocks]

    def __len__(self) -> int:
        return len(self.stocks)

    def report(self, limit: int = 20) -> str:
        lines = [
            "=" * 100,
            f"NIFTY 500 MOMENTUM SHORTLIST  ({'cache' if self.from_cache else 'live'})",
            "=" * 100,
            f"  universe {self.universe_size} NSE stocks (top-by-market-cap proxy for "
            f"Nifty 500) -> {len(self.stocks)} selected",
            f"  fetched {self.fetched_at:%Y-%m-%d %H:%M} " if self.fetched_at else "",
            "",
        ]
        for i, s in enumerate(self.stocks[:limit], 1):
            lines.append(f"  {i:>3}. {s.describe()}")
        if len(self.stocks) > limit:
            lines.append(f"       ... and {len(self.stocks) - limit} more")
        if self.excluded:
            lines += ["", "  Excluded"]
            for reason, count in sorted(self.excluded.items(), key=lambda kv: -kv[1]):
                lines.append(f"    {count:>5}  {reason}")
        lines.append("=" * 100)
        return "\n".join(lines)


@dataclass
class ScreenCriteria:
    """Thresholds. Liquidity mirrors D6; the rest shapes the shortlist."""

    universe_size: int = 500            # the Nifty 500 proxy
    top_n: int = 200
    min_price: float = 50.0
    max_price: float = 20_000.0
    min_turnover: float = 50_00_00_000.0        # ₹50 crore average daily value
    min_rvol: float = 0.0
    require_positive_momentum: bool = True      # long-only, D3
    max_volatility_m: float = 12.0              # refuse untradeably wild names

    def describe(self) -> str:
        return (f"universe={self.universe_size} top={self.top_n} "
                f"price={self.min_price:.0f}-{self.max_price:.0f} "
                f"turnover>=₹{self.min_turnover / 1e7:.0f}cr "
                f"vol<={self.max_volatility_m:.0f}%")


def _cache_path(day: Optional[dt.date] = None) -> Path:
    day = day or dt.date.today()
    root = Path(config.SCRIP_MASTER_DIR) / "tv_screen"
    return root / f"nifty500_momentum_{day.isoformat()}.json"


def fetch_universe(size: int = 500, *, timeout: float = 25.0) -> List[ScreenedStock]:
    """One request for the top ``size`` NSE stocks by market cap.

    Raises nothing the caller must handle — network problems propagate as
    :class:`OSError` subclasses and are caught by :func:`screen`.
    """
    body = {
        "filter": [
            {"left": "exchange", "operation": "equal", "right": "NSE"},
            {"left": "type", "operation": "equal", "right": "stock"},
            {"left": "is_primary", "operation": "equal", "right": True},
        ],
        "options": {"lang": "en"},
        "symbols": {"query": {"types": []}, "tickers": []},
        "columns": list(COLUMNS),
        "sort": {"sortBy": "market_cap_basic", "sortOrder": "desc"},
        "range": [0, int(size)],
    }
    request = urllib.request.Request(
        SCANNER_URL, data=json.dumps(body).encode(), headers=HEADERS)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read())

    out: List[ScreenedStock] = []
    for row in payload.get("data", []):
        values = row.get("d") or []
        if len(values) < len(COLUMNS):
            continue
        try:
            out.append(ScreenedStock(
                symbol=str(row.get("s", "")),
                name=str(values[0] or ""),
                close=float(values[1] or 0.0),
                volume=float(values[2] or 0.0),
                perf_1m=float(values[3] or 0.0),
                perf_3m=float(values[4] or 0.0),
                volatility_m=float(values[5] or 0.0),
                rvol=float(values[6] or 0.0),
                avg_volume_10d=float(values[7] or 0.0),
                sector=str(values[8] or ""),
                market_cap=float(values[9] or 0.0),
            ))
        except (TypeError, ValueError):
            # One malformed row must not cost the other four hundred and ninety-nine.
            continue
    return out


def apply_screen(stocks: Sequence[ScreenedStock],
                 criteria: Optional[ScreenCriteria] = None) -> ScreenResult:
    """Filter and rank. Exclusions are tallied by reason, never silent (§5.7)."""
    criteria = criteria or ScreenCriteria()
    kept: List[ScreenedStock] = []
    excluded: Dict[str, int] = {}

    def drop(reason: str) -> None:
        excluded[reason] = excluded.get(reason, 0) + 1

    for s in stocks:
        if not s.symbol or s.close <= 0:
            drop("no price")
        elif not (criteria.min_price <= s.close <= criteria.max_price):
            drop("price outside range")
        elif s.turnover < criteria.min_turnover:
            drop("below liquidity floor")
        elif s.volatility_m > criteria.max_volatility_m:
            drop("too volatile to size sanely")
        elif criteria.require_positive_momentum and s.perf_3m <= 0:
            drop("no positive momentum (long-only, D3)")
        elif s.rvol < criteria.min_rvol:
            drop("insufficient relative volume")
        else:
            kept.append(s)

    kept.sort(key=lambda s: s.risk_adjusted, reverse=True)
    return ScreenResult(stocks=kept[:criteria.top_n], universe_size=len(stocks),
                        fetched_at=dt.datetime.now(), excluded=excluded)


def screen(criteria: Optional[ScreenCriteria] = None, *,
           use_cache: bool = True, day: Optional[dt.date] = None) -> ScreenResult:
    """Fetch (or reuse today's cache), filter and rank.

    **Never raises.** A screener that is down returns an empty result, and the caller
    keeps whatever watchlist it already had. Refusing to trade because a convenience
    endpoint is unreachable would be a worse failure than the one it is reporting.
    """
    criteria = criteria or ScreenCriteria()
    path = _cache_path(day)

    if use_cache and path.exists():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            stocks = [ScreenedStock(**row) for row in raw["stocks"]]
            result = apply_screen(stocks, criteria)
            result.from_cache = True
            log.info("Momentum shortlist from cache (%s).", path.name)
            return result
        except Exception as exc:  # noqa: BLE001 - a bad cache is not fatal
            log.warning("Could not read the screen cache (%s) — refetching.", exc)

    try:
        stocks = fetch_universe(criteria.universe_size)
    except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError) as exc:
        log.error("TradingView screener unavailable (%s). Returning an empty shortlist; "
                  "the existing watchlist is unchanged.", exc)
        return ScreenResult(universe_size=0, fetched_at=dt.datetime.now())

    if not stocks:
        log.error("The screener answered but returned no rows — treating as unavailable "
                  "rather than as an empty market.")
        return ScreenResult(universe_size=0, fetched_at=dt.datetime.now())

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(
            {"fetched_at": dt.datetime.now().isoformat(),
             "stocks": [s.__dict__ for s in stocks]}, indent=1), encoding="utf-8")
    except OSError as exc:
        log.warning("Could not cache the screen (%s) — continuing.", exc)

    return apply_screen(stocks, criteria)


def to_instrument_ids(result: ScreenResult, master: Any = None) -> List[str]:
    """Map ``NSE:RELIANCE`` to canonical ``nse_cm:<token>`` ids.

    Requires the instrument master: the canonical id is built from the exchange token,
    never from the display name (B4). Without a master this returns nothing rather than
    guessing, because a guessed token is an order on the wrong instrument.
    """
    if master is None:
        log.error("No instrument master — cannot resolve %d screened symbols to "
                  "canonical ids. Refusing to guess (B4).", len(result))
        return []

    ids: List[str] = []
    for stock in result.stocks:
        instrument = None
        for candidate in (f"{stock.name}-EQ", stock.name):
            instrument = master.by_symbol(candidate) if hasattr(
                master, "by_symbol") else None
            if instrument is not None:
                break
        if instrument is not None:
            ids.append(instrument.instrument_id)
    if len(ids) < len(result):
        log.warning("Resolved %d of %d screened symbols to instrument ids.",
                    len(ids), len(result))
    return ids
