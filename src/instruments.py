"""Instrument master, liquidity screen, and universe validation.

Fixes two defects that would each have broken the first real order (DESIGN.md 0.1):

  * **B4** — the execution engine sent the *display name* from the tick
    (e.g. ``"Nifty 50"``) as ``trading_symbol``. Every order would have been rejected.
    Internal events now carry an opaque ``instrument_id``; the broker's ``trading_symbol``
    is resolved here, once, at the execution boundary.
  * **B5** — the shipped default universe was the Nifty 50 *index*, which cannot be
    bought in the cash segment at all. :func:`validate_universe` now rejects anything
    that isn't genuinely tradable, at startup, before a single tick is processed.

The universe is a **screen, not a list** (DESIGN.md D6). Liquidity and prices drift, so
hardcoding ten symbols guarantees they are wrong within a year. What is stable is the
*criteria*; the resulting list is regenerated and version-stamped so every backtest
records which universe it actually ran against.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

log = logging.getLogger("ligerbot.instruments")

# Cash-segment series that represent something you can actually buy.
# "EQ" covers ordinary equities and ETFs (NIFTYBEES is EQ); "BE" is trade-to-trade,
# which settles delivery-only and is therefore useless for an intraday MIS bot.
TRADABLE_SERIES = {"EQ"}
DELIVERY_ONLY_SERIES = {"BE", "BZ", "IL"}


@dataclass(frozen=True)
class Instrument:
    """One tradable instrument, normalized from the broker's scrip master.

    ``instrument_id`` is our stable internal handle and the only identifier that travels
    on the event bus. Everything broker-specific stops at this boundary.
    """

    instrument_id: str
    token: str
    trading_symbol: str
    name: str
    exchange_segment: str
    series: str
    lot_size: int
    tick_size: float
    isin: str = ""
    # Screen inputs; populated from market data, not the scrip master.
    last_price: Optional[float] = None
    avg_daily_value: Optional[float] = None
    is_fno: bool = False
    # Instruments that tend to stop out together. DESIGN.md D2 assumes NSE large caps
    # are strongly correlated intraday and sizes for the worst case; this is what lets
    # the risk engine actually *act* on that assumption rather than only allow for it.
    correlation_group: str = ""

    @property
    def is_tradable_cash(self) -> bool:
        """Can this be bought intraday in the cash segment?

        The check that would have caught the Nifty-50-index default: an index has no
        tradable series and no meaningful lot/tick, so it fails here.
        """
        return (
            bool(self.trading_symbol)
            and self.series in TRADABLE_SERIES
            and self.lot_size >= 1
            and self.tick_size > 0
        )

    def round_to_tick(self, price: float) -> float:
        """Snap a price to the instrument's tick size.

        Orders at sub-tick prices are rejected by the exchange, so every limit price
        and stop level passes through here before it goes anywhere near the broker.
        """
        if self.tick_size <= 0:
            return round(price, 2)
        return round(round(price / self.tick_size) * self.tick_size, 4)


@dataclass
class ScreenCriteria:
    """Liquidity screen thresholds (DESIGN.md D6)."""

    min_avg_daily_value: float = 200_00_00_000.0  # Rs 200 crore
    max_price: float = 5_000.0
    min_price: float = 50.0
    require_fno: bool = True
    target_size: int = 12

    def describe(self) -> str:
        return (
            f"adv>={self.min_avg_daily_value / 1e7:.0f}cr "
            f"price={self.min_price:.0f}-{self.max_price:.0f} "
            f"fno={self.require_fno} target={self.target_size}"
        )


@dataclass
class Universe:
    """A screened, version-stamped set of instruments.

    The stamp matters: a backtest run against a different universe is a different
    experiment, and without recording which one it used the results aren't reproducible.
    """

    instruments: List[Instrument]
    screened_on: dt.date
    criteria_summary: str
    version: str = field(default="", init=False)

    def __post_init__(self) -> None:
        ids = ",".join(sorted(i.instrument_id for i in self.instruments))
        digest = hashlib.sha256(
            f"{ids}|{self.criteria_summary}".encode("utf-8")
        ).hexdigest()[:12]
        object.__setattr__(self, "version", f"{self.screened_on.isoformat()}-{digest}")

    def __len__(self) -> int:
        return len(self.instruments)

    @property
    def ids(self) -> List[str]:
        return [i.instrument_id for i in self.instruments]


# --------------------------------------------------------------------------
# Loading the scrip master
# --------------------------------------------------------------------------
# Kotak's scrip master column names. Kept as a mapping rather than inlined so a change
# on the broker's side is a one-line fix here instead of a hunt through the parser.
KOTAK_COLUMNS = {
    "token": ("pSymbol", "pTrdSymbol", "token"),
    "trading_symbol": ("pTrdSymbol", "pSymbol", "trading_symbol"),
    "name": ("pSymbolName", "pDesc", "name"),
    "series": ("pGroup", "pSeries", "series"),
    "lot_size": ("lLotSize", "lot_size"),
    # dTickSize first: a working integration on this machine (D:\JEANS), which mirrors
    # openalgo's verified approach, reads dTickSize. lTickSize was this module's original
    # guess and is kept only as a fallback.
    "tick_size": ("dTickSize", "lTickSize", "tick_size"),
    "isin": ("pISIN", "isin"),
}

# Fallback when scrip_master() does not hand back a usable URL. Pattern taken from a
# working integration; the date component is today's, so a stale cache is impossible.
SCRIP_MASTER_URL_TEMPLATE = (
    "https://lapi.kotaksecurities.com/wso2-scripmaster/v1/prod/"
    "{date}/transformed-v1/{segment}-v1.csv"
)


def _first_present(row: Dict[str, str], keys: Sequence[str]) -> str:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return str(value).strip()
    return ""


def _normalize_tick_size(raw: str) -> float:
    """Kotak reports tick size in paise-hundredths (``5`` -> Rs 0.05).

    Sanity-bounded rather than trusted: a scrip master format change that silently
    shifted this by 100x would otherwise produce plausible-looking but wrong prices on
    every order. Anything outside a believable NSE tick range falls back to Rs 0.05.
    """
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 0.05
    if value <= 0:
        return 0.05
    tick = value / 100.0
    if not (0.0001 <= tick <= 10.0):
        log.warning("Implausible tick size %r (-> %.4f); defaulting to 0.05.", raw, tick)
        return 0.05
    return tick


def parse_scrip_master(rows: Iterable[Dict[str, str]], segment: str = "nse_cm") -> List[Instrument]:
    """Normalize scrip-master rows into :class:`Instrument` objects.

    Malformed rows are skipped with a count rather than raising — a single bad line in a
    50,000-row broker file should not stop the bot from starting.
    """
    out: List[Instrument] = []
    skipped = 0
    for row in rows:
        try:
            token = _first_present(row, KOTAK_COLUMNS["token"])
            trading_symbol = _first_present(row, KOTAK_COLUMNS["trading_symbol"])
            if not token or not trading_symbol:
                skipped += 1
                continue

            lot_raw = _first_present(row, KOTAK_COLUMNS["lot_size"]) or "1"
            out.append(Instrument(
                instrument_id=f"{segment}:{token}",
                token=token,
                trading_symbol=trading_symbol,
                name=_first_present(row, KOTAK_COLUMNS["name"]) or trading_symbol,
                exchange_segment=segment,
                series=_first_present(row, KOTAK_COLUMNS["series"]).upper(),
                lot_size=max(1, int(float(lot_raw))),
                tick_size=_normalize_tick_size(_first_present(row, KOTAK_COLUMNS["tick_size"])),
                isin=_first_present(row, KOTAK_COLUMNS["isin"]),
            ))
        except (ValueError, TypeError, KeyError):
            skipped += 1
    if skipped:
        log.warning("Skipped %d malformed scrip-master rows.", skipped)
    log.info("Parsed %d instruments from the %s scrip master.", len(out), segment)
    return out


def load_scrip_master_csv(path: str | Path, segment: str = "nse_cm") -> List[Instrument]:
    """Load and normalize a scrip-master CSV downloaded from the broker."""
    with open(path, newline="", encoding="utf-8-sig") as handle:
        return parse_scrip_master(csv.DictReader(handle), segment=segment)


def resolve_scrip_master_url(
    neo_client: Any = None, segment: str = "nse_cm", *, day: Optional[dt.date] = None
) -> str:
    """Find the URL of today's scrip-master CSV.

    ``scrip_master()`` returns a **URL string**, not rows — the SDK resolves the location
    and does not parse the file. This module originally assumed rows, which is why B4 had
    no path to success: there was nothing to feed the parser.

    Falls back to the published URL pattern if the SDK gives nothing usable, so a single
    flaky call does not block startup.
    """
    day = day or dt.date.today()
    if neo_client is not None:
        try:
            result = neo_client.scrip_master(exchange_segment=segment)
            if isinstance(result, str) and result.startswith("http"):
                return result
            # Some versions return a list or an envelope of per-segment paths.
            if isinstance(result, dict):
                for value in result.values():
                    if isinstance(value, str) and value.startswith("http"):
                        return value
            if isinstance(result, list):
                for value in result:
                    if isinstance(value, str) and segment in value:
                        return value
            log.warning("scrip_master() returned no usable URL (%r) — using the "
                        "published fallback pattern.", str(result)[:120])
        except Exception as exc:  # noqa: BLE001 - any SDK failure just falls back
            log.warning("scrip_master() failed (%s) — using the published fallback "
                        "pattern.", exc)
    return SCRIP_MASTER_URL_TEMPLATE.format(date=day.isoformat(), segment=segment)


def download_scrip_master(
    url: str, segment: str = "nse_cm", *, timeout: float = 60.0
) -> List[Instrument]:
    """Download and parse the scrip-master CSV.

    Header cleaning is not cosmetic: the published file carries stray spaces and
    semicolons in its column names, so ``pSymbol`` may arrive as ``pSymbol;`` or
    ``p Symbol`` and every lookup would miss.
    """
    import csv as _csv
    import io

    import requests

    log.info("Downloading the %s scrip master from %s", segment, url)
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()

    text = response.text
    first_newline = text.find("\n")
    if first_newline > 0:
        header = text[:first_newline].replace(" ", "").replace(";", "")
        text = header + text[first_newline:]

    rows = list(_csv.DictReader(io.StringIO(text)))
    log.info("Scrip master: %d raw row(s).", len(rows))
    return parse_scrip_master(rows, segment=segment)


def load_or_download_master(
    neo_client: Any = None,
    *,
    segment: str = "nse_cm",
    cache_dir: Optional[str | Path] = None,
    day: Optional[dt.date] = None,
    force_refresh: bool = False,
) -> "InstrumentMaster":
    """Today's instrument master, from cache if fresh, else downloaded.

    Cached per day because the scrip master changes daily and a stale one silently maps
    symbols to the wrong tokens.
    """
    day = day or dt.date.today()
    cache_dir = cache_dir or getattr(__import__("config"), "SCRIP_MASTER_DIR",
                                     "state/scrip_master")
    cache_path = InstrumentMaster.cache_path(cache_dir, day, segment)

    if not force_refresh:
        cached = InstrumentMaster.load_cache(cache_path)
        if cached is not None and len(cached):
            log.info("Loaded %d instruments from today's cache (%s).",
                     len(cached), cache_path)
            return cached

    url = resolve_scrip_master_url(neo_client, segment, day=day)
    master = InstrumentMaster(download_scrip_master(url, segment))
    if len(master):
        master.save_cache(cache_path)
    return master


class InstrumentMaster:
    """Bidirectional lookup over the scrip master, cached to disk for the trading day.

    Cached on disk rather than in Redis (a deviation from DESIGN.md 3.4): the scrip
    master is a large, static, daily blob, so it belongs in a file the modules can each
    mmap cheaply, not in the memory of the event bus that carries live orders.
    """

    def __init__(self, instruments: Sequence[Instrument]) -> None:
        self._by_id: Dict[str, Instrument] = {}
        self._by_symbol: Dict[str, Instrument] = {}
        self._by_token: Dict[str, Instrument] = {}
        for inst in instruments:
            self._by_id[inst.instrument_id] = inst
            self._by_symbol[inst.trading_symbol.upper()] = inst
            self._by_token[inst.token] = inst

    def __len__(self) -> int:
        return len(self._by_id)

    def by_id(self, instrument_id: str) -> Optional[Instrument]:
        return self._by_id.get(instrument_id)

    def by_symbol(self, trading_symbol: str) -> Optional[Instrument]:
        return self._by_symbol.get(trading_symbol.upper())

    def by_token(self, token: str) -> Optional[Instrument]:
        return self._by_token.get(str(token))

    def resolve(self, ref: str) -> Optional[Instrument]:
        """Look up by whichever identifier the caller happens to hold."""
        return self.by_id(ref) or self.by_symbol(ref) or self.by_token(ref)

    def require(self, instrument_id: str) -> Instrument:
        """Look up or raise.

        Used on the execution path, where a silent ``None`` would become a malformed
        order rather than an obvious failure.
        """
        inst = self.by_id(instrument_id)
        if inst is None:
            raise KeyError(
                f"Unknown instrument_id {instrument_id!r}. It is not in the scrip "
                f"master ({len(self._by_id)} instruments loaded) — refusing to guess."
            )
        return inst

    # -- cache -------------------------------------------------------------
    @staticmethod
    def cache_path(base_dir: str | Path, day: dt.date, segment: str = "nse_cm") -> Path:
        return Path(base_dir) / f"scrip_master_{segment}_{day.isoformat()}.json"

    def save_cache(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = [asdict(i) for i in self._by_id.values()]
        path.write_text(json.dumps(payload), encoding="utf-8")
        log.info("Cached %d instruments to %s", len(payload), path)

    @classmethod
    def load_cache(cls, path: str | Path) -> Optional["InstrumentMaster"]:
        path = Path(path)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return cls([Instrument(**row) for row in payload])
        except (OSError, ValueError, TypeError) as exc:
            log.warning("Instrument cache at %s unreadable (%s) — will refetch.", path, exc)
            return None


# --------------------------------------------------------------------------
# Screening and validation
# --------------------------------------------------------------------------
def validate_universe(
    instruments: Sequence[Instrument],
    *,
    strict: bool = True,
) -> List[Instrument]:
    """Drop anything not genuinely tradable intraday in the cash segment.

    This is the startup gate that B5 needed. In ``strict`` mode a rejection raises,
    because a misconfigured universe should stop the bot at boot — not surface later as
    a stream of exchange rejections during market hours.
    """
    good: List[Instrument] = []
    problems: List[str] = []

    for inst in instruments:
        if inst.series in DELIVERY_ONLY_SERIES:
            problems.append(
                f"{inst.instrument_id} ({inst.name}): series {inst.series!r} is "
                f"delivery-only (trade-to-trade) — MIS intraday is not permitted."
            )
        elif not inst.is_tradable_cash:
            problems.append(
                f"{inst.instrument_id} ({inst.name}): not tradable in the cash segment "
                f"(series={inst.series!r} lot={inst.lot_size} tick={inst.tick_size}). "
                f"Indices cannot be bought directly — use an ETF such as NIFTYBEES."
            )
        else:
            good.append(inst)

    if problems:
        message = "Universe validation rejected %d instrument(s):\n  %s" % (
            len(problems), "\n  ".join(problems)
        )
        if strict:
            raise ValueError(message)
        log.warning(message)

    return good


def apply_screen(
    instruments: Sequence[Instrument],
    criteria: Optional[ScreenCriteria] = None,
    *,
    screened_on: Optional[dt.date] = None,
) -> Universe:
    """Rank tradable instruments by liquidity and take the top ``target_size``.

    Instruments missing the market-data fields the screen needs (price, traded value)
    are excluded rather than assumed good — screening on absent data is how an illiquid
    name ends up in a universe it has no business being in.
    """
    criteria = criteria or ScreenCriteria()
    screened_on = screened_on or dt.date.today()

    eligible: List[Instrument] = []
    for inst in validate_universe(instruments, strict=False):
        if inst.avg_daily_value is None or inst.last_price is None:
            continue
        if inst.avg_daily_value < criteria.min_avg_daily_value:
            continue
        # Price band: high-priced shares make risk-based sizing intolerably coarse,
        # because a target notional can only be expressed in whole shares.
        if not (criteria.min_price <= inst.last_price <= criteria.max_price):
            continue
        if criteria.require_fno and not inst.is_fno:
            continue
        eligible.append(inst)

    eligible.sort(key=lambda i: i.avg_daily_value or 0.0, reverse=True)
    selected = eligible[: criteria.target_size]

    if len(selected) < criteria.target_size:
        log.warning(
            "Screen produced %d instruments, below the target of %d. Statistical power "
            "in backtesting scales with trade count, which scales with universe size "
            "(DESIGN.md D5) — consider relaxing the criteria.",
            len(selected), criteria.target_size,
        )

    universe = Universe(
        instruments=selected,
        screened_on=screened_on,
        criteria_summary=criteria.describe(),
    )
    log.info("Universe %s: %d instruments [%s]",
             universe.version, len(universe), ", ".join(i.trading_symbol for i in selected))
    return universe


# Sector groupings for the correlation filter. Coarse on purpose: the aim is to catch
# "three banks is really one bet", not to model a covariance matrix. A realised-
# correlation estimator would be better, but it needs history we do not yet have (D5) —
# and a rough grouping applied today beats a precise one that arrives after the drawdown.
SECTOR_GROUPS: Dict[str, tuple] = {
    "banking": ("HDFCBANK", "ICICIBANK", "SBIN", "AXISBANK", "KOTAKBANK", "INDUSINDBK",
                "BANKBARODA", "PNB", "FEDERALBNK", "IDFCFIRSTB", "AUBANK"),
    "it": ("TCS", "INFY", "WIPRO", "HCLTECH", "TECHM", "LTIM", "MPHASIS", "COFORGE"),
    "auto": ("MARUTI", "TATAMOTORS", "M&M", "BAJAJ-AUTO", "HEROMOTOCO", "EICHERMOT",
             "ASHOKLEY", "TVSMOTOR"),
    "energy": ("RELIANCE", "ONGC", "IOC", "BPCL", "GAIL", "HINDPETRO", "OIL"),
    "metals": ("TATASTEEL", "JSWSTEEL", "HINDALCO", "VEDL", "SAIL", "NMDC", "JINDALSTEL"),
    "pharma": ("SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "AUROPHARMA", "LUPIN"),
    "fmcg": ("HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "DABUR", "MARICO"),
    "financials": ("BAJFINANCE", "BAJAJFINSV", "HDFCLIFE", "SBILIFE", "ICICIPRULI",
                   "CHOLAFIN", "SHRIRAMFIN"),
    # Index ETFs correlate with everything; grouping them together stops the bot holding
    # an index ETF alongside a basket that is effectively the same exposure.
    "index_etf": ("NIFTYBEES", "BANKBEES", "JUNIORBEES", "SETFNIF50"),
}

_SYMBOL_TO_GROUP: Dict[str, str] = {
    symbol: group for group, symbols in SECTOR_GROUPS.items() for symbol in symbols
}


def correlation_group_for(trading_symbol: str) -> str:
    """Sector group for a symbol, or "" if ungrouped (and so unconstrained).

    Strips the common ``-EQ`` suffix so scrip-master symbols match.
    """
    base = trading_symbol.upper().split("-")[0].strip()
    return _SYMBOL_TO_GROUP.get(base, "")


def assign_correlation_groups(instruments: Sequence[Instrument]) -> List[Instrument]:
    """Return copies with ``correlation_group`` populated from the sector map."""
    from dataclasses import replace

    return [
        replace(i, correlation_group=i.correlation_group
                or correlation_group_for(i.trading_symbol))
        for i in instruments
    ]


def min_position_granularity(inst: Instrument, target_notional: float) -> float:
    """Fraction of ``target_notional`` represented by a single share.

    The sizing-precision check behind D6's price ceiling. At 0.5 the position can only
    be 1 or 2 shares, so realised risk can be double the intended risk. Callers should
    treat anything above ~0.1 as a warning that this instrument is too coarse to size
    accurately at the current equity.
    """
    if target_notional <= 0 or inst.last_price is None or inst.last_price <= 0:
        return 1.0
    return min(1.0, inst.last_price / target_notional)
