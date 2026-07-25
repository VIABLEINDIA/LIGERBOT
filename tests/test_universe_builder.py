"""Universe builder tests (DESIGN.md D6).

The hole this closes was functional, not cosmetic: D6 specified "a screen, not a list", and
the screening logic existed — but nothing fetched the two inputs it filters on. The scrip
master carries no price and no volume, so `apply_screen` correctly excluded everything and
returned an **empty universe**. Even with the probe run and the master downloaded, there
would have been nothing to subscribe to and nothing to backfill.

Two properties are load-bearing here:

* **The screen fails closed.** An instrument whose liquidity cannot be established is
  excluded, never assumed adequate. Trading an illiquid name because its data was missing
  is worse than trading a shorter list.
* **Provenance travels with the universe.** A screen run on a one-day quote proxy is a
  weaker claim than one run on sixty days of recorded bars, and must not be reported as
  though the two were equivalent.
"""
from __future__ import annotations

import datetime as dt

import pytest

from src.instruments import Instrument, InstrumentMaster, ScreenCriteria, parse_scrip_master
from src.universe_builder import (
    LiquidityReading, LiquiditySource, build_universe, enrich, liquidity_from_quote,
    measure_liquidity,
)

DAY = dt.date(2026, 7, 23)


def scrip_rows(n: int = 3):
    names = ["RELIANCE", "HDFCBANK", "INFY", "SBIN", "TCS", "ITC"][:n]
    return [{"pSymbol": str(2000 + i), "pTrdSymbol": f"{name}-EQ",
             "pSymbolName": name, "pGroup": "EQ", "lLotSize": "1", "dTickSize": "5"}
            for i, name in enumerate(names)]


class FakeNeo:
    """Returns quotes keyed by token, as the SDK does.

    The payload is stored as ``_by_token`` rather than ``quotes`` — an attribute of that
    name would shadow the method, and ``safe_call`` would (correctly) report that the
    client has no callable ``quotes``.
    """

    def __init__(self, by_token=None, raises=None):
        self._by_token = by_token or {}
        self.raises = raises
        self.calls = 0

    def quotes(self, instrument_tokens=None, quote_type=None):
        self.calls += 1
        if self.raises:
            raise self.raises
        rows = []
        for entry in instrument_tokens or []:
            token = entry["instrument_token"]
            if token in self._by_token:
                rows.append({"tk": token, **self._by_token[token]})
        return {"data": rows}


class TestTheHoleItCloses:
    def test_bare_scrip_master_screens_to_nothing(self):
        """The bug: the master carries no price or volume, so everything is excluded."""
        from src.instruments import apply_screen

        parsed = parse_scrip_master(scrip_rows())
        assert all(i.last_price is None for i in parsed)
        assert len(apply_screen(parsed, ScreenCriteria(target_size=12))) == 0

    def test_enriched_master_screens_to_a_real_universe(self):
        parsed = parse_scrip_master(scrip_rows())
        readings = {
            i.instrument_id: LiquidityReading(
                i.instrument_id, last_price=1000.0,
                avg_daily_value=500_00_00_000.0,
                source=LiquiditySource.RECORDED_BARS, days_observed=20)
            for i in parsed
        }
        from src.instruments import apply_screen

        enriched = enrich(parsed, readings)
        universe = apply_screen(enriched, ScreenCriteria(require_fno=False, target_size=12))
        assert len(universe) == 3


class TestQuoteProxy:
    def test_traded_value_from_price_and_volume(self):
        reading = liquidity_from_quote("nse_cm:1", {"ltp": "1500.5", "v": "100000"})
        assert reading.avg_daily_value == pytest.approx(1500.5 * 100_000)
        assert reading.source is LiquiditySource.QUOTE_PROXY

    def test_alternate_field_names(self):
        reading = liquidity_from_quote("nse_cm:1",
                                       {"last_traded_price": "100", "volume": "50"})
        assert reading.usable

    def test_comma_formatted_numbers(self):
        assert liquidity_from_quote("nse_cm:1", {"ltp": "1,500", "v": "1,00,000"}).usable

    def test_missing_price_is_unusable(self):
        reading = liquidity_from_quote("nse_cm:1", {"v": "100000"})
        assert not reading.usable
        assert reading.source is LiquiditySource.UNAVAILABLE

    def test_missing_volume_keeps_price_but_is_unusable(self):
        """Fails closed: no volume means liquidity is unknown, not adequate."""
        reading = liquidity_from_quote("nse_cm:1", {"ltp": "100"})
        assert reading.last_price == 100.0
        assert reading.avg_daily_value is None
        assert not reading.usable

    def test_proxy_is_marked_untrustworthy(self):
        assert not LiquiditySource.QUOTE_PROXY.trustworthy
        assert LiquiditySource.RECORDED_BARS.trustworthy
        assert LiquiditySource.HISTORY.trustworthy


class TestMeasureLiquidity:
    def test_quotes_are_used_when_no_bars_exist(self, tmp_path):
        from src.bar_store import ParquetBarStore

        parsed = parse_scrip_master(scrip_rows(2))
        neo = FakeNeo({i.token: {"ltp": "1000", "v": "500000"} for i in parsed})
        readings = measure_liquidity(
            parsed, neo_client=neo, store=ParquetBarStore(tmp_path, "1m"))
        assert all(r.source is LiquiditySource.QUOTE_PROXY for r in readings.values())

    def test_every_instrument_gets_a_reading(self, tmp_path):
        """Even unquoted ones — absent means UNAVAILABLE, never silently skipped."""
        from src.bar_store import ParquetBarStore

        parsed = parse_scrip_master(scrip_rows(3))
        neo = FakeNeo({parsed[0].token: {"ltp": "1000", "v": "500000"}})
        readings = measure_liquidity(
            parsed, neo_client=neo, store=ParquetBarStore(tmp_path, "1m"))
        assert len(readings) == 3
        unavailable = [r for r in readings.values()
                       if r.source is LiquiditySource.UNAVAILABLE]
        assert len(unavailable) == 2

    def test_a_failed_quote_batch_excludes_rather_than_assumes(self, tmp_path):
        from src.bar_store import ParquetBarStore

        parsed = parse_scrip_master(scrip_rows(2))
        neo = FakeNeo(raises=ConnectionError("down"))
        readings = measure_liquidity(
            parsed, neo_client=neo, store=ParquetBarStore(tmp_path, "1m"))
        assert all(not r.usable for r in readings.values())

    def test_no_broker_still_returns_readings(self, tmp_path):
        from src.bar_store import ParquetBarStore

        parsed = parse_scrip_master(scrip_rows(2))
        readings = measure_liquidity(
            parsed, neo_client=None, store=ParquetBarStore(tmp_path, "1m"))
        assert len(readings) == 2


class TestRecordedBarsPreferred:
    def _store_with_bars(self, tmp_path, instrument_id, days=5):
        import datetime as dtm

        from src import market_calendar as cal
        from src.bar_store import ParquetBarStore
        from src.bars import Bar

        store = ParquetBarStore(tmp_path, "1m")
        trading_days = cal.trading_days_between(
            dtm.date(2026, 7, 1), dtm.date(2026, 7, 31))[:days]
        bars = []
        for day in trading_days:
            start = cal.at(day, cal.SESSION_OPEN)
            for minute in range(10):
                bar_start = start + dtm.timedelta(minutes=minute)
                bars.append(Bar(instrument_id, bar_start,
                                bar_start + dtm.timedelta(minutes=1),
                                1000.0, 1001.0, 999.0, 1000.0,
                                volume=10_000.0, vwap=1000.0, tick_count=100))
        store.write(bars)
        return store

    def test_recorded_bars_beat_the_quote_proxy(self, tmp_path):
        parsed = parse_scrip_master(scrip_rows(1))
        instrument = parsed[0]
        store = self._store_with_bars(tmp_path, instrument.instrument_id)

        neo = FakeNeo({instrument.token: {"ltp": "9999", "v": "1"}})
        readings = measure_liquidity([instrument], neo_client=neo, store=store)
        reading = readings[instrument.instrument_id]
        assert reading.source is LiquiditySource.RECORDED_BARS
        assert reading.days_observed == 5
        assert neo.calls == 0, "no quote call needed when bars already answer it"

    def test_recorded_value_is_an_average_across_days(self, tmp_path):
        parsed = parse_scrip_master(scrip_rows(1))
        instrument = parsed[0]
        store = self._store_with_bars(tmp_path, instrument.instrument_id)
        reading = measure_liquidity([instrument], store=store)[instrument.instrument_id]
        # 10 bars/day x 10,000 volume x 1000 price
        assert reading.avg_daily_value == pytest.approx(10 * 10_000 * 1000.0)


class TestBuildUniverse:
    def test_returns_a_universe_and_its_provenance(self, tmp_path):
        from src.bar_store import ParquetBarStore

        parsed = parse_scrip_master(scrip_rows(3))
        master = InstrumentMaster(parsed)
        neo = FakeNeo({i.token: {"ltp": "1000", "v": "5000000"} for i in parsed})

        universe, provenance = build_universe(
            master, neo_client=neo, store=ParquetBarStore(tmp_path, "1m"),
            criteria=ScreenCriteria(require_fno=False, target_size=12), day=DAY)

        assert len(universe) == 3
        assert provenance["candidates"] == 3
        assert provenance["selected"] == 3

    def test_a_proxy_universe_is_flagged_untrustworthy(self, tmp_path):
        """A one-day proxy must not be reported as though it were an average."""
        from src.bar_store import ParquetBarStore

        parsed = parse_scrip_master(scrip_rows(2))
        neo = FakeNeo({i.token: {"ltp": "1000", "v": "5000000"} for i in parsed})
        _, provenance = build_universe(
            InstrumentMaster(parsed), neo_client=neo,
            store=ParquetBarStore(tmp_path, "1m"),
            criteria=ScreenCriteria(require_fno=False, target_size=12), day=DAY)
        assert provenance["weakest_source_used"] == "quote_proxy"
        assert provenance["trustworthy"] is False

    def test_empty_universe_is_reported_not_hidden(self, tmp_path, caplog):
        from src.bar_store import ParquetBarStore

        parsed = parse_scrip_master(scrip_rows(2))
        with caplog.at_level("ERROR"):
            universe, provenance = build_universe(
                InstrumentMaster(parsed), neo_client=None,
                store=ParquetBarStore(tmp_path, "1m"), day=DAY)
        assert len(universe) == 0
        assert "fails closed" in caplog.text

    def test_correlation_groups_are_assigned(self, tmp_path):
        from src.bar_store import ParquetBarStore

        parsed = parse_scrip_master(scrip_rows(3))
        neo = FakeNeo({i.token: {"ltp": "1000", "v": "5000000"} for i in parsed})
        universe, _ = build_universe(
            InstrumentMaster(parsed), neo_client=neo,
            store=ParquetBarStore(tmp_path, "1m"),
            criteria=ScreenCriteria(require_fno=False, target_size=12), day=DAY)
        groups = {i.correlation_group for i in universe.instruments}
        assert "banking" in groups and "it" in groups

    def test_untradable_instruments_never_reach_the_screen(self, tmp_path):
        from src.bar_store import ParquetBarStore

        index = Instrument("nse_cm:11536", "11536", "NIFTY 50", "Nifty 50",
                           "nse_cm", "", 0, 0.0)
        parsed = parse_scrip_master(scrip_rows(1)) + [index]
        neo = FakeNeo({i.token: {"ltp": "1000", "v": "5000000"} for i in parsed})
        universe, provenance = build_universe(
            InstrumentMaster(parsed), neo_client=neo,
            store=ParquetBarStore(tmp_path, "1m"),
            criteria=ScreenCriteria(require_fno=False, target_size=12), day=DAY)
        assert provenance["candidates"] == 1
        assert "NIFTY 50" not in [i.trading_symbol for i in universe.instruments]
