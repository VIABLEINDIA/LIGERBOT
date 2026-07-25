"""Instrument master, screen and universe-validation tests — the fix for B4 and B5."""
from __future__ import annotations

import datetime as dt

import pytest

from src.instruments import (
    Instrument, InstrumentMaster, ScreenCriteria, apply_screen, load_scrip_master_csv,
    min_position_granularity, parse_scrip_master, validate_universe,
)


def equity(symbol="RELIANCE", token="2885", price=1_300.0, adv=500_00_00_000.0,
           series="EQ", fno=True, tick=0.05, lot=1) -> Instrument:
    return Instrument(
        instrument_id=f"nse_cm:{token}", token=token, trading_symbol=symbol,
        name=symbol, exchange_segment="nse_cm", series=series,
        lot_size=lot, tick_size=tick, last_price=price,
        avg_daily_value=adv, is_fno=fno,
    )


def index() -> Instrument:
    """The Nifty 50 index — the untradable default that shipped as B5."""
    return Instrument(
        instrument_id="nse_cm:11536", token="11536", trading_symbol="NIFTY 50",
        name="Nifty 50", exchange_segment="nse_cm", series="",
        lot_size=0, tick_size=0.0, last_price=24_500.0,
    )


class TestTradability:
    def test_equity_is_tradable(self):
        assert equity().is_tradable_cash

    def test_index_is_not_tradable(self):
        # B5: you cannot buy the index in the cash segment.
        assert not index().is_tradable_cash

    def test_zero_lot_or_tick_is_not_tradable(self):
        assert not equity(lot=0).is_tradable_cash
        assert not equity(tick=0.0).is_tradable_cash

    def test_non_eq_series_is_not_tradable(self):
        assert not equity(series="BE").is_tradable_cash


class TestValidateUniverse:
    def test_rejects_the_index_loudly(self):
        with pytest.raises(ValueError, match="NIFTYBEES"):
            validate_universe([index()])

    def test_rejects_trade_to_trade_series(self):
        with pytest.raises(ValueError, match="delivery-only"):
            validate_universe([equity(series="BE")])

    def test_non_strict_mode_filters_instead_of_raising(self):
        kept = validate_universe([equity(), index()], strict=False)
        assert [i.trading_symbol for i in kept] == ["RELIANCE"]

    def test_valid_universe_passes_through(self):
        universe = [equity("RELIANCE"), equity("INFY", token="1594")]
        assert len(validate_universe(universe)) == 2


class TestScreen:
    def test_filters_by_liquidity(self):
        candidates = [
            equity("LIQUID", token="1", adv=500_00_00_000.0),
            equity("ILLIQUID", token="2", adv=10_00_00_000.0),
        ]
        universe = apply_screen(candidates, ScreenCriteria(target_size=10))
        assert [i.trading_symbol for i in universe.instruments] == ["LIQUID"]

    def test_filters_by_price_ceiling(self):
        # The constraint that is easy to miss: a Rs 30,000 share cannot be sized
        # accurately against a risk-derived target notional.
        candidates = [equity("CHEAP", token="1", price=1_000.0),
                      equity("DEAR", token="2", price=30_000.0)]
        universe = apply_screen(candidates, ScreenCriteria(target_size=10))
        assert [i.trading_symbol for i in universe.instruments] == ["CHEAP"]

    def test_filters_by_price_floor(self):
        candidates = [equity("PENNY", token="1", price=5.0), equity("NORMAL", token="2")]
        universe = apply_screen(candidates, ScreenCriteria(target_size=10))
        assert [i.trading_symbol for i in universe.instruments] == ["NORMAL"]

    def test_fno_requirement(self):
        candidates = [equity("INFNO", token="1", fno=True),
                      equity("NOTFNO", token="2", fno=False)]
        assert len(apply_screen(candidates, ScreenCriteria(require_fno=True))) == 1
        assert len(apply_screen(candidates, ScreenCriteria(require_fno=False))) == 2

    def test_excludes_instruments_missing_screen_data(self):
        # Screening on absent data is how an illiquid name sneaks into the universe.
        missing = Instrument(
            instrument_id="nse_cm:9", token="9", trading_symbol="UNKNOWN", name="UNKNOWN",
            exchange_segment="nse_cm", series="EQ", lot_size=1, tick_size=0.05,
        )
        assert len(apply_screen([missing, equity()], ScreenCriteria())) == 1

    def test_ranks_by_liquidity_and_truncates(self):
        candidates = [
            equity(f"SYM{i}", token=str(i), adv=(i + 1) * 100_00_00_000.0)
            for i in range(8)
        ]
        universe = apply_screen(candidates, ScreenCriteria(target_size=3))
        assert [i.trading_symbol for i in universe.instruments] == ["SYM7", "SYM6", "SYM5"]

    def test_index_never_survives_the_screen(self):
        universe = apply_screen([index(), equity()], ScreenCriteria(target_size=10))
        assert "NIFTY 50" not in [i.trading_symbol for i in universe.instruments]

    def test_universe_is_version_stamped(self):
        # Backtests must record which universe they ran against, or they aren't
        # reproducible.
        day = dt.date(2026, 7, 23)
        first = apply_screen([equity()], ScreenCriteria(), screened_on=day)
        same = apply_screen([equity()], ScreenCriteria(), screened_on=day)
        different = apply_screen(
            [equity(), equity("INFY", token="1594")], ScreenCriteria(), screened_on=day
        )
        assert first.version == same.version
        assert first.version != different.version
        assert first.version.startswith("2026-07-23-")


class TestInstrumentMaster:
    def test_lookup_by_every_identifier(self):
        master = InstrumentMaster([equity()])
        assert master.by_id("nse_cm:2885").trading_symbol == "RELIANCE"
        assert master.by_symbol("reliance").token == "2885"
        assert master.by_token("2885").trading_symbol == "RELIANCE"
        assert master.resolve("RELIANCE") is not None

    def test_require_raises_on_unknown(self):
        # On the execution path a silent None becomes a malformed order.
        master = InstrumentMaster([equity()])
        with pytest.raises(KeyError, match="refusing to guess"):
            master.require("nse_cm:doesnotexist")

    def test_cache_round_trip(self, tmp_path):
        master = InstrumentMaster([equity(), equity("INFY", token="1594")])
        path = tmp_path / "master.json"
        master.save_cache(path)
        restored = InstrumentMaster.load_cache(path)
        assert len(restored) == 2
        assert restored.by_symbol("INFY").token == "1594"

    def test_missing_cache_returns_none(self, tmp_path):
        assert InstrumentMaster.load_cache(tmp_path / "nope.json") is None

    def test_corrupt_cache_returns_none_rather_than_raising(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{not json", encoding="utf-8")
        assert InstrumentMaster.load_cache(path) is None


class TestScripMasterParsing:
    def test_parses_kotak_columns(self):
        rows = [{
            "pSymbol": "2885", "pTrdSymbol": "RELIANCE-EQ", "pSymbolName": "RELIANCE",
            "pGroup": "EQ", "lLotSize": "1", "lTickSize": "5", "pISIN": "INE002A01018",
        }]
        parsed = parse_scrip_master(rows)
        assert len(parsed) == 1
        assert parsed[0].tick_size == pytest.approx(0.05)  # 5 -> Rs 0.05
        assert parsed[0].instrument_id == "nse_cm:2885"

    def test_skips_malformed_rows_without_failing(self):
        # One bad line in a 50,000-row broker file must not stop startup.
        rows = [
            {"pSymbol": "", "pTrdSymbol": ""},
            {"pSymbol": "2885", "pTrdSymbol": "RELIANCE-EQ", "lLotSize": "1",
             "lTickSize": "5"},
        ]
        assert len(parse_scrip_master(rows)) == 1

    def test_implausible_tick_size_falls_back(self):
        rows = [{"pSymbol": "1", "pTrdSymbol": "X", "lTickSize": "99999999"}]
        assert parse_scrip_master(rows)[0].tick_size == 0.05

    def test_loads_from_csv(self, tmp_path):
        csv_path = tmp_path / "scrip.csv"
        csv_path.write_text(
            "pSymbol,pTrdSymbol,pSymbolName,pGroup,lLotSize,lTickSize\n"
            "2885,RELIANCE-EQ,RELIANCE,EQ,1,5\n"
            "1594,INFY-EQ,INFY,EQ,1,5\n",
            encoding="utf-8",
        )
        parsed = load_scrip_master_csv(csv_path)
        assert {i.trading_symbol for i in parsed} == {"RELIANCE-EQ", "INFY-EQ"}


class TestTickRounding:
    def test_rounds_to_the_tick(self):
        inst = equity(tick=0.05)
        assert inst.round_to_tick(100.123) == pytest.approx(100.10)
        assert inst.round_to_tick(100.14) == pytest.approx(100.15)

    def test_already_aligned_price_is_unchanged(self):
        assert equity(tick=0.05).round_to_tick(100.05) == pytest.approx(100.05)


class TestGranularity:
    def test_expensive_share_is_coarse(self):
        # Rs 30,000 share against a Rs 60,000 target: only 2 shares fit, so realised
        # risk can be far from intended.
        assert min_position_granularity(equity(price=30_000.0), 60_000.0) == pytest.approx(0.5)

    def test_cheap_share_is_fine_grained(self):
        assert min_position_granularity(equity(price=100.0), 60_000.0) < 0.01

    def test_handles_missing_price(self):
        inst = Instrument("nse_cm:1", "1", "X", "X", "nse_cm", "EQ", 1, 0.05)
        assert min_position_granularity(inst, 10_000.0) == 1.0
