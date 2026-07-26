"""TradingView momentum shortlist tests.

This module exists to solve a bootstrapping problem: `momentum_screen` needs sixty sessions
of price history per name and the Parquet store starts **empty**, filling at one session per
day. Waiting three months to learn which stocks are trending is not a plan. One HTTP request
returns three thousand NSE stocks with performance, volatility and relative volume attached.

Everything worth testing here follows from that being a *convenience* rather than a
dependency:

* **It never raises.** A screener that is down returns an empty result and the caller keeps
  the watchlist it already had. Refusing to trade because an undocumented endpoint is
  unreachable would be a worse failure than the one being reported.
* **Empty means unavailable, not "no stocks qualify".** Those need opposite responses, and
  conflating them would have a network blip silently empty the watchlist.
* **Exclusions are tallied.** A screen that quietly returns forty names when asked for two
  hundred is indistinguishable from a broken one (§5.7).
* **Symbols are never guessed into instrument ids.** A guessed token is an order on the
  wrong instrument (B4).

The network is never touched. Every test drives `apply_screen` or injects a fake fetch.
"""
from __future__ import annotations

import datetime as dt
import json
import urllib.error

import pytest

import config
from src.tv_screener import (
    ScreenCriteria, ScreenResult, ScreenedStock, apply_screen, screen,
    to_instrument_ids,
)


def stock(name="RELIANCE", *, close=1300.0, perf_3m=12.0, volatility_m=2.0,
          rvol=1.0, avg_volume_10d=10_000_000.0, sector="Energy Minerals",
          perf_1m=4.0, market_cap=1e13) -> ScreenedStock:
    return ScreenedStock(
        symbol=f"NSE:{name}", name=name, close=close, volume=avg_volume_10d,
        perf_1m=perf_1m, perf_3m=perf_3m, volatility_m=volatility_m, rvol=rvol,
        avg_volume_10d=avg_volume_10d, sector=sector, market_cap=market_cap)


@pytest.fixture(autouse=True)
def _isolated_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "SCRIP_MASTER_DIR", str(tmp_path))


# ---------------------------------------------------------------------------
class TestRiskAdjustedRanking:
    def test_the_key_is_performance_over_volatility(self):
        s = stock(perf_3m=12.0, volatility_m=3.0)
        assert s.risk_adjusted == pytest.approx(4.0)

    def test_a_steadier_riser_outranks_a_wilder_one_with_a_bigger_move(self):
        """Ranking on raw performance selects for volatility, and those names blow
        through ATR stops at a rate the cost model says is unaffordable."""
        result = apply_screen([
            stock("WILD", perf_3m=30.0, volatility_m=10.0),   # ra = 3.0
            stock("CALM", perf_3m=12.0, volatility_m=2.0),    # ra = 6.0
        ])
        assert [s.name for s in result.stocks] == ["CALM", "WILD"]

    def test_zero_volatility_does_not_divide_by_zero(self):
        assert stock(volatility_m=0.0).risk_adjusted == 0.0

    def test_turnover_is_price_times_average_volume(self):
        assert stock(close=100.0, avg_volume_10d=1_000_000.0).turnover == 1e8


class TestFiltering:
    def test_penny_stocks_are_excluded(self):
        result = apply_screen([stock("CHEAP", close=5.0)])
        assert len(result) == 0
        assert "price outside range" in result.excluded

    def test_illiquid_names_are_excluded(self):
        result = apply_screen([stock("THIN", avg_volume_10d=100.0)])
        assert "below liquidity floor" in result.excluded

    def test_untradeably_volatile_names_are_excluded(self):
        """A 40%-monthly-volatility name cannot be sized sanely against a 0.5% risk
        budget without the stop being either absurdly wide or instantly hit."""
        result = apply_screen([stock("WILD", volatility_m=40.0)])
        assert "too volatile to size sanely" in result.excluded

    def test_falling_stocks_are_excluded_when_long_only(self):
        result = apply_screen([stock("FALLING", perf_3m=-8.0)])
        assert "no positive momentum (long-only, D3)" in result.excluded

    def test_the_long_only_filter_can_be_disabled(self):
        result = apply_screen([stock("FALLING", perf_3m=-8.0)],
                              ScreenCriteria(require_positive_momentum=False))
        assert len(result) == 1

    def test_the_rvol_floor_filters(self):
        result = apply_screen([stock("QUIET", rvol=0.2)],
                              ScreenCriteria(min_rvol=0.8))
        assert "insufficient relative volume" in result.excluded

    def test_a_row_without_a_price_is_excluded(self):
        result = apply_screen([stock("BROKEN", close=0.0)])
        assert "no price" in result.excluded

    def test_exclusions_are_tallied_by_reason(self):
        """A screen that quietly returns forty names when asked for two hundred is
        indistinguishable from a broken one (§5.7)."""
        result = apply_screen([stock(f"P{i}", close=5.0) for i in range(7)])
        assert result.excluded["price outside range"] == 7


class TestTopN:
    def test_it_returns_at_most_top_n(self):
        universe = [stock(f"S{i}", perf_3m=float(i)) for i in range(50)]
        assert len(apply_screen(universe, ScreenCriteria(top_n=10))) == 10

    def test_two_hundred_is_the_requested_shape(self):
        universe = [stock(f"S{i}", perf_3m=1.0 + i * 0.1) for i in range(400)]
        assert len(apply_screen(universe, ScreenCriteria(top_n=200))) == 200

    def test_the_best_are_kept(self):
        universe = [stock(f"S{i}", perf_3m=float(i + 1), volatility_m=1.0)
                    for i in range(20)]
        assert apply_screen(universe, ScreenCriteria(top_n=1)).stocks[0].name == "S19"

    def test_the_universe_size_is_recorded(self):
        result = apply_screen([stock(f"S{i}") for i in range(5)])
        assert result.universe_size == 5


class TestItNeverRaises:
    """A screener being down is not a reason to stop trading a watchlist that exists."""

    def test_a_network_failure_returns_an_empty_result(self, monkeypatch, caplog):
        def dead(*a, **k):
            raise urllib.error.URLError("no route to host")

        monkeypatch.setattr("src.tv_screener.fetch_universe", dead)
        with caplog.at_level("ERROR"):
            result = screen(use_cache=False)
        assert len(result) == 0
        assert "unchanged" in caplog.text

    def test_a_timeout_returns_an_empty_result(self, monkeypatch):
        def slow(*a, **k):
            raise TimeoutError("timed out")

        monkeypatch.setattr("src.tv_screener.fetch_universe", slow)
        assert len(screen(use_cache=False)) == 0

    def test_malformed_json_returns_an_empty_result(self, monkeypatch):
        def garbage(*a, **k):
            raise json.JSONDecodeError("bad", "", 0)

        monkeypatch.setattr("src.tv_screener.fetch_universe", garbage)
        assert len(screen(use_cache=False)) == 0

    def test_zero_rows_is_treated_as_unavailable_not_as_an_empty_market(
            self, monkeypatch, caplog):
        """Those need opposite responses. Conflating them would have a network blip
        silently empty the watchlist."""
        monkeypatch.setattr("src.tv_screener.fetch_universe", lambda *a, **k: [])
        with caplog.at_level("ERROR"):
            result = screen(use_cache=False)
        assert result.universe_size == 0
        assert "rather than as an empty market" in caplog.text


class TestCaching:
    def test_a_successful_fetch_is_cached(self, monkeypatch):
        monkeypatch.setattr("src.tv_screener.fetch_universe",
                            lambda *a, **k: [stock("A"), stock("B")])
        screen(use_cache=False)

        calls = []

        def must_not_fetch(*a, **k):
            calls.append(1)
            return []

        monkeypatch.setattr("src.tv_screener.fetch_universe", must_not_fetch)
        second = screen(use_cache=True)
        assert calls == [], "it refetched instead of using the cache"
        assert second.from_cache is True
        assert len(second) == 2

    def test_the_cache_means_a_session_does_not_depend_on_the_endpoint_at_0900(
            self, monkeypatch):
        monkeypatch.setattr("src.tv_screener.fetch_universe",
                            lambda *a, **k: [stock("A")])
        screen(use_cache=False)

        def dead(*a, **k):
            raise urllib.error.URLError("down")

        monkeypatch.setattr("src.tv_screener.fetch_universe", dead)
        assert len(screen(use_cache=True)) == 1

    def test_a_corrupt_cache_falls_back_to_fetching(self, monkeypatch, tmp_path,
                                                    caplog):
        from src.tv_screener import _cache_path

        path = _cache_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("not json at all", encoding="utf-8")

        monkeypatch.setattr("src.tv_screener.fetch_universe",
                            lambda *a, **k: [stock("A")])
        with caplog.at_level("WARNING"):
            result = screen(use_cache=True)
        assert len(result) == 1
        assert "refetching" in caplog.text

    def test_the_cache_is_keyed_by_day(self):
        from src.tv_screener import _cache_path

        assert _cache_path(dt.date(2026, 3, 2)) != _cache_path(dt.date(2026, 3, 3))


class TestSymbolResolution:
    def _master(self):
        from src.instruments import InstrumentMaster, parse_scrip_master

        return InstrumentMaster(parse_scrip_master([
            {"pSymbol": "2885", "pTrdSymbol": "RELIANCE-EQ", "pSymbolName": "RELIANCE",
             "pGroup": "EQ", "lLotSize": "1", "dTickSize": "5"},
        ]))

    def test_it_resolves_to_canonical_ids(self):
        result = apply_screen([stock("RELIANCE")])
        assert to_instrument_ids(result, self._master()) == ["nse_cm:2885"]

    def test_without_a_master_it_refuses_rather_than_guessing(self, caplog):
        """A guessed token is an order on the wrong instrument (B4)."""
        result = apply_screen([stock("RELIANCE")])
        with caplog.at_level("ERROR"):
            assert to_instrument_ids(result, None) == []
        assert "B4" in caplog.text

    def test_unresolvable_symbols_are_dropped_and_counted(self, caplog):
        result = apply_screen([stock("RELIANCE"), stock("NOTLISTED")])
        with caplog.at_level("WARNING"):
            ids = to_instrument_ids(result, self._master())
        assert ids == ["nse_cm:2885"]
        assert "Resolved 1 of 2" in caplog.text


class TestReporting:
    def test_the_report_labels_the_universe_as_a_proxy(self):
        """It is the top N NSE stocks by market cap, not the official constituent list,
        and will drift from it between rebalances."""
        result = apply_screen([stock("A")])
        assert "proxy for Nifty 500" in result.report()

    def test_it_distinguishes_a_cached_run_from_a_live_one(self):
        result = apply_screen([stock("A")])
        assert "(live)" in result.report()
        result.from_cache = True
        assert "(cache)" in result.report()

    def test_every_ranking_component_is_visible(self):
        text = stock("RELIANCE").describe()
        for token in ("3M=", "vol=", "ra=", "rvol="):
            assert token in text

    def test_the_symbols_list_is_exposed_for_subscription(self):
        result = apply_screen([stock("A"), stock("B")])
        assert result.symbols == ["NSE:A", "NSE:B"]
