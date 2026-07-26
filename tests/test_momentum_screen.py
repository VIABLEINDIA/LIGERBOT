"""Momentum ranking tests.

The existing liquidity screen answers *"can this be traded?"*. This answers *"is it worth
trading today?"*, and the tests are built around the four decisions that make it a ranking
rather than a sort:

* **Risk-adjusted, not raw.** Sorting by biggest move selects for volatility, and volatile
  names blow through ATR stops at a rate the cost model already says is unaffordable.
* **Recent bars skipped.** Short-term reversal contaminates raw momentum; a stock that ran
  hard yesterday is as likely to give it back as continue.
* **Trend quality scored separately.** A 20% earnings gap has enormous "momentum" and no
  trend. Ranking it highly hands a pullback strategy an instrument with no pullbacks in it.
* **Fails closed.** No history means excluded, never ranked at the bottom.
"""
from __future__ import annotations

import datetime as dt
import math

import pytest

from src.momentum_screen import (
    Direction, MIN_BARS, MomentumCriteria, RankedUniverse, rank, score, trend_quality,
)


def steady(n: int = 80, start: float = 100.0, per_bar: float = 0.004) -> list[float]:
    """A clean compounding advance — the shape momentum is supposed to find."""
    return [start * (1 + per_bar) ** i for i in range(n)]


def choppy(n: int = 80, start: float = 100.0, amplitude: float = 0.05) -> list[float]:
    """Same endpoints, no trend: a sine wave with a slight drift."""
    return [start * (1 + 0.004 * i + amplitude * math.sin(i / 3.0)) for i in range(n)]


def gapped(n: int = 80, start: float = 100.0, jump: float = 0.25) -> list[float]:
    """Flat, one enormous gap, flat again. Huge return, no trend."""
    half = n // 2
    return [start] * half + [start * (1 + jump)] * (n - half)


class TestRiskAdjustment:
    def test_the_ranking_key_is_return_over_volatility(self):
        s = score("x", steady(80, per_bar=0.004))
        assert s.risk_adjusted == pytest.approx(s.lookback_return / s.volatility, rel=1e-6)

    def test_a_calm_riser_outranks_a_wilder_one_with_a_bigger_move(self):
        """The whole point. Raw return would pick the volatile name, whose stops then get
        hit at a rate the cost model says is unaffordable."""
        calm = steady(80, per_bar=0.004)                       # smooth
        wild = choppy(80, amplitude=0.08)                      # bigger swings

        universe = rank({"calm": calm, "wild": wild},
                        criteria=MomentumCriteria(skip_bars=0, top_n=2))
        assert universe.instrument_ids[0] == "calm"

    def test_zero_volatility_does_not_divide_by_zero(self):
        s = score("flat", [100.0] * 80)
        assert s.risk_adjusted == 0.0
        assert not s.usable


class TestSkippingRecentBars:
    def test_the_skipped_window_is_excluded_from_the_return(self):
        """A spike in the skipped bars must not inflate the score."""
        base = steady(80, per_bar=0.002)
        spiked = base + [base[-1] * 1.30]

        without = score("x", base, criteria=MomentumCriteria(skip_bars=0))
        with_spike = score("x", spiked, criteria=MomentumCriteria(skip_bars=1))
        assert with_spike.lookback_return == pytest.approx(without.lookback_return,
                                                           rel=1e-9)

    def test_skip_zero_includes_everything(self):
        series = steady(80)
        s = score("x", series, criteria=MomentumCriteria(skip_bars=0, lookback_bars=80))
        assert s.bars_used == 80

    def test_skipping_more_than_the_series_is_survivable(self):
        s = score("x", [100.0, 101.0], criteria=MomentumCriteria(skip_bars=10))
        assert not s.usable


class TestTrendQuality:
    def test_a_clean_advance_scores_near_one(self):
        assert trend_quality(steady(80)) > 0.99

    def test_a_choppy_series_scores_lower(self):
        assert trend_quality(choppy(80, amplitude=0.08)) < trend_quality(steady(80))

    def test_a_single_gap_scores_poorly_despite_a_huge_return(self):
        """The case that matters: enormous momentum by any return measure, and nothing
        for a pullback strategy to work with."""
        series = gapped(80, jump=0.25)
        s = score("x", series, criteria=MomentumCriteria(skip_bars=0))
        assert s.lookback_return > 0.20
        assert s.trend_quality < 0.80

    def test_a_flat_series_has_no_trend_rather_than_an_error(self):
        assert trend_quality([100.0] * 50) == 0.0

    def test_degenerate_input_is_survivable(self):
        assert trend_quality([]) == 0.0
        assert trend_quality([100.0]) == 0.0
        assert trend_quality([0.0, -5.0]) == 0.0

    def test_it_breaks_ties_between_equal_risk_adjusted_scores(self):
        clean = steady(80, per_bar=0.003)
        noisy = [c * (1 + 0.01 * ((-1) ** i)) for i, c in enumerate(clean)]
        universe = rank({"clean": clean, "noisy": noisy},
                        criteria=MomentumCriteria(skip_bars=0, top_n=2))
        assert universe.instrument_ids[0] == "clean"


class TestItFailsClosed:
    def test_too_little_history_is_excluded_not_ranked_last(self):
        """Trading a name because its data was missing is the mistake the liquidity
        screen already refuses to make."""
        universe = rank({"short": steady(5), "long": steady(80)},
                        criteria=MomentumCriteria(skip_bars=0))
        assert universe.instrument_ids == ["long"]
        assert "short" in universe.excluded
        assert "insufficient history" in universe.excluded["short"]

    def test_the_minimum_bar_count_is_enforced(self):
        assert not score("x", steady(MIN_BARS - 1),
                         criteria=MomentumCriteria(skip_bars=0)).usable
        assert score("x", steady(MIN_BARS),
                    criteria=MomentumCriteria(skip_bars=0)).usable

    def test_an_empty_universe_is_reported_not_hidden(self):
        universe = rank({"a": steady(5), "b": steady(3)},
                        criteria=MomentumCriteria(skip_bars=0))
        assert len(universe) == 0
        assert len(universe.excluded) == 2
        assert "Excluded" in universe.report()

    def test_no_candidates_at_all_is_survivable(self):
        assert len(rank({})) == 0


class TestDirectionFilter:
    def test_long_only_excludes_downtrends(self):
        """D3: long-only for v1. A falling stock has momentum and nothing this bot can
        do with it."""
        down = [100.0 * (0.996 ** i) for i in range(80)]
        universe = rank({"up": steady(80), "down": down},
                        criteria=MomentumCriteria(skip_bars=0,
                                                  require_direction=Direction.UP))
        assert universe.instrument_ids == ["up"]
        assert universe.excluded["down"] == "direction down"

    def test_direction_can_be_disabled(self):
        down = [100.0 * (0.996 ** i) for i in range(80)]
        universe = rank({"down": down},
                        criteria=MomentumCriteria(skip_bars=0, require_direction=None))
        assert len(universe) == 1

    def test_a_flat_series_is_neither_up_nor_down(self):
        series = [100.0 + 0.0001 * i for i in range(80)]
        assert score("x", series,
                     criteria=MomentumCriteria(skip_bars=0)).direction is Direction.FLAT


class TestRelativeVolume:
    def test_rvol_compares_recent_volume_to_the_window(self):
        closes = steady(80)
        volumes = [1000.0] * 75 + [3000.0] * 5
        s = score("x", closes, volumes, criteria=MomentumCriteria(skip_bars=0))
        assert s.rvol > 2.0

    def test_flat_volume_gives_rvol_near_one(self):
        s = score("x", steady(80), [1000.0] * 80,
                  criteria=MomentumCriteria(skip_bars=0))
        assert s.rvol == pytest.approx(1.0, abs=0.05)

    def test_missing_volume_defaults_to_neutral(self):
        """Absent volume must not exclude an instrument — that is a data gap, not a
        signal about the instrument."""
        assert score("x", steady(80),
                     criteria=MomentumCriteria(skip_bars=0)).rvol == 1.0

    def test_the_rvol_floor_filters(self):
        closes = steady(80)
        universe = rank(
            {"quiet": closes, "busy": closes},
            {"quiet": [1000.0] * 80, "busy": [1000.0] * 75 + [5000.0] * 5},
            criteria=MomentumCriteria(skip_bars=0, min_rvol=1.5))
        assert universe.instrument_ids == ["busy"]


class TestTopN:
    def test_it_returns_at_most_top_n(self):
        universe = rank({f"i{n}": steady(80, per_bar=0.001 * (n + 1)) for n in range(50)},
                        criteria=MomentumCriteria(skip_bars=0, top_n=10))
        assert len(universe) == 10

    def test_the_best_are_the_ones_kept(self):
        candidates = {f"i{n}": steady(80, per_bar=0.001 * (n + 1)) for n in range(20)}
        universe = rank(candidates, criteria=MomentumCriteria(skip_bars=0, top_n=3))
        assert universe.instrument_ids[0] == "i19"

    def test_two_hundred_is_a_supported_size(self):
        """The requested shape: screen 200, trade 3. The ranking is a watchlist, not a
        portfolio — the risk engine still gates every individual entry."""
        candidates = {f"i{n}": steady(80, per_bar=0.0005 * (n + 1)) for n in range(500)}
        universe = rank(candidates, criteria=MomentumCriteria(skip_bars=0, top_n=200))
        assert len(universe) == 200

    def test_scores_are_ordered_best_first(self):
        candidates = {f"i{n}": steady(80, per_bar=0.001 * (n + 1)) for n in range(10)}
        universe = rank(candidates, criteria=MomentumCriteria(skip_bars=0, top_n=10))
        keys = [s.risk_adjusted for s in universe.scores]
        assert keys == sorted(keys, reverse=True)


class TestReporting:
    def test_the_report_explains_the_ranking(self):
        """A ranking nobody can interrogate is one nobody can debug when it starts
        choosing badly."""
        universe = rank({"a": steady(80), "b": choppy(80)},
                        criteria=MomentumCriteria(skip_bars=0), day=dt.date(2026, 3, 2))
        text = universe.report()
        assert "2026-03-02" in text
        assert "R²" in text and "rvol" in text

    def test_exclusions_are_tallied_by_reason(self):
        universe = rank({"a": steady(5), "b": steady(4), "c": steady(80)},
                        criteria=MomentumCriteria(skip_bars=0))
        assert "2" in universe.report()

    def test_every_component_survives_into_the_score(self):
        s = score("x", steady(80), [1000.0] * 80, criteria=MomentumCriteria(skip_bars=0))
        for token in ("ret=", "vol=", "risk_adj=", "R²=", "rvol="):
            assert token in s.describe()


class TestStoreIntegration:
    def test_it_ranks_from_recorded_bars(self, tmp_path):
        """Our own Parquet store: no broker call, and it improves every day the bot runs."""
        import datetime as dtm

        from src import market_calendar as cal
        from src.bar_store import ParquetBarStore
        from src.bars import Bar
        from src.momentum_screen import rank_from_store

        store = ParquetBarStore(tmp_path, "1m")
        days = cal.trading_days_between(dtm.date(2026, 1, 1), dtm.date(2026, 4, 30))[:40]

        bars = []
        for i, day in enumerate(days):
            price = 100.0 * (1.004 ** i)
            start = cal.at(day, cal.SESSION_OPEN)
            for minute in range(5):
                bar_start = start + dtm.timedelta(minutes=minute)
                bars.append(Bar("nse_cm:2885", bar_start,
                                bar_start + dtm.timedelta(minutes=1),
                                price, price * 1.001, price * 0.999, price,
                                volume=10_000.0, vwap=price, tick_count=50))
        store.write(bars)

        universe = rank_from_store(
            ["nse_cm:2885"], store,
            criteria=MomentumCriteria(skip_bars=0, lookback_bars=40, top_n=10),
            day=days[-1], lookback_days=200)
        assert universe.instrument_ids == ["nse_cm:2885"]

    def test_a_missing_instrument_is_skipped_not_fatal(self, tmp_path):
        from src.bar_store import ParquetBarStore
        from src.momentum_screen import rank_from_store

        universe = rank_from_store(["nse_cm:9999"], ParquetBarStore(tmp_path, "1m"),
                                   day=dt.date(2026, 3, 2))
        assert len(universe) == 0
