"""Can a strategy trade at all on bars of a given size?

This exists because of a defect found the expensive way: a backtest on real NSE data
reported **zero trades**, from the strategy *and* from the negative control, and that read
as a market observation until it was traced.

Session-anchored indicators reset in `on_session_start`, so **warmup must fit inside a
single session** — it does not accumulate across days. At 15-minute bars an NSE session
holds 25 bars; `trend_pullback` needs 28 and `sma_crossover` needs 50. Both are then
structurally incapable of *ever* trading, and nothing said so. The bot would run all day,
log normally, pass every health check, and take no trades.

That is the same shape as the feed that reconnects forever and the consumer that falls
behind: healthy-looking and useless. The strategy engine now refuses to start rather than
producing that silence.
"""
from __future__ import annotations

import pytest

import config
from src import market_calendar as cal
from src.strategies.sma_crossover import SmaCrossover
from src.strategies.trend_pullback import TrendPullback
from src.strategy_base import Strategy

SESSION_SECONDS = 375 * 60


class Warmup(Strategy):
    """A strategy with a settable warmup and no behaviour."""

    name = "warmup_probe"

    def __init__(self, warmup: int = 0):
        super().__init__(warmup=warmup)
        self._warmup = warmup

    @property
    def warmup_bars(self) -> int:
        return self._warmup

    def on_bar(self, bar, ctx):
        return []


class TestBarsPerSession:
    @pytest.mark.parametrize("seconds,expected", [
        (60, 375), (300, 75), (900, 25), (1800, 12), (3600, 6),
    ])
    def test_the_arithmetic_matches_the_nse_session(self, seconds, expected):
        assert Warmup().bars_per_session(seconds) == expected

    def test_it_uses_the_calendar_not_a_hardcoded_number(self):
        span = (cal.SESSION_CLOSE.hour * 60 + cal.SESSION_CLOSE.minute) - (
            cal.SESSION_OPEN.hour * 60 + cal.SESSION_OPEN.minute)
        assert Warmup().bars_per_session(60) == span


class TestTheDefectItCatches:
    """The precise cases that produced zero trades on real data."""

    def test_trend_pullback_cannot_trade_at_15_minutes(self):
        ok, note = TrendPullback().check_interval(900)
        assert ok is False
        assert "NEVER trade" in note

    def test_sma_crossover_cannot_trade_at_15_minutes(self):
        ok, _ = SmaCrossover().check_interval(900)
        assert ok is False

    def test_both_can_trade_at_5_minutes(self):
        assert TrendPullback().check_interval(300)[0] is True
        assert SmaCrossover().check_interval(300)[0] is True

    def test_the_shipped_configuration_is_feasible(self):
        """The one that matters: whatever STRATEGY_BAR_SECONDS is set to must work for
        the configured strategy, or the bot silently does nothing."""
        ok, _ = TrendPullback().check_interval(config.STRATEGY_BAR_SECONDS)
        assert ok, f"STRATEGY_BAR_SECONDS={config.STRATEGY_BAR_SECONDS} is unusable"

    def test_the_message_names_a_workable_interval(self):
        """An error that says only 'this is broken' makes the reader do arithmetic at
        exactly the moment they are least able to."""
        _, note = TrendPullback().check_interval(900)
        assert "<=" in note and "s)" in note

    def test_the_suggested_interval_actually_works(self):
        strategy = TrendPullback()
        assert strategy.check_interval(strategy._max_interval_seconds())[0] is True


class TestBoundaries:
    def test_warmup_equal_to_session_length_is_refused(self):
        """Exactly enough warmup leaves zero usable bars, which is not 'just barely
        works' — it is 'never trades'."""
        assert Warmup(warmup=25).check_interval(900)[0] is False

    def test_one_usable_bar_is_permitted_but_flagged(self):
        ok, note = Warmup(warmup=24).check_interval(900)
        assert ok is True
        assert "usable bars per session" in note

    def test_a_strategy_with_no_warmup_is_always_feasible(self):
        assert Warmup(warmup=0).check_interval(3600) == (True, "")

    def test_a_comfortable_margin_produces_no_note(self):
        assert Warmup(warmup=10).check_interval(60) == (True, "")

    def test_usable_bars_goes_negative_when_impossible(self):
        assert Warmup(warmup=50).usable_bars_per_session(900) == -25


class TestTheEngineRefuses:
    def test_it_does_not_start_on_an_unusable_interval(self, monkeypatch, caplog):
        """Refusing is the point. Running would mean a full session of healthy-looking
        silence."""
        import fakeredis

        from src import event_bus
        from src.strategy_engine import StrategyEngine

        client = fakeredis.FakeStrictRedis(server=fakeredis.FakeServer(),
                                           decode_responses=True)
        monkeypatch.setattr(event_bus, "get_client", lambda *a, **k: client)
        monkeypatch.setattr(event_bus, "ping", lambda *a, **k: True)
        monkeypatch.setattr(config, "STRATEGY_BAR_SECONDS", 900)

        engine = StrategyEngine(TrendPullback())

        def must_not_consume(**kwargs):
            raise AssertionError("entered the consume loop on an unusable interval")

        monkeypatch.setattr(engine.position_updates, "read", must_not_consume)
        with caplog.at_level("ERROR"):
            engine.run()
        assert "CANNOT TRADE AT THIS BAR INTERVAL" in caplog.text

    def test_it_starts_normally_on_a_usable_interval(self, monkeypatch):
        import fakeredis

        from src import event_bus
        from src.strategy_engine import StrategyEngine

        client = fakeredis.FakeStrictRedis(server=fakeredis.FakeServer(),
                                           decode_responses=True)
        monkeypatch.setattr(event_bus, "get_client", lambda *a, **k: client)
        monkeypatch.setattr(event_bus, "ping", lambda *a, **k: True)
        monkeypatch.setattr(config, "STRATEGY_BAR_SECONDS", 300)

        engine = StrategyEngine(TrendPullback())
        reached = []

        class Stop(RuntimeError):
            pass

        def read(**kwargs):
            reached.append(1)
            raise Stop()

        monkeypatch.setattr(engine.position_updates, "read", read)
        with pytest.raises(Stop):
            engine.run()
        assert reached == [1]
