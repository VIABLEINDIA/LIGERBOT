"""Strategy engine tests, focused on the session boundary.

Found during the Phase 5 review. The live resampler only completes a bucket when the
*next* one opens, so the final bucket of a session has nothing to close it. Two
consequences, both live/backtest divergences:

1. The strategy never saw the last coarse bar of the day live, while the backtester did —
   a divergence at exactly the point square-off logic runs.
2. Worse: that stale bucket surfaced on the *next* session's first bar, **after**
   ``on_session_start`` had already reset the session-anchored indicators. Day two's VWAP
   was therefore anchored on day one's close, and the strategy received a bar stamped a
   whole session earlier.

Neither was caught by unit tests of the resampler in isolation, because in isolation the
behaviour is correct — a streaming resampler cannot know a session ended. The fix belongs
to the engine, and so do these tests.
"""
from __future__ import annotations

import datetime as dt
from typing import List

import fakeredis
import pytest

import config
from src import event_bus
from src import market_calendar as cal
from src.bars import Bar
from src.strategy_base import Strategy, StrategyContext

DAY1 = dt.date(2026, 7, 23)   # Thursday
DAY2 = dt.date(2026, 7, 24)   # Friday


class Recorder(Strategy):
    """Records every bar and session reset, in order."""

    name = "recorder"

    def __init__(self, **params):
        super().__init__(**params)
        self.events: List[tuple] = []

    def on_session_start(self, day):
        self.events.append(("session_start", day))

    def on_bar(self, bar, ctx: StrategyContext):
        self.events.append(("bar", bar.bar_start, bar.close))
        return []

    @property
    def bars_seen(self):
        return [e for e in self.events if e[0] == "bar"]


@pytest.fixture
def engine(monkeypatch):
    server = fakeredis.FakeServer()
    monkeypatch.setattr(
        event_bus, "get_client",
        lambda *a, **k: fakeredis.FakeStrictRedis(server=server, decode_responses=True))
    monkeypatch.setattr(event_bus, "ping", lambda *a, **k: True)
    monkeypatch.setattr(config, "STRATEGY_BAR_SECONDS", 300)

    from src.strategy_engine import StrategyEngine

    return StrategyEngine(Recorder())


def minute_bar(day: dt.date, offset: int, price: float) -> dict:
    start = cal.at(day, cal.SESSION_OPEN) + dt.timedelta(minutes=offset)
    return Bar("nse_cm:1", start, start + dt.timedelta(minutes=1),
               open=price, high=price + 1, low=price - 1, close=price,
               volume=1000.0, vwap=price, tick_count=50).to_event()


def feed_session(engine, day: dt.date, minutes: int, base: float = 100.0) -> None:
    for i in range(minutes):
        engine._handle_bar(minute_bar(day, i, base + i * 0.1))


class TestSessionBoundary:
    def test_final_bucket_of_the_session_is_emitted(self, engine):
        """375 one-minute bars must produce 75 five-minute bars, not 74."""
        feed_session(engine, DAY1, 375)
        # The last bucket is still held — nothing has closed it yet.
        assert len(engine.strategy.bars_seen) == 74

        # The next session's first bar must flush it.
        engine._handle_bar(minute_bar(DAY2, 0, 500.0))
        assert len(engine.strategy.bars_seen) == 75

    def test_stale_bucket_is_dispatched_before_the_session_resets(self, engine):
        """The ordering bug: yesterday's bar must not land after today's reset.

        If it does, the session-anchored indicators (VWAP, opening range, relative
        volume) have already been cleared, and yesterday's close silently becomes the
        first observation of today's series.
        """
        feed_session(engine, DAY1, 375)
        engine._handle_bar(minute_bar(DAY2, 0, 500.0))

        events = engine.strategy.events
        reset_indices = [i for i, e in enumerate(events)
                         if e[0] == "session_start" and e[1] == DAY2]
        assert reset_indices, "day 2 session never started"
        day2_reset = reset_indices[0]

        stale = [i for i, e in enumerate(events)
                 if e[0] == "bar" and e[1].date() == DAY1 and i > day2_reset]
        assert not stale, (
            f"{len(stale)} bar(s) from {DAY1} were delivered AFTER "
            f"on_session_start({DAY2}) — the new day's VWAP would be anchored on "
            f"yesterday's close")

    def test_every_bar_is_dispatched_within_its_own_session(self, engine):
        feed_session(engine, DAY1, 375)
        feed_session(engine, DAY2, 60)

        events = engine.strategy.events
        current = None
        for event in events:
            if event[0] == "session_start":
                current = event[1]
            else:
                assert event[1].date() == current, (
                    f"bar dated {event[1].date()} dispatched during session {current}")

    def test_no_bars_are_lost_across_the_boundary(self, engine):
        feed_session(engine, DAY1, 375)
        feed_session(engine, DAY2, 375)
        # 75 per session, and the final bucket of day 2 is still held.
        assert len(engine.strategy.bars_seen) == 149

    def test_session_start_fires_once_per_day(self, engine):
        feed_session(engine, DAY1, 30)
        feed_session(engine, DAY2, 30)
        starts = [e for e in engine.strategy.events if e[0] == "session_start"]
        assert [s[1] for s in starts] == [DAY1, DAY2]


class TestLiveMatchesBacktestResampling:
    """Resampling exists in two implementations, and they must not drift.

    - backtest: ``src.backtest.bar_source.resample_frame``  (batch, pandas)
    - live:     ``src.strategy_engine.BarResampler``        (streaming)

    Their shapes are genuinely different — one sees a whole frame, the other one bar at a
    time — so sharing an implementation is awkward. Keeping both is defensible only while
    something proves they agree; otherwise every backtest is measured against different
    bars than live sees, which is the divergence DESIGN.md 2.1 exists to prevent.
    """

    @staticmethod
    def _live(bars, interval):
        from src.strategy_engine import BarResampler

        resampler = BarResampler(interval)
        out = []
        for bar in bars:
            done = resampler.add(bar)
            if done is not None:
                out.append(done)
        out.extend(resampler.flush())      # include the final held bucket
        return out

    @staticmethod
    def _backtest(bars, interval):
        import pandas as pd

        from src.backtest.bar_source import resample_frame

        frame = pd.DataFrame([{
            "instrument_id": b.instrument_id, "bar_start": b.bar_start,
            "bar_end": b.bar_end, "open": b.open, "high": b.high, "low": b.low,
            "close": b.close, "volume": b.volume, "vwap": b.vwap,
            "tick_count": b.tick_count, "synthetic": b.synthetic,
        } for b in bars])
        frame["bar_start"] = pd.to_datetime(frame["bar_start"])
        frame["bar_end"] = pd.to_datetime(frame["bar_end"])
        return resample_frame(frame, interval)

    def _assert_agree(self, bars, interval):
        import pandas as pd

        live = self._live(bars, interval)
        backtest = self._backtest(bars, interval)
        assert len(live) == len(backtest), (
            f"bar count differs: live {len(live)} vs backtest {len(backtest)}")

        for i, lb in enumerate(live):
            row = backtest.iloc[i]
            assert pd.Timestamp(row["bar_start"]).tz_localize(None) == \
                pd.Timestamp(lb.bar_start).tz_localize(None), f"bar {i} start"
            for field in ("open", "high", "low", "close", "volume", "tick_count"):
                assert float(row[field]) == pytest.approx(
                    float(getattr(lb, field))), f"bar {i} {field}"
            assert float(row["vwap"]) == pytest.approx(lb.vwap, abs=1e-4), f"bar {i} vwap"
            assert bool(row["synthetic"]) == bool(lb.synthetic), f"bar {i} synthetic"

    def _bars(self, count, *, gaps=(), synthetic=()):
        bars = []
        price = 100.0
        for i in range(count):
            if i in gaps:
                continue
            price += (1.0 if i % 3 else -0.5)
            start = cal.at(DAY1, cal.SESSION_OPEN) + dt.timedelta(minutes=i)
            bars.append(Bar("nse_cm:1", start, start + dt.timedelta(minutes=1),
                            open=price, high=price + 0.8, low=price - 0.6,
                            close=price + 0.2, volume=1000.0 + i * 10,
                            vwap=price + 0.1, tick_count=50 + i,
                            synthetic=(i in synthetic)))
        return bars

    @pytest.mark.parametrize("interval", [60, 180, 300, 900])
    def test_agree_on_clean_data(self, interval):
        self._assert_agree(self._bars(60), interval)

    def test_agree_with_missing_minutes(self):
        self._assert_agree(self._bars(30, gaps={7, 8, 9}), 300)

    def test_agree_on_synthetic_bars(self):
        self._assert_agree(self._bars(30, synthetic={5, 6, 7, 8, 9}), 300)

    def test_agree_on_a_partial_final_bucket(self):
        """32 bars at 5-minute means the last bucket holds only 2 bars."""
        self._assert_agree(self._bars(32), 300)

    def test_agree_on_a_full_session(self):
        self._assert_agree(self._bars(375), 300)


class TestResamplerFlush:
    def test_flush_returns_held_buckets(self):
        from src.strategy_engine import BarResampler

        resampler = BarResampler(300)
        for i in range(3):
            resampler.add(Bar.from_event(minute_bar(DAY1, i, 100.0 + i)))
        assert resampler.held_session_day() == DAY1

        flushed = resampler.flush()
        assert len(flushed) == 1
        assert flushed[0].bar_start.date() == DAY1
        # Idempotent: nothing is held afterwards.
        assert resampler.flush() == []
        assert resampler.held_session_day() is None

    def test_flush_merges_correctly(self):
        from src.strategy_engine import BarResampler

        resampler = BarResampler(300)
        prices = [100.0, 103.0, 99.0, 101.0]
        for i, price in enumerate(prices):
            resampler.add(Bar.from_event(minute_bar(DAY1, i, price)))
        bar = resampler.flush()[0]
        assert bar.open == prices[0]
        assert bar.close == prices[-1]
        assert bar.high == max(prices) + 1
        assert bar.low == min(prices) - 1

    def test_flush_with_nothing_held(self):
        from src.strategy_engine import BarResampler

        assert BarResampler(300).flush() == []

    def test_pass_through_interval_holds_nothing(self):
        """At the store interval there is no bucketing, so nothing can be stranded."""
        from src.strategy_engine import BarResampler

        resampler = BarResampler(60)
        resampler.add(Bar.from_event(minute_bar(DAY1, 0, 100.0)))
        assert resampler.held_session_day() is None
        assert resampler.flush() == []
