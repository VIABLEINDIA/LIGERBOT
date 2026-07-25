"""Fill-model tests — the invariants that keep a backtest honest."""
from __future__ import annotations

import datetime as dt

import pytest

from src import market_calendar as cal
from src.backtest.costs import CostModel, SlippageModel
from src.backtest.sim_broker import FillReason, SimBroker
from src.bars import Bar
from src.risk_engine import Intent, OrderRequest, Side

DAY = dt.date(2026, 7, 23)


def t(hour: int, minute: int) -> dt.datetime:
    return dt.datetime.combine(DAY, dt.time(hour, minute), tzinfo=cal.IST)


def bar(minute: int, o: float, h: float, l: float, c: float,
        volume: float = 100_000.0, instrument="nse_cm:1") -> Bar:
    return Bar(instrument, t(11, minute), t(11, minute + 1), o, h, l, c,
               volume=volume, vwap=(h + l + c) / 3, tick_count=100)


def order(quantity=100, side=Side.BUY, intent=Intent.OPEN_LONG,
          stop=99.0, target=None, instrument="nse_cm:1") -> OrderRequest:
    return OrderRequest(
        instrument_id=instrument, side=side, quantity=quantity, intent=intent,
        ref_price=100.0, stop_loss=stop, take_profit=target,
    )


@pytest.fixture
def broker() -> SimBroker:
    # No slippage/liquidity friction unless a test is specifically about it.
    return SimBroker(CostModel(), SlippageModel(slippage_bps=0.0, half_spread_bps=0.0),
                     enforce_liquidity=False)


class TestNextBarExecution:
    """The anti-look-ahead invariant (DESIGN.md 2.3 rule 1)."""

    def test_fill_uses_the_following_bars_open(self, broker):
        """The engine's ordering: process bar t, strategy decides, submit; bar t+1 fills.

        The decision was made from bar t's close (100.5), so filling anywhere near that
        price would be look-ahead. It must fill at bar t+1's open (102.0), including the
        overnight/interval gap that a real order would have suffered.
        """
        submit_bar = bar(0, 100.0, 101.0, 99.0, 100.5)
        broker.process_bar(submit_bar)
        broker.submit(order(), submit_bar.bar_end)

        fills = broker.process_bar(bar(1, 102.0, 103.0, 101.0, 102.5))
        entry = [f for f in fills if f.intent.is_open]
        assert len(entry) == 1
        assert entry[0].price == pytest.approx(102.0)  # next open, not 100.5

    def test_nothing_fills_on_a_synthetic_bar(self, broker):
        """A gap-filler bar means nothing traded — there was no price to trade at."""
        broker.submit(order(), t(11, 0))
        synthetic = Bar("nse_cm:1", t(11, 1), t(11, 2), 100.0, 100.0, 100.0, 100.0,
                        volume=0.0, vwap=100.0, tick_count=0, synthetic=True)
        assert broker.process_bar(synthetic) == []

    def test_fill_price_is_the_open_never_the_close(self, broker):
        broker.submit(order(), t(11, 0))
        fills = broker.process_bar(bar(1, 105.0, 110.0, 104.0, 109.0))
        assert fills[0].price == pytest.approx(105.0)

    def test_duplicate_pending_order_is_rejected(self, broker):
        broker.submit(order(), t(11, 0))
        broker.submit(order(), t(11, 0))
        assert len(broker.rejected) == 1


class TestStopsAndTargets:
    def test_stop_fills_at_the_stop_when_touched(self, broker):
        broker.submit(order(stop=99.0), t(11, 0))
        broker.process_bar(bar(1, 100.0, 100.5, 99.8, 100.2))  # entry at 100
        fills = broker.process_bar(bar(2, 100.0, 100.2, 98.5, 99.0))
        stopped = [f for f in fills if f.reason is FillReason.STOP_LOSS]
        assert len(stopped) == 1
        assert stopped[0].price == pytest.approx(99.0)

    def test_gap_through_the_stop_fills_at_the_open(self, broker):
        """Stops are not guaranteed prices. Modelling them as exact understates tail risk."""
        broker.submit(order(stop=99.0), t(11, 0))
        broker.process_bar(bar(1, 100.0, 100.5, 99.8, 100.2))
        fills = broker.process_bar(bar(2, 95.0, 95.5, 94.0, 94.5))
        gapped = [f for f in fills if f.reason is FillReason.GAP_THROUGH_STOP]
        assert len(gapped) == 1
        assert gapped[0].price == pytest.approx(95.0)  # the open, well below the stop

    def test_target_fills_when_reached(self, broker):
        broker.submit(order(stop=99.0, target=105.0), t(11, 0))
        broker.process_bar(bar(1, 100.0, 100.5, 99.8, 100.2))
        fills = broker.process_bar(bar(2, 100.5, 106.0, 100.0, 105.5))
        assert [f for f in fills if f.reason is FillReason.TAKE_PROFIT]

    def test_ambiguous_bar_resolves_to_the_stop(self, broker):
        """When a bar contains both stop and target, assume the stop hit first.

        A bar records only its extremes, not its path, so either assumption is a guess.
        The optimistic guess is how backtests lie.
        """
        broker.submit(order(stop=99.0, target=105.0), t(11, 0))
        broker.process_bar(bar(1, 100.0, 100.5, 99.8, 100.2))
        fills = broker.process_bar(bar(2, 100.0, 106.0, 98.0, 101.0))
        exits = [f for f in fills if not f.intent.is_open]
        assert len(exits) == 1
        assert exits[0].reason is FillReason.STOP_LOSS

    def test_position_can_be_stopped_on_its_entry_bar(self, broker):
        # Entry fills at the open, then the same bar's range is checked — exactly as
        # could happen in reality.
        broker.submit(order(stop=99.0), t(11, 0))
        fills = broker.process_bar(bar(1, 100.0, 100.2, 98.0, 98.5))
        assert len(fills) == 2
        assert fills[0].intent.is_open
        assert fills[1].reason is FillReason.STOP_LOSS

    def test_no_exit_when_levels_are_untouched(self, broker):
        broker.submit(order(stop=99.0, target=105.0), t(11, 0))
        broker.process_bar(bar(1, 100.0, 100.5, 99.8, 100.2))
        assert broker.process_bar(bar(2, 100.2, 101.0, 99.5, 100.8)) == []


class TestSlippage:
    def test_buys_fill_worse_than_the_bar_open(self):
        broker = SimBroker(CostModel(), SlippageModel(slippage_bps=10.0),
                           enforce_liquidity=False)
        broker.submit(order(side=Side.BUY), t(11, 0))
        fills = broker.process_bar(bar(1, 100.0, 101.0, 99.0, 100.5))
        assert fills[0].price > 100.0
        assert fills[0].slippage_per_share > 0

    def test_sells_fill_worse_too(self):
        broker = SimBroker(CostModel(), SlippageModel(slippage_bps=10.0),
                           enforce_liquidity=False)
        broker.submit(order(side=Side.SELL, intent=Intent.OPEN_SHORT, stop=101.0),
                      t(11, 0))
        fills = broker.process_bar(bar(1, 100.0, 101.0, 99.0, 100.5))
        assert fills[0].price < 100.0


class TestLiquidity:
    def test_order_is_trimmed_to_a_share_of_bar_volume(self):
        broker = SimBroker(CostModel(), SlippageModel(slippage_bps=0.0, half_spread_bps=0.0),
                           max_volume_participation=0.10, enforce_liquidity=True)
        broker.submit(order(quantity=10_000), t(11, 0))
        fills = broker.process_bar(bar(1, 100.0, 101.0, 99.0, 100.5, volume=1_000.0))
        assert fills[0].quantity == 100  # 10% of 1,000

    def test_zero_volume_on_a_real_bar_still_fills(self):
        """Zero volume on a real bar means 'no volume data', not 'nothing traded'.

        Suppressing fills here would make the backtester unusable on any feed that
        doesn't report volume.
        """
        broker = SimBroker(enforce_liquidity=True)
        broker.submit(order(quantity=100, stop=95.0), t(11, 0))
        fills = broker.process_bar(bar(1, 100.0, 101.0, 99.0, 100.5, volume=0.0))
        entry = [f for f in fills if f.intent.is_open]
        assert len(entry) == 1
        assert entry[0].quantity == 100

    def test_thin_bar_rejects_the_order(self):
        broker = SimBroker(enforce_liquidity=True, max_volume_participation=0.10)
        broker.submit(order(quantity=100), t(11, 0))
        # 10% of 5 shares rounds to zero — nothing can fill.
        assert broker.process_bar(bar(1, 100.0, 101.0, 99.0, 100.5, volume=5.0)) == []
        assert broker.rejected

    def test_liquidity_can_be_disabled(self):
        broker = SimBroker(enforce_liquidity=False)
        broker.submit(order(quantity=10_000), t(11, 0))
        fills = broker.process_bar(bar(1, 100.0, 101.0, 99.0, 100.5, volume=10.0))
        assert fills[0].quantity == 10_000


class TestForceClose:
    def test_square_off_closes_at_the_bar_close(self, broker):
        broker.submit(order(), t(11, 0))
        entry_bar = bar(1, 100.0, 101.0, 99.5, 100.5)
        broker.process_bar(entry_bar)
        fill = broker.force_close("nse_cm:1", bar(2, 100.5, 101.0, 100.0, 100.8))
        assert fill is not None
        assert fill.reason is FillReason.SQUARE_OFF
        assert fill.price == pytest.approx(100.8)

    def test_force_close_with_no_position_returns_none(self, broker):
        assert broker.force_close("nse_cm:1", bar(1, 100.0, 101.0, 99.0, 100.0)) is None

    def test_force_close_cancels_any_pending_order(self, broker):
        broker.submit(order(), t(11, 0))
        broker.process_bar(bar(1, 100.0, 101.0, 99.5, 100.5))
        broker.submit(order(intent=Intent.CLOSE_LONG, side=Side.SELL), t(11, 2))
        broker.force_close("nse_cm:1", bar(3, 100.0, 101.0, 99.0, 100.0))
        assert "nse_cm:1" not in broker.pending


class TestExcursions:
    def test_mae_and_mfe_track_the_worst_and_best(self, broker):
        broker.submit(order(stop=90.0), t(11, 0))
        broker.process_bar(bar(1, 100.0, 100.0, 100.0, 100.0))
        broker.process_bar(bar(2, 100.0, 104.0, 97.0, 101.0))
        position = broker.positions["nse_cm:1"]
        assert position.mae == pytest.approx(3.0)   # 100 -> 97
        assert position.mfe == pytest.approx(4.0)   # 100 -> 104

    def test_bars_held_increments(self, broker):
        broker.submit(order(stop=90.0), t(11, 0))
        broker.process_bar(bar(1, 100.0, 100.0, 100.0, 100.0))
        broker.process_bar(bar(2, 100.0, 100.5, 99.5, 100.0))
        broker.process_bar(bar(3, 100.0, 100.5, 99.5, 100.0))
        assert broker.positions["nse_cm:1"].bars_held == 3


class TestClosedPositionHandoff:
    def test_closed_position_is_recoverable_for_the_trade_record(self, broker):
        broker.submit(order(stop=99.0), t(11, 0))
        broker.process_bar(bar(1, 100.0, 100.5, 99.8, 100.2))
        broker.process_bar(bar(2, 100.0, 100.2, 98.5, 99.0))  # stopped
        assert "nse_cm:1" in broker.closed_positions
        assert broker.closed_positions["nse_cm:1"].entry_price == pytest.approx(100.0)
        assert not broker.has_position("nse_cm:1")
