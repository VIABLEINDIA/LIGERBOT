"""Golden-file pipeline test (DESIGN.md 3.10).

> *Golden-file pipeline test: fixed bar sequence in → exact expected orders out. This is
> the regression net for every future refactor.*

Every other test in this suite asserts a **property** — risk never exceeds the cap, fills
never precede the signal, costs are subtracted. Properties are the right tool for rules you
can state. They are useless against the failure this file exists to catch: a refactor that
quietly changes *which trades happen* while keeping every stated property true.

That failure has already happened here more than once. The live/backtest resampler
divergence at the session boundary broke no invariant — it just anchored day two's VWAP on
day one's close. The wall-clock-vs-bar-time phase check broke no invariant either. Both
were found by accident. A trace that has to be re-approved by a human is how you stop
finding them by accident.

## What makes this test worth having

**The input is hand-built, not random.** `generate_history` with a fixed seed would be
reproducible, but nobody can look at its output and say whether it is *right*. The price
path here is piecewise-linear between named control points, so the trace tells a story a
reviewer can check against the scenario: price ramps until the crossover fires, breaks down
through the stop, recovers into the second session, and is squared off at the deadline.

**Regeneration is opt-in and loud.** `LIGERBOT_UPDATE_GOLDEN=1` rewrites the file; nothing
else does. A golden test that regenerates itself on failure asserts only that the code
agrees with itself, which is worth precisely nothing. The CI hygiene job runs without that
variable set.

**The file is checked for teeth.** `TestTheGoldenFileHasTeeth` mutates inputs that *should*
change behaviour and asserts the trace moves. A golden file that matches no matter what you
do is a rubber stamp, and this is the check that it is not one.
"""
from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

import pandas as pd
import pytest

from src import market_calendar as cal
from src.backtest.bar_source import InMemoryBarSource
from src.backtest.costs import CostModel, SlippageModel
from src.backtest.engine import BacktestConfig, BacktestEngine
from src.backtest.trace import Trace
from src.risk_engine import RiskLimits
from src.strategies.sma_crossover import SmaCrossover

GOLDEN = Path(__file__).resolve().parent / "golden" / "pipeline_trace.txt"
UPDATE = os.environ.get("LIGERBOT_UPDATE_GOLDEN") == "1"

DAY_ONE = dt.date(2026, 3, 2)      # Monday
# Wednesday, not Tuesday: 2026-03-03 is Holi and the exchange is shut. Picking the two
# sessions off the calendar rather than off the weekday is itself part of what this
# scenario exercises — the gap between them is real, and the strategy must survive it.
DAY_TWO = dt.date(2026, 3, 4)
RELIANCE = "nse_cm:2885"
HDFCBANK = "nse_cm:1333"
INFY = "nse_cm:1594"
UNIVERSE = [RELIANCE, HDFCBANK, INFY]


# ---------------------------------------------------------------------------
# A hand-built price path. No RNG anywhere: the same bytes on every machine,
# every Python version, forever.
# ---------------------------------------------------------------------------
def piecewise(control_points: list[tuple[int, float]], length: int) -> list[float]:
    """Linear interpolation between (bar_index, price) control points."""
    out: list[float] = []
    for i in range(length):
        for (x0, y0), (x1, y1) in zip(control_points, control_points[1:]):
            if x0 <= i <= x1:
                span = x1 - x0
                out.append(y0 if span == 0 else y0 + (y1 - y0) * (i - x0) / span)
                break
        else:
            out.append(control_points[-1][1])
    return out


def session_frame(day: dt.date, closes: list[float]) -> pd.DataFrame:
    """Minute bars whose OHLC is derived deterministically from the close path."""
    window = cal.session_window(day)
    assert window is not None, f"{day} is not a trading day"
    open_at = window[0]

    rows = []
    previous = closes[0]
    for i, close in enumerate(closes):
        start = open_at + dt.timedelta(minutes=i)
        # A fixed 8bp bar range around the open→close move, so highs and lows are
        # reproducible and stops trigger at predictable prices.
        span = max(abs(close - previous), previous * 0.0004)
        rows.append({
            "bar_start": start,
            "bar_end": start + dt.timedelta(minutes=1),
            "open": round(previous, 2),
            "high": round(max(previous, close) + span * 0.25, 2),
            "low": round(min(previous, close) - span * 0.25, 2),
            "close": round(close, 2),
            "volume": 5_000.0 + (i % 7) * 250.0,
            "vwap": round((previous + close) / 2, 2),
            "tick_count": 40 + (i % 5),
            "synthetic": False,
        })
        previous = close

    frame = pd.DataFrame(rows)
    frame["bar_start"] = pd.to_datetime(frame["bar_start"])
    frame["bar_end"] = pd.to_datetime(frame["bar_end"])
    return frame


BARS_PER_SESSION = 375  # 09:15 → 15:30


def scenario_history() -> dict[str, pd.DataFrame]:
    """Two sessions built to exercise every path that matters.

    The first draft of this scenario produced a trace with no stop-loss and no rejection
    in it — two profitable days and nothing refused. It would have locked in the happy
    path and guarded nothing worth guarding. The paths below are shaped so that:

    * **RELIANCE** rallies into a crossover entry, then breaks down hard enough on day one
      to take out its stop (a *losing* trade, so the trace covers a red day too).
    * **HDFCBANK** trends gently all day and is still open at 15:10, forcing a square-off.
    * **INFY** crosses over *third*, by which point the position cap is full — so the risk
      engine refuses it and the refusal is recorded.
    """
    reliance_d1 = piecewise(
        # The break is ONE bar wide, and it has to be. A gentler slide lets the 5-period
        # average cross back under the 20 first, so the strategy exits politely on a
        # signal and the stop is never reached — which is what the first two drafts of
        # this scenario did. A stop only wins the race when the move outruns the average,
        # so this models the case stops actually exist for: a news-driven collapse inside
        # a single minute, with the engine's fills-before-strategy ordering meaning the
        # stop is checked against this bar's low before the strategy is even shown it.
        [(0, 1300.0), (60, 1300.0), (130, 1338.0), (140, 1338.0),
         (141, 1272.0), (374, 1281.0)], BARS_PER_SESSION)
    reliance_d2 = piecewise(
        [(0, 1288.0), (90, 1288.0), (170, 1322.0), (374, 1331.0)], BARS_PER_SESSION)

    hdfc_d1 = piecewise(
        [(0, 1650.0), (70, 1650.0), (150, 1690.0), (374, 1694.0)], BARS_PER_SESSION)
    hdfc_d2 = piecewise(
        [(0, 1692.0), (120, 1692.0), (210, 1723.0), (374, 1719.0)], BARS_PER_SESSION)

    # Deliberately the slowest to trigger, so it arrives at a full book.
    infy_d1 = piecewise(
        [(0, 1480.0), (80, 1480.0), (175, 1516.0), (374, 1521.0)], BARS_PER_SESSION)
    infy_d2 = piecewise(
        [(0, 1519.0), (100, 1519.0), (190, 1544.0), (374, 1540.0)], BARS_PER_SESSION)

    def both(d1: list[float], d2: list[float]) -> pd.DataFrame:
        return pd.concat([session_frame(DAY_ONE, d1), session_frame(DAY_TWO, d2)],
                         ignore_index=True)

    return {
        RELIANCE: both(reliance_d1, reliance_d2),
        HDFCBANK: both(hdfc_d1, hdfc_d2),
        INFY: both(infy_d1, infy_d2),
    }


def run_scenario(**overrides) -> str:
    """The pipeline under test: bars → strategy → risk → broker → fills."""
    strategy = SmaCrossover(
        short_period=overrides.pop("short_period", 5),
        long_period=overrides.pop("long_period", 20),
        stop_pct=overrides.pop("stop_pct", 0.01),
    )
    config = BacktestConfig(
        starting_equity=overrides.pop("starting_equity", 500_000.0),
        # Two concurrent positions, not the default three, so the third crossover of the
        # day meets a full book and the refusal path lands in the trace.
        risk_limits=overrides.pop("risk_limits", RiskLimits(max_open_positions=2)),
        cost_model=overrides.pop("cost_model", CostModel()),
        slippage=overrides.pop("slippage", SlippageModel()),
        skip_quality_gate=True,
        enforce_liquidity=False,
        compound=False,
        **overrides,
    )
    trace = Trace()
    engine = BacktestEngine(strategy, config, trace=trace)
    engine.run(InMemoryBarSource(scenario_history()), UNIVERSE, DAY_ONE, DAY_TWO)
    return trace.render()


# ---------------------------------------------------------------------------
class TestGoldenTrace:
    def test_pipeline_matches_the_approved_trace(self):
        produced = run_scenario()

        if UPDATE:
            GOLDEN.parent.mkdir(parents=True, exist_ok=True)
            GOLDEN.write_text(produced, encoding="utf-8")
            pytest.skip(f"golden file rewritten: {GOLDEN}")

        assert GOLDEN.exists(), (
            f"{GOLDEN} is missing. Regenerate with LIGERBOT_UPDATE_GOLDEN=1 and "
            f"*read the diff* before committing it.")

        expected = GOLDEN.read_text(encoding="utf-8")
        if produced != expected:
            import difflib

            diff = "\n".join(difflib.unified_diff(
                expected.splitlines(), produced.splitlines(),
                fromfile="approved", tofile="produced", lineterm=""))
            pytest.fail(
                "The pipeline no longer produces the approved trace.\n\n"
                "This is not automatically a bug — but it IS a behaviour change, and it "
                "needs a human to say which.\nIf the new behaviour is correct, rerun with "
                "LIGERBOT_UPDATE_GOLDEN=1 and commit the diff as part of the change that "
                f"caused it.\n\n{diff}")

    def test_the_run_is_deterministic(self):
        """Same input twice, same bytes. Anything else makes the golden file noise."""
        assert run_scenario() == run_scenario()

    def test_the_scenario_actually_trades(self):
        """A golden file over a trace with no fills would pass forever and prove nothing."""
        produced = run_scenario()
        assert produced.count("FILL") >= 4, produced

    def test_every_path_that_matters_is_exercised(self):
        """The first draft covered none of these. A trace of two green days guards nothing.

        A stop-loss is the only exit that *loses* money, a square-off is the only one the
        strategy does not choose, and a rejection is the risk engine doing the one job it
        exists for. If the scenario stops covering any of them, the golden file has
        quietly narrowed to the happy path.
        """
        produced = run_scenario()
        for marker in ("stop_loss", "square_off", "REJECT"):
            assert marker in produced, (
                f"{marker} never occurs — the scenario has narrowed and the golden file "
                f"no longer protects that path")


class TestTheGoldenFileHasTeeth:
    """A golden file that matches no matter what you change is a rubber stamp."""

    @pytest.mark.parametrize("label,overrides", [
        ("wider stop", {"stop_pct": 0.02}),
        ("faster signal", {"short_period": 3}),
        ("tighter risk", {"risk_limits": RiskLimits(risk_per_trade=0.001)}),
        ("tighter position cap", {"risk_limits": RiskLimits(max_open_positions=1)}),
        ("more slippage", {"slippage": SlippageModel(slippage_bps=25.0)}),
        ("smaller account", {"starting_equity": 100_000.0}),
    ])
    def test_changing_behaviour_changes_the_trace(self, label, overrides):
        baseline = run_scenario()
        assert run_scenario(**overrides) != baseline, (
            f"{label} did not move the trace — the golden file is not sensitive to it")


class TestRegenerationIsDeliberate:
    def test_the_golden_file_is_committed(self):
        assert GOLDEN.exists()
        assert GOLDEN.stat().st_size > 0

    def test_regeneration_requires_an_explicit_environment_variable(self):
        """CI must never silently re-approve. Absent the flag, the test compares."""
        assert os.environ.get("LIGERBOT_UPDATE_GOLDEN") != "1", (
            "LIGERBOT_UPDATE_GOLDEN is set in this environment — the golden test is "
            "rewriting rather than checking.")

    def test_the_trace_is_human_readable(self):
        """Whoever reviews a diff has to be able to tell what changed and why."""
        text = GOLDEN.read_text(encoding="utf-8")
        assert text.startswith("#"), "no provenance header"
        for token in ("strategy", "equity", "costs", "slippage"):
            assert token in text.split("\n\n")[0], f"header omits {token}"
