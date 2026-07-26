"""Which component is losing the money?

`metrics.py` reports *how much* was lost. This reports *what is at fault*, because those
are different questions and only the second one tells you what to change.

The distinction is not academic. The first real-data backtest (§5.12) showed friction at
80% of the loss on twelve names, which pointed squarely at costs. Widening the sample to
thirty-nine showed gross P&L negative before a single rupee of friction — the opposite
diagnosis, from the same strategy on the same days. Averages had been hiding it.

**What settled it was distributional, not average.** An average MFE of +0.29R is equally
consistent with *"entries go nowhere"* and *"exits give winners back"*, and those demand
opposite fixes. The fact that decided it was a count: **no trade reached +1.0R and then
finished negative.** That ruled out the exit logic outright and left the entry.

So this module computes the handful of facts that separate the failure modes, and applies
them in an order that matters:

1. **Too few trades** → refuse. Thirty-six trades over four days will support any story.
2. **Net positive** → nothing to diagnose.
3. **Gross positive, net negative** → friction. Trade less, not differently.
4. **Winners reached target then gave it back** → exits.
5. **Positions never moved in our favour** → entry. Nothing downstream can fix this.
6. **Immediate full-loss stop-outs** → stops sitting inside the noise.

The order encodes a precedence, not a preference: friction is checked before entry because
a strategy whose gross is positive does not have an entry problem, however poor its net.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Sequence


class Fault(Enum):
    NONE = "none"
    INCONCLUSIVE = "inconclusive"
    FRICTION = "friction"
    EXITS = "exits"
    ENTRY = "entry"
    STOPS = "stops"
    MIXED = "mixed"


#: An excursion below this never gave the trade a chance to work.
MOVED_THRESHOLD_R = 0.25
#: Reaching this and finishing negative means the exit gave something back.
GAVE_BACK_THRESHOLD_R = 1.0
#: A stop hit this quickly was sitting inside the bar-to-bar noise.
QUICK_STOP_BARS = 2


@dataclass(frozen=True)
class Attribution:
    fault: Fault
    detail: str
    trade_count: int = 0
    excursion_sample: int = 0
    actual_win_rate: float = 0.0
    break_even_win_rate: float = 0.0
    reward_risk: float = 0.0
    avg_win_r: float = 0.0
    avg_loss_r: float = 0.0
    net_r: float = 0.0
    gross_r: float = 0.0            # after slippage, before charges
    frictionless_r: float = 0.0     # before slippage AND charges
    slippage_r: float = 0.0
    charges_r: float = 0.0
    never_moved_pct: float = 0.0
    gave_back_count: int = 0
    quick_stop_pct: float = 0.0
    median_mfe_r: float = 0.0

    @property
    def win_rate_shortfall(self) -> float:
        """How far the win rate is from where the reward:risk requires it to be.

        The actionable number: it says whether the gap is a tuning distance or a
        different-strategy distance.
        """
        return self.break_even_win_rate - self.actual_win_rate

    def report(self) -> str:
        lines = [
            "=" * 74,
            f"FAILURE ATTRIBUTION — {self.fault.value.upper()}",
            "=" * 74,
            f"  {self.detail}",
            "",
        ]
        if self.fault is Fault.INCONCLUSIVE:
            lines.append("=" * 74)
            return "\n".join(lines)

        lines += [
            f"  trades              {self.trade_count}",
            f"  net                 {self.net_r:+.3f}R per trade",
            f"  gross               {self.gross_r:+.3f}R  (before charges)",
            f"  frictionless        {self.frictionless_r:+.3f}R  (before slippage too)",
            f"    slippage          {self.slippage_r:.3f}R"
            f"   charges {self.charges_r:.3f}R",
            f"  win rate            {self.actual_win_rate:.1%}"
            f"   (break-even needs {self.break_even_win_rate:.1%})",
            f"  reward:risk         {self.reward_risk:.2f}:1"
            f"   (avg win {self.avg_win_r:+.3f}R, avg loss {self.avg_loss_r:+.3f}R)",
        ]
        if self.excursion_sample:
            lines += [
                "",
                f"  median MFE          {self.median_mfe_r:+.2f}R",
                f"  never reached +{MOVED_THRESHOLD_R}R  "
                f"{self.never_moved_pct:.0%} of trades",
                f"  reached +{GAVE_BACK_THRESHOLD_R}R then lost  "
                f"{self.gave_back_count} trade(s)",
            ]
        lines.append("=" * 74)
        return "\n".join(lines)


def _median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def diagnose(trades: Sequence, *, min_trades: int = 20,
             hurdle_r: float = 0.12) -> Attribution:
    """Attribute a losing book to a component. Never raises."""
    count = len(trades)
    if count < min_trades:
        return Attribution(
            fault=Fault.INCONCLUSIVE, trade_count=count,
            detail=(f"{count} trade(s) is too few to attribute anything — "
                    f"{min_trades} is the floor. A sample this size will support "
                    f"whichever story it is asked to."))

    rs = [t.r_multiple for t in trades]
    wins = [r for r in rs if r > 0]
    losses = [r for r in rs if r <= 0]

    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    reward_risk = (avg_win / abs(avg_loss)) if avg_loss else 0.0
    break_even = (abs(avg_loss) / (avg_win + abs(avg_loss))
                  if (avg_win + abs(avg_loss)) > 0 and losses else 0.0)
    win_rate = len(wins) / count
    net_r = sum(rs) / count

    # Three levels, matching metrics.py's vocabulary exactly so the two reports cannot
    # contradict each other:
    #
    #   net           after slippage and charges — what actually happened
    #   gross         after slippage, before charges
    #   frictionless  before both — the signal's raw edge
    #
    # Getting this wrong once already produced a confidently wrong verdict. Slippage is
    # baked into entry and exit prices, so it is *already inside* `r_multiple`; adding
    # back only the charges left a book that was +0.30R frictionless reading as "no
    # dominant cause", which would send someone to redesign a signal that worked.
    risk_total = sum(abs(t.risk_per_share) * abs(t.quantity) for t in trades)
    avg_risk = (risk_total / count) if (risk_total and count) else 0.0

    charges = sum(t.costs.total for t in trades)
    slippage = sum((t.entry_slippage_per_share + t.exit_slippage_per_share)
                   * abs(t.quantity) for t in trades)

    charges_r = (charges / avg_risk / count) if avg_risk else 0.0
    slippage_r = (slippage / avg_risk / count) if avg_risk else 0.0
    gross_r = net_r + charges_r
    frictionless_r = gross_r + slippage_r
    friction_r = charges_r + slippage_r

    # Excursion, only for trades where an R is meaningful.
    usable = [t for t in trades if t.risk_per_share > 0]
    mfes = [t.mfe / t.risk_per_share for t in usable]
    never_moved = (sum(1 for m in mfes if m < MOVED_THRESHOLD_R) / len(mfes)
                   if mfes else 0.0)
    gave_back = sum(1 for t in usable
                    if t.mfe / t.risk_per_share >= GAVE_BACK_THRESHOLD_R
                    and t.r_multiple < 0)
    quick_stops = (sum(1 for t in trades
                       if t.bars_held <= QUICK_STOP_BARS and t.r_multiple <= -0.9)
                   / count)

    common = dict(
        trade_count=count, excursion_sample=len(usable),
        actual_win_rate=win_rate, break_even_win_rate=break_even,
        reward_risk=reward_risk, avg_win_r=avg_win, avg_loss_r=avg_loss,
        net_r=net_r, gross_r=gross_r, frictionless_r=frictionless_r,
        slippage_r=slippage_r, charges_r=charges_r,
        never_moved_pct=never_moved,
        gave_back_count=gave_back, quick_stop_pct=quick_stops,
        median_mfe_r=_median(mfes),
    )

    if net_r >= 0:
        return Attribution(
            fault=Fault.NONE,
            detail=f"Profitable at {net_r:+.3f}R per trade — nothing to attribute.",
            **common)

    # Friction first, and the test is FRICTIONLESS, not gross. A book that makes money
    # before execution cost does not have a signal problem however poor its net, and the
    # earlier version — which tested gross and so ignored slippage — misattributed
    # exactly that case.
    if frictionless_r > 0:
        dominant = "slippage" if slippage_r > charges_r else "charges"
        return Attribution(
            fault=Fault.FRICTION,
            detail=(f"Frictionless is {frictionless_r:+.3f}R but net is {net_r:+.3f}R — "
                    f"execution cost of {friction_r:.3f}R per trade is eating the edge "
                    f"(hurdle ~{hurdle_r}R), mostly {dominant} "
                    f"(slippage {slippage_r:.3f}R, charges {charges_r:.3f}R). The signal "
                    f"is not the problem: trade less often, or hold for larger moves so "
                    f"the same cost is a smaller fraction of each trade."),
            **common)

    if gave_back >= max(1, int(0.3 * len(usable))):
        return Attribution(
            fault=Fault.EXITS,
            detail=(f"{gave_back} of {len(usable)} trades reached "
                    f"+{GAVE_BACK_THRESHOLD_R}R and still finished negative. The entry "
                    f"is finding moves; the exit is handing them back."),
            **common)

    # Stops are checked BEFORE entry, and the order is load-bearing rather than
    # arbitrary. A trade stopped out inside two bars *also* shows a low best-case
    # excursion — there was no time for one. So "never moved" is a CONSEQUENCE of the
    # quick stop, not independent evidence about the signal, and attributing it to the
    # entry would send someone to redesign a setup that was never given a chance.
    #
    # This is what made the real-data verdict (§5.12.1) trustworthy: nothing was stopped
    # within three bars and the median hold was fifteen, so the low excursion could not
    # be blamed on the stop and the entry was genuinely the remaining explanation.
    if quick_stops >= 0.35:
        return Attribution(
            fault=Fault.STOPS,
            detail=(f"{quick_stops:.0%} of trades took a full loss within "
                    f"{QUICK_STOP_BARS} bars. The stop is sitting inside the noise, so "
                    f"the trade is dead before the thesis has a chance — the low "
                    f"excursion is a symptom of that, not of the entry."),
            **common)

    if never_moved >= 0.35:
        return Attribution(
            fault=Fault.ENTRY,
            detail=(f"{never_moved:.0%} of trades never reached "
                    f"+{MOVED_THRESHOLD_R}R and the median best case was "
                    f"{_median(mfes):+.2f}R against a stop a full R away, with no "
                    f"pattern of early stop-outs to explain it. Positions are not moving "
                    f"in our favour — nothing downstream of the entry can fix that."),
            **common)

    return Attribution(
        fault=Fault.MIXED,
        detail=(f"Losing {net_r:+.3f}R per trade with no single dominant cause. "
                f"Reward:risk of {reward_risk:.2f}:1 needs a {break_even:.0%} win rate "
                f"and got {win_rate:.0%}."),
        **common)
