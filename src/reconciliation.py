"""Paper-vs-backtest reconciliation (DESIGN.md 2.6, Phase 4).

The highest-value test in the project. Running paper and backtest over the *same* sessions
and comparing them is the only check that can catch a whole class of errors nothing else
sees: a backtest that is subtly optimistic, a live path that diverges from the modelled
one, a strategy whose indicators warm up differently in production.

**Divergence is not the finding — attribution is.** "Paper made 12,000 less than the
backtest" tells you almost nothing and invites the wrong conclusion. The useful answer is
*which* of these it was, because each has a different fix:

======================  ======================================================
Cause                   What it means
======================  ======================================================
Missing trades          The strategy signalled in replay but not live, or the
                        risk engine refused it live. Usually a feed gap, a
                        stale-feed block, or a halt.
Extra trades            Live took a trade the backtest did not — often an
                        indicator warmup difference.
Fill price divergence   The slippage model is wrong, or execution is slower
                        than modelled.
Exit divergence         Same entry, different exit. Usually the pessimistic
                        intrabar rule resolving differently against real ticks.
Cost divergence         The cost model does not match the contract note.
======================  ======================================================

A reconciliation that reports only a total is a reconciliation nobody can act on.
"""
from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.session_recorder import RecordedTrade, SessionRecord, SessionStore

log = logging.getLogger("ligerbot.reconciliation")


@dataclass
class TradePair:
    """A backtest trade matched to its paper counterpart."""

    backtest: RecordedTrade
    paper: RecordedTrade

    @property
    def entry_divergence_bps(self) -> float:
        if self.backtest.entry_price <= 0:
            return 0.0
        return 10_000.0 * (self.paper.entry_price - self.backtest.entry_price) / \
            self.backtest.entry_price

    @property
    def exit_divergence_bps(self) -> float:
        if self.backtest.exit_price <= 0:
            return 0.0
        return 10_000.0 * (self.paper.exit_price - self.backtest.exit_price) / \
            self.backtest.exit_price

    @property
    def pnl_divergence(self) -> float:
        return self.paper.net_pnl - self.backtest.net_pnl

    @property
    def cost_divergence(self) -> float:
        return self.paper.costs - self.backtest.costs

    @property
    def same_exit_reason(self) -> bool:
        return self.paper.exit_reason == self.backtest.exit_reason


@dataclass
class Attribution:
    """Where the P&L divergence came from, in rupees."""

    missing_trades: float = 0.0     # backtest traded, paper did not
    extra_trades: float = 0.0       # paper traded, backtest did not
    fill_prices: float = 0.0        # matched trades, different fills
    costs: float = 0.0              # matched trades, different charges

    @property
    def total(self) -> float:
        return self.missing_trades + self.extra_trades + self.fill_prices + self.costs

    def as_rows(self) -> List[Tuple[str, float]]:
        return [
            ("missing trades (in backtest, not paper)", self.missing_trades),
            ("extra trades (in paper, not backtest)", self.extra_trades),
            ("fill-price divergence on matched trades", self.fill_prices),
            ("cost divergence on matched trades", self.costs),
        ]


@dataclass
class ReconciliationTolerance:
    """What counts as tracking closely enough to proceed to live capital."""

    min_match_rate: float = 0.80          # of backtest trades also taken in paper
    max_entry_divergence_bps: float = 15.0
    max_expectancy_divergence_r: float = 0.05
    max_unexplained_fraction: float = 0.20  # of total |divergence|
    entry_match_window_minutes: float = 10.0


@dataclass
class ReconciliationResult:
    days_compared: int = 0
    days_skipped: List[str] = field(default_factory=list)
    pairs: List[TradePair] = field(default_factory=list)
    missing_in_paper: List[RecordedTrade] = field(default_factory=list)
    extra_in_paper: List[RecordedTrade] = field(default_factory=list)
    attribution: Attribution = field(default_factory=Attribution)

    backtest_pnl: float = 0.0
    paper_pnl: float = 0.0
    backtest_expectancy_r: float = 0.0
    paper_expectancy_r: float = 0.0
    backtest_trades: int = 0
    paper_trades: int = 0

    tolerance: ReconciliationTolerance = field(default_factory=ReconciliationTolerance)

    # -- derived -----------------------------------------------------------
    @property
    def pnl_divergence(self) -> float:
        return self.paper_pnl - self.backtest_pnl

    @property
    def match_rate(self) -> float:
        if not self.backtest_trades:
            return 0.0
        return len(self.pairs) / self.backtest_trades

    @property
    def expectancy_divergence_r(self) -> float:
        return self.paper_expectancy_r - self.backtest_expectancy_r

    @property
    def mean_entry_divergence_bps(self) -> float:
        if not self.pairs:
            return 0.0
        return sum(abs(p.entry_divergence_bps) for p in self.pairs) / len(self.pairs)

    @property
    def exit_reason_agreement(self) -> float:
        if not self.pairs:
            return 0.0
        return sum(1 for p in self.pairs if p.same_exit_reason) / len(self.pairs)

    @property
    def unexplained(self) -> float:
        """Divergence the attribution could not account for.

        Should be near zero. A large residual means the attribution logic itself is
        wrong, which matters more than the divergence — it would mean the diagnosis
        cannot be trusted either.
        """
        return self.pnl_divergence - self.attribution.total

    @property
    def failures(self) -> List[str]:
        problems: List[str] = []
        t = self.tolerance
        if self.match_rate < t.min_match_rate:
            problems.append(
                f"only {self.match_rate:.0%} of backtest trades appeared in paper "
                f"(need {t.min_match_rate:.0%})")
        if self.mean_entry_divergence_bps > t.max_entry_divergence_bps:
            problems.append(
                f"mean entry divergence {self.mean_entry_divergence_bps:.1f}bps "
                f"exceeds {t.max_entry_divergence_bps:.0f}bps — the slippage model is "
                f"wrong or execution is slower than modelled")
        if abs(self.expectancy_divergence_r) > t.max_expectancy_divergence_r:
            problems.append(
                f"expectancy differs by {self.expectancy_divergence_r:+.3f}R "
                f"(limit {t.max_expectancy_divergence_r:.3f}R)")
        denominator = abs(self.pnl_divergence)
        if denominator > 1e-9:
            residual = abs(self.unexplained) / denominator
            if residual > t.max_unexplained_fraction:
                problems.append(
                    f"{residual:.0%} of the divergence is unattributed — the "
                    f"reconciliation itself cannot be trusted until this is explained")
        return problems

    @property
    def passed(self) -> bool:
        return self.days_compared > 0 and not self.failures

    def report(self) -> str:
        rule = "=" * 78
        lines = [rule, "PAPER vs BACKTEST RECONCILIATION (DESIGN.md 2.6)", rule]

        if self.days_compared == 0:
            lines += ["", "  No comparable sessions.",
                      "  Both a paper run and a backtest over the same days are required.",
                      rule]
            return "\n".join(lines)

        lines += [
            "",
            f"  Sessions compared    {self.days_compared:>12}",
            f"  Sessions skipped     {len(self.days_skipped):>12}"
            + (f"   ({', '.join(self.days_skipped[:5])})" if self.days_skipped else ""),
            "",
            "P&L",
            f"  Backtest             {self.backtest_pnl:>+12,.2f}   "
            f"({self.backtest_trades} trades, {self.backtest_expectancy_r:+.3f}R)",
            f"  Paper                {self.paper_pnl:>+12,.2f}   "
            f"({self.paper_trades} trades, {self.paper_expectancy_r:+.3f}R)",
            f"  Divergence           {self.pnl_divergence:>+12,.2f}",
            "",
            "Attribution — where the divergence came from",
        ]
        for label, amount in self.attribution.as_rows():
            lines.append(f"  {label:<44} {amount:>+12,.2f}")
        lines.append(f"  {'unexplained residual':<44} {self.unexplained:>+12,.2f}")

        lines += [
            "",
            "Agreement",
            f"  Trade match rate     {self.match_rate:>11.0%}   "
            f"({len(self.pairs)}/{self.backtest_trades} backtest trades matched)",
            f"  Mean entry divergence{self.mean_entry_divergence_bps:>11.1f}bps",
            f"  Exit-reason agreement{self.exit_reason_agreement:>11.0%}",
            f"  Expectancy gap       {self.expectancy_divergence_r:>+11.3f}R",
            "",
        ]

        if self.missing_in_paper:
            lines.append(f"  {len(self.missing_in_paper)} backtest trade(s) never "
                         f"happened in paper — check feed gaps, stale-feed blocks and "
                         f"halts on those days.")
        if self.extra_in_paper:
            lines.append(f"  {len(self.extra_in_paper)} paper trade(s) the backtest did "
                         f"not take — usually an indicator warmup difference.")

        lines.append("")
        if self.passed:
            lines += ["  VERDICT: paper tracks the backtest within tolerance.",
                      "  This clears the Phase 4 reconciliation gate. It does NOT by "
                      "itself clear",
                      "  live trading — that also needs the full session count and the "
                      "§2.6 gates."]
        else:
            lines.append("  VERDICT: BLOCKED — paper does not track the backtest.")
            for problem in self.failures:
                lines.append(f"    - {problem}")
            lines += ["",
                      "  A divergence here means the model is wrong somewhere. Find out",
                      "  where before trusting either side — DESIGN.md 2.6 treats this as",
                      "  the highest-value test in the project for that reason."]
        lines.append(rule)
        return "\n".join(lines)


def _match_trades(
    backtest: List[RecordedTrade],
    paper: List[RecordedTrade],
    window_minutes: float,
) -> Tuple[List[TradePair], List[RecordedTrade], List[RecordedTrade]]:
    """Greedily pair trades by instrument and entry time.

    Timestamps will not be identical — paper fills on a real clock, the backtest on bar
    boundaries — so matching is by nearest entry within a window rather than by equality.
    """
    unmatched_paper = list(paper)
    pairs: List[TradePair] = []
    missing: List[RecordedTrade] = []

    for bt in backtest:
        bt_entry = _parse(bt.entry_at)
        candidates = [
            (abs((_parse(p.entry_at) - bt_entry).total_seconds()), p)
            for p in unmatched_paper
            if p.instrument_id == bt.instrument_id and p.direction == bt.direction
        ]
        candidates = [(gap, p) for gap, p in candidates if gap <= window_minutes * 60]
        if not candidates:
            missing.append(bt)
            continue
        candidates.sort(key=lambda item: item[0])
        best = candidates[0][1]
        unmatched_paper.remove(best)
        pairs.append(TradePair(backtest=bt, paper=best))

    return pairs, missing, unmatched_paper


def _parse(value: str) -> dt.datetime:
    try:
        parsed = dt.datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return dt.datetime.min.replace(tzinfo=dt.timezone.utc)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=dt.timezone.utc)


def reconcile(
    store: SessionStore,
    *,
    tolerance: Optional[ReconciliationTolerance] = None,
    paper_source: str = "paper",
    backtest_source: str = "backtest",
) -> ReconciliationResult:
    """Compare every session both sources recorded."""
    tolerance = tolerance or ReconciliationTolerance()
    result = ReconciliationResult(tolerance=tolerance)

    all_backtest: List[RecordedTrade] = []
    all_paper: List[RecordedTrade] = []

    for day in store.common_days(paper_source, backtest_source):
        paper_record = store.load(paper_source, day)
        backtest_record = store.load(backtest_source, day)
        if paper_record is None or backtest_record is None:
            continue

        # A halted day stopped partway through; the backtest of that day did not.
        # Comparing them yields a divergence with a known, uninteresting cause.
        if not paper_record.comparable:
            result.days_skipped.append(f"{day} (halted: {paper_record.halt_reason[:40]})")
            continue

        result.days_compared += 1
        all_backtest.extend(backtest_record.trades)
        all_paper.extend(paper_record.trades)

    result.backtest_trades = len(all_backtest)
    result.paper_trades = len(all_paper)
    result.backtest_pnl = sum(t.net_pnl for t in all_backtest)
    result.paper_pnl = sum(t.net_pnl for t in all_paper)
    result.backtest_expectancy_r = (
        sum(t.r_multiple for t in all_backtest) / len(all_backtest)
        if all_backtest else 0.0)
    result.paper_expectancy_r = (
        sum(t.r_multiple for t in all_paper) / len(all_paper) if all_paper else 0.0)

    pairs, missing, extra = _match_trades(
        all_backtest, all_paper, tolerance.entry_match_window_minutes)
    result.pairs = pairs
    result.missing_in_paper = missing
    result.extra_in_paper = extra

    # Attribute the divergence. Signs are chosen so the parts sum toward
    # (paper - backtest): a trade the backtest took and paper did not removes that
    # trade's P&L from paper's side.
    result.attribution = Attribution(
        missing_trades=-sum(t.net_pnl for t in missing),
        extra_trades=sum(t.net_pnl for t in extra),
        costs=-sum(p.cost_divergence for p in pairs),
        fill_prices=sum(p.pnl_divergence + p.cost_divergence for p in pairs),
    )
    return result
