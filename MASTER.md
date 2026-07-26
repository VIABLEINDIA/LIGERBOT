# LIGERBOT — Master Reference

> **One document, everything.** Architecture, every module, every settled decision, the
> risk and cost models, both strategies, the evidence ladder from backtest to live capital,
> every defect testing caught, and the operational procedures.
>
> `DESIGN.md` is the *reasoning* — why each decision went the way it did, at the length the
> argument needed. `README.md` is the *quick start*. This is the **complete reference**:
> what exists, what it does, and what is still unknown.

**Repository:** <https://github.com/VIABLEINDIA/LIGERBOT> ·
**Market:** NSE equities & indices, intraday only (MIS, flat by session close) ·
**Broker:** Kotak Neo API v2 ·
**Scale:** 46 modules / 11,124 lines source · 10,177 lines tests · **1,340 tests, 95% coverage**

---

## 1. What this is, and what it is not

### What is built

An event-driven intraday trading system: eight processes communicating only over Redis
Streams, with a backtest harness that drives **the same strategy and risk objects the live
system uses**. Bar building, risk engine, order state machine, paper broker, broker
reconciliation, go-live guard, structured logging, health endpoints and dashboards all
exist and are tested.

### What is not

**No strategy in this repository has been validated on real market data.** That is not a
caveat at the bottom of the page; it is the single most important fact about the project.

- `trend_pullback` v1 is implemented and **fails the go-live gates**.
- `sma_crossover` is a deliberate **negative control** — it is *supposed* to lose money.
- The go-live guard **blocks**, on seven outstanding prerequisites. That refusal is the
  system working, not a bug to route around.
- The Kotak capability probe has **not been run**. Historical data depth is unknown, and
  the `limits()` field mapping in `src/account.py` remains an unverified guess.

Every "complete and verified" in this document means *the machinery* is verified. None of
them means the strategy makes money.

### The honest summary

The machinery is in good shape. The evidence is absent. Those are different things, and
conflating them is how people lose money with well-engineered software.

---

## 2. Architecture

Eight independent processes. They never call each other — they read and write Redis
Streams. A crashed strategy engine cannot take down execution.

```
  ┌─────────────────────────────────────────────────────────────────────┐
  │                       REDIS STREAMS (the spine)                     │
  └─────────────────────────────────────────────────────────────────────┘
   market_ticks   market_bars   trade_signals  approved_orders  filled_orders
        ▲              ▲              ▲               ▲              ▲
        │              │              │               │              │
  ┌─────┴──────┐ ┌─────┴──────┐ ┌─────┴──────┐ ┌──────┴─────┐ ┌──────┴──────┐
  │ 1 Ingestion│→│ 2 Bar      │→│ 3 Strategy │→│ 4 Risk     │→│ 5 Execution │
  │  Kotak WS  │ │   Builder  │ │   Engine   │ │   Manager  │ │  OR 5b Paper│
  └────────────┘ └─────┬──────┘ └────────────┘ └────────────┘ └──────┬──────┘
                       │                              ▲              │
                       ▼                              │ position_updates
                 Parquet store              ┌─────────┴────────┐     │
                 (the dataset)              │ 7 Position Mgr   │◀────┘
                                            │  broker = truth  │
                                            └──────────────────┘
                       ┌──────────────────┐
                       │ 6 Storage Logger │  all streams → InfluxDB → Grafana
                       │ passive observer │
                       └──────────────────┘
```

### Streams

| Stream | Producer | Consumers |
|---|---|---|
| `market_ticks` | ingestion | bar builder, storage |
| `market_bars` | bar builder | strategy engine, paper broker, storage |
| `trade_signals` | strategy engine | risk manager, storage |
| `approved_orders` | risk manager | execution engine **or** paper broker, storage |
| `filled_orders` | execution / paper broker | position manager, storage |
| `position_updates` | position manager | risk manager, strategy engine, storage |
| `alerts` | any module | operator, storage |
| `dead_letter` | event bus | operator |

Trimmed at `STREAM_MAXLEN = 100,000`. Consumption uses **consumer groups with explicit
acks**, not `$` cursors — a restart resumes rather than silently skipping the backlog.

### Two rules that shape everything

**Exactly one module fills orders.** In `paper` mode that is `paper_broker`; in `dry_run`
and `live` it is `execution_engine`. Both consume `approved_orders` from separate groups,
so running both would fill every order twice **while looking entirely healthy**. Each
refuses to start in the other's mode.

**Every gate blocks new risk and none blocks exits.** Halts, stale feeds, session phase,
drawdown breaches — all stop entries. None ever prevents getting flat. A safety feature
that traps you in a position is not a safety feature.

---

## 3. Every module

### Core pipeline

| Module | Lines | Purpose |
|---|---|---|
| `data_ingestion.py` | 216 | Kotak WebSocket → `market_ticks`. Bounded reconnection, halts loudly when exhausted. |
| `bar_builder.py` | 205 | Ticks → closed bars → stream **and** Parquet. Only *closed* bars are published. |
| `strategy_engine.py` | 232 | Bars → signals. Resamples 1m → 5m; flushes the previous session before rolling. |
| `risk_manager.py` | 278 | Signals → sized orders. Thin adapter around the pure risk engine. |
| `execution_engine.py` | 397 | Orders → Kotak. Idempotency, rate limiting, order state machine. |
| `paper_broker.py` | 158 | Orders → simulated fills using the backtester's model. Paper mode only. |
| `position_manager.py` | 397 | Fills → the book. Reconciles against the broker; the broker wins. |
| `storage_logger.py` | 90 | All streams → InfluxDB. Passive; never backpressures trading. |

### Pure logic (no I/O — shared by backtest and live)

| Module | Lines | Purpose |
|---|---|---|
| `risk_engine.py` | 461 | Every risk rule. Provably cannot exceed the open-risk cap. |
| `bars.py` | 346 | Time-bar aggregation, gap filling, VWAP. |
| `indicators.py` | 351 | Incremental EMA, ADX, ATR, RSI, RVOL, VWAP. |
| `order_state.py` | 377 | `PENDING→SENT→ACKED→PARTIAL→FILLED\|REJECTED\|CANCELLED\|EXPIRED` + idempotency. |
| `market_calendar.py` | 253 | NSE holidays, session phases, square-off clock. |
| `instruments.py` | 454 | Scrip master, liquidity screen, sector groups. |
| `strategy_base.py` | 115 | The `Strategy` interface — deliberately narrow. |

### Backtest harness

| Module | Lines | Purpose |
|---|---|---|
| `backtest/engine.py` | 292 | Fills → session control → strategy. That order enforces next-bar execution. |
| `backtest/sim_broker.py` | 259 | Fill model: next-bar open, pessimistic intrabar resolution. |
| `backtest/costs.py` | 188 | The full Indian cost stack. |
| `backtest/portfolio.py` | 254 | Trade ledger and equity curve. |
| `backtest/metrics.py` | 291 | Expectancy, profit factor, drawdown, by-month/instrument breakdowns. |
| `backtest/gates.py` | 255 | §2.6 go-live gates and the Phase 4 gates. |
| `backtest/walk_forward.py` | 293 | Walk-forward, locked holdout, trial counting. |
| `backtest/bar_source.py` | 348 | Bar sources + the data-quality gate. |
| `backtest/synthetic.py` | 135 | Synthetic data for testing the harness itself. |
| `backtest/trace.py` | 97 | Deterministic decision log — drives the golden-file test. |

### Infrastructure and safety

| Module | Lines | Purpose |
|---|---|---|
| `event_bus.py` | 340 | Streams, consumer groups, dead-lettering, backlog alerts. |
| `kill_switch.py` | 141 | Redis halt flag + operator CLI. Fails **closed**. |
| `live_guard.py` | 286 | Blocks live trading on seven prerequisites. |
| `live_scaling.py` | 222 | Evidence-gated position-size ladder. |
| `feed_health.py` | 265 | Staleness watchdog (TTL keys) + reconnect policy. |
| `auth_session.py` | 204 | **One** login per day, shared. TOTP is single-use. |
| `auth.py` / `kotak_api.py` | 136 / 113 | Login flow; guards around the SDK's actual behaviour. |
| `account.py` | 212 | Equity retrieval and the session snapshot (D1). |
| `reconciliation.py` | 298 | Paper vs backtest, with divergence attribution. |
| `session_recorder.py` | 180 | Per-session results, for reconciliation. |
| `briefing.py` | 265 | Pre-open go/no-go and post-close report. |
| `alerting.py` | 176 | Six conditions, deduped, multi-sink, failure-isolated. |
| `influx_writer.py` | 254 | Batched, bounded, drop-oldest archiving. |
| `logging_setup.py` | 179 | Structured logs + correlation-ID threading. |
| `health.py` | 241 | Per-module health endpoints. |
| `universe_builder.py` | 268 | Scrip master → tradable universe, with provenance. |
| `momentum_screen.py` | 170 | Ranks a universe: return ÷ volatility, trend quality (R²), RVOL. |
| `tv_screener.py` | 158 | Nifty 500 momentum shortlist — the day-one bootstrap. |
| `bar_store.py` | 205 | Parquet dataset, written from day one. |

---

## 4. Settled decisions

### D1 — Equity is read live from Kotak, never configured
Computed as **cash + MTM**, never from margin (which embeds ~5× MIS leverage). Pinned for
the session so intraday P&L cannot feed back into sizing. **Fails closed**: if equity
cannot be established, the module halts rather than guessing, because a wrong figure
mis-sizes every trade that day *by the same factor* — and no percentage-based cap notices,
since they all share the wrong base.

### D2 — Risk parameters, made internally consistent
| Parameter | Value |
|---|---|
| Risk per trade | **0.5%** of session equity |
| Total open risk cap | **1.5%** — the primary control |
| Max daily drawdown | **2.0%** (halt) |
| Max open positions | 3 |
| Max positions per correlation group | 1 |
| Max exposure per trade | 75% of equity |
| Max gross exposure | 200% of equity |
| Min stop distance | 0.1% |
| Short selling | **disabled** (D3) |

The cap is on the **sum of distance-to-stop**, not position count — counting positions
bounds nothing. A **worst-case daily gate** refuses entries once realised losses *plus open
risk* would breach 2%, so the limit bounds the day rather than tripping after the fact.

### D3 — Long-only for v1
Shorting intraday in India carries different margin treatment and borrow constraints.
Deferred, not forgotten. Session direction (up/down/flat) is tracked in metrics precisely
because a long-only strategy looks good in a rising market for reasons unrelated to skill.

### D4 — Store 1-minute bars, trade 5-minute
Measured: friction ≈ `0.00085 / stop_pct` in R terms, implying an **economic stop floor
around 0.7%**. One-minute ATR stops are structurally uneconomic — they produced *zero*
qualifying trades. Storing the finer interval keeps the option open; trading the coarser
one keeps the trades economic.

### D5 — Historical data: Kotak Neo only
No paid vendor. Mitigations: (1) measure actual depth with a probe **before** committing
downstream work; (2) **self-record to Parquet from day one**, because it is the
slowest-maturing asset in the project. *Status: the probe has not been run.*

### D6 — Universe: a liquidity screen, not a fixed list
Screened on price and average daily traded value, with sector groups for the concentration
filter. **Fails closed** — an instrument whose liquidity cannot be established is excluded,
never assumed adequate. Liquidity is measured from `HISTORY` → `RECORDED_BARS` →
`QUOTE_PROXY` in descending order of trust, and **which source was used travels with the
universe**, because a screen run on a one-day proxy is a weaker claim than one run on sixty
days of recorded bars.

---

## 5. Risk model

### Sizing

```
quantity = floor( (equity × risk_per_trade) / |entry − stop| )
```

A signal **without a stop is rejected outright**, never sized by notional. Risk is recorded
from the *actual fill price*, not the reference price: a position that slipped on entry
carries more risk than was budgeted, and the cap must see the real figure.

### The order gates apply

1. **Session phase** — judged at the signal's own `bar_time`, not wall-clock now (the
   backtest does the same; using `now` would judge a queued signal against a different
   window than the one it was generated in).
2. **Signal age** — an hours-old signal is refused even if its phase was valid.
3. **Kill switch** — checked per order, effective without a restart.
4. **Feed staleness** — TTL keys expire on their own; no writer marks anything dead.
5. **Risk engine** — open-risk cap, worst-case daily gate, position count, correlation
   group, exposure caps, absolute rupee backstops.

### Escalation

| Condition | Response |
|---|---|
| Open risk would exceed 1.5% | refuse the entry |
| Realised loss + open risk would breach 2% | refuse all new entries |
| Realised loss breaches 2% | **halt** — exits only |
| Reconciliation mismatch ≥ threshold | **halt** — the book is unverifiable |
| Equity unresolvable | **halt** — sizing is impossible |
| Feed reconnection exhausted | **halt** — loudly |
| 15:10 | forced square-off, no strategy opinion consulted |

---

## 6. Cost model — the number to beat

```
brokerage  min(₹20, 0.03%) per order
STT        0.025% on the SELL leg
exchange   0.00297%
SEBI       0.0001%
stamp      0.003% on the BUY leg
GST        18% on (brokerage + exchange)
slippage   2.5 bps/leg, min 1 bp, ×2 within 15 min of open/close
```

**Measured hurdle: ≈ 0.12R per round trip.** The harness independently reproduced this at
0.107R. Every backtest number in this project is net of it, and results are reported split
into **frictionless / slippage / charges** so the drag is never hidden inside a gross P&L
figure.

The consequence, spelled out: at ~12.5% cost drag, a 2:1 reward-to-risk strategy needs only
a **37%** win rate to profit; a 1:1 strategy needs **56%**. Win rate alone is meaningless.
The objective is **expectancy net of costs**, with maximum drawdown as the binding
constraint.

---

## 7. Strategies

### `sma_crossover` — the negative control

10/50 SMA crossover. It exists **to lose money**. If the harness ever reports it as
profitable, the harness is broken — look-ahead, missing costs, optimistic fills. Measured:
**−0.097R per trade**. It is a test instrument, not a trading strategy.

### `trend_pullback` v1 — the reference strategy

Trade *with* an established trend, enter on a pullback, exit on a measured stop.

| Parameter | Default | Role |
|---|---|---|
| `ema_fast` / `ema_slow` | 9 / 21 | trend direction |
| `adx_period` / `adx_min` | 14 / 20.0 | trend *strength* — no trend, no trade |
| `atr_period` / `atr_mult` | 14 / 3.0 | stop distance |
| `min_stop_pct` | 0.007 | the D4 economic floor |
| `atr_ceiling_pct` | 0.05 | refuse untradeably volatile names |
| `rvol_min` | 0.8 | require participation |
| `rsi_max` | 75.0 | do not buy exhaustion |
| `pullback_atr` | 0.5 | how deep a pullback qualifies |
| `trail_after_r` / `trail_atr_mult` | 1.0 / 1.5 | trail once 1R is banked |
| `time_stop_bars` / `time_stop_min_r` | 20 / 0.5 | exit trades that go nowhere |

**Status: unvalidated, and currently failing the gates.** Its positive control was weak,
and synthetic data could not distinguish *"the generator lacks the structure it targets"*
from *"the strategy detects it weakly"*. **Only real data settles that.**

---

## 8. The evidence ladder

Nothing advances a stage without evidence from the one before.

```
   Backtest ──gates §2.6──▶ Paper ──Phase 4 gates──▶ Live (minimum size) ──ladder──▶ Scale
```

### §2.6 gates — backtest → paper

| Gate | Threshold |
|---|---|
| Net expectancy after costs | > 0R |
| Profit factor | > 1.3 |
| Max drawdown | within risk appetite |
| Out-of-sample trades | ≥ 200 |
| Survives doubled slippage | > 0R |
| Walk-forward OOS expectancy | positive |
| In-sample degradation | contained |
| Trial count | **disclosed** |
| Not driven by one month or instrument | concentration check |

*Current status: **BLOCKED** — 7 failures on synthetic data. The qualifier matters: these
are not results, they are the gates correctly refusing a strategy that has never been shown
real prices.*

### Phase 4 gates — paper → live

| Gate | Threshold | Blocking? |
|---|---|---|
| Paper sessions completed | ≥ 20 | yes |
| Paper tracks the backtest | reconciliation passes | yes |
| Divergence is **explained** | unattributed residual small | yes |
| Halt rate | < 15% | advisory |

*Current status: **0 of 20 sessions**.*

The third gate is the subtle one. A large divergence that is *fully attributed* is a
diagnosis; a small one that is *not* attributed is a mystery — and the mystery is the
problem, because it means the model is wrong somewhere nobody has named.

**A strategy can pass §2.6 and fail Phase 4. That failure is the single most valuable
signal the project can produce**, because it means the backtest was measuring something the
live system does not do. Finding that out with paper money is the entire point.

### Live guard

`TRADING_MODE=live` is **deliberately not sufficient** — an environment variable is one
typo away from committing real capital. The guard additionally requires demonstrated
backtest and paper evidence, a verified broker field mapping, and a human-written
authorisation file. *Current status: **BLOCKED** on 7 prerequisites.*

---

## 9. Defect log — what testing caught

Every one of these was found by a test or a harness, **not by reading the code**. That is
the argument for the test suite, stated as evidence rather than as principle.

| # | Defect | Why it mattered |
|---|---|---|
| 1 | Synthetic-bar coverage fabricated (76% → 0–17%) | `shutdown()` flushed to `now()`, manufacturing bars for periods the bot was not running. **It silently inflated the quality of the dataset every backtest is built on.** |
| 2 | Drawdown limit was a *trip threshold*, not a cap | −3.4% was reachable against a 2% limit. Fixed by the worst-case gate. |
| 3 | Friction hidden inside gross P&L | Made every result look better than it was. Now split three ways. |
| 4 | 1-minute ATR stops structurally uneconomic | Zero qualifying trades. Settled D4. |
| 5 | Live path judged session phase on wall-clock | Backtest used `bar_time`. A queued signal was judged against the wrong window. |
| 6 | Rejection reasons named the wrong subsystem | During an incident, sends whoever is on the keyboard down the wrong path. |
| 7 | Resampler divergence at the session boundary | The final bucket surfaced *after* `on_session_start` reset VWAP — **anchoring day two on day one's close.** Broke no stated invariant. |
| 8 | `DRY_RUN` filled instantly at signal price | Overstated results ~0.22R/trade against a 0.12R budget. |
| 9 | `TRADING_MODE`/`DRY_RUN` divergence | Silently disabled broker equity **and** reconciliation in paper mode. |
| 10 | Fixing paper mode created a TOTP collision | 3 modules logging in within one 30-second window; two always fail. |
| 11 | NSE 2026 holidays wrong | 11 of 245 days misclassified. |
| 12 | Universe screen returned empty | Nothing populated `last_price` / `avg_daily_value`. |
| 13 | Three of six alert conditions never raised | Feed stale, order rejected, consumer backlog. |
| 14 | Influx writer silently dropped 7 of 9 position fields | Including `open_positions` and `total_open_risk` — **dashboard panels would have been empty forever**, and an empty risk panel reads as "nothing is wrong". |
| 15 | `set_backlog(None)` raised into the trading loop | A *health-reporting* call could take a trading module down. |
| 16 | `allow_reuse_address` on Windows | Two modules could bind the same port; both report success, one silently answers nobody. |

**The pattern**: almost every defect looks healthy from outside. That is why "the process is
running" is worthless as a health signal, and why the test suite targets the paths that only
execute when something is already going wrong.

---

## 10. Operations

### Modes

```bash
TRADING_MODE=dry_run    # nothing fills. Wiring check ONLY — not paper trading.
TRADING_MODE=paper      # realistic fills: next-bar open, slippage, costs.
TRADING_MODE=live       # real orders. Only after the Phase 4 gate passes.
```

⚠️ **`dry_run` is not paper trading.** It ignores next-bar delay, slippage and costs.

### Start

```bash
docker compose up -d                      # redis + influxdb + grafana
TRADING_MODE=paper python run_all.py      # orchestrator picks the right filler
```

Or individually — order matters less than it used to, since consumer groups mean a late
starter picks up the backlog:

```bash
python -m src.storage_logger      python -m src.position_manager
python -m src.paper_broker        # OR src.execution_engine — never both
python -m src.risk_manager        python -m src.strategy_engine
python -m src.bar_builder         python -m src.data_ingestion   # last
```

### Monitor

| What | Where |
|---|---|
| Trading dashboard | <http://127.0.0.1:3000> → *LIGERBOT — Trading* |
| Operations dashboard | same → *LIGERBOT — Operations* |
| Module health | `curl 127.0.0.1:980{0..7}/health` |
| Pre-open go/no-go | `python -m src.briefing morning` |
| Post-close report | `python -m src.briefing evening` |
| Live-trading status | `python -m src.live_guard check` |

Health status semantics: `ok` · `degraded` (503 — wedged loop or backlog) · **`halted`
(200 — the kill switch working *is* the system working; 503 would tell an orchestrator to
restart the halt away)**.

### Stop

```bash
python -m src.kill_switch halt "reason"    # blocks new entries; exits still permitted
python -m src.kill_switch status
python -m src.kill_switch clear
```

Modules **fail closed** when they cannot read the kill switch — an unreachable Redis means
they are already refusing entries.

### Recover

| Symptom | Action |
|---|---|
| Order in doubt after a transit error | It stays `SENT`, never `REJECTED`. The ack timeout expires it; reconciliation establishes the truth. **Do not re-send manually.** |
| Reconciliation mismatch | The bot halts. Check `positions()` against the app before clearing. |
| Dead-lettered message | `XRANGE dead_letter - +`. On `approved_orders` it means a trade was never placed. |
| Feed died | Bounded reconnect, then halt. Check ingestion's log and health endpoint. |
| Session expired mid-day | Re-authentication is automatic; the shared session is republished. |

### Configuration

`.env` (gitignored) from `.env.example`. Never commit it. `LIVE_MAX_DAILY_LOSS` ships
**blank** on purpose so a clone cannot inherit someone else's risk tolerance.

Health endpoints and Grafana bind to **loopback** — they expose positions, P&L and open
risk. Widening that should be a deliberate act.

---

## 11. Roadmap

### Blocked on one command

```bash
.\.venv-kotak\Scripts\python.exe -m tools.probe_kotak_history
```

Read-only, places no orders. It settles **four unknowns at once**:

1. **`history()` depth** — the D5 question. Under ~6 months and the 200-trade gate is not
   reachable from backfill; a year or more and §2.6 may be reachable directly; nothing at
   all and self-recording forward becomes the only source.
2. **`limits()` field names** — the last unverified mapping, and the one that mis-sizes
   every trade by the same factor if wrong.
3. **`positions()` / `order_report()` shapes** — pinning the best-effort parsers.
4. **`scrip_master()` shape** — for the universe builder.

### Then, in order

1. Fix `account.py` against the real response; sanity-check parsed equity against the app.
2. Build a universe from a real scrip master.
3. Backfill whatever history exists; run the data-quality gate.
4. **Walk-forward `trend_pullback` v1 against the §2.6 gates.**

Step 4 is where the real answer lives.

### The outcome nobody should flinch from

`trend_pullback` v1 may have no edge. If it fails the gates on real data, the options are
**iterate** or **stop** — and stopping is a legitimate outcome, not a wasted effort. The
machinery would have done its job by telling you honestly that there is nothing there,
which is worth considerably more than a system that lets you find out with money.

### Deliberately not built

| Item | Why |
|---|---|
| Shorting | D3 — different margin and borrow treatment. Deferred, not forgotten. |
| Options / futures | Cash intraday first. |
| ML models | No edge demonstrated with rules yet; ML on an unvalidated premise multiplies the problem. |
| Multi-broker abstraction | One broker not yet working end to end. |
| Code from `openalgo` / `nautilus_trader` | **AGPL / LGPL.** This repo is public; copying either would force those licences onto the whole project. Capabilities can be reimplemented clean-room if needed. |

---

## Appendix — reference

**Session clock (IST):** pre-open 09:00 · open 09:15 · entries 09:30–14:45 ·
**square-off 15:10** (ahead of the broker's ~15:20 MIS cutoff) · close 15:30

**Test commands**
```bash
python -m pytest -q                                    # 1,340 tests
python -m pytest -q --cov=src --cov-report=term        # 95%
python demo_phase0.py demo_phase3.py demo_phase4.py demo_phase5.py
LIGERBOT_UPDATE_GOLDEN=1 python -m pytest tests/test_golden_pipeline.py   # re-approve, then READ THE DIFF
```

**Key documents**
| File | Contains |
|---|---|
| `MASTER.md` | this document |
| `DESIGN.md` | the reasoning behind every decision, at length |
| `README.md` | quick start |
| `tests/golden/pipeline_trace.txt` | the approved end-to-end decision trace |

**Environment isolation.** The Kotak SDK is *not* in `requirements.txt` — it hard-pins
`certifi`, `urllib3`, `websockets` and `pandas`, and installing it into a shared environment
downgrades them for every other project. That is not hypothetical; it happened during
development and broke this project's own test suite at C level. Use `.venv-kotak`.
