# LIGERBOT — Event-Driven Algorithmic Trading Bot

An advanced, event-driven **microservices** trading bot built on the **Kotak Neo Live API**,
following the architectural blueprint of five decoupled modules communicating over a
**Redis Streams** event bus.

> 📖 **[MASTER.md](MASTER.md) is the complete reference** — architecture, all 46 modules,
> the six settled decisions, risk and cost models, both strategies, the backtest → paper →
> live evidence ladder, every defect testing caught, and the operational runbook.
> This README is the quick start; [DESIGN.md](DESIGN.md) is the reasoning.

> ⚠️ **Risk disclaimer:** Algorithmic trading involves substantial risk of loss.
> **No strategy in this repository has been validated on real market data.** The bundled
> SMA-crossover is a deliberate **negative control** that loses money by design;
> `trend_pullback` v1 is implemented but currently **fails the go-live gates**. Run in
> paper/simulation mode. You are solely responsible for any trades this software places.
>
> A previous version of this file set a "60% win rate" target. That was the wrong
> objective and has been removed: win rate is meaningless without the reward-to-risk
> ratio attached. At 12.5% cost drag, a 2:1 strategy needs only a 37% win rate to profit,
> while a 1:1 strategy needs 56%. The objective is **expectancy net of costs**, with
> maximum drawdown as the binding constraint.

---

## Architecture

```
                         ┌──────────────────────────────────────────────┐
                         │                REDIS STREAMS                   │
                         │            (the Event Bus / spine)             │
                         └──────────────────────────────────────────────┘
   market_ticks ▲            trade_signals ▲        approved_orders ▲       filled_orders ▲
               │                          │                        │                     │
   ┌───────────┴──────┐   ┌───────────────┴────┐   ┌───────────────┴───┐   ┌─────────────┴────┐
   │ 1. Data Ingestion│──▶│ 3. Strategy Engine │──▶│ 4. Risk Manager   │──▶│ 5. Execution Eng │
   │ (Kotak Neo WS)   │   │ (SMA 10/50 cross)  │   │ (position sizing) │   │ (Kotak Neo REST) │
   └──────────────────┘   └────────────────────┘   └───────────────────┘   └──────────────────┘
                                                                                   │
                         ┌─────────────────────────────────────────────────────────┘
                         ▼
                  ┌────────────────────┐
                  │ 6. Storage/Logger  │  (InfluxDB time-series archive → Grafana)
                  │  passive observer  │
                  └────────────────────┘
```

Each module is an independent process. They never call each other directly — they only
read/write Redis Streams. This isolates failures (a crashed strategy engine can't take
down execution) and lets you scale, restart, or replace any module independently.

| # | Module | File | Consumes | Produces |
|---|--------|------|----------|----------|
| 1 | Data Ingestion | `src/data_ingestion.py` | Kotak Neo WebSocket | `market_ticks` |
| 2 | Event Bus | `src/event_bus.py` | (Redis Streams helper library) | — |
| 3 | Strategy Engine | `src/strategy_engine.py` | `market_ticks` | `trade_signals` |
| 4 | Risk Manager | `src/risk_manager.py` | `trade_signals` | `approved_orders` |
| 5 | Execution Engine | `src/execution_engine.py` | `approved_orders` | `filled_orders` |
| 6 | Storage / Logger | `src/storage_logger.py` | all streams | InfluxDB |

A test consumer (`src/consumer.py`) is included to verify the ingestion → event-bus wiring.

---

## Prerequisites

- **Python 3.10+**
- **Docker** (for Redis and InfluxDB) — or standalone installs
- A **Kotak Neo** trading account with **Trade API** access enabled (generate the
  consumer key / secret from the Trade API card in the Neo app/web).

---

## Quick start

### 1. Clone & install

```bash
pip install -r requirements.txt
```

The **Kotak SDK is deliberately not in `requirements.txt`.** `neo_api_client` hard-pins
`certifi==2022.12.7`, `urllib3==1.26.14`, `websockets==8.1` and `pandas==2.2.3`, so
installing it into a shared environment downgrades those for every other project there.
That is not hypothetical — it happened during development and broke this project's own test
suite at C level, along with several unrelated tools.

Nothing in `src/` imports the SDK at module scope, so the whole suite runs without it. Only
the processes that actually talk to the broker need it, and they should get it in isolation:

```bash
python -m venv .venv-kotak
.venv-kotak/bin/pip install -r requirements.txt -r requirements-broker.txt
.venv-kotak/bin/python -m tools.probe_kotak_history
```

### 2. Bring up the infrastructure (Redis + InfluxDB)

```bash
docker compose up -d
```

Then open <http://localhost:8086> to finish the InfluxDB setup — create an org
(e.g. `ligerbot`), a bucket (e.g. `trading_logs`), and an API token. Copy the token
into your `.env`.

### 3. Configure credentials

```bash
cp .env.example .env
# edit .env with your Kotak Neo credentials + InfluxDB token
```

### 4. Run the modules

Launch everything with the orchestrator — it starts the right executor for the mode:

```bash
TRADING_MODE=dry_run python run_all.py    # wiring check, nothing fills
TRADING_MODE=paper   python run_all.py    # realistic simulated fills
```

Or run each in its own terminal. Order matters less than it used to — consumer groups
mean a module that starts late picks up the backlog rather than skipping it:

```bash
python -m src.storage_logger      # archive everything
python -m src.position_manager    # positions, P&L, broker reconciliation
python -m src.paper_broker        # paper fills   (OR src.execution_engine for live)
python -m src.risk_manager        # gate signals into orders
python -m src.strategy_engine     # generate signals
python -m src.bar_builder         # ticks -> bars
python -m src.data_ingestion      # start the live feed (do this last)
```

Run **either** `paper_broker` **or** `execution_engine`, never both — both would fill
every order and double-count every trade.

### Choosing a mode

```bash
TRADING_MODE=dry_run    # nothing fills. Wiring check only.
TRADING_MODE=paper      # realistic fills: next-bar open, slippage, costs. Phase 4.
TRADING_MODE=live       # real orders. Only after the Phase 4 gate passes.
```

⚠️ **Do not use `dry_run` as paper trading.** It fills nothing, and the *old* DRY_RUN
behaviour — instant fills at the signal price — overstated results by ~0.22R per trade
against a ~0.12R friction budget. That is what `paper` mode exists to replace; see
[Phase 4](#phase-4--paper-trading).

---

## Safety features

**Working today (Phase 0 complete — see [DESIGN.md](DESIGN.md)):**

- **Kill switch** — `TRADING_MODE` (`dry_run`/`paper`) stops any real order from leaving
  the machine, plus a live Redis halt flag that needs no restart (Phase 3).
- **Total open-risk cap** (1.5% of session equity) — the primary control. Caps the *sum of
  distance-to-stop* across open positions, because counting positions bounds nothing.
- **Per-trade risk cap** (0.5%) drives position sizing. A signal without a stop-loss is
  rejected outright rather than silently sized by notional exposure.
- **Worst-case daily gate** — new entries are refused once realised losses *plus open risk*
  would breach the 2% daily limit, so the limit bounds the day rather than merely tripping
  after the fact.
- **Max daily drawdown** halt (2% of session equity), now driven by real P&L.
- **Equity read from the broker** each session and pinned for the day, computed as
  `cash + MTM` — never from margin, which embeds ~5x MIS leverage.
- **Universe validation** at startup rejects anything not tradable intraday in the cash
  segment (including indices).
- **Self-owned square-off** at 15:10, ahead of the broker's ~15:20 MIS auto-cutoff.
- **Client-side rate limiting** on the execution path to respect broker API limits.

**Added in Phase 3 (hardening):**

- **Durable event delivery.** Consumer groups with explicit `XACK`. Events queued while a
  module is down are delivered on restart, and unacked work is reclaimed after a crash.
  *(An earlier README claimed this already worked. It did not — every module read from
  `"$"` and silently discarded queued orders. That was defect B6.)*
- **Idempotent orders.** Each order carries a deterministic client id derived from its
  signal, with a Redis-backed dedupe set. At-least-once delivery plus idempotency gives
  exactly-once *effects*.
- **A real order state machine.** `PENDING → SENT → ACKED → PARTIAL → FILLED`, with
  terminal branches for rejection, cancellation and expiry. Acceptance is no longer
  mistaken for a fill.
- **Position manager with broker reconciliation.** The broker is authoritative; a
  disagreement halts trading rather than being corrected away.
- **Feed staleness detection.** Per-instrument liveness with TTL expiry. A stale feed
  blocks entries and never blocks exits. Bounded reconnection with backoff.
- **Kill switch.** A Redis flag halting new risk across every module without a restart.
  Fails closed if Redis is unreachable.
- **Bounded streams.** `MAXLEN` on every stream, so Redis cannot grow until it refuses
  writes mid-session.

**Added in Phase 4 (paper trading):**

- **Realistic paper fills** driving the backtester's own `SimBroker`, so paper and
  backtest cannot diverge in fill semantics.
- **Paper-vs-backtest reconciliation** with divergence attribution and an unattributed
  residual check.
- **Daily briefings** — a morning go/no-go and an evening post-close report.

**Added in Phase 5 (live safeguards):**

- **Live guard.** `TRADING_MODE=live` is *not* sufficient. Live additionally requires
  passed backtest gates, a passed paper gate, a completed broker probe, a loaded
  instrument master, resolved equity, and a human-written authorisation file that expires
  after 7 days and must match the account's capital. No override exists.
- **Scaling ladder.** Starts at 10% size; promotion needs sessions *and* trades *and*
  expectancy; demotion skips straight to the floor.
- **Absolute rupee backstops** that hold even when the percentage limits are computed from
  a wrong equity figure.

**Still not done:**

- ⚠️ **No strategy has been validated on real data.** `trend_pullback` v1 currently
  **fails the go-live gates** (DESIGN.md 2.6). Hardening makes the machinery trustworthy;
  it says nothing about whether the strategy makes money.
- ⚠️ **Zero paper sessions completed.** The Phase 4 gate needs 20+ (40+ if the backtest
  evidence was thin). That is calendar time against live data — not something code can
  shorten.
- ⚠️ **The Kotak probe has never been run.** Historical depth is unmeasured, and the
  equity field names in `src/account.py` are still unverified guesses.

See `config.py` for every tunable parameter and [DESIGN.md](DESIGN.md) for the full
roadmap and the defect register.

---

## Project layout

```
LIGERBOT/
├── README.md
├── DESIGN.md                 # architecture, decisions, defect register, roadmap
├── requirements.txt
├── docker-compose.yml
├── .env.example
├── .gitignore
├── conftest.py               # puts the repo root on sys.path for pytest
├── config.py                 # central config (reads .env)
├── run_all.py                # optional multi-process orchestrator
├── demo_simulate.py          # end-to-end pipeline demo (fakeredis)
├── demo_phase0.py            # Phase 0 verification — no Redis, broker or market hours
├── demo_phase1.py            # Phase 1 verification — backtest harness + negative control
├── demo_phase2.py            # Phase 2 verification — strategy v1 + go-live gates
├── demo_phase3.py            # Phase 3 verification — hardening, chaos, kill switch
├── demo_phase4.py            # Phase 4 verification — paper trading machinery
├── demo_phase5.py            # Phase 5 verification — live safeguards (shows refusal)
├── tools/
│   └── probe_kotak_history.py  # measures Kotak's real history depth  [run first]
├── src/
│   ├── __init__.py
│   ├── auth.py               # Kotak Neo authentication helper
│   ├── event_bus.py          # Redis Streams helpers
│   ├── market_calendar.py    # NSE sessions, holidays, phases        [Phase 0]
│   ├── instruments.py        # instrument master + universe screen    [Phase 0]
│   ├── account.py            # broker equity + session snapshot       [Phase 0]
│   ├── bars.py               # pure time-bar aggregation              [Phase 0]
│   ├── bar_store.py          # Parquet historical store               [Phase 0]
│   ├── bar_builder.py        # Module 2b — ticks to bars              [Phase 0]
│   ├── risk_engine.py        # pure risk logic (no I/O)               [Phase 0]
│   ├── order_state.py        # order lifecycle + idempotency          [Phase 3]
│   ├── position_manager.py   # Module 7 — positions, P&L, reconcile   [Phase 3]
│   ├── feed_health.py        # staleness, reconnect, heartbeat        [Phase 3]
│   ├── kill_switch.py        # halt flag + CLI                        [Phase 3]
│   ├── paper_broker.py       # realistic paper fills                  [Phase 4]
│   ├── session_recorder.py   # per-session results                    [Phase 4]
│   ├── reconciliation.py     # paper vs backtest + attribution        [Phase 4]
│   ├── briefing.py           # morning / evening reports              [Phase 4]
│   ├── live_guard.py         # hard gate before live trading          [Phase 5]
│   ├── live_scaling.py       # position-size ladder                   [Phase 5]
│   ├── strategy_base.py      # Strategy interface + registry          [Phase 1]
│   ├── indicators.py         # incremental indicator library         [Phase 2]
│   ├── strategies/           # sma_crossover (control), trend_pullback (v1)
│   ├── backtest/             # cost model, fills, metrics, walk-forward, gates
│   ├── data_ingestion.py     # Module 1
│   ├── consumer.py           # test consumer for the event bus
│   ├── strategy_engine.py    # Module 3 (reference SMA — see note)
│   ├── risk_manager.py       # Module 4 (Redis wrapper; logic moving to risk_engine)
│   ├── execution_engine.py   # Module 5
│   └── storage_logger.py     # Module 6
└── tests/                    # 1340 tests, 95% coverage (incl. tests/golden/ — §3.10)
```

> **On the bundled SMA-crossover strategy:** it is retained as an interface example and as
> the backtester's *negative control* — a correct cost model must show it **losing money**
> (DESIGN.md 2.5). It computes averages over raw ticks rather than time bars, so its
> horizon changes with liquidity. It is not a strategy to trade.

## Phase 0 — foundations

Verify the whole foundation layer with no infrastructure at all:

```bash
python demo_phase0.py    # calendar, universe, bars, Parquet, equity, risk
python -m pytest -q      # 1340 tests
```

New in Phase 0:

| Module | Purpose |
|--------|---------|
| `src/market_calendar.py` | NSE holidays, session windows, and the phase machine that decides when entries and exits are permitted. |
| `src/instruments.py` | Scrip-master parsing, token ↔ trading-symbol mapping, liquidity screen, and the startup gate that rejects untradable instruments. |
| `src/bars.py` | Pure tick→bar aggregation. Shared byte-for-byte between live and backtest, which is what makes backtest results meaningful. |
| `src/bar_store.py` | Partitioned Parquet store. Recording starts now so the dataset is mature by the time Phase 2 needs it. |
| `src/bar_builder.py` | The runner: `market_ticks` → `market_bars` + Parquet. |
| `src/account.py` | Broker equity retrieval and the pinned session snapshot. |
| `src/risk_engine.py` | Pure risk logic — sizing, the open-risk cap, and the worst-case daily gate. |

## Phase 1 — backtest harness

```bash
python demo_phase1.py    # costs, quality gate, negative control, walk-forward
```

| Module | Purpose |
|--------|---------|
| `src/backtest/costs.py` | The Indian intraday cost stack — brokerage, STT, exchange, GST, SEBI, stamp duty — charged per leg, plus a pessimistic slippage model. |
| `src/backtest/bar_source.py` | The `BarSource` seam (Parquet / in-memory / live) and the data-quality gate. |
| `src/backtest/sim_broker.py` | Fill simulation: next-bar-open execution, pessimistic intrabar resolution, gaps through stops. |
| `src/backtest/portfolio.py` | Trade ledger and equity curve, with friction split out from P&L. |
| `src/backtest/metrics.py` | Full metric set plus breakdowns by hour, month, instrument and session direction. |
| `src/backtest/engine.py` | Drives bars through the **same** strategy and risk-engine objects the live system uses. |
| `src/backtest/walk_forward.py` | Rolling optimise-then-test, trial counting, locked holdout, parameter-stability table. |
| `src/backtest/synthetic.py` | Driftless random-walk generator, for validating the harness itself. |
| `src/strategy_base.py` | The `Strategy` interface and registry. |
| `src/strategies/sma_crossover.py` | The **negative control** — not a strategy to trade. |
| `tools/probe_kotak_history.py` | Measures what Kotak's API actually serves. **Run this first.** |

### What the harness guarantees

- **No look-ahead.** Signals from bar *t* fill at bar *t+1*'s open. Property-tested: a
  strategy's decisions over bars `0..t` are byte-identical whether or not later bars exist
  in the source.
- **Friction is visible.** Reports split *frictionless* P&L (what the signal was worth)
  from slippage and from charges. Reporting only net hides whether a strategy had no edge
  or had one that costs ate — different problems, different fixes.
- **Pessimism where the data is silent.** A bar records only its extremes, so when its
  range contains both stop and target, the stop is assumed to have hit first. Gaps through
  a stop fill at the open, not at the stop.
- **Bad data is refused, not warned about.** Unadjusted corporate actions, sparse
  sessions, inconsistent OHLC and excess gap-fill all block a run.

### The negative control

`sma_crossover` exists to be *wrong*. On a driftless random walk it must lose money after
costs — and it does, at −0.097R per trade. The attribution is the real check:

```
Frictionless expectancy  +0.010R   the signal itself (~0 = no edge, by construction)
Friction                  0.107R   vs the ~0.12R hurdle predicted in DESIGN.md 5.2
Net expectancy           -0.097R
```

The harness reproduces the analytical cost hurdle through full fill simulation, from a
completely independent path. **If this control ever shows a profit, the harness is broken
and nothing else it reports can be believed.**

## Phase 2 — strategy and validation gates

```bash
python demo_phase2.py    # indicators, strategy, interval sweep, controls, gates
```

| Module | Purpose |
|--------|---------|
| `src/indicators.py` | Incremental EMA, SMA, session VWAP, ATR, RSI, ADX, opening range, relative volume. O(1) per update, replay-deterministic, explicit warmup. |
| `src/strategies/trend_pullback.py` | Strategy v1 — long-only trend-pullback with ATR risk, regime filter and three exits. |
| `src/backtest/gates.py` | Automated §2.6 go/no-go evaluation. |
| `src/backtest/bar_source.py` | (extended) `resample_frame` / `ResampledBarSource` for the D4 interval sweep. |

### The economic stop floor

Building v1 exposed a constraint the design had missed. Risk-based sizing means
`notional = risk_budget / stop_distance`, so a **tighter stop produces a larger position** —
and friction scales with notional while the risk budget does not:

```
friction_in_R  ≈  0.00085 / stop_pct
```

A 0.4% stop costs 0.21R per trade; a 0.8% stop costs 0.11R. To stay under the ~0.12R hurdle
the stop must be **at least ~0.7% of price**. The interval sweep confirms it:

| interval | avg stop | friction |
|---------:|---------:|---------:|
| 1m | — | *no qualifying trades* |
| 3m | 0.76% | 0.131R |
| 5m | 0.88% | 0.117R |
| 15m | 1.63% | 0.072R |

This settles D4 on firmer ground than "fewer round trips": 1-minute ATR stops are
structurally uneconomic regardless of signal quality. The strategy now refuses any trade
whose stop falls below the floor.

### Both controls

A negative control alone is only half a test — a strategy that never trades passes it. So
there is also a **positive control**: synthetic data with genuine return persistence, plus a
unit test that puts a textbook trend-pullback in front of the strategy and asserts it takes
the trade. That separates *"the strategy is blind"* from *"the data has nothing to find"*.

The unit-level control passes. The aggregate one is weak — frictionless expectancy rises
with injected momentum (−0.034R → +0.046R) but stays small against friction. Whether that is
the generator lacking the structure v1 targets, or v1 detecting weakly, **synthetic data
cannot say**. Real data can.

### Gate status

On synthetic data the gates **BLOCK, with 7 failures** — which is correct. A strategy that
passed on a random walk would mean the gates were broken, not that the strategy works.
Phase 2's exit criterion is the gates passing **out-of-sample on real market data**, and
that is still blocked on the Kotak probe.

## Phase 3 — production hardening

```bash
python demo_phase3.py    # B6, crash recovery, breaker, staleness, kill switch
python demo_simulate.py  # the whole pipeline end to end, in-process
```

| Module | Purpose |
|--------|---------|
| `src/event_bus.py` | (extended) `StreamConsumer` / `MultiStreamConsumer` — consumer groups, `XACK`, `XAUTOCLAIM`, dead-letter routing, `MAXLEN` trimming. |
| `src/order_state.py` | Order state machine, deterministic client order ids, Redis-backed dedupe. |
| `src/position_manager.py` | Module 7 — the source of truth for positions and P&L, with broker reconciliation. |
| `src/feed_health.py` | Per-instrument staleness, TTL liveness keys, bounded reconnection, heartbeats. |
| `src/kill_switch.py` | Redis halt flag + CLI. Fails closed. |

### Loss and duplication are different failures

They have different causes and need different fixes, and fixing only one trades a silent
failure for a loud one:

| | cause | fix |
|---|---|---|
| **Orders lost** | at-most-once delivery (`"$"` cursors) | consumer groups + `XACK` |
| **Orders duplicated** | at-least-once delivery *without* idempotency | deterministic client order id + dedupe set |

At-least-once delivery **plus** idempotency gives exactly-once effects. Getting only the
first half would turn a silent dropped order into a loud double-fired one.

### Everything fails closed, and nothing traps a position

Two rules run through the whole system. Every gate — kill switch, feed staleness, halted
breaker, reconciliation mismatch, unreachable Redis — blocks **new risk** and permits
**exits**. A switch that also blocked exits would strand positions in exactly the
situation where someone reached for it.

```bash
python -m src.kill_switch status
python -m src.kill_switch halt "investigating fills"
python -m src.kill_switch clear
```

## Phase 4 — paper trading

```bash
python demo_phase4.py              # fills, recording, reconciliation, briefings, gate

TRADING_MODE=paper python run_all.py
python -m src.briefing morning     # each pre-open
python -m src.briefing evening     # each post-close
```

| Module | Purpose |
|--------|---------|
| `src/paper_broker.py` | Realistic paper fills, driving the **backtester's own** `SimBroker`. |
| `src/session_recorder.py` | Per-session results — trades, rejections, halts — for both paper and backtest. |
| `src/reconciliation.py` | Paper-vs-backtest comparison with divergence **attribution**. |
| `src/briefing.py` | Morning go/no-go and evening post-close reports. |

### Three trading modes

`TRADING_MODE` is the single source of truth; `DRY_RUN` is **derived** from it:

| Mode | Broker session | Real orders | Fills |
|---|---|---|---|
| `dry_run` | no | no | instant, at the signal price — a wiring check |
| `paper` | **yes** | no | simulated: next-bar open, slippage, costs, liquidity cap |
| `live` | yes | **yes** | the exchange |

**Paper mode needs a real broker session.** Equity comes from the broker (D1) and the
position manager reconciles against it — only *order placement* is simulated. When
`TRADING_MODE` and `DRY_RUN` were independent settings they could disagree, and paper mode
silently ran on a *configured* equity figure with reconciliation disabled. Since Phase 4
exists to reconcile paper against backtest, that invalidated the very sessions it was
accumulating. They can no longer diverge.

An unrecognised `TRADING_MODE` fails at import rather than defaulting — a typo there would
otherwise decide whether real orders are sent.

`run_all.py` starts **exactly one** filler per mode, and each filler refuses to run in a
mode it does not own. Both consume `approved_orders` from separate consumer groups, so
running both would fill every order twice — while looking entirely healthy.

### Why paper mode had to be rebuilt

`DRY_RUN` filled instantly at the *signal's* price — no next-bar delay, no slippage, no
costs. On a realistic 15bps gap that overstates results by **~0.22R per trade**, against a
total friction budget of ~0.12R, always favourably. Paper would have beaten the backtest
for entirely artificial reasons and the reconciliation would have read that as the backtest
being conservative.

An optimistic paper mode is worse than none — it manufactures confidence rather than
merely failing to provide it. Paper now runs the same `SimBroker` the backtester does, so
the two cannot diverge in fill semantics.

### Attribution, not just divergence

"Paper made 12,000 less" cannot be acted on. Each cause has a different fix:

| Cause | What it means |
|---|---|
| Missing trades | Signalled in replay but not live — feed gap, stale-feed block, or halt |
| Extra trades | Live traded where the backtest did not — usually indicator warmup |
| Fill-price divergence | Slippage model wrong, or execution slower than modelled |
| Cost divergence | Cost model doesn't match the contract note |

The report also shows the **unattributed residual**, and a large one fails the gate on its
own — if the diagnosis can't be trusted, it matters more than the thing being diagnosed.

## Phase 5 — live safeguards

```bash
python demo_phase5.py                 # shows the guard REFUSING, which is correct today
python -m src.live_guard check        # what still blocks live trading
python -m src.live_guard authorise --capital 300000 --by "your name"
```

| Module | Purpose |
|--------|---------|
| `src/live_guard.py` | Hard gate. `TRADING_MODE=live` is not sufficient on its own. |
| `src/live_scaling.py` | Position-size ladder — minimum size, evidence-gated promotion. |

### Live trading cannot be enabled by an environment variable

An env var is one typo, one copied `.env`, one careless export away from real capital. The
guard also requires **evidence** (§2.6 gates and the Phase 4 paper gate passed),
**correctness** (broker probe run, so the equity field mapping is verified rather than
guessed), and **intent** — an authorisation file naming the date, capital and person.

The authorisation expires after 7 days, and is rejected if the account holds materially
more than was authorised. Every check defaults to *not passed*, and **there is no
override** — a bypass becomes the thing someone reaches for at 09:14 in a hurry.

### The ladder is deliberately asymmetric

| | requires |
|---|---|
| **Promotion** | minimum sessions **and** trades **and** expectancy still tracking |
| **Demotion** | 3 consecutive losing sessions, a halt, or expectancy through the floor — straight to the floor, no stepping down |

46 profitable sessions to reach full size in the demo; three bad ones to lose it.
Promoting too slowly costs a little upside; promoting too quickly costs capital.

### Absolute backstops

Every D2 limit is a *fraction of equity* — exactly wrong if the equity figure is wrong.
Equity mis-read 10× high turns the "2% daily limit" into 20% of the real account.
`LIVE_MAX_DAILY_LOSS` and `LIVE_MAX_ORDERS_PER_DAY` bound the damage in rupees regardless.
Live mode only.

### The order is not negotiable

```
1. python -m tools.probe_kotak_history        <- still the blocker
2. backfill + walk-forward until §2.6 gates pass
3. TRADING_MODE=paper, 20+ sessions, reconcile daily
4. python -m src.live_guard authorise ...
5. TRADING_MODE=live at minimum size
```

---

## Disclaimer

This is educational/reference software. No warranty. Trading decisions and losses are
entirely your own responsibility.
