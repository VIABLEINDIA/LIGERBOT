# LIGERBOT — Design Document

**Target market:** NSE equities & index instruments, **intraday only** (MIS, flat by session close).
**Status:** machinery built and tested; **the strategy is unvalidated and live trading is
blocked.** Phases 0–5 are implemented — bar building, backtest harness, risk engine,
paper broker, reconciliation, live guard — behind 1,239 tests at 95% coverage. What does
*not* exist is evidence: no real market data has been through it, `trend_pullback` v1 has
never been walk-forward tested against anything but synthetic prices, and the go-live guard
refuses on seven outstanding prerequisites. That refusal is the system working.

Read §4 for per-phase status. Every "complete and verified" there means the *machinery* is
verified; none of them means the strategy makes money.

**Scope of this document:** design and rationale. It was written before the code and now
records both — the six resolved decisions (§5.1), the order things were built (§4), and
§5.2 onward, which is a log of defects found *after* they were built. That section is the
most useful part of this file: it is what testing caught that reading never would have.

**Settled:** equity read live from the broker each session (D1) · 2% daily loss limit with a
1.5% total open-risk cap and 0.5% per trade (D2) · long-only v1 (D3) · 1-minute bars stored,
5-minute traded (D4) · Kotak-only history with four compensating mitigations (D5) ·
8–12 name liquidity screen replacing the untradable index default (D6).

---

## 0. Where we actually are

The five-module event-driven pipeline works: `demo_simulate.py` pushes synthetic ticks
through `StrategyEngine → RiskManager → ExecutionEngine → StorageLogger` over fakeredis
and events land on all four streams. That is a real achievement and the architecture
(Redis Streams spine, process isolation) is sound and worth keeping.

But the current system **cannot generate a trustworthy edge and cannot safely place a
real order.** Before designing forward, here is the honest audit, because the design
below is shaped entirely by these gaps.

### 0.1 Blocking defects in what exists

| # | Where | Problem | Consequence |
|---|-------|---------|-------------|
| B1 | `src/strategy_engine.py:51` | SMAs are computed over **raw ticks**, not time bars. `SMA_LONG=50` means "last 50 ticks" — possibly 2 seconds of data. | The strategy has no defined time horizon. Its behaviour changes with tick rate, which changes with liquidity and time of day. It is not a strategy, it's noise. |
| B2 | `src/risk_manager.py:36,65` | `realized_pnl_today` is initialised to `0.0` and **never written to by anything**. | `_daily_drawdown_breached()` can never return True. The headline safety feature — max daily drawdown halt — is dead code. |
| B3 | `src/execution_engine.py:115` | Publishes to `filled_orders` on broker **acceptance**, never on actual fill. Nothing consumes `filled_orders` except the archiver. | The loop is open: the bot never learns what it actually owns or what it made/lost. This is the root cause of B2. |
| B4 | `src/execution_engine.py:93` | `trading_symbol=instrument`, where `instrument` is the display name carried from the tick (e.g. `"Nifty 50"`). | Every live order would be rejected. There is no instrument master mapping token ↔ trading symbol ↔ lot/tick size. |
| B5 | `config.py:50` | Default universe is `11536:Nifty 50:nse_cm` — the **Nifty 50 index is not tradable** in the cash segment. | The default configuration describes an instrument that cannot be bought. Index exposure intraday requires F&O (out of scope) or an ETF such as NIFTYBEES. |
| B6 | All modules (`last_id = "$"`) | Every consumer starts at "only messages from now on" and never persists a cursor. | `README.md:120` claims "idempotent stream cursors so a restarted module never re-processes old events." In reality a restart **silently discards every queued event**. Losing an approved order is worse than replaying one. |
| B7 | `src/risk_manager.py:97` | The strategy emits no `stop_loss`, so sizing always falls to the exposure branch. | `RISK_PER_TRADE` (the documented 1% risk cap) is never actually applied. Sizing is pure notional exposure. |
| B8 | `src/risk_manager.py:118` | Net signed quantity conflates "close long" with "open short". | A BUY→SELL crossover pair nets to zero and pops the position, so the bot cannot distinguish reversing from flattening, and the `MAX_OPEN_POSITIONS` cap is measuring the wrong thing. |
| B9 | Nothing anywhere | No market-hours check, no NSE holiday calendar, no end-of-day square-off. | MIS positions get force-squared-off by the broker at ~15:20 at whatever price the market offers. The bot must own its exit, not the broker. |
| B10 | `src/data_ingestion.py:86` | `_on_close` logs a warning and does nothing. No reconnect, no staleness watchdog. | The WebSocket dies, the process stays alive, the strategy keeps trading on a frozen last price. This is the classic way an intraday bot bleeds out. |
| B11 | `src/risk_manager.py:59` | Default config: `qty = (100_000 × 0.10) / 24_500 = 0` → rejected. | The shipped defaults produce zero trades. `demo_simulate.py:37` has to raise equity to ₹50L to make anything happen. |
| B12 | `src/storage_logger.py:57` | `SYNCHRONOUS` Influx writes, one point per tick, no batching or trimming; Redis streams have no `MAXLEN`. | Under real tick volume the archiver blocks and Redis memory grows without bound. |

**Consequence for the plan:** the strategy layer cannot be designed on top of ticks, and no
strategy can be evaluated until a backtester with a realistic cost model exists. That
dictates the build order in §4 — foundations, then backtester, then strategy, then hardening.

Every defect above is assigned to a phase in §4. B4, B5, B9 and B11 are resolved in Phase 0
by decisions D1 and D6 (§5.1); B1, B7 and B8 in Phases 0–2; B2, B3, B6, B10 and B12 in Phase 3.

### 0.2 What we keep

- The Redis Streams event-bus architecture and process isolation.
- The module boundaries and stream vocabulary.
- `src/event_bus.py` publish/decode helpers (the read path needs replacing, §3.1).
- The `DRY_RUN` kill switch concept (`config.py:67`) — it gets extended, not replaced.

---

## 1. Strategy layer

### 1.1 Core principle: one strategy class, two transports

The single most important design rule in this document:

> **The exact same `Strategy` object must run unchanged in backtest, paper, and live.**
> Only what feeds it bars differs.

Any strategy that is reimplemented for backtesting will diverge from the live one, and
every backtest result becomes a lie. This constraint drives the interfaces below.

### 1.2 Bars, not ticks

Insert a **bar builder** between ingestion and strategy.

```
market_ticks ──▶ [ Bar Builder ] ──▶ market_bars ──▶ [ Strategy Engine ]
```

- New module `src/bar_builder.py`, new stream `market_bars`.
- Aggregates ticks into fixed time bars (default 1-minute, configurable `BAR_INTERVAL`).
- Emits a bar **only when it closes**, tagged `closed=true`. A strategy must never see a
  forming bar — that is the most common source of accidental look-ahead.
- Bar schema: `instrument_id, bar_start, bar_end, open, high, low, close, volume, vwap, tick_count`.
- Gap handling: if no ticks arrived in an interval, emit a synthetic flat bar
  (`o=h=l=c=prev_close, volume=0`) so indicator windows stay time-aligned. Mark it
  `synthetic=true` so the strategy can choose to distrust it.
- Session-anchored: the first bar of each trading day resets VWAP and any session state.

Why a separate module rather than doing this inside the strategy: multiple strategies and
the backtester all need identical bars, and the archived `market_bars` stream becomes the
historical dataset for backtesting (§2.2).

### 1.3 The `Strategy` interface

```
class Strategy(ABC):
    name: str
    params: dict                      # everything tunable, for sweeps + provenance
    warmup_bars: int                  # bars required before signals are trusted

    def on_bar(self, bar: Bar, ctx: StrategyContext) -> list[Signal]: ...
    def on_session_start(self, day: date) -> None: ...
    def on_session_end(self, day: date) -> list[Signal]: ...   # forced flat
```

`StrategyContext` gives read-only access to current position, session clock, and time
remaining in the session. It gives **no** access to Redis, the broker, or account equity —
strategies decide *direction and levels*, never size. Sizing is the risk manager's job and
must stay there.

### 1.4 Signal schema (replaces the current thin payload)

The current signal (`strategy_engine.py:79`) carries direction but no risk levels, which is
why B7 exists. The new schema:

| Field | Purpose |
|-------|---------|
| `intent` | `OPEN_LONG` / `CLOSE_LONG` / `OPEN_SHORT` / `CLOSE_SHORT` — **not** bare BUY/SELL. Fixes B8. |
| `instrument_id` | Canonical id from the instrument master, not a display name. Fixes B4. |
| `ref_price` | Close of the signalling bar (for logging/slippage attribution only). |
| `stop_loss` | **Mandatory on every OPEN.** Drives risk-based sizing. Fixes B7. |
| `take_profit` | Optional; null means "managed by exit rules". |
| `bar_time` | Close time of the bar that produced the signal — the audit anchor. |
| `strategy_name`, `strategy_version`, `params_hash` | Provenance: which exact configuration produced this. |
| `reason` | Short human string, e.g. `"ema_cross_up + above_vwap + adx>20"`. Invaluable in post-mortems. |

A signal without a `stop_loss` on an OPEN intent is **rejected by the risk manager**, not
silently sized by exposure. Making this a hard contract is what makes the 1%-risk rule real.

### 1.5 Indicator library

New `src/indicators.py`. Requirements:

- **Incremental / O(1) per update.** The current `sma()` (`strategy_engine.py:31`) does
  `list(values)[-period:]` and re-sums on every tick — O(N) per instrument per tick. Use
  running sums for SMA, recursive updates for EMA/RSI/ATR.
- Each indicator is a small stateful object with `update(bar) -> float | None`, returning
  `None` until warmed up.
- Every indicator must be **replay-deterministic**: same bar sequence in, same values out,
  no wall-clock or randomness.

Initial set (chosen for intraday equities specifically):

| Indicator | Why it's in the intraday set |
|-----------|------------------------------|
| EMA(fast/slow) | Trend direction, more responsive than SMA on short bars. |
| **Session VWAP** | The single most-watched intraday reference on NSE. Institutional flow anchors to it; price above/below VWAP is the cleanest intraday bias filter available. |
| ATR(14) | Volatility-normalised stop distance — the input that makes risk-based sizing meaningful across instruments at different price levels. |
| ADX(14) or ATR%-of-price | **Regime filter.** Suppresses entries in chop. |
| RSI(14) | Exhaustion / no-chase filter. |
| Opening-range high/low (first 15 min) | Classic, well-documented intraday NSE reference level. |
| Relative volume | Avoid trading illiquid drift. |

### 1.6 Reference strategy v1 — trend-pullback with regime filter

The bundled SMA-crossover is a **known intraday loser**: a raw moving-average cross on
short bars is a chop-maximising machine. It buys every false break and pays costs on each.
It stays in the repo as an interface example and as the backtester's *negative control* —
a correct backtester must show it losing money after costs. If it shows a profit, the
backtester is wrong.

Strategy v1 design:

**Bias (per session, per instrument)**
- Long entry permitted when `close > session_VWAP` and `EMA_fast > EMA_slow`.
- **v1 is long-only (D3).** The mirrored short logic is specified but gated behind
  `ALLOW_SHORT=false`; the risk manager rejects short intents while the flag is off, so
  enabling it later needs no schema or pipeline change.

**Regime gate**
- No entries unless `ADX(14) > adx_min` (default 20) — i.e. a trend actually exists.
- No entries when `ATR% < atr_floor` (dead tape) or `> atr_ceiling` (news chaos).

**Entry**
- With bias established, enter on a **pullback that holds**: price retraces toward
  EMA_fast/VWAP and then closes back in the direction of bias. Entering on pullbacks
  rather than on the cross itself is what avoids paying the spread at every false break.
- Optional confirmation: relative volume above its session median.

**Exit** (all three active simultaneously)
1. **Initial stop:** `entry ∓ (atr_mult × ATR)`, default `atr_mult = 1.5`. This is the
   `stop_loss` that ships on the signal and drives sizing.
2. **Trailing stop:** once `+1R` is reached, trail at `atr_mult × ATR` from the extreme.
3. **Time stop:** exit if the trade hasn't reached `+0.5R` within `N` bars (default 20) —
   capital tied up in a non-mover is capital not working, and intraday time is finite.

**Session rules (hard, non-overridable by any strategy)**
- No entries before **09:30** — the opening auction and first 15 minutes are a different
  statistical regime.
- No new entries after **14:45**.
- **All positions flat by 15:10**, via the bot's own market orders. Never let the broker's
  ~15:20 MIS auto-square-off be the exit — that is an uncontrolled fill at an
  adversarial moment. Fixes B9.

**On the "60% win rate" goal in the README.** That target should be deleted. Win rate alone
is not an objective — a strategy can win 90% of the time and be ruinous. The objective
this design optimises is **expectancy net of the full Indian cost stack** (§2.3), with
max drawdown as the binding constraint. A 45%-win-rate strategy at 2R average win is a
far better business than a 65%-win-rate strategy at 0.4R.

### 1.7 Strategy registry

Strategies are registered by name and instantiated from config
(`STRATEGY=trend_pullback_v1`), so the backtester, paper runner, and live engine all
select the same way. Each run records `(strategy_name, version, params_hash)` on every
signal, so any trade in InfluxDB is traceable to the exact configuration that produced it.

---

## 2. Backtesting & validation harness

**This gets built before strategy v1 is written.** Without it, strategy design is guessing.

### 2.1 Architecture

```
  BarSource ──▶ BacktestClock ──▶ Strategy ──▶ RiskEngine ──▶ SimBroker ──▶ Portfolio
  (parquet)          │            (SAME class as live)   (SAME class)      │
                     └────────────────── metrics ◀─────────────────────────┘
```

The critical move: **extract the pure decision logic out of the Redis-coupled modules.**

- `RiskManager` (`src/risk_manager.py`) splits into `RiskEngine` (pure: signal + state in,
  decision out, no I/O) and a thin Redis wrapper. The backtester uses `RiskEngine` directly.
- Same for the strategy — already pure if it only touches `on_bar`.

This guarantees the backtest exercises the *same* sizing, drawdown, and position-cap code
that runs live. Any risk rule tested only in the backtester and reimplemented for live is
a rule that will eventually differ.

### 2.2 Historical data

- Canonical store: **Parquet, partitioned by `instrument/date`**, holding 1-minute bars.
  Columns match the `market_bars` schema exactly (§1.2).
- `BarSource` interface with three adapters:
  - `ParquetBarSource` — backtesting.
  - `BrokerHistoricalSource` — bulk backfill from the broker's historical endpoint.
  - `LiveStreamBarSource` — reads `market_bars` from Redis; used by paper/live.
- Data quality gate on ingest, and it must be strict: missing-bar detection, zero-volume
  runs, price spikes beyond `n × ATR`, and **corporate-action adjustment** (splits and
  bonuses silently destroy equity backtests). Every dataset is stamped with a quality
  report; a backtest on unvalidated data refuses to run.
- Ideal dataset for a confident conclusion: **2+ years** of 1-minute bars across the
  intended universe, spanning at least one high-volatility and one low-volatility regime.

**Reality check (D5).** Backfill comes from Kotak Neo only, and its coverage is expected to
fall well short of the ideal above. The actual depth is unknown and measuring it is the
first task of Phase 1. Where history is thin, the compensations are: self-recording forward
from Phase 0 onward, widening the universe to buy trade count (D6), and extending the paper
period rather than lowering the gates (§2.6). The `BarSource` abstraction means a vendor
feed can be added later without touching the strategy, should D5 need revisiting.

### 2.3 Execution simulation — the part that decides everything

Most retail intraday strategies are profitable in a naive backtest and unprofitable live,
and the entire difference is here. The cost model is not a detail to add later; it is the
gate.

**Fill model**
- A signal generated on bar `t` (using only its close) executes at bar `t+1` **open**.
  Never at bar `t` close. This is the anti-look-ahead invariant, and it is asserted in
  tests, not merely intended.
- Market orders: fill at `next_open ± slippage`.
- Slippage model: `max(half_spread, slippage_bps × price)`, with `slippage_bps` configurable
  and deliberately pessimistic. Widen it for the first and last 15 minutes of the session.
- Stops are checked against bar `low`/`high`. When a bar's range contains both stop and
  target, resolve **pessimistically** (assume the stop hit first). Optimistic intrabar
  resolution is the second-most-common way a backtest lies.
- Optional liquidity cap: reject/partial-fill any order exceeding `x%` of the bar's volume.

**Indian intraday cost stack — modelled explicitly, per leg**

| Component | Notes |
|-----------|-------|
| Brokerage | Per-order, typically `min(flat, %)` — model the broker's actual plan. |
| STT | Sell side only for intraday equity. |
| Exchange transaction charges | Both legs. |
| GST | On (brokerage + transaction charges). |
| SEBI turnover fee | Both legs. |
| Stamp duty | Buy side only. |

These are hardcoded as a `CostModel` with the rates in config, and every backtest reports
**gross P&L, total costs, and net P&L separately.** A strategy is evaluated on net.
For a high-frequency intraday strategy, costs routinely exceed gross edge — reporting only
net without showing the cost line hides *why* something failed.

### 2.4 Metrics

Per run: net & gross P&L, CAGR, **max drawdown** (₹ and %), Sharpe, Sortino, Calmar,
win rate, profit factor, **expectancy per trade in R**, average win/loss in R, trade count,
trades/day, average holding time, exposure %, largest loss, max consecutive losses,
MAE/MFE distributions, and a cost breakdown.

Also required, because they catch flaws aggregate stats hide:
- Equity curve + underwater (drawdown) plot.
- P&L by hour-of-day — reveals whether the edge is real or is one time-of-day artefact.
- P&L by month — reveals whether it's one lucky quarter.
- Trade-level ledger exported to CSV for manual inspection.

### 2.5 Validation protocol — the anti-overfitting rules

A single backtest over a single period is worthless. The protocol:

1. **Split the data up front.** The most recent ~20% is a locked holdout. It is touched
   **once**, at the very end. If the holdout is examined and then parameters change, the
   holdout is burned and a new one must be carved out.
2. **Walk-forward analysis.** Rolling windows (e.g. 6-month optimise → 2-month test,
   stepped forward). Report only the stitched out-of-sample curve. In-sample results are
   never quoted as evidence of anything.
3. **Parameter robustness over parameter optimality.** Prefer a broad plateau of decent
   performance over a sharp peak. A parameter surface with a single spike is an overfit,
   not a discovery. Publish the heatmap.
4. **Count the trials.** Log how many parameter combinations were evaluated. Testing 500
   configs and reporting the best guarantees a good-looking result from pure noise; the
   reported Sharpe must be discounted accordingly.
5. **Negative control.** The SMA-crossover reference must show a loss after costs. If it
   doesn't, fix the backtester before believing anything else it says.
6. **Sensitivity check.** Re-run with slippage doubled. A strategy that survives only at
   optimistic slippage is not deployable.

### 2.6 Go-live gates

No real capital until **all** of these pass on out-of-sample data:

- [ ] Positive net expectancy after the full cost stack — i.e. gross expectancy above the
      **~0.12R** cost hurdle derived in §5.2.
- [ ] Profit factor > 1.3 out-of-sample.
- [ ] Max drawdown within the 2% daily / configured aggregate risk appetite (D2).
- [ ] Minimum 200 out-of-sample trades (below this, results are not statistically meaningful).
- [ ] Survives the doubled-slippage sensitivity run.
- [ ] Walk-forward OOS curve is not driven by a single month or a single instrument.
- [ ] Long-only bias is visible, not hidden: performance broken out by up / down / flat
      sessions (D3), confirming the edge is not simply market beta.
- [ ] Paper-traded live for **20+ trading sessions** — extended to **40+** if the Kotak
      history proves thin enough that the 200-trade bar was met only marginally (D5
      mitigation 4) — with paper results reconciled against a backtest over those same
      sessions. Divergence between the two means the model is wrong, and it is investigated
      before proceeding — this reconciliation is the highest-value test in the entire document.

**These thresholds do not move to accommodate limited data.** If the data cannot support
them, the answer is a longer forward test, not a lower bar.

---

## 3. Production hardening

### 3.1 Event bus: consumer groups replace `"$"` cursors

Fixes B6. This is the highest-severity infrastructure item — it means a restart currently
drops approved orders on the floor.

- Replace `XREAD` + in-memory `last_id` with **`XREADGROUP`** using a per-module consumer
  group, and explicit **`XACK` only after the event is fully processed.**
- Unacked messages are recovered on restart via `XAUTOCLAIM`.
- Poison messages (repeatedly failing) route to a `dead_letter` stream with the error, and
  raise an alert rather than blocking the consumer or vanishing.
- **`XADD ... MAXLEN ~ N`** on every stream. Fixes half of B12.
- Update `README.md:120` — the current idempotency claim is false today and must not stay
  in the docs unqualified.

### 3.2 Closing the position/P&L loop

Fixes B2 and B3, the pair that makes the drawdown circuit breaker fake.

- New `src/position_manager.py`, the **single source of truth** for positions, equity, and P&L.
- On startup and every N seconds, **reconcile against the broker** (`positions()`,
  `order_report()`, `limits()`). The broker is authoritative; local state is a cache. Any
  mismatch logs loudly and triggers a halt if it exceeds a threshold — a bot that disagrees
  with the broker about what it owns must not keep trading.
- **Equity retrieval (D1).** Fetches account funds from `limits()` and computes
  `equity = cash_balance + unrealised_MTM` — deliberately *not* available margin, which
  includes ~5× MIS leverage and would inflate every position size accordingly. Margin
  headroom is tracked separately, as a placement constraint rather than a sizing input.
- **Session equity snapshot (D1).** `session_equity` is captured once at
  `on_session_start` and is what sizes every trade and defines the day's loss limit.
  Intraday P&L moves the drawdown counter but never the sizing base — resizing off live
  equity would systematically put on the largest position right after the best run-up.
- Fails closed: if `limits()` errors or returns an implausible value (zero, negative, or a
  >50% jump from the prior session), refuse new entries and alert rather than falling back
  to a config default. Exits remain permitted.
- Publishes `position_updates`; the risk manager consumes it and updates
  `realized_pnl_today` and `equity` for real.
- Only then does the drawdown circuit breaker actually work.

### 3.3 Order lifecycle

Fixes B3.

- Explicit state machine: `PENDING → SENT → ACKED → PARTIAL → FILLED | REJECTED | CANCELLED | EXPIRED`.
- An order poller reconciles open orders against the broker's order report.
- `filled_orders` is emitted on **actual fills** (including each partial), carrying real
  fill price and quantity — not on submission acknowledgement.
- **Idempotency:** every order carries a deterministic client order id derived from
  `(signal_id, instrument, intent)`, passed in the broker's `tag` field, plus a Redis
  dedupe set. Combined with §3.1, a crashed execution engine resumes without either
  double-firing or dropping orders.
- Timeout handling: an order not acked within N seconds is queried, then cancelled.

### 3.4 Instrument master

Fixes B4 and B5.

- Daily download of the NSE scrip master at startup, cached in Redis.
- Maps `instrument_id ↔ token ↔ trading_symbol ↔ exchange_segment ↔ lot_size ↔ tick_size ↔ freeze_qty`.
- All internal events carry `instrument_id`; the execution engine resolves to
  `trading_symbol` at the boundary. Display names never leave the logging layer.
- Prices round to `tick_size`, quantities respect lot/freeze limits, before any order is sent.
- Universe validation at startup: reject any configured instrument that is not tradable in
  the cash segment. This is what would have caught the Nifty-50-index default.

### 3.5 Feed health

Fixes B10.

- Heartbeat: ingestion publishes a liveness beat every N seconds.
- **Staleness watchdog:** if no tick for an instrument within `X × expected_interval`
  during market hours, mark the instrument stale.
- Stale instrument → risk manager **blocks new entries** on it but still permits exits.
  Never trade on a frozen price; always be able to get out.
- WebSocket auto-reconnect with exponential backoff and re-subscription; alert after
  `n` consecutive failures.
- Sequence/timestamp sanity checks to catch out-of-order or replayed ticks.

### 3.6 Market calendar & session control

Fixes B9.

- `src/market_calendar.py`: NSE holiday list, session times, early-close days.
- Modules refuse to place entry orders outside `09:15–15:30` on a trading day.
- A session scheduler drives `on_session_start` / `on_session_end` and enforces the
  15:10 hard-flat (§1.6).

### 3.7 Kill switches

Three independent layers, because one is not enough:

1. `DRY_RUN` (exists) — no order ever leaves the machine.
2. **`ligerbot:halt` Redis key** — checked by risk manager and execution engine on every
   event. A small CLI sets it. Halts new entries instantly across all modules without
   requiring a restart; exits still permitted.
3. **Automatic halts** — daily drawdown breach (now real, §3.2), broker reconciliation
   mismatch, feed staleness across the whole universe, repeated order rejections,
   consecutive-loss limit.

Every halt is loud: log at ERROR, write to the archive, and push an alert.

### 3.8 Auth & session management

- A single auth/session service holds the Neo session and shares the token via Redis, rather
  than each module authenticating independently (`execution_engine.py:51`,
  `data_ingestion.py:92` currently each call `authenticate_neo()`).
- Track token expiry; re-login proactively before expiry and on any 401.
- Never log tokens, MPIN, or TOTP secrets.
- Verify `.gitignore` covers `.env` before any credential is ever written to disk.

### 3.9 Storage & observability

- Batched, asynchronous Influx writes with a bounded queue (fixes the rest of B12);
  drop-oldest on overflow, and never let archiving backpressure the trading path.
- Retention: full-fidelity ticks for a short window, 1-minute bars retained long-term.
- Structured JSON logs with a correlation id threaded `signal → order → fill`, so any
  trade can be reconstructed end to end. **Built** — `src/logging_setup.py`.

  Five processes touch every trade, so reconstructing *which* signal became *which* order
  meant reading five logs and matching on timestamps and instrument names. Both collide.

  Threading the id by hand was rejected: it would mean touching every call site and would
  be forgotten exactly once — in the error path, where it matters. It lives in a
  `ContextVar` instead, bound by `StreamConsumer.handle()` from the message being handled.
  That is the one place worth instrumenting, because it is the single boundary every
  cross-process unit of work passes through, so **every module gets it without a single
  call site changing**. Falls back to `client_order_id`, then the stream entry id, so an
  untagged message is still followable.

  Three properties are load-bearing and each has tests:

  - **The id never leaks between messages.** Leaking would be worse than not having it: it
    would attribute one order's failure to a different order — confidently wrong rather
    than silent. Restored on exit, including when the handler raises.
  - **Logging cannot break trading.** A formatter that raises on an unserialisable object
    would turn a log line into an outage. Every failure path degrades to a plainer line.
  - **Secrets never reach the log.** Structured logging invites passing rich context, and
    a broker session is a dict containing a bearer token. Sensitive keys are redacted at
    the formatter, recursively, so no call site has to remember.

  `LOG_FORMAT` defaults to `text` deliberately — during a live session a human is watching
  a terminal, and JSON is unreadable at a glance. Text still shows the id, appended as
  `<LB-2885-1042>`.
- Health endpoint per module (last event processed, lag, consumer-group backlog).
  **Built** — `src/health.py`, on `127.0.0.1:9800+offset`, one fixed port per module.

  The naive version answers *"is the process alive"*, which is worthless here because
  **alive is the failure mode**. The feed that reconnected forever, the consumer that fell
  behind until the stream trimmed past it, the socket that died while downstream kept
  computing on a frozen price — all of those pass a liveness check. That is why they were
  expensive. So the check rests on two clocks:

  | | meaning | failure? |
  |---|---|---|
  | `last_loop_at` | the consume loop turned | **yes** — stale means *wedged*, not busy |
  | `last_event_at` | work actually happened | **no** — a quiet market is not a fault |

  Conflating them would fire an alert every lunchtime, and an alert people learn to ignore
  is worse than none.

  **A halted bot returns 200.** The kill switch working is the system working; a 503 would
  tell an orchestrator to restart the process and destroy the halt. Degraded returns 503 so
  a monitor does the right thing without parsing the body. `/live` is deliberately weaker —
  it answers only "the process responds" — and is documented as such so nobody wires up the
  weak check believing they have the strong one.

  It binds to **loopback by default**: the payload carries positions, P&L and equity, and
  exposing that on every interface because it was the convenient default is a real
  disclosure. The server runs on a daemon thread, a port it cannot bind is logged and
  shrugged off, and a handler that raises returns 500 — a monitoring feature that can take
  down execution is not worth having.
- Grafana dashboards: equity curve, open positions, stream lag, order reject rate, feed
  staleness, P&L vs. the backtest's expectation for the same period. **Partly built** —
  `grafana/`, provisioned read-only, on loopback.

  Three of the six have Influx data behind them and are built: **P&L and costs**, **open
  positions and total open risk**, **order flow and reject rate**. Three do not:

  | asked for | why it is not here |
  |---|---|
  | stream lag | a property of a *process*, not an event — lives on the health endpoint |
  | feed staleness | same; and archiving it would mean the archiver reporting on itself |
  | P&L vs. backtest | reconciliation writes to `state/sessions/`, not Influx |

  The operations dashboard **says so on its face**, with pointers to where each actually
  lives. That matters more than it sounds: a panel querying data nobody writes renders
  empty forever, and **an empty panel on a risk dashboard reads as "nothing is wrong"**.
  An honest gap beats a panel that lies by omission.

  **Building this found a real defect.** `BatchingInfluxWriter` filters against a fixed
  allow-list and drops anything absent *silently*. Seven of the nine fields in a
  `position_updates` snapshot were being discarded — including `open_positions` and
  `total_open_risk`, two of the things listed above. `correlation_id` was dropped too,
  which would have made the threading built for §3.9 useless in the permanent record.
  Every one of those panels would have been empty from the day it shipped.

  `tests/test_grafana_dashboards.py` now parses the Flux in every panel and asserts each
  measurement and field is one the writer actually archives, and — the direction that found
  the bug — that nothing the position book publishes is dropped. Reverting the field list
  fails seven tests.
- Alerts on: halt triggered, feed stale, order rejected, reconciliation mismatch, module
  down, stream backlog growing.

### 3.10 Testing

- Unit tests for indicators (against known-good reference values), the cost model,
  position sizing, and the fill model.
- Property test for the anti-look-ahead invariant: a strategy's output for bars `0..t` must
  be identical whether or not bars `t+1..n` exist in the source.
- Golden-file pipeline test: fixed bar sequence in → exact expected orders out. This is the
  regression net for every future refactor. **Built** — `tests/test_golden_pipeline.py`
  against `tests/golden/pipeline_trace.txt`, driven by the `Trace` recorder in
  `src/backtest/trace.py`.

  Every other test here asserts a *property*. Properties are the right tool for rules you
  can state, and useless against a refactor that quietly changes **which trades happen**
  while keeping every stated property true — which is exactly the shape of the resampler
  divergence (§5.4) and the wall-clock phase check (§5.3). Both were found by accident.

  Three things make it worth having rather than ceremonial:

  - **The input is hand-built, not seeded random.** A reviewer can read the trace against
    the scenario and say whether it is *right*, not merely whether it changed.
  - **Regeneration is opt-in** (`LIGERBOT_UPDATE_GOLDEN=1`) and never happens in CI. A
    golden file that regenerates on failure asserts only that the code agrees with itself.
  - **The file is checked for teeth.** Six mutations that *should* move the trace are
    asserted to move it, so it cannot silently degrade into a rubber stamp.

  The scenario deliberately covers the paths that hurt: a stop-loss (the only exit that
  loses money), a forced square-off (the only one the strategy does not choose), a risk
  refusal, a re-entry after being stopped, and a holiday gap between the two sessions.
- Chaos tests: kill each module mid-flight and assert no order is lost **and** none is
  duplicated.
- Reconciliation test: backtest vs. paper over the same sessions must agree within tolerance.

---

## 4. Implementation order

Sequenced so each phase is independently verifiable and nothing is built on an
unvalidated foundation.

### Phase 0 — Foundations (blocks everything)
1. Instrument master + liquidity screen + universe validation (§3.4, D6) — fixes B4, B5.
2. Bar builder + `market_bars` stream (§1.2) — fixes B1.
3. **Parquet writer wired to the bar builder on day one (D5 mitigation 2).** Out of order
   relative to Phase 1, deliberately: self-recording is the slowest-to-mature asset in the
   project, so it starts accumulating before anything else is built.
4. Market calendar & session control (§3.6) — fixes B9.
5. Equity retrieval + `session_equity` snapshot (§3.2, D1) — fixes B11.
6. Extract `RiskEngine` as pure logic from `RiskManager`, with the D2 parameter set (§2.1).

**Exit criteria:** simulated feed produces correct, gap-filled 1-minute bars persisted to
Parquet; session boundaries fire; live equity is retrieved and snapshotted; `RiskEngine` is
unit-tested with zero I/O and provably cannot exceed 1.5% total open risk.

**Status: complete and verified.** Run `python demo_phase0.py`. The risk invariant survives
contact with the full backtest engine too — over a six-month synthetic run the open-risk cap
refused 176 signals and the worst-case daily gate refused a further 139.

### Phase 1 — Backtest harness
7. **Measure Kotak's actual historical depth (D5 mitigation 1).** First task in the phase —
   a throwaway probe script. Its result may force revisiting D5, so nothing downstream is
   committed until it lands.
8. `BarSource` + Parquet store + data-quality gate (§2.2).
9. Historical backfill for the screened universe.
10. `SimBroker` fill model + full Indian cost model (§2.3), calibrated to the real
    brokerage plan (§5.3 item 1).
11. Metrics + reports (§2.4), including the up/down/flat session breakdown (D3).
12. Walk-forward runner (§2.5).

**Exit criteria:** the SMA-crossover reference is backtested and **shows a loss after
costs** (§2.5 rule 5). Until the negative control fails correctly, the harness is not trusted.

**Status: harness complete and verified.** The negative control loses −0.097R per trade on
a driftless random walk. The attribution is the stronger result: frictionless expectancy
+0.010R (no edge, as constructed) minus 0.107R of friction — the harness independently
reproduces §5.2's analytical ~0.12R hurdle through full fill simulation, from a completely
separate code path. Run `python demo_phase1.py`.

**Amendment — friction must be split, not just netted.** The first working version reported
"cost drag" of 0.059R against §5.2's predicted 0.12R, and the gap was an accounting artefact:
slippage is baked into fill prices, so it was hiding inside gross P&L rather than appearing
as friction. Reports now separate **frictionless P&L** (what the signal was worth) from
slippage and from explicit charges. This is not cosmetic — netting them together makes "the
strategy has no edge" and "the edge was eaten by execution" look identical, and those call
for completely different responses.

**Still blocked on item 7.** Everything above is validated against synthetic data, which can
verify the *harness* but cannot validate any *strategy*. The Kotak probe
(`tools/probe_kotak_history.py`) needs live credentials and has not been run.

### Phase 2 — Strategy
13. Incremental indicator library (§1.5).
14. `Strategy` interface, registry, new signal schema (§1.3–1.4, §1.7) — fixes B7, B8.
15. Trend-pullback v1, long-only (§1.6, D3).
16. Bar-interval sweep — 1/3/5/15-minute (D4).
17. Walk-forward validation → §2.6 gates.

**Exit criteria:** all §2.6 backtest gates pass out-of-sample. If they don't, iterate here —
do **not** proceed to hardening on a strategy without a demonstrated edge.

**Status: machinery complete; strategy unvalidated and cannot be validated yet.** The
indicator library, trend-pullback v1, the interval sweep and the automated gate evaluation
all work. On synthetic data the gates **BLOCK with 7 failures**, which is the correct
outcome — a strategy that passed on a random walk would prove the gates were broken.
Run `python demo_phase2.py`.

#### The economic stop floor — a constraint that was missing from §1.6

Implementing the strategy exposed a relationship the design did not account for. Risk-based
sizing sets `notional = risk_budget / stop_distance`, so a *tighter* stop produces a
*larger* position, while friction scales with notional and the risk budget does not:

```
friction_in_R  ≈  0.00085 / stop_pct
```

A 0.4% stop therefore costs 0.21R per trade; a 0.8% stop costs 0.11R. **To stay under the
~0.12R hurdle the stop must be at least ~0.7% of price.** Measured across the interval sweep
this holds tightly — 3-minute bars produced 0.76% stops and 0.131R friction; 15-minute bars
produced 1.63% stops and 0.072R friction.

Two consequences:

1. `TrendPullback` now refuses any trade whose stop falls below `min_stop_pct` (0.7%
   default). Refusing a trade that is structurally uneconomic is cheaper than taking it.
2. **This resolves D4 empirically.** 1-minute ATR stops land near 0.2–0.3% of price and are
   structurally unprofitable regardless of signal quality — the sweep produced *zero*
   qualifying trades at 1-minute. The "trade 5-minute" default stands, with `atr_mult`
   raised from 1.5 to 3.0 so 5-minute stops clear the floor.

#### Positive control — and what it could not settle

§2.5 rule 5 specified a negative control. Implementing v1 showed that is only half a test:
a strategy that never trades also passes it. Phase 2 adds a **positive control** — synthetic
data with genuine AR(1) return persistence, plus a unit-level test that places a textbook
trend-pullback directly in front of the strategy and asserts it takes the trade.

The unit-level control passes cleanly. The aggregate one is **weak**: frictionless expectancy
rises with injected momentum (−0.034R → +0.031R → +0.046R at momentum 0, 0.3, 0.6) but the
effect stays small relative to friction. Two explanations remain open and synthetic data
cannot separate them:

- the AR(1) generator does not produce the trend-then-pullback *structure* v1 targets, or
- v1's edge detection is genuinely weak.

Real data distinguishes them. Tuning parameters until the synthetic numbers improved would
be fitting noise — precisely the failure §2.5 exists to prevent — so the parameters stand
as derived.

### Phase 3 — Hardening
18. Consumer groups + DLQ + stream trimming (§3.1) — fixes B6.
19. Full position manager + broker reconciliation (§3.2) — fixes B2, B3.
20. Order state machine + idempotency (§3.3).
21. Feed health & reconnect (§3.5) — fixes B10.
22. Kill switches (§3.7), auth service (§3.8), observability (§3.9), test suite (§3.10).

**Exit criteria:** chaos tests pass; the drawdown circuit breaker demonstrably trips in a
fault-injection test.

**Status: complete and verified.** `tests/test_chaos.py` kills modules mid-flight (after
the broker call, before the ack — the window where belief and reality diverge) and asserts
both that no order is lost *and* that none is duplicated. The breaker fires under fault
injection. Run `python demo_phase3.py`.

#### Two findings from wiring it up

**The live path was judging session phase against the wrong clock.** The risk manager
called `cal.phase()` with no argument — wall-clock *now* — while the backtester judges at
`cal.phase(bar.bar_end)`. In live trading these usually coincide, so it looked correct; but
any queueing, restart or replay made the live path disagree with the backtested one. That
is precisely the divergence §2.1 exists to prevent, and it was invisible until the pipeline
was exercised end to end. The adapter now judges at the signal's own `bar_time` — which
in turn *requires* a separate staleness check (`MAX_SIGNAL_AGE_SECONDS`), since otherwise
an hours-old signal would pass on the strength of when it was generated.

**Rejection reasons named the wrong subsystem.** The risk engine reports "session phase
does not permit new entries" whenever `allows_entry` is false — but the adapter lowers that
flag for four different reasons (phase, stale signal, kill switch, dead feed). An operator
reading the log during an incident would have chased the wrong cause. The adapter now
reports the specific gate that blocked.

Neither was caught by unit tests; both surfaced only when the modules ran together.

### Phase 4 — Paper trading
23. Run live-data paper trading for 20+ sessions — **40+ if backtest evidence was thin**
    (D5 mitigation 4).
24. Reconcile paper vs. backtest daily; investigate every divergence.
25. **Daily briefing report** (morning pre-open and evening post-close): positions, P&L
    against expectation, gate status, halts, feed health, reconciliation results. A paper
    period nobody reads is not a test — the divergence in item 24 only gets investigated
    if something puts it in front of a human every day.

**Exit criteria:** paper P&L tracks backtest expectation within tolerance.

**Status: machinery complete; the sessions themselves cannot be manufactured.** The paper
broker, session recorder, reconciliation engine, briefings and Phase 4 gate all work and
are tested. What remains is 20-40 sessions of calendar time against live market data —
there is no way to shorten that and no substitute for it. Run `python demo_phase4.py`.

#### DRY_RUN would have made the whole phase meaningless

Building this exposed a defect that would have invalidated Phase 4 entirely. `DRY_RUN`
filled orders **instantly at the signal's reference price** — no next-bar delay, no
slippage, no costs, no liquidity check. Paper trading on that path would have beaten the
backtest for purely artificial reasons, and the reconciliation would have read the gap as
"the backtest is conservative" rather than "paper mode is lying".

Measured on a realistic 15bps gap between signal close and next open, the overstatement is
**~0.22R per trade** — nearly double the entire ~0.12R friction budget from §5.2, and
always in the favourable direction.

The fix is structural rather than a correction to the numbers: `src/paper_broker.py` drives
the backtester's own `SimBroker`. Paper and backtest cannot diverge in fill semantics
because they are the same code. `DRY_RUN` is now one of three explicit modes
(`dry_run` / `paper` / `live`) and means "nothing fills at all" — a wiring check, not a
simulation.

An optimistic paper mode is worse than no paper mode: it manufactures confidence instead of
merely failing to provide it.

#### Attribution, not divergence

A reconciliation reporting "paper made 12,000 less than the backtest" invites the wrong
conclusion and cannot be acted on. `src/reconciliation.py` decomposes the gap into missing
trades, extra trades, fill-price divergence and cost divergence — each with a different
fix — and reports the **unattributed residual**. A large residual fails the gate on its own,
because it means the diagnosis cannot be trusted either, which matters more than the
divergence being diagnosed.

### Phase 5 — Live, minimum size
26. Live with the smallest tradable size and a hard daily loss cap.
27. Scale only after a sustained period matching paper behaviour.

**Status: safeguards complete; live trading is blocked, correctly.** The guard, the scaling
ladder and the absolute backstops are built and tested. Live trading itself is refused —
every prerequisite fails today. Run `python demo_phase5.py`.

#### `TRADING_MODE=live` is deliberately not sufficient

An environment variable is one typo, one copied `.env`, one careless export away from
committing real capital. `src/live_guard.py` additionally requires:

1. **Evidence** — the §2.6 backtest gates and the Phase 4 paper gate must have passed.
2. **Correctness** — the broker probe must have run, so the equity field names in
   `src/account.py` are verified rather than guessed.
3. **Intent** — an authorisation *file* naming the date, capital and person. It expires
   after 7 days (a decision made weeks ago is not today's decision) and is rejected if the
   account holds materially more than was authorised (authorising for a small account and
   pointing the bot at a large one is how a test becomes a position).

Every check defaults to **not passed**, and there is no override or `--force`. A bypass
becomes the thing someone reaches for at 09:14 on a morning they are in a hurry. The
execution engine calls the guard before arming, so the refusal is structural rather than
advisory.

#### The scaling ladder is asymmetric on purpose

Promotion requires a minimum number of sessions **and** trades at the current rung, with
expectancy still tracking — five quiet days prove nothing about fills. Demotion skips
straight to the floor on three consecutive losing sessions, a halt, or expectancy falling
through the floor. In the demo it took 46 profitable sessions to reach full size and three
bad ones to lose it.

That asymmetry is the design: promoting too slowly costs a little upside, promoting too
quickly costs capital. Scaling multiplies the equity *base*, never the risk *rules*, so
every proportional guarantee from D2 survives the ramp unchanged.

#### Absolute backstops, because percentages inherit the equity error

Every limit in D2 is a fraction of equity — which is exactly wrong when the equity figure
itself is wrong. An unverified broker field mapping mis-sizes every trade by the same
factor, and each percentage cap scales *with* that error rather than catching it.

Worked example from the demo: equity mis-read 10× high turns the "2% daily limit" into 20%
of the real account. `LIVE_MAX_DAILY_LOSS` and `LIVE_MAX_ORDERS_PER_DAY` bound the damage in
rupees regardless. They apply in live mode only — in backtest and paper they would cap
activity for reasons unrelated to the strategy and distort the results.

**`LIVE_MAX_DAILY_LOSS` ships unset, and the live guard blocks until it is chosen.** There
is no sensible default: the figure states how much a particular operator is willing to lose
in a day, which no library can guess. Shipping a number would be worse than shipping none —
it would be a limit nobody decided on, inherited silently by anyone who cloned the repo.

Every other monetary figure in this document is **derived rather than chosen**, and so says
nothing about any account: the ~₹47 fixed cost per round trip and the ₹2L viability floor
both fall out of the broker's fee structure (§5.2); the ₹200cr liquidity threshold and
~₹5,000 share-price ceiling are universal screen criteria (D6). `TOTAL_EQUITY` is a round
illustrative value used only where no broker exists to ask — backtests, demos and tests.

---

## 5. Decisions

### 5.1 Resolved

#### D1 — Equity is retrieved live from Kotak, never configured

`TOTAL_EQUITY` as a static config value is removed as the source of truth. The position
manager (§3.2) fetches account funds from the Kotak Neo `limits()` endpoint and derives
equity from it. This is consistent with the standing principle that the broker is
authoritative (§6.3).

Three design constraints follow, and all three are non-obvious:

1. **Size off own capital, not buying power.** MIS gives roughly 5× intraday leverage, so
   available margin substantially exceeds cash. Sizing off margin would silently violate
   the 0.5% risk rule by a factor of five. Define:
   `equity = cash_balance + unrealised_MTM`, explicitly *not* available margin. Margin is
   checked separately, as a constraint on whether an order can be placed at all.

2. **Snapshot equity at session start; hold it fixed all day.** Sizing off continuously
   updating equity looks correct and is a trap: a profitable morning inflates position
   sizes right before any mean reversion, so the largest position is systematically taken
   at the worst moment. Instead, capture `session_equity` in `on_session_start` and use it
   for all sizing and for the daily drawdown limit for the whole session. It is refreshed
   once per day, at the open. Intraday P&L moves the drawdown counter, never the sizing base.

3. **Fail closed on retrieval failure.** If `limits()` fails or returns something
   implausible (zero, negative, or a >50% jump from the previous session's close), do
   **not** fall back to a config default and do **not** guess. Refuse to open new positions
   and alert. A wrong equity figure sizes every trade in the session wrongly. Config
   retains a `MAX_EQUITY_CAP` purely as a sanity ceiling — if the API ever returns
   something absurd, the cap bounds the damage.

This also resolves B11: sizing can no longer be zero because of a stale hardcoded figure.

#### D2 — Risk parameters, made internally consistent

Daily loss limit set at 2% of session equity. `MAX_OPEN_POSITIONS` is demoted from primary
control to secondary sanity cap, and replaced by a **total open-risk cap** — the sum of
distance-to-stop across all open positions. Counting positions does not bound risk, because
three wide-stop positions carry far more exposure than three tight-stop ones.

| Parameter | Value | Derivation |
|-----------|-------|------------|
| `MAX_DAILY_DRAWDOWN` | **2.0%** | Chosen. |
| `MAX_OPEN_RISK` | **1.5%** | Must sit below the daily limit, so that every open position stopping out simultaneously cannot breach the day's limit before the breaker can act. |
| `RISK_PER_TRADE` | **0.5%** | `MAX_OPEN_RISK / MAX_OPEN_POSITIONS`. |
| `MAX_OPEN_POSITIONS` | **3** | Retained as a secondary cap only. |

Consistency check: worst case is three concurrent full-risk positions stopping out together
= 1.5%, inside the 2% limit, leaving room for one further trade before the breaker halts
the day. The breaker trips after four full-risk losses.

**Amendment (found during Phase 0 implementation).** The above holds only for positions
opened from a clean slate. A breaker that watches *realised* P&L alone is a trip threshold,
not a cap: it can sit at -1.9% realised with 1.5% of risk still open, neither figure
breaching on its own, and a simultaneous stop-out ends the day at **-3.4%**. The stated 2%
limit would not have held.

The fix is to gate new entries on the **worst case** — realised losses plus all open risk —
rather than on realised losses alone. Risk is then counted the moment it is committed, not
when it lands. `RiskEngine.projected_loss()` implements this, and it is a *rejection* of new
entries rather than a hard halt, because it is reversible: as positions close, headroom
returns. The hard halt still fires on realised P&L. Both are fuzz-tested in
`tests/test_risk_engine.py::TestWorstCaseBound`.

The simultaneity assumption is deliberately pessimistic and correct here: NSE large caps are
strongly correlated intraday, so three concurrent longs behave close to one position of
three times the size. Nothing is diversified away on a market-wide down move.

**Amendment — the correlation assumption is now enforced, not merely allowed for.** The
above *assumed* correlation and sized for it, but nothing prevented the bot holding three
bank stocks and treating them as three independent positions. The open-risk cap bounded the
total while quietly mislabelling the concentration. `RiskLimits.max_positions_per_group`
(default 1) now caps positions within a correlated group, with sector groupings in
`src/instruments.py`. Deliberately coarse: the aim is to catch "three banks is really one
bet", not to model a covariance matrix. A realised-correlation estimator would be better but
needs history we do not have yet (D5) — and a rough grouping applied today beats a precise
one that arrives after the drawdown.

#### D3 — Long-only for v1

Shorting is deferred to v2. Halves the surface to validate and avoids NSE short-ban list
handling and the different sell-side cost treatment. Accepted cost: no participation in
down-trending sessions, so expect a materially lower trade count and a strategy whose
results correlate with market direction. The backtest reports performance separately for
up/down/flat sessions so this bias is visible rather than hidden.

The `Signal.intent` vocabulary still includes `OPEN_SHORT`/`CLOSE_SHORT` (§1.4) so v2 needs
no schema migration; the risk manager simply rejects short intents while `ALLOW_SHORT=false`.

#### D4 — Bar interval: store 1-minute, trade 5-minute

These are two decisions, previously conflated. The **store** is 1-minute regardless — 1-min
aggregates up to any coarser interval, never the reverse, so storing finer preserves options
at negligible cost. The **trading** interval defaults to 5-minute: fewer bars means fewer
round trips, and cost drag (§5.2) is the largest single threat to viability. Since interval
is a strategy parameter, Phase 2 sweeps 1/3/5/15-minute empirically at no additional cost.

**Confirmed in Phase 2, for a reason stronger than "fewer round trips".** The sweep showed
1-minute ATR stops land near 0.2–0.3% of price, and since `notional = risk_budget /
stop_distance`, such tight stops force positions so large that friction swamps any signal.
At 1-minute the strategy produced **zero** qualifying trades once the economic stop floor
was applied. 5-minute stands, with `atr_mult` raised 1.5 → 3.0 so its stops clear the ~0.7%
floor. See "the economic stop floor" under Phase 2 in §4.

#### D5 — Historical data: Kotak Neo only

Chosen: no paid vendor, no external dependency. This is a real constraint with real
consequences, and the design compensates rather than pretends otherwise.

**The risk:** if Kotak's history is as thin as expected, the §2.6 gate of 200+ out-of-sample
trades over 2+ years is unreachable, and a strategy validated on a short, single-regime
window is barely validated at all. A backtest over 60 days of one market regime tells you
how the strategy did in that regime, not whether it has an edge.

**Four mitigations, all free:**

1. **Establish the actual depth first.** The literal first task of Phase 1 is a throwaway
   script that queries the Kotak historical endpoint and answers: how far back, at what
   granularity, how many instruments per call, what rate limits. Every subsequent decision
   depends on this and it is currently unknown. If depth turns out to be under ~6 months of
   1-minute data, that materially changes the plan and we revisit this decision with real
   numbers rather than assumptions.

2. **Start self-recording immediately.** The bar builder writes to the Parquet store from
   the moment Phase 0 lands, in parallel with all other work. This costs nothing, runs
   unattended, and by the time Phase 2 needs data the dataset has been growing for weeks.
   Self-recorded bars are also the highest-fidelity data available — they are exactly what
   the live system will see.

3. **Trade breadth for depth.** Statistical power comes from trade count, not calendar
   span. Ten instruments over 60 days yields roughly ten times the trades of one instrument
   over 60 days. This is the main lever available, and it drives the universe decision (D6).
   It does *not* substitute for regime diversity — ten correlated names in one quarter is
   still one market regime — but it fixes the sample-size half of the problem.

4. **Hold the gates; extend the paper period instead.** The §2.6 thresholds are not relaxed
   to fit the available data. Where backtest evidence is thin, the paper-trading period
   (§4 Phase 4) extends from 20 sessions to **40+**, shifting the burden of proof onto
   forward testing. Slower, but forward results on unseen data are stronger evidence than
   any backtest.

#### D6 — Universe: a liquidity screen, not a fixed list

Replaces the untradable Nifty 50 index default (B5). The universe is defined by a **screen
run at build time**, not a hardcoded list, because liquidity and prices drift:

- Present in the F&O list — a reliable liquidity proxy, and it forward-protects the v2
  short side from ban-list surprises.
- Average daily traded value above ₹200 crore.
- **Share price under ~₹5,000.** This constraint is easy to miss and matters at smaller
  capital: risk-based sizing produces a target notional, and a ₹30,000/share stock can only
  be bought in absurdly coarse increments, so realised risk diverges badly from intended risk.
- Spread across sectors. This reduces correlation only modestly intraday, but modestly is
  better than not at all.
- Target size **8–12 names**, per D5 mitigation 3. Plus `NIFTYBEES` for index exposure,
  since the index itself cannot be traded in the cash segment.

The screen is re-run monthly and the universe is version-stamped, so any backtest records
which universe it ran against. A starter set meeting these criteria at time of writing would
be names like RELIANCE, HDFCBANK, ICICIBANK, INFY, TCS, SBIN, AXISBANK, ITC, LT, TATAMOTORS
— but this list must be **regenerated from live data at build time**, not copied from here.

### 5.2 The number the strategy has to beat

Falling out of D1–D6, with the cost stack from §2.3 (discount broker, ~₹20/order flat) and
an ATR stop around 0.8% of price:

```
Round-trip cost  ≈  ₹47  +  0.085% of notional     (incl. ~0.05% slippage)
Target notional  =  risk_amount / stop_distance    ≈  0.625 × equity
Cost as % of the amount risked  ≈  11–15%
```

The fixed ₹47 dominates below about ₹5L equity; above that, costs asymptote to ~11% of risk.

**So gross expectancy must exceed ~0.12R just to break even.** Concretely, at a 1.5:1
reward-to-risk ratio the strategy needs a **~45% win rate** to clear costs; at 2:1 it needs
**~37%**. These are the pass marks Phase 2 is aiming at, and they are the reason the
README's "60% win rate" framing was the wrong target — win rate is meaningless without the
R-multiple attached to it.

### 5.3 Still open

None of these block Phase 0.

1. **Brokerage plan.** The cost model in §2.3 is parameterised, but the actual per-order
   rate needs to come from the real account. A percentage-based plan instead of flat-fee
   changes the small-capital arithmetic significantly. *Needed by Phase 1.*
2. **Kotak historical depth.** Unknown until measured — the first task of Phase 1 (D5
   mitigation 1). Genuinely may force revisiting D5.

   *Status: the probe is written and its dependencies are installed (isolated in
   `.venv-kotak/`, because the Kotak SDK hard-pins old shared packages and installing it
   globally breaks other projects). It fails cleanly on the missing credential and is
   waiting only on `.env`.*
3. **Minimum viable equity threshold.** Since equity is now dynamic (D1), the system needs a
   floor below which it refuses to trade, because cost drag exceeds any plausible edge.
   §5.2 suggests somewhere near ₹2L. Set it precisely once the real brokerage plan is known.

### 5.4 Paper mode was silently degraded

`TRADING_MODE` was introduced in Phase 4 but never propagated: five call sites still
branched on `DRY_RUN`, and the two settings were independent. With `TRADING_MODE=paper` and
`DRY_RUN` at its default:

- **`risk_manager`** skipped authentication, so session equity fell back to the configured
  `TOTAL_EQUITY` instead of the broker's real figure — defeating D1 entirely.
- **`position_manager`** skipped authentication, so **reconciliation was disabled**.

Both silently. Phase 4's exit criterion is reconciling paper results against a backtest over
the same sessions, so this would have invalidated every session as it was recorded — and the
symptom would only have appeared weeks later, as an unexplained divergence.

**The fix is structural, not a set of corrected call sites.** `TRADING_MODE` is now primary
and `DRY_RUN` is *derived* from it, so they cannot disagree. An unrecognised mode raises at
import rather than defaulting, because a typo there decides whether real orders are sent.

Three predicates replace the boolean, and they are not interchangeable:

| | `dry_run` | `paper` | `live` |
|---|---|---|---|
| `needs_broker_session()` | no | **yes** | yes |
| `sends_real_orders()` | no | no | **yes** |
| `simulates_fills()` | no | **yes** | no |

The distinction that was missing: **paper mode needs a real broker session.** Equity and
reconciliation are real; only order placement is simulated. That is the whole point of the
phase — a paper run against a fabricated equity figure tests nothing.

**Each filler now refuses the modes it does not own.** The execution engine and the paper
broker both consume `approved_orders` from *separate consumer groups*, so both would receive
every order and both would fill it — double-counting every trade while appearing healthy.
`run_all.py` already started only one; now neither will start in the wrong mode even if
launched by hand.

### 5.5 One login per session — and how fixing paper mode made it urgent

§3.8 specified a shared broker session and it was never built: four modules each called
`authenticate_neo()` independently. Every call performs a full `totp_login` +
`totp_validate`, and **a TOTP code is single-use within a 30-second window**. `run_all.py`
staggers module startup by one second, so modules starting together generate the *same*
code and all but one are rejected as replays.

**Fixing paper mode (§5.4) turned this from latent to blocking.** Before that fix, paper
authenticated in one module — the other two skipped the broker, which was the bug. After
it, paper authenticates in three. So correcting one defect exposed a second that had been
hidden behind it, and paper mode would not have started.

**A session is not one token.** The SDK's `configuration` carries `sid`, `bearer_token`,
`edit_token`, `edit_sid`, `view_token` and `base_url`, all populated by the two-step login.
Passing only `access_token` to a fresh `NeoAPI` leaves the rest unset and authenticated
calls fail in ways that look like permission errors. `src/auth_session.py` captures the
whole set rather than guessing which fields are load-bearing.

**Coordination without a separate process.** Whichever module needs a session first takes a
Redis lock, logs in once, and publishes it; the others restore it. Exactly one TOTP is
consumed however many modules start at once. Design details that matter:

- **The lock has a TTL.** A crash mid-login expires rather than deadlocking every module
  for the rest of the day.
- **A waiting caller gives up rather than racing.** Starting a competing login would burn
  a second code in the same window and be rejected anyway.
- **Sessions are keyed by trading day**, so a new day always re-logs in — a stale token
  fails mid-session rather than at startup, which is far worse.
- **No Redis degrades to a direct login.** A broken cache should cost coordination, not
  stop a single module from working.
- **An unauthenticated capture is refused.** Sharing one would hand other modules a client
  that fails later and less obviously than it would have failed at the source.

This also supplies something that was entirely missing: **expiry recovery.**
`kotak_api.KotakSessionExpired` existed but nothing acted on it, so a session expiring
mid-day left modules retrying a call that would fail identically forever. The execution
engine's reconciliation path now re-establishes the session instead.

### 5.6 Alerting, and the second half of B12

**Alerting (§3.9).** A halt logged `ERROR` to stdout and nothing else. During a six-hour
session nobody is watching that terminal, so a halt at 11:03 stayed invisible until someone
next looked — as did stale feeds, reconciliation mismatches and dead-lettered orders. Three
properties shaped `src/alerting.py`:

- **No sink may raise into the caller.** A webhook timing out is not a reason to stop
  managing positions. Every sink failure is caught and logged, and the remaining sinks
  still run.
- **Repetition must not become noise.** A stale feed evaluated every second would emit an
  alert every second, and a stream nobody can read is the same as no alerts. Alerts carry a
  dedup key and are suppressed for a cooldown; the suppressed count is reported when the
  condition next fires, so the volume stays visible without the flood.
- **Alerts persist.** They reach a Redis stream as well as the log, so the evening briefing
  reports what happened during a session nobody watched — the actual use case.

The webhook sink is a generic JSON POST rather than a Telegram or Slack integration:
vendor lock-in in an alert path is a poor trade for a few lines saved.

**Archiving (B12, second half).** The archiver wrote one point per event synchronously.
The failure is indirect and worth stating in full: it blocks on Influx → stops acking → its
pending list grows → the stream reaches `MAXLEN` and trims entries that were never archived
→ the archive develops holes, silently, exactly when the most is happening.

So the fix required a decision, not just batching: **the archive is best-effort by design.**
Falling behind must cost archive completeness, never consumer progress — because a stalled
archiver eventually costs both. Concretely:

- Points are enqueued and the caller returns immediately; a background thread flushes.
- The queue is bounded and **drops oldest** on overflow. Oldest rather than newest is
  deliberate: during a burst, recent data describes the state you are currently in, while
  old data describes one that has already passed.
- The storage logger acks once a point is **queued**, not once it is written. Holding the
  ack until a possibly-dead backend confirms would grow the pending list until the stream
  trimmed past it — losing far more than the occasional dropped point.
- Drops are counted and alerted above a threshold. Silent archive gaps are worse than
  noisy ones.
- A failed write is **not** requeued. Retrying a failing backend indefinitely is how the
  queue fills in the first place.

### 5.7 The universe screen could not produce a universe

A functional hole, not a cosmetic one. D6 specified "a screen, not a list", and both the
screening logic and the criteria were built — but **nothing fetched the two inputs the
screen filters on.** The scrip master carries no price and no traded volume, so
`apply_screen` correctly excluded every candidate and returned an empty universe. Even with
the probe run and the master downloaded, there would have been nothing to subscribe to and
nothing to backfill.

`src/universe_builder.py` supplies the missing inputs. `last_price` is straightforward —
`quotes()` returns it. **Average daily value is the hard one**, because it needs history,
which is exactly what the probe is still measuring. Three sources are tried in descending
order of trust:

| Source | Trust | Notes |
|---|---|---|
| `HISTORY` | full | real daily bars — pending the probe |
| `RECORDED_BARS` | full | our own Parquet store, needs no broker call, improves daily (D5 mitigation 2) |
| `QUOTE_PROXY` | **weak** | today's traded value so far — one day, and understated early in a session |

**Which source was used is recorded on the universe.** A screen run on a one-day proxy is a
weaker claim than one run on sixty days of recorded bars, and reporting them identically
would let a provisional universe pass as a validated one.

The screen **fails closed**: an instrument whose liquidity cannot be established is
excluded, never assumed adequate. Trading an illiquid name because its data was missing is
a worse outcome than trading a shorter list.

### 5.8 The three unraised alerts

§3.9 listed six alert conditions. Halt, dead-letter, reconciliation mismatch and archive
overflow were wired; three were not. Each shares a shape: **the bot looks healthy while
quietly not working**, which is the most expensive way for a failure to present.

- **Feed stale** — entries are blocked and exits still permitted, but nothing said so.
  Deduplicated per instrument, because `evaluate()` runs every loop iteration and would
  otherwise alert every second for as long as the feed stayed down. A whole-feed outage
  reads differently from one bad symbol.
- **Order rejected** — one rejection is a bad price or a margin shortfall; a *run* of them
  is systemic (wrong trading symbol, expired session, unusable product code) and the
  strategy will keep generating signals into it. The dedup key is the *reason*, so distinct
  causes surface separately while a repeating one does not flood.
- **Consumer backlog** — nothing monitored consumer-group lag at all. A module falling
  behind still runs, still logs, still passes any liveness check; the backlog is the only
  visible symptom, and on `approved_orders` it means trades going unplaced.

### 5.9 Cross-referencing other Kotak integrations on this machine

Four other projects on this machine talk to Kotak Neo (`D:\APEXBOT`,
`D:\Aishwarya\apex-trading-bot`, `D:\JEANS`, `D:\BALU`). Reading them was worth it — one
comment pointed at a defect that would have stopped the bot dead — but it also showed the
limits of borrowed knowledge.

**`src/auth.py` was broken in three ways.** Verified by introspecting the *installed*
`neo_api_client` v2.0.0 rather than trusting any project's code:

| Previous call | Reality |
|---|---|
| `NeoAPI(consumer_secret=...)` | No such parameter. Signature is `(environment, access_token, neo_fin_key, consumer_key)`; `consumer_secret` survives only in a docstring and commented-out code. Raises `TypeError`. |
| `client.login(mobilenumber=...)` | No such method. It is `totp_login(mobile_number=..., ucc=, totp=)` — underscored. |
| `client.session_2fa(OTP=...)` | No such method. It is `totp_validate(mpin=...)`. |

The bot could not have authenticated at all. The probe would have failed with a `TypeError`
rather than anything informative. Three independent sources now agree on the corrected flow.
This is the same lesson as the NSE holiday list: **a plausible-looking API call written from
recall is not an API call.**

**Order-status field names gained, and corroborated.** `nOrdNo`, `ordSt`, `fldQty`,
`avgPrc`, `unFldSz` via `order_report()` — used identically by two independent
integrations. That made the second half of B3 buildable: `poll_open_orders` now reconciles
working orders against the broker instead of only expiring unacknowledged ones, so partial
fills and post-acknowledgement rejections are detected. Unrecognised statuses map to `None`
rather than being guessed at.

**Two things nobody had.** Worth recording precisely because it bounds what borrowing can
achieve:

- **`limits()` field names.** All four projects were checked. One returns the raw dict
  unexamined; another flags its Kotak field names as unverified in its own README. So
  `src/account.py`'s equity mapping remains a guess *everywhere*, and only a live call
  settles it. This is the mapping that mis-sizes every trade by the same factor if wrong.
- **Scrip-master download.** The nearest implementation across all four projects is a
  `NotImplementedError` with a TODO — so there was nothing to borrow. **Since written, it
  has been built here** (`resolve_scrip_master_url` / `download_scrip_master` /
  `load_or_download_master` in `src/instruments.py`, covered by
  `tests/test_scrip_master.py`), and B4 now has a success path rather than only a
  fail-safe. The bullet is kept because it still bounds what borrowing achieved.

**Licences ruled out the richest source.** `D:\APEXBOT\external\` vendors `nautilus_trader`
(LGPL) and `openalgo` (AGPL) — the latter has verified adapters for many Indian brokers.
Copying from either into this now-public repo would force its licence onto the whole
project, so neither was read for code.

**The probe now dumps every remaining unknown.** `limits()`, `positions()`,
`order_report()`, `scrip_master()` and `search_scrip()` response shapes, so a single
credentialed run settles the two open mappings instead of requiring a second round trip.

### 5.10 Post-build review findings

A deliberately adversarial pass over the finished system, motivated by the track record:
a real defect had surfaced in *every* phase, and always through testing rather than
reading. That pattern is evidence more exist, not reassurance.

**A live/backtest divergence at the session boundary — the most consequential find.**
Resampling 1-minute bars to the strategy interval happens in two places: `resample_frame`
(batch, used by the backtester) and `BarResampler` (streaming, used live). The streaming
version only completes a bucket when the *next* one opens, so:

1. The final bucket of each session was never emitted live — 74 five-minute bars instead
   of 75 — while the backtester emitted all 75. A divergence at precisely the point
   square-off logic runs.
2. Worse, that stale bucket then surfaced on the **next** session's first bar, *after*
   `on_session_start` had already reset the session-anchored indicators. Day two's VWAP
   was therefore anchored on day one's close, and the strategy received a bar stamped a
   whole session earlier — with `StrategyContext.now` wrong by a full trading day.

Neither was visible in unit tests of the resampler alone, because in isolation its
behaviour is *correct*: a streaming resampler cannot know a session ended. The bug belonged
to the engine's ordering. `BarResampler` now exposes `flush()` and `held_session_day()`, and
`StrategyEngine._handle_bar` closes out the previous session's bucket **before** rolling.

**Two resamplers is a standing risk, now bounded by a test.** Their shapes differ enough
(whole-frame versus one-bar-at-a-time) that sharing an implementation is awkward, so both
remain — but `TestLiveMatchesBacktestResampling` asserts they agree across intervals, gaps,
synthetic bars, partial final buckets and a full 375-bar session. Without that, the claim
that backtest and live share behaviour would rest on nothing.

**26 numeric boundary probes found nothing.** Cost-model monotonicity and the flat-fee cap;
long-versus-short round trips diverging correctly once entry ≠ exit; sizing never exceeding
the per-trade budget across stop distances from 0.2% to 10%; profits correctly creating
headroom in the worst-case gate; the position book's reversal-through-zero case realising
P&L only on the closed portion and re-pricing the remainder. All held.

### 5.11 Resolved after Phase 5

**The NSE holiday calendar was wrong, and it mattered.** The list shipped in Phase 0 was
written from recall and marked provisional. Cross-checked against two independent published
calendars in July 2026, **2025 was exactly right and 2026 was substantially wrong**: three
dates off by a single day (Holi, Ram Navami, Mahavir Jayanti), two spurious closures, and
six missing entirely — including a late-added Maharashtra municipal election day.

Eleven days out of 245 were misclassified. The failure mode is quiet rather than loud: a
missing holiday makes the bar builder emit synthetic gap-fill bars for a day the market
never opened, and every backtest spanning that day inherits fabricated data that looks
entirely normal. Five spurious closures would have discarded real trading days.

This is why `covers_year()` warns on unlisted years and `NSE_HOLIDAYS_FILE` exists — but
the deeper lesson is that "provisional, verify before use" sitting in a comment is not a
control. The corrected dates are now pinned by tests, including tests asserting that the
previously-wrong dates *are* trading days.

**Muhurat trading is now an explicit decision rather than an accident.** Diwali Laxmi Pujan
holds a short ceremonial session, often on a day the market is otherwise closed
(2026-11-08 is a Sunday). Real trades occur, but it runs roughly an hour at non-standard
times, so every session constant in `market_calendar.py` is wrong for it. The bot skips it
deliberately — recorded in `MUHURAT_SESSIONS` so the reasoning survives, rather than being
an incidental consequence of the weekend check.

---

## 6. Design principles

1. **The same code decides in backtest and in live.** Any divergence invalidates all testing.
2. **Fail closed.** Every uncertainty — stale feed, reconciliation mismatch, unknown state —
   resolves to *no new risk*. Exits are always permitted.
3. **The broker is the source of truth** for positions, orders, and cash. Local state is a cache.
4. **Costs are modelled before edge is claimed.** Gross P&L is not a result.
5. **Out-of-sample or it didn't happen.**
6. **Every trade is traceable** to the exact strategy version, parameters, and bar that
   produced it.
7. **Own the exit.** Never let the broker's auto-square-off be the strategy's exit.

### 5.12 The first backtest on real market data

Real NSE 5-minute bars, sourced from TradingView as a bootstrap (the Parquet store is
still empty and the Kotak probe has not run). **39 liquid momentum names, 4 sessions,
11,700 bars.** The universe came from the scanner shortlist filtered to D6's real
liquidity floor (Rs 200cr ADV) plus current participation, so these are names a bot could
actually trade rather than thin movers.

| | trades | expectancy | 95% CI | PF | win rate | net |
|---|---|---|---|---|---|---|
| `trend_pullback` v1 | 36 | **-0.2682R** | [-0.503, -0.034] | 0.397 | 25.0% | -24,794 |
| `sma_crossover` (negative control) | 12 | +0.0465R | [-0.439, +0.532] | 1.177 | 33.3% | +1,425 |

**The negative control beat the strategy.** That is the headline, and it is not a
comfortable one.

**It is not a friction problem.** This is the finding that changes the diagnosis. An
earlier 12-name run showed friction as 80% of the loss, which pointed at costs. Widening
the sample shows **gross P&L before any friction is -15,653, or -435 per trade**. Friction
itself came in at 254 per trade — **0.099R against the ~0.12R design budget**, so the cost
model is behaving exactly as §5.2 says it should. The entry logic is losing money on its
own.

**The regime was hostile, and that is a real caveat rather than an excuse.** Averaged
across the 39 names the window was **-0.79%**, with two up sessions and two down. D3 exists
precisely because a long-only strategy looks good in a rising market for reasons unrelated
to skill — and the converse applies here. A 5% account loss against a 0.79% market decline
is still a large amplification, but four days is one regime and cannot settle anything.

**The safety machinery worked.** The daily drawdown breaker tripped twice and halted new
entries; 219 signals were refused across the run (74 worst-case gate, 66 position cap, 59
open-risk cap, 20 post-halt). Those refusals are the system doing its job, and the loss
would have been materially larger without them.

**What this does and does not license.** It does not clear or fail the §2.6 gates — the
data-quality gate blocks the sample outright at 4 trading days against a 20-day minimum. It
does move the question. Before this run the honest position was "no idea"; the plausible
range spanned "essentially break-even before costs" to "structurally broken". The upper
bound of the confidence interval is now **negative**, and gross is negative, so the
break-even end of that range is excluded for this window.

**The tempting mistake is to tune on it.** Thirty-six trades over four days is exactly the
sample size that will happily yield a parameter set with positive expectancy and no
predictive content. That is what §2.5's walk-forward, locked holdout and trial counting
exist to prevent, and the discipline matters most at the moment the result is
disappointing. The next input is more data, not more fitting.

#### 5.12.1 Autopsy — *how* it loses

Mechanism, not parameter search. On 36 trades a search would produce a configuration with
positive expectancy and no predictive content; characterising the failure mode generates
hypotheses to test on data that does not exist yet.

```
EXIT REASON        25 signal      avg -0.098R      (69% of trades)
                    7 stop_loss   avg -1.071R
                    4 square_off  avg +0.071R

EXCURSION          MFE median +0.29R   p75 +0.84R
                   42% of trades never reached +0.25R
                   0 of 36 reached +1.0R and then finished negative

SHAPE              9 winners / 27 losers
                   avg win +0.707R   avg loss -0.593R   ->  R:R 1.19:1
                   needs 46% win rate to break even; got 25%
```

**The exits are not the problem, and the excursion data proves it.** The obvious suspicion
with a 25% win rate and 69% of exits coming from the strategy's own signal logic is that it
is cutting trades that would have worked. It is not: **no trade reached +1.0R and then
finished negative**, and MFE sits at a median of +0.29R. The trailing stop gives nothing
back because there is nothing to give back.

**The stops are not the problem either.** Seven stop-outs at -1.071R is the stop doing
exactly what it was sized to do, and no trade was stopped within three bars — so entries
are not being taken immediately in front of adverse moves.

**The entry is the problem.** Forty-two per cent of positions never move a quarter of an R
in favour, and the median best-case excursion is +0.29R against a stop a full R away. The
pullback-in-an-uptrend setup is not being followed by continuation in this sample. That is
a statement about the *signal*, and it is the one thing the cost model, the risk engine and
the exit logic cannot compensate for.

**It is broad rather than concentrated.** The worst instrument contributed -1.42R and the
best +0.78R across 39 names, so this is not one bad symbol dragging an otherwise sound
strategy — which makes a systematic cause more likely than luck.

**The R:R arithmetic is the summary.** At 1.19:1 the strategy needs a 46% win rate to break
even and produced 25%. Either winners must run substantially further or the entry must be
far more selective. Both are hypotheses for the next dataset, **not** changes to make on the
strength of four days.
