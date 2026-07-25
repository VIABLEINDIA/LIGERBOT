"""Execution engine tests — the module that actually places orders.

This module sat at **25% coverage** while `RiskEngine` sat at 96%. That is backwards from
where the risk lives: the risk engine computes a number and returns, and if it is wrong a
test catches it. The execution engine holds a broker session, runs unattended for six and a
half hours, and converts decisions into irreversible external side effects. 156 of its 209
statements had never been executed by a test — including every path that runs when
something goes wrong, which is precisely when it matters.

The demos exercised this module end to end and exited 0, which is why the gap went
unnoticed. A demo walks the happy path. Everything below is a path the happy case never
reaches.

Grouped by the property being protected, not by method:

* **Idempotency** — what makes at-least-once delivery safe rather than dangerous.
* **B3: acceptance is not execution** — the original defect, and the easiest to reintroduce.
* **B4: never guess a symbol** — a guessed symbol is an order on the wrong instrument.
* **Ambiguity is preserved, not resolved** — a send that fails in transit may still have
  reached the exchange, and the engine must not decide that it did not.
* **Mode separation** — paper mode must refuse, or every trade fills twice.
"""
from __future__ import annotations

import datetime as dt

import fakeredis
import pytest

import config
from src import event_bus, kotak_api
from src.execution_engine import ExecutionEngine, RateLimiter
from src.instruments import InstrumentMaster, parse_scrip_master
from src.order_state import BrokerOrderStatus, Fill, ManagedOrder, OrderStatus
from src.risk_engine import Intent, Side


@pytest.fixture
def client():
    return fakeredis.FakeStrictRedis(server=fakeredis.FakeServer(), decode_responses=True)


@pytest.fixture(autouse=True)
def _bus(monkeypatch, client):
    monkeypatch.setattr(event_bus, "get_client", lambda *a, **k: client)
    monkeypatch.setattr(event_bus, "ping", lambda *a, **k: True)


@pytest.fixture(autouse=True)
def _isolate_alerter():
    """The alerter is a process-wide singleton holding a Redis handle and a dedup cache.

    Left alone it keeps the *first* test's fakeredis, so later assertions read an empty
    stream and the failure looks like "the alert was never raised" rather than "it went
    somewhere else".
    """
    from src.alerting import reset_alerter

    reset_alerter()
    yield
    reset_alerter()


@pytest.fixture
def master():
    return InstrumentMaster(parse_scrip_master([
        {"pSymbol": "2885", "pTrdSymbol": "RELIANCE-EQ", "pSymbolName": "RELIANCE",
         "pGroup": "EQ", "lLotSize": "1", "dTickSize": "5"},
    ]))


@pytest.fixture
def engine(master):
    return ExecutionEngine(master)


class FakeNeo:
    """Records what was sent, and returns whatever the test tells it to."""

    def __init__(self, response=None, raises=None):
        self.response = response if response is not None else {"stat": "Ok",
                                                               "nOrdNo": "24070100001"}
        self.raises = raises
        self.sent: list[dict] = []

    def place_order(self, **kwargs):
        self.sent.append(kwargs)
        if self.raises:
            raise self.raises
        return self.response


def approved(**overrides) -> dict:
    fields = {
        "client_order_id": "LB-001",
        "instrument_id": "nse_cm:2885",
        "side": "BUY",
        "intent": "OPEN_LONG",
        "quantity": "10",
        "order_type": "MARKET",
        "price": "1300.0",
    }
    fields.update(overrides)
    return fields


def published(client) -> list[dict]:
    return [row[1] for row in client.xrange(config.STREAM_FILLED_ORDERS)]


def live(monkeypatch):
    monkeypatch.setattr(config, "TRADING_MODE", "live")


def dry_run(monkeypatch):
    monkeypatch.setattr(config, "TRADING_MODE", "dry_run")


# ---------------------------------------------------------------------------
class TestIdempotency:
    """At-least-once delivery is safe only because of this. Without it, every
    reclaimed-but-unacked entry becomes a duplicate order at the exchange."""

    def test_a_redelivered_order_is_not_sent_twice(self, engine, monkeypatch, client):
        live(monkeypatch)
        engine.neo = FakeNeo()
        engine._handle(approved())
        engine._handle(approved())          # same client_order_id
        assert len(engine.neo.sent) == 1

    def test_the_duplicate_is_not_silently_dropped(self, engine, monkeypatch, caplog):
        live(monkeypatch)
        engine.neo = FakeNeo()
        engine._handle(approved())
        with caplog.at_level("WARNING"):
            engine._handle(approved())
        assert "already sent" in caplog.text

    def test_distinct_orders_both_send(self, engine, monkeypatch):
        live(monkeypatch)
        engine.neo = FakeNeo()
        engine._handle(approved(client_order_id="LB-001"))
        engine._handle(approved(client_order_id="LB-002"))
        assert len(engine.neo.sent) == 2

    def test_an_order_without_a_client_order_id_is_refused(self, engine):
        """Without one, a redelivery cannot be recognised — so it is a hard error."""
        fields = approved()
        del fields["client_order_id"]
        with pytest.raises(ValueError, match="client_order_id"):
            engine._handle(fields)

    def test_an_order_without_an_instrument_is_refused(self, engine):
        fields = approved()
        del fields["instrument_id"]
        with pytest.raises(ValueError, match="instrument"):
            engine._handle(fields)

    @pytest.mark.parametrize("quantity", ["0", "-5"])
    def test_a_nonpositive_quantity_is_refused(self, engine, quantity):
        with pytest.raises(ValueError, match="quantity"):
            engine._handle(approved(quantity=quantity))


class TestHaltBehaviour:
    """A halt stops new risk. It must never trap an existing position."""

    def test_a_halt_refuses_to_open(self, engine, monkeypatch):
        live(monkeypatch)
        engine.neo = FakeNeo()
        engine.kill_switch.halt("daily loss limit")
        engine._handle(approved(intent="OPEN_LONG"))
        assert engine.neo.sent == []

    def test_a_halt_still_permits_closing(self, engine, monkeypatch):
        """The dangerous failure is a halt that leaves you unable to get flat."""
        live(monkeypatch)
        engine.neo = FakeNeo()
        engine.kill_switch.halt("daily loss limit")
        engine._handle(approved(intent="CLOSE_LONG", side="SELL"))
        assert len(engine.neo.sent) == 1

    def test_the_halt_reason_is_logged(self, engine, monkeypatch, caplog):
        live(monkeypatch)
        engine.neo = FakeNeo()
        engine.kill_switch.halt("feed outage")
        with caplog.at_level("ERROR"):
            engine._handle(approved())
        assert "feed outage" in caplog.text


class TestSymbolResolutionB4:
    """Defect B4: the engine used to send the display name as the trading symbol."""

    def test_no_instrument_master_refuses_rather_than_guessing(self, monkeypatch):
        live(monkeypatch)
        engine = ExecutionEngine(None)
        engine.neo = FakeNeo()
        with pytest.raises(RuntimeError, match="B4"):
            engine._trading_symbol("nse_cm:2885")

    def test_an_unknown_instrument_raises(self, engine):
        with pytest.raises(KeyError):
            engine._trading_symbol("nse_cm:9999")

    def test_the_canonical_symbol_is_what_reaches_the_broker(self, engine, monkeypatch):
        live(monkeypatch)
        engine.neo = FakeNeo()
        engine._handle(approved())
        assert engine.neo.sent[0]["trading_symbol"] == "RELIANCE-EQ"

    def test_the_client_order_id_is_carried_to_the_broker_as_the_tag(self, engine,
                                                                    monkeypatch):
        """Idempotency has to survive at the broker too, not just in our registry."""
        live(monkeypatch)
        engine.neo = FakeNeo()
        engine._handle(approved(client_order_id="LB-042"))
        assert engine.neo.sent[0]["tag"] == "LB-042"


class TestAcceptanceIsNotExecutionB3:
    """The original defect: `stat: Ok` was published to filled_orders as a fill."""

    def test_an_accepted_order_is_acked_not_filled(self, engine, monkeypatch, client):
        live(monkeypatch)
        engine.neo = FakeNeo({"stat": "Ok", "nOrdNo": "999"})
        engine._handle(approved())

        order = engine.registry.get("LB-001")
        assert order.status is OrderStatus.ACKED
        assert order.filled_quantity == 0
        assert all(row["status"] != "FILLED" for row in published(client))

    def test_the_broker_order_id_is_recorded(self, engine, monkeypatch):
        live(monkeypatch)
        engine.neo = FakeNeo({"stat": "Ok", "nOrdNo": "24070100077"})
        engine._handle(approved())
        assert engine.registry.get("LB-001").broker_order_id == "24070100077"

    def test_a_rejection_is_marked_rejected(self, engine, monkeypatch):
        live(monkeypatch)
        engine.neo = FakeNeo({"stat": "Not_Ok", "errMsg": "insufficient margin"})
        engine._handle(approved())
        order = engine.registry.get("LB-001")
        assert order.status is OrderStatus.REJECTED
        assert "margin" in order.error

    def test_a_rejection_raises_an_alert(self, engine, monkeypatch, client):
        live(monkeypatch)
        engine.neo = FakeNeo({"stat": "Not_Ok", "errMsg": "invalid symbol"})
        engine._handle(approved())
        alerts = client.xrange(config.STREAM_ALERTS)
        assert any("invalid symbol" in row[1]["message"] for row in alerts)


class TestTransitFailureStaysAmbiguous:
    """The single most important behaviour in this module.

    If `place_order` raises, the order may have reached the exchange anyway — a timeout on
    the response says nothing about whether the request arrived. Marking it REJECTED would
    be a *decision* that it did not, and the engine would then happily re-send, doubling a
    live position. It stays SENT so the ack timeout expires it and reconciliation
    establishes what actually happened.
    """

    def test_a_transit_exception_leaves_the_order_sent(self, engine, monkeypatch):
        live(monkeypatch)
        engine.neo = FakeNeo(raises=TimeoutError("read timed out"))
        engine._handle(approved())
        assert engine.registry.get("LB-001").status is OrderStatus.SENT

    def test_it_is_not_marked_rejected(self, engine, monkeypatch):
        live(monkeypatch)
        engine.neo = FakeNeo(raises=ConnectionError("connection reset"))
        engine._handle(approved())
        assert engine.registry.get("LB-001").status is not OrderStatus.REJECTED

    def test_the_error_is_recorded_and_published(self, engine, monkeypatch, client):
        live(monkeypatch)
        engine.neo = FakeNeo(raises=TimeoutError("read timed out"))
        engine._handle(approved())
        assert "timed out" in engine.registry.get("LB-001").error
        assert published(client), "the ambiguous state must still be broadcast"

    def test_the_duplicate_guard_still_holds_after_a_transit_failure(self, engine,
                                                                    monkeypatch):
        """The dangerous sequence: send fails ambiguously, entry is redelivered, and a
        second order goes out for a position that may already exist."""
        live(monkeypatch)
        engine.neo = FakeNeo(raises=TimeoutError("read timed out"))
        engine._handle(approved())
        engine._handle(approved())
        assert len(engine.neo.sent) == 1


class TestDryRunIsNotPaperTrading:
    def test_dry_run_sends_nothing_to_the_broker(self, engine, monkeypatch):
        dry_run(monkeypatch)
        engine.neo = FakeNeo()
        engine._handle(approved())
        assert engine.neo.sent == []

    def test_dry_run_fills_instantly_and_says_so(self, engine, monkeypatch, client):
        dry_run(monkeypatch)
        engine._handle(approved())
        rows = published(client)
        # Lower-case: the bus JSON-encodes the field, it is not a Python repr.
        assert rows[-1]["dry_run"] == "true"
        assert rows[-1]["mode"] == "dry_run"

    def test_dry_run_needs_no_instrument_master(self, monkeypatch):
        """It must stay usable as a wiring check on a machine with no scrip master."""
        dry_run(monkeypatch)
        engine = ExecutionEngine(None)
        engine._handle(approved())
        assert engine.registry.get("LB-001").filled_quantity == 10


class TestReconciliation:
    def _acked(self, engine, broker_id="B1", quantity=10):
        # A stub session, so `_ensure_broker()` never reaches a real login.
        engine.neo = FakeNeo()
        order = ManagedOrder(client_order_id="LB-001", instrument_id="nse_cm:2885",
                             side=Side.BUY, intent=Intent.OPEN_LONG, quantity=quantity)
        order.mark_sent()
        order.mark_acked(broker_id)
        engine.registry.register(order)
        return order

    def _status(self, **overrides):
        fields = dict(broker_order_id="B1", raw_status="complete", filled_quantity=10,
                      pending_quantity=0, average_price=1301.5)
        fields.update(overrides)
        return BrokerOrderStatus(**fields)

    def test_a_broker_fill_is_applied(self, engine):
        order = self._acked(engine)
        engine._apply_broker_status(order, self._status())
        assert order.filled_quantity == 10
        assert order.status is OrderStatus.FILLED

    def test_each_partial_emits_its_own_event(self, engine, client):
        """A position built from three partials must produce three fills, not one
        aggregate — the position book keys on each increment."""
        order = self._acked(engine, quantity=30)
        for filled in (10, 20, 30):
            engine._apply_broker_status(
                order, self._status(filled_quantity=filled, pending_quantity=30 - filled))
        fills = [row for row in published(client) if int(row.get("filled_quantity", 0)) > 0]
        assert len(fills) == 3
        assert [int(row["filled_quantity"]) for row in fills] == [10, 20, 30]

    def test_no_duplicate_fill_when_the_report_repeats(self, engine, client):
        """order_report() is polled repeatedly; the same completed order appears every
        time and must not re-fill on each poll."""
        order = self._acked(engine)
        engine._apply_broker_status(order, self._status())
        before = len(published(client))
        engine._apply_broker_status(order, self._status())
        assert len(published(client)) == before
        assert order.filled_quantity == 10

    def test_an_unrecognised_status_is_left_untouched(self, engine, caplog):
        """Guessing FILLED or REJECTED from a status we do not understand is worse than
        surfacing it."""
        order = self._acked(engine)
        with caplog.at_level("WARNING"):
            engine._apply_broker_status(
                order, self._status(raw_status="zzz-unknown", filled_quantity=0))
        assert order.status is OrderStatus.ACKED
        assert "unrecognised" in caplog.text

    def test_a_post_ack_rejection_is_applied(self, engine):
        order = self._acked(engine)
        engine._apply_broker_status(order, self._status(
            raw_status="rejected", filled_quantity=0, rejection_reason="RMS block"))
        assert order.status is OrderStatus.REJECTED
        assert "RMS" in order.error

    def test_a_cancellation_is_applied(self, engine):
        order = self._acked(engine)
        engine._apply_broker_status(
            order, self._status(raw_status="cancelled", filled_quantity=0))
        assert order.status is OrderStatus.CANCELLED

    def test_an_illegal_transition_is_logged_not_forced(self, engine, caplog):
        """Our state machine and the broker disagreeing is information. Forcing the
        transition would destroy it."""
        order = self._acked(engine)
        order.add_fill(Fill(10, 1300.0, dt.datetime.now()))   # terminal: FILLED
        with caplog.at_level("ERROR"):
            engine._apply_broker_status(
                order, self._status(raw_status="cancelled", filled_quantity=0))
        assert order.status is OrderStatus.FILLED

    def test_an_acked_order_missing_from_the_report_is_surfaced(self, engine, monkeypatch,
                                                               caplog):
        """Our view and the broker's disagree. That is not a transient to be swallowed."""
        live(monkeypatch)
        self._acked(engine, broker_id="B-GONE")
        monkeypatch.setattr(kotak_api, "safe_call", lambda *a, **k: {"data": []})
        with caplog.at_level("ERROR"):
            engine._reconcile_live_orders()
        assert "disagree" in caplog.text

    def test_an_unacked_order_is_skipped_not_flagged(self, engine, monkeypatch, caplog):
        """No broker id yet means the ack timeout owns it, not reconciliation."""
        live(monkeypatch)
        engine.neo = FakeNeo()
        order = ManagedOrder(client_order_id="LB-002", instrument_id="nse_cm:2885",
                             side=Side.BUY, intent=Intent.OPEN_LONG, quantity=10)
        order.mark_sent()
        engine.registry.register(order)
        monkeypatch.setattr(kotak_api, "safe_call", lambda *a, **k: {"data": []})
        with caplog.at_level("ERROR"):
            engine._reconcile_live_orders()
        assert "disagree" not in caplog.text


class TestReconciliationFailureModes:
    def _acked(self, engine):
        engine.neo = FakeNeo()
        order = ManagedOrder(client_order_id="LB-001", instrument_id="nse_cm:2885",
                             side=Side.BUY, intent=Intent.OPEN_LONG, quantity=10)
        order.mark_sent()
        order.mark_acked("B1")
        engine.registry.register(order)
        return order

    def test_a_failed_report_does_not_assume_completion(self, engine, monkeypatch, caplog):
        """The order stays exactly as it was. Assuming it completed would be inventing
        a fill; assuming it died would be inventing a cancellation."""
        live(monkeypatch)
        order = self._acked(engine)

        def boom(*a, **k):
            raise kotak_api.KotakAPIError("order_report unavailable")

        monkeypatch.setattr(kotak_api, "safe_call", boom)
        with caplog.at_level("ERROR"):
            engine._reconcile_live_orders()
        assert order.status is OrderStatus.ACKED
        assert "NOT assumed complete" in caplog.text

    def test_an_expired_session_triggers_reauthentication(self, engine, monkeypatch):
        """Retrying a call that failed for session expiry would fail identically forever."""
        live(monkeypatch)
        self._acked(engine)

        def expired(*a, **k):
            raise kotak_api.KotakSessionExpired("2fa required")

        refreshed = []
        monkeypatch.setattr(kotak_api, "safe_call", expired)
        monkeypatch.setattr("src.auth_session.refresh_after_expiry",
                            lambda *a, **k: refreshed.append(1) or "new-session")
        engine._reconcile_live_orders()
        assert refreshed == [1]
        assert engine.neo == "new-session"

    def test_reconciliation_is_skipped_when_no_orders_are_live(self, engine, monkeypatch):
        live(monkeypatch)
        engine.neo = FakeNeo()
        called = []
        monkeypatch.setattr(kotak_api, "safe_call",
                            lambda *a, **k: called.append(1) or {"data": []})
        engine.poll_open_orders()
        assert called == [], "no live orders means no reason to call the broker"

    def test_reconciliation_runs_before_expiry(self, engine, monkeypatch):
        """An order the broker has already filled must not be expired just because our
        own ack timeout elapsed."""
        live(monkeypatch)
        order = self._acked(engine)
        monkeypatch.setattr(config, "ORDER_ACK_TIMEOUT_SECONDS", 0.0)
        monkeypatch.setattr(kotak_api, "safe_call", lambda *a, **k: {"data": [{
            "nOrdNo": "B1", "ordSt": "complete", "fldQty": "10", "unFldSz": "0",
            "avgPrc": "1301.5"}]})
        engine.poll_open_orders()
        assert order.status is OrderStatus.FILLED
        assert order.status is not OrderStatus.EXPIRED


class TestModeSeparation:
    def test_paper_mode_refuses_to_run(self, engine, monkeypatch, caplog):
        """Both this and paper_broker consume approved_orders from separate groups, so
        both would receive every order and both would fill it — double-counting every
        trade while looking entirely healthy."""
        monkeypatch.setattr(config, "TRADING_MODE", "paper")
        with caplog.at_level("ERROR"):
            assert engine._check_mode() is False
        assert "twice" in caplog.text

    @pytest.mark.parametrize("mode", ["dry_run", "live"])
    def test_other_modes_are_permitted(self, engine, monkeypatch, mode):
        monkeypatch.setattr(config, "TRADING_MODE", mode)
        assert engine._check_mode() is True

    def test_run_stops_when_redis_is_unreachable(self, engine, monkeypatch, caplog):
        monkeypatch.setattr(event_bus, "ping", lambda *a, **k: False)
        with caplog.at_level("ERROR"):
            engine.run()      # must return, not loop
        assert "Redis not reachable" in caplog.text

    def test_run_stops_in_paper_mode(self, engine, monkeypatch):
        """Armed with a tripwire: if the refusal ever regresses, this must FAIL rather
        than hang. A test whose failure mode is an infinite loop stops being a test."""
        monkeypatch.setattr(config, "TRADING_MODE", "paper")

        def must_not_reach(**kwargs):
            raise AssertionError("run() entered the consume loop in paper mode")

        monkeypatch.setattr(engine.consumer, "claim_stale", must_not_reach)
        monkeypatch.setattr(engine.consumer, "read", must_not_reach)
        engine.run()

    def test_live_mode_is_not_sufficient_to_arm(self, engine, monkeypatch, caplog):
        """An environment variable is one typo away from committing real capital."""
        live(monkeypatch)
        with caplog.at_level("ERROR"):
            cleared = engine._check_live_clearance()
        assert cleared is False
        assert "LIVE TRADING BLOCKED" in caplog.text

    def test_non_live_modes_skip_the_guard(self, engine, monkeypatch):
        dry_run(monkeypatch)
        assert engine._check_live_clearance() is True


class TestAckTimeoutExpiry:
    """The safety net under the ambiguous transit failure.

    An order left SENT because `place_order` raised is not resolved by hoping. The ack
    timeout expires it and publishes that fact, which is what lets reconciliation and the
    position manager establish the truth instead of the engine guessing.
    """

    def _sent(self, engine):
        order = ManagedOrder(client_order_id="LB-001", instrument_id="nse_cm:2885",
                             side=Side.BUY, intent=Intent.OPEN_LONG, quantity=10)
        order.mark_sent()
        engine.registry.register(order)
        return order

    def test_an_unacknowledged_order_expires(self, engine, monkeypatch):
        dry_run(monkeypatch)
        order = self._sent(engine)
        monkeypatch.setattr(config, "ORDER_ACK_TIMEOUT_SECONDS", 0.0)
        engine.poll_open_orders()
        assert order.status is OrderStatus.EXPIRED

    def test_the_expiry_is_published(self, engine, monkeypatch, client):
        """Expiring silently would leave every other module believing it is still working."""
        dry_run(monkeypatch)
        self._sent(engine)
        monkeypatch.setattr(config, "ORDER_ACK_TIMEOUT_SECONDS", 0.0)
        engine.poll_open_orders()
        assert any(row["status"] == "EXPIRED" for row in published(client))

    def test_a_fresh_order_is_left_alone(self, engine, monkeypatch):
        dry_run(monkeypatch)
        order = self._sent(engine)
        monkeypatch.setattr(config, "ORDER_ACK_TIMEOUT_SECONDS", 3600.0)
        engine.poll_open_orders()
        assert order.status is OrderStatus.SENT

    def test_a_broker_ack_moves_a_sent_order(self, engine):
        order = self._sent(engine)
        engine.neo = FakeNeo()
        engine._apply_broker_status(order, BrokerOrderStatus(
            broker_order_id="B7", raw_status="open", filled_quantity=0,
            pending_quantity=10, average_price=0.0))
        assert order.status is OrderStatus.ACKED
        assert order.broker_order_id == "B7"


class TestFailureIsolation:
    def test_a_broken_alerter_does_not_stop_the_order_being_recorded(self, engine,
                                                                    monkeypatch, caplog):
        """Alerting is observability. It must never be able to take down execution."""
        live(monkeypatch)
        engine.neo = FakeNeo({"stat": "Not_Ok", "errMsg": "margin"})

        def broken(*a, **k):
            raise ConnectionError("alert sink down")

        monkeypatch.setattr("src.alerting.get_alerter", broken)
        with caplog.at_level("ERROR"):
            engine._handle(approved())
        assert engine.registry.get("LB-001").status is OrderStatus.REJECTED
        assert "Could not raise the rejection alert" in caplog.text


class TestTheMainLoop:
    def test_one_pass_wires_every_stage(self, engine, monkeypatch):
        """`run()` is the only thing that calls these in production, and nothing had ever
        executed it. A stage dropped from the loop would be invisible."""
        dry_run(monkeypatch)
        calls = []

        class Stop(RuntimeError):
            pass

        monkeypatch.setattr(engine.consumer, "claim_stale",
                            lambda **k: calls.append("claim") or [])
        monkeypatch.setattr(engine.consumer, "read",
                            lambda **k: calls.append("read") or [])
        monkeypatch.setattr(engine, "poll_open_orders", lambda: calls.append("poll"))
        monkeypatch.setattr(engine.registry, "purge_terminal",
                            lambda *a, **k: calls.append("purge"))

        def backlog(*a, **k):
            calls.append("backlog")
            raise Stop()                      # break out after exactly one pass

        monkeypatch.setattr(engine.consumer, "check_backlog", backlog)
        with pytest.raises(Stop):
            engine.run()
        assert calls == ["claim", "read", "poll", "purge", "backlog"]

    def test_an_approved_order_read_from_the_stream_is_executed(self, engine, monkeypatch):
        """End to end through the consumer, not just a direct `_handle` call."""
        dry_run(monkeypatch)

        class Stop(RuntimeError):
            pass

        monkeypatch.setattr(engine.consumer, "claim_stale", lambda **k: [])
        monkeypatch.setattr(engine.consumer, "read",
                            lambda **k: [("1-1", approved(client_order_id="LB-777"))])

        def backlog(*a, **k):
            raise Stop()

        monkeypatch.setattr(engine.consumer, "check_backlog", backlog)
        with pytest.raises(Stop):
            engine.run()
        assert engine.registry.get("LB-777") is not None

    def test_a_cleared_guard_arms_live_trading(self, engine, monkeypatch, caplog):
        """The block path was tested; the *pass* path is the one that risks money."""
        live(monkeypatch)

        class Report:
            cleared = True

            def render(self):
                return "all clear"

        monkeypatch.setattr("src.live_guard.evaluate", lambda **k: Report())
        with caplog.at_level("WARNING"):
            assert engine._check_live_clearance() is True
        assert "cleared" in caplog.text.lower()


class TestStartup:
    def test_main_survives_a_missing_instrument_cache(self, monkeypatch):
        """A missing or corrupt cache must degrade to a warning, not a crash on boot."""
        import src.execution_engine as mod

        monkeypatch.setattr(mod.InstrumentMaster, "load_cache",
                            lambda *a, **k: (_ for _ in ()).throw(OSError("no such file")))
        started = []
        monkeypatch.setattr(mod.ExecutionEngine, "run", lambda self: started.append(self))
        mod.main()
        assert started and started[0].instruments is None

    def test_main_passes_a_loaded_master_to_the_engine(self, monkeypatch, master):
        import src.execution_engine as mod

        monkeypatch.setattr(mod.InstrumentMaster, "load_cache", lambda *a, **k: master)
        started = []
        monkeypatch.setattr(mod.ExecutionEngine, "run", lambda self: started.append(self))
        mod.main()
        assert started[0].instruments is master

    def test_live_without_an_instrument_master_is_flagged_loudly(self, monkeypatch,
                                                                caplog):
        """It cannot resolve symbols, so every live order will be refused (B4). Better to
        say so at startup than to discover it on the first signal."""
        live(monkeypatch)
        engine = ExecutionEngine(None)

        class Stop(RuntimeError):
            pass

        monkeypatch.setattr(engine, "_check_live_clearance", lambda: True)
        monkeypatch.setattr(engine.consumer, "claim_stale", lambda **k: [])

        def stop(**k):
            raise Stop()

        monkeypatch.setattr(engine.consumer, "read", stop)
        with caplog.at_level("ERROR"), pytest.raises(Stop):
            engine.run()
        assert "B4" in caplog.text


class TestRateLimiter:
    def test_it_spaces_calls(self):
        # Tolerance is deliberate: Windows timer granularity is ~15ms against a 20ms
        # interval. The property worth asserting is that it sleeps at all — a tighter
        # bound would buy nothing and fail intermittently on CI, and a flaky test is
        # worse than a lenient one.
        limiter = RateLimiter(max_per_second=50)
        import time as _t

        start = _t.monotonic()
        limiter.wait()
        limiter.wait()
        assert _t.monotonic() - start >= limiter.min_interval * 0.5

    def test_a_zero_rate_does_not_divide_by_zero(self):
        assert RateLimiter(0).min_interval == 1.0

    def test_the_limiter_is_applied_to_live_orders(self, engine, monkeypatch):
        live(monkeypatch)
        engine.neo = FakeNeo()
        waits = []
        monkeypatch.setattr(engine.rate_limiter, "wait", lambda: waits.append(1))
        engine._handle(approved())
        assert waits == [1]
