"""Structured logging and correlation-ID threading (DESIGN.md 3.9).

Five processes participate in every trade. Reconstructing *which* signal became *which*
order currently means reading five logs and matching on timestamps and instrument names —
and both of those collide. The correlation id fixes that, but only if it is actually
present on the lines that matter, which is the hard part: threading it by hand through
every call site would be forgotten exactly once, in the error path.

Three properties carry the weight here, and none of them is "the output is JSON":

* **The id is bound automatically at the bus boundary and never leaks.** Leaking would be
  worse than not having it, because it would attribute one order's failure to a different
  order — a confidently wrong answer instead of no answer.
* **Logging cannot break trading.** A formatter that raises on an unserialisable object
  turns a log line into an outage. Every failure path degrades instead.
* **Secrets never reach the log.** Structured logging invites passing rich context, and a
  broker session is a dict containing a bearer token. This is the test that stops a
  convenience feature becoming a credential leak.
"""
from __future__ import annotations

import datetime as dt
import io
import json
import logging

import fakeredis
import pytest

import config
from src import event_bus
from src.logging_setup import (
    JsonFormatter, REDACTED, TextFormatter, configure, correlation,
    current_correlation_id, redact,
)


@pytest.fixture
def client():
    return fakeredis.FakeStrictRedis(server=fakeredis.FakeServer(), decode_responses=True)


@pytest.fixture(autouse=True)
def _restore_logging():
    """Put the root logger back exactly as pytest had it."""
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    yield
    root.handlers[:] = handlers
    root.setLevel(level)


def capture(component="test", *, fmt="json"):
    """Configure logging to a buffer and return (logger, buffer)."""
    stream = io.StringIO()
    original = config.LOG_FORMAT
    config.LOG_FORMAT = fmt
    try:
        logger = configure(component, stream=stream)
    finally:
        config.LOG_FORMAT = original
    return logger, stream


def lines(stream) -> list[dict]:
    return [json.loads(line) for line in stream.getvalue().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
class TestCorrelationThreading:
    def test_no_id_by_default(self):
        assert current_correlation_id() is None

    def test_the_id_appears_on_every_line_inside_the_block(self):
        logger, stream = capture()
        with correlation("LB-042"):
            logger.info("placing")
            logger.info("placed")
        assert [row["correlation_id"] for row in lines(stream)] == ["LB-042", "LB-042"]

    def test_it_is_restored_not_cleared_on_exit(self):
        """Nesting must work, or a handler that logs after calling another handler loses
        its own id."""
        with correlation("outer"):
            with correlation("inner"):
                assert current_correlation_id() == "inner"
            assert current_correlation_id() == "outer"
        assert current_correlation_id() is None

    def test_it_is_restored_even_when_the_body_raises(self):
        """The error path is exactly where a leaked id would do its damage."""
        with pytest.raises(ValueError):
            with correlation("LB-1"):
                raise ValueError("boom")
        assert current_correlation_id() is None

    def test_absent_id_simply_omits_the_field(self):
        logger, stream = capture()
        logger.info("no correlation here")
        assert "correlation_id" not in lines(stream)[0]


class TestBusBoundaryBinding:
    """The single place worth instrumenting: every cross-process unit of work passes
    through `StreamConsumer.handle`, so binding there threads the id across all five
    modules without a single call site changing."""

    def _consumer(self, client):
        return event_bus.StreamConsumer(client, "s1", "g1", "c1", max_deliveries=3)

    def test_the_handler_sees_the_messages_correlation_id(self, client):
        consumer = self._consumer(client)
        event_bus.publish(client, "s1", {"correlation_id": "LB-777", "x": 1})
        entry_id, fields = consumer.read(count=1, block_ms=10)[0]

        seen = []
        consumer.handle(entry_id, fields, lambda f: seen.append(current_correlation_id()))
        assert seen == ["LB-777"]

    def test_it_falls_back_to_the_client_order_id(self, client):
        consumer = self._consumer(client)
        event_bus.publish(client, "s1", {"client_order_id": "LB-9", "x": 1})
        entry_id, fields = consumer.read(count=1, block_ms=10)[0]

        seen = []
        consumer.handle(entry_id, fields, lambda f: seen.append(current_correlation_id()))
        assert seen == ["LB-9"]

    def test_a_message_with_neither_still_gets_a_traceable_id(self, client):
        """The entry id is at least unique, so an untagged message is still followable."""
        consumer = self._consumer(client)
        event_bus.publish(client, "s1", {"x": 1})
        entry_id, fields = consumer.read(count=1, block_ms=10)[0]

        seen = []
        consumer.handle(entry_id, fields, lambda f: seen.append(current_correlation_id()))
        assert seen == [entry_id]

    def test_the_id_does_not_leak_between_messages(self, client):
        """The bug that would make this whole mechanism worse than useless: attributing
        one order's failure to the order handled before it."""
        consumer = self._consumer(client)
        event_bus.publish(client, "s1", {"correlation_id": "LB-1"})
        event_bus.publish(client, "s1", {"correlation_id": "LB-2"})

        seen = []
        for entry_id, fields in consumer.read(count=2, block_ms=10):
            consumer.handle(entry_id, fields,
                            lambda f: seen.append(current_correlation_id()))
        assert seen == ["LB-1", "LB-2"]
        assert current_correlation_id() is None

    def test_a_failing_handler_does_not_leak_its_id(self, client):
        consumer = self._consumer(client)
        event_bus.publish(client, "s1", {"correlation_id": "LB-BAD"})
        entry_id, fields = consumer.read(count=1, block_ms=10)[0]
        consumer.handle(entry_id, fields,
                        lambda f: (_ for _ in ()).throw(ValueError("boom")))
        assert current_correlation_id() is None

    def test_the_retry_warning_carries_the_id(self, client):
        """The line you actually need during an incident."""
        logger, stream = capture("event_bus")
        consumer = self._consumer(client)
        event_bus.publish(client, "s1", {"correlation_id": "LB-55"})
        entry_id, fields = consumer.read(count=1, block_ms=10)[0]
        consumer.handle(entry_id, fields,
                        lambda f: (_ for _ in ()).throw(ValueError("boom")))
        assert any(row.get("correlation_id") == "LB-55" for row in lines(stream))


class TestSecretsNeverReachTheLog:
    """Structured logging invites passing rich context objects. A broker session is a dict
    containing a bearer token. This is what stops a convenience feature leaking one."""

    @pytest.mark.parametrize("key", [
        "bearer_token", "view_token", "edit_token", "KOTAK_MPIN", "consumer_key",
        "KOTAK_TOTP_SECRET", "password", "sid", "api_key", "authorization",
    ])
    def test_sensitive_keys_are_redacted(self, key):
        assert redact({key: "hunter2"})[key] == REDACTED

    def test_redaction_is_case_insensitive(self):
        assert redact({"Bearer_Token": "x"})["Bearer_Token"] == REDACTED

    def test_nested_secrets_are_redacted(self):
        out = redact({"session": {"inner": {"mpin": "1234"}}})
        assert out["session"]["inner"]["mpin"] == REDACTED

    def test_secrets_inside_lists_are_redacted(self):
        out = redact({"sessions": [{"bearer_token": "a"}, {"bearer_token": "b"}]})
        assert [s["bearer_token"] for s in out["sessions"]] == [REDACTED, REDACTED]

    def test_ordinary_values_survive(self):
        out = redact({"instrument_id": "nse_cm:2885", "quantity": 10})
        assert out == {"instrument_id": "nse_cm:2885", "quantity": 10}

    def test_a_session_passed_as_context_is_redacted_on_the_way_out(self):
        logger, stream = capture()
        logger.info("restored session",
                    extra={"session": {"sid": "abc123", "bearer_token": "eyJ..."}})
        text = stream.getvalue()
        assert "abc123" not in text
        assert "eyJ" not in text
        assert REDACTED in text

    def test_a_cyclic_structure_does_not_hang(self):
        payload = {"a": 1}
        payload["self"] = payload
        assert redact(payload)          # depth-bounded; must return


class TestLoggingCannotBreakTrading:
    def test_an_unserialisable_object_still_logs(self):
        class Awkward:
            def __repr__(self):
                raise RuntimeError("no repr for you")

            def __str__(self):
                raise RuntimeError("no str either")

        logger, stream = capture()
        logger.info("with context", extra={"thing": Awkward()})
        assert lines(stream), "the line was lost entirely"

    def test_a_formatter_failure_degrades_rather_than_raises(self):
        record = logging.LogRecord("x", logging.INFO, __file__, 1, "msg", None, None)
        record.getMessage = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
        output = JsonFormatter("test").format(record)
        assert json.loads(output)["msg"] == "log formatting failed"

    def test_exceptions_are_captured_not_swallowed(self):
        logger, stream = capture()
        try:
            raise ValueError("the actual problem")
        except ValueError:
            logger.exception("handler failed")
        assert "the actual problem" in lines(stream)[0]["exception"]


class TestOutputShape:
    def test_json_lines_are_one_object_each(self):
        logger, stream = capture()
        for i in range(3):
            logger.info("line %d", i)
        raw = [line for line in stream.getvalue().splitlines() if line.strip()]
        assert len(raw) == 3
        assert all(json.loads(line)["level"] == "INFO" for line in raw)

    def test_the_component_is_recorded(self):
        logger, stream = capture("execution")
        logger.info("x")
        assert lines(stream)[0]["component"] == "execution"

    def test_arguments_are_interpolated(self):
        logger, stream = capture()
        logger.info("filled %d @ %.2f", 10, 1300.5)
        assert lines(stream)[0]["msg"] == "filled 10 @ 1300.50"

    def test_extra_context_is_carried(self):
        logger, stream = capture()
        logger.info("order", extra={"instrument_id": "nse_cm:2885", "quantity": 10})
        context = lines(stream)[0]["context"]
        assert context["instrument_id"] == "nse_cm:2885"
        assert context["quantity"] == 10

    def test_datetimes_are_serialised(self):
        logger, stream = capture()
        logger.info("at", extra={"when": dt.datetime(2026, 3, 2, 10, 0)})
        assert "2026-03-02T10:00:00" in lines(stream)[0]["context"]["when"]

    def test_the_timestamp_is_iso(self):
        logger, stream = capture()
        logger.info("x")
        dt.datetime.fromisoformat(lines(stream)[0]["ts"])   # must parse


class TestTextRemainsTheDefault:
    def test_the_configured_default_is_text(self, monkeypatch):
        """A human watching a terminal during a live session needs to read it. JSON is
        opt-in for when something downstream parses these instead."""
        import importlib

        monkeypatch.delenv("LOG_FORMAT", raising=False)
        reloaded = importlib.reload(config)
        try:
            assert reloaded.LOG_FORMAT == "text"
        finally:
            importlib.reload(config)

    def test_text_output_is_human_readable(self):
        logger, stream = capture("execution", fmt="text")
        logger.info("Order accepted")
        assert "Order accepted" in stream.getvalue()
        assert "{" not in stream.getvalue()

    def test_text_output_still_shows_the_correlation_id(self):
        logger, stream = capture("execution", fmt="text")
        with correlation("LB-7"):
            logger.info("placing")
        assert "<LB-7>" in stream.getvalue()

    def test_an_invalid_format_is_rejected_at_import(self, monkeypatch):
        import importlib

        monkeypatch.setenv("LOG_FORMAT", "yaml")
        with pytest.raises(ValueError, match="LOG_FORMAT"):
            importlib.reload(config)
        monkeypatch.delenv("LOG_FORMAT")
        importlib.reload(config)


class TestConfigureIsWellBehaved:
    def test_it_does_not_duplicate_handlers(self):
        stream = io.StringIO()
        for _ in range(3):
            logger = configure("test", stream=stream)
        logger.info("once")
        assert len([l for l in stream.getvalue().splitlines() if l.strip()]) == 1

    def test_it_leaves_foreign_handlers_alone(self):
        """Under pytest this is `caplog`'s handler. Removing it would turn "the line was
        not emitted" and "the line went elsewhere" into the same observation."""
        root = logging.getLogger()
        foreign = logging.StreamHandler(io.StringIO())
        root.addHandler(foreign)
        configure("test", stream=io.StringIO())
        assert foreign in root.handlers

    def test_caplog_still_works_after_configuring(self, caplog):
        configure("test", stream=io.StringIO())
        with caplog.at_level("WARNING"):
            logging.getLogger("ligerbot.test").warning("visible to caplog")
        assert "visible to caplog" in caplog.text
