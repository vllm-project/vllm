# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``@enable_tracing`` decorator and trace gate."""

# Standard
from unittest.mock import patch

# Third Party
import pytest

# First Party
from lmcache.v1.mp_observability.event import EventType
from lmcache.v1.mp_observability.trace.decorator import (
    enable_tracing,
    is_tracing_enabled,
    publish_call_event,
    set_tracing_enabled,
)


@pytest.fixture(autouse=True)
def reset_gate():
    """Ensure the trace gate is off before and after every test."""
    set_tracing_enabled(False)
    yield
    set_tracing_enabled(False)


class TestGate:
    def test_default_off(self):
        assert is_tracing_enabled() is False

    def test_set_and_clear(self):
        set_tracing_enabled(True)
        assert is_tracing_enabled() is True
        set_tracing_enabled(False)
        assert is_tracing_enabled() is False


class TestDisabledOverhead:
    """When tracing is off, the decorator must not touch the EventBus."""

    def test_no_publish_when_disabled(self):
        @enable_tracing()
        def f(x, y):
            return x + y

        with patch(
            "lmcache.v1.mp_observability.trace.decorator.get_event_bus"
        ) as fake_bus:
            assert f(1, 2) == 3
            fake_bus.assert_not_called()

    def test_no_signature_bind_when_disabled(self):
        """Signature is bound at decoration time only when ENABLED.

        We assert this indirectly: a function whose signature would
        raise on bind (e.g. wrong arity) still completes its real call
        successfully when the gate is off.
        """
        calls = []

        @enable_tracing()
        def f(*args, **kwargs):
            calls.append((args, kwargs))

        f(1, 2, k=3)
        assert calls == [((1, 2), {"k": 3})]


class TestEntryPublish:
    def test_publishes_when_enabled(self):
        seen = []

        @enable_tracing(qualname="pkg.mod.Cls.method")
        def method(self, a, b=2):
            return a + b

        set_tracing_enabled(True)
        with patch(
            "lmcache.v1.mp_observability.trace.decorator.get_event_bus"
        ) as fake_bus:
            fake_bus.return_value.publish = lambda ev: seen.append(ev)
            assert method("self_obj", 10) == 12

        assert len(seen) == 1
        ev = seen[0]
        assert ev.event_type == EventType.TRACE_CALL
        assert ev.metadata["qualname"] == "pkg.mod.Cls.method"
        # ``self`` is dropped; defaults are filled in.
        assert ev.metadata["args"] == {"a": 10, "b": 2}

    def test_default_qualname(self):
        @enable_tracing()
        def some_func(x):
            return x

        # Nested defs include ``<locals>`` in __qualname__ so we
        # compare against the actual current name to avoid coupling
        # the test to the test method's path.
        assert some_func.__lmc_trace_qualname__ == (
            f"{some_func.__module__}.{some_func.__qualname__}"
        )

    def test_capture_filter(self):
        seen = []

        @enable_tracing(capture=["a"])
        def f(a, b, c):
            return a + b + c

        set_tracing_enabled(True)
        with patch(
            "lmcache.v1.mp_observability.trace.decorator.get_event_bus"
        ) as fake_bus:
            fake_bus.return_value.publish = lambda ev: seen.append(ev)
            f(1, 2, 3)
        assert seen[0].metadata["args"] == {"a": 1}

    def test_redact_filter(self):
        seen = []

        @enable_tracing(redact=["password"])
        def login(user, password):
            return True

        set_tracing_enabled(True)
        with patch(
            "lmcache.v1.mp_observability.trace.decorator.get_event_bus"
        ) as fake_bus:
            fake_bus.return_value.publish = lambda ev: seen.append(ev)
            login("alice", "hunter2")
        assert seen[0].metadata["args"] == {"user": "alice"}

    def test_entry_only_published_on_exception(self):
        """Even when the wrapped function raises, the entry event has
        already been published.  No exit event is emitted."""
        seen = []

        @enable_tracing()
        def boom(x):
            raise RuntimeError("nope")

        set_tracing_enabled(True)
        with patch(
            "lmcache.v1.mp_observability.trace.decorator.get_event_bus"
        ) as fake_bus:
            fake_bus.return_value.publish = lambda ev: seen.append(ev)
            with pytest.raises(RuntimeError):
                boom(7)
        assert len(seen) == 1
        assert seen[0].metadata["args"] == {"x": 7}


class TestPublishCallEvent:
    def test_no_publish_when_disabled(self):
        with patch(
            "lmcache.v1.mp_observability.trace.decorator.get_event_bus"
        ) as fake_bus:
            publish_call_event("a.b", {"x": 1})
            fake_bus.assert_not_called()

    def test_publishes_when_enabled(self):
        seen = []
        set_tracing_enabled(True)
        with patch(
            "lmcache.v1.mp_observability.trace.decorator.get_event_bus"
        ) as fake_bus:
            fake_bus.return_value.publish = lambda ev: seen.append(ev)
            publish_call_event("a.b", {"x": 1})
        md = seen[0].metadata
        assert md["qualname"] == "a.b"
        assert md["args"] == {"x": 1}
        # ``t_mono`` is stamped at publish time so the recorder's record
        # timestamp is co-temporal with ``Event.timestamp``.
        assert isinstance(md["t_mono"], float)
        assert md["t_mono"] > 0
