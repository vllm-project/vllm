# SPDX-License-Identifier: Apache-2.0

"""Tests for :class:`CallDispatcher` and the default v1 registrations."""

# Future
from __future__ import annotations

# Standard
from typing import Any

# Third Party
import pytest

# First Party
from lmcache.cli.commands.trace._dispatch import (
    CallDispatcher,
    ReplayContext,
    build_default_dispatcher,
)
from lmcache.v1.distributed.api import ObjectKey

_SM_PREFIX = "lmcache.v1.distributed.storage_manager.StorageManager"


class _FakeSM:
    """Minimal StorageManager stand-in for dispatcher tests.

    Records each call into ``self.calls`` so assertions can match
    forwarded arguments exactly.  For the context-manager entry, it
    returns a tiny object whose ``__enter__``/``__exit__`` push into
    ``self.events`` so the FIFO ordering can be verified.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.events: list[str] = []

    def reserve_write(self, **kw: Any) -> dict[Any, Any]:
        self.calls.append(("reserve_write", kw))
        return {}

    def finish_write(self, **kw: Any) -> None:
        self.calls.append(("finish_write", kw))

    def submit_prefetch_task(self, **kw: Any) -> None:
        self.calls.append(("submit_prefetch_task", kw))

    def finish_read_prefetched(self, **kw: Any) -> None:
        self.calls.append(("finish_read_prefetched", kw))

    def read_prefetched_results(self, keys: list[ObjectKey]) -> "_FakeCM":
        return _FakeCM(self, keys)


class _FakeCM:
    def __init__(self, parent: _FakeSM, keys: list[ObjectKey]) -> None:
        self._parent = parent
        self._keys = keys

    def __enter__(self) -> None:
        self._parent.events.append(f"enter-{len(self._keys)}")

    def __exit__(self, *_exc: object) -> None:
        self._parent.events.append(f"exit-{len(self._keys)}")


def _key(i: int) -> ObjectKey:
    return ObjectKey(chunk_hash=bytes([i]), model_name="test", kv_rank=0)


class TestCallDispatcher:
    def test_register_and_has(self):
        d = CallDispatcher()
        assert not d.has("x")
        d.register("x", lambda c, a: None)
        assert d.has("x")

    def test_duplicate_registration_raises(self):
        d = CallDispatcher()
        d.register("x", lambda c, a: None)
        with pytest.raises(ValueError):
            d.register("x", lambda c, a: None)

    def test_dispatch_unknown_qualname_raises_keyerror(self):
        d = CallDispatcher()
        with pytest.raises(KeyError):
            d.dispatch("no.such.qualname", ReplayContext(sm=_FakeSM()), {})

    def test_registered_qualnames_returns_new_list(self):
        d = CallDispatcher()
        d.register("a", lambda c, a: None)
        qns = d.registered_qualnames()
        qns.append("b")
        assert "b" not in d.registered_qualnames()


class TestDefaultDispatcher:
    def test_registers_all_v1_qualnames(self):
        d = build_default_dispatcher()
        expected = {
            f"{_SM_PREFIX}.reserve_write",
            f"{_SM_PREFIX}.finish_write",
            f"{_SM_PREFIX}.submit_prefetch_task",
            f"{_SM_PREFIX}.finish_read_prefetched",
            f"{_SM_PREFIX}.read_prefetched_results.__enter__",
            f"{_SM_PREFIX}.read_prefetched_results.__exit__",
        }
        assert set(d.registered_qualnames()) == expected

    def test_simple_method_forwarded_with_kwargs(self):
        sm = _FakeSM()
        ctx = ReplayContext(sm=sm)
        d = build_default_dispatcher()
        d.dispatch(
            f"{_SM_PREFIX}.reserve_write",
            ctx,
            {"keys": [_key(1)], "layout_desc": "LAYOUT", "mode": "new"},
        )
        assert sm.calls == [
            (
                "reserve_write",
                {"keys": [_key(1)], "layout_desc": "LAYOUT", "mode": "new"},
            ),
        ]

    def test_read_prefetched_enter_exit_fifo(self):
        """Two overlapping contexts with identical keys exit in FIFO order."""
        sm = _FakeSM()
        ctx = ReplayContext(sm=sm)
        d = build_default_dispatcher()

        keys = [_key(1), _key(2)]
        enter = f"{_SM_PREFIX}.read_prefetched_results.__enter__"
        exit_ = f"{_SM_PREFIX}.read_prefetched_results.__exit__"

        d.dispatch(enter, ctx, {"keys": keys})
        d.dispatch(enter, ctx, {"keys": keys})
        assert sm.events == ["enter-2", "enter-2"]
        # Both contexts share the same key tuple.
        assert tuple(keys) in ctx.open_read_contexts
        assert len(ctx.open_read_contexts[tuple(keys)]) == 2

        d.dispatch(exit_, ctx, {"keys": keys})
        assert sm.events == ["enter-2", "enter-2", "exit-2"]
        d.dispatch(exit_, ctx, {"keys": keys})
        assert sm.events == ["enter-2", "enter-2", "exit-2", "exit-2"]
        # Fully drained → dict cleaned up.
        assert ctx.open_read_contexts == {}

    def test_exit_without_matching_enter_is_warning_only(self, caplog):
        sm = _FakeSM()
        ctx = ReplayContext(sm=sm)
        d = build_default_dispatcher()

        with caplog.at_level("WARNING"):
            d.dispatch(
                f"{_SM_PREFIX}.read_prefetched_results.__exit__",
                ctx,
                {"keys": [_key(1)]},
            )
        # No crash, no effect on sm.
        assert sm.events == []
