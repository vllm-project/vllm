# SPDX-License-Identifier: Apache-2.0
"""Public-API unit tests for ``LMCacheMPConnector.store_kv_async``. The MQ and
GPU boundaries are stubbed; no live daemon or CUDA device needed. Covers the
async store contract added for SGLang MP mode: ``store_kv_async`` always
returns a pollable future -- an already-completed one on the no-op paths
(unhealthy connector / no chunk-aligned range), and the daemon's real
completion future on the happy path -- and never blocks on the result."""

# Standard
import threading

# Third Party
import pytest
import torch

# The adapter imports ``sglang`` at module load; skip cleanly where it's absent
# (sglang is an optional integration, not a hard LMCache dependency).
pytest.importorskip("sglang")

# First Party
from lmcache.integration.sglang import multi_process_adapter as adapter_mod
from lmcache.integration.sglang.multi_process_adapter import (
    LMCacheMPConnector,
    _completed_future,
)
from lmcache.integration.sglang.sglang_adapter import StoreMetadata
from lmcache.v1.multiprocess.futures import MessagingFuture

_CHUNK_SIZE = 256


def _make_connector(healthy: bool = True) -> LMCacheMPConnector:
    """Build a connector without running ``__init__`` (which opens ZMQ).

    Sets only the attributes the store paths touch; anything a given
    test needs beyond this it stubs itself.
    """
    conn = object.__new__(LMCacheMPConnector)
    conn._health_event = threading.Event()
    if healthy:
        conn._health_event.set()
    conn._lmcache_chunk_size = _CHUNK_SIZE
    conn._mq_timeout = 5.0
    return conn


def _store_metadata(num_tokens: int) -> StoreMetadata:
    return StoreMetadata(
        last_node=None,
        token_ids=list(range(num_tokens)),
        kv_indices=torch.empty(0, dtype=torch.int64),
        offset=0,
        request_id="req-test",
    )


class _SpyFuture(MessagingFuture):
    """Future that records whether the caller blocked on ``result``."""

    def __init__(self) -> None:
        super().__init__()
        self.result_called = False

    def result(self, timeout=None):
        self.result_called = True
        return super().result(timeout)


class _FakeRaw:
    """Stand-in for ``send_lmcache_request``'s return; hands back a preset
    future from ``to_cuda_future`` so no CUDA event is needed."""

    def __init__(self, future: MessagingFuture) -> None:
        self._future = future

    def to_cuda_future(self, device=None) -> MessagingFuture:
        return self._future


class _FakeEvent:
    def __init__(self, interprocess: bool = False) -> None:
        pass

    def record(self, stream) -> None:
        pass

    def ipc_handle(self) -> bytes:
        return b"fake-ipc-handle"


class _FakeTorchDev:
    Event = _FakeEvent

    @staticmethod
    def current_stream():
        return object()


def test_completed_future_resolves_to_given_result() -> None:
    done_true = _completed_future(True)
    assert done_true.query() is True
    assert done_true.result(timeout=0) is True

    done_false = _completed_future(False)
    assert done_false.query() is True
    assert done_false.result(timeout=0) is False


def test_store_kv_async_unhealthy_returns_failed_future_no_send(monkeypatch) -> None:
    conn = _make_connector(healthy=False)

    def _fail_send(*args, **kwargs):
        pytest.fail("send_lmcache_request must not be called when unhealthy")

    monkeypatch.setattr(adapter_mod, "send_lmcache_request", _fail_send)

    future = conn.store_kv_async(_store_metadata(num_tokens=4 * _CHUNK_SIZE))

    assert isinstance(future, MessagingFuture)
    assert future.query() is True
    # Unhealthy connector stored nothing -> the future must report failure.
    assert future.result(timeout=0) is False


def test_store_kv_async_no_aligned_range_returns_completed_future_no_send(
    monkeypatch,
) -> None:
    conn = _make_connector(healthy=True)

    def _fail_send(*args, **kwargs):
        pytest.fail("send_lmcache_request must not be called with no aligned range")

    monkeypatch.setattr(adapter_mod, "send_lmcache_request", _fail_send)

    # Fewer tokens than one chunk -> aligned_end == 0 -> no wire send.
    future = conn.store_kv_async(_store_metadata(num_tokens=_CHUNK_SIZE - 1))

    assert isinstance(future, MessagingFuture)
    assert future.result(timeout=0) is True


def test_store_kv_async_happy_path_returns_daemon_future_without_blocking(
    monkeypatch,
) -> None:
    conn = _make_connector(healthy=True)
    conn.mq_client = object()  # type: ignore[assignment]
    conn.instance_id = 123
    conn.device = "cpu"
    # Stub the helpers store_kv_async calls so we exercise only its own logic.
    conn._slot_mapping_to_block_ids = lambda kv_indices: [0, 1]  # type: ignore[method-assign,assignment]
    conn._create_key = lambda *args, **kwargs: "fake-key"  # type: ignore[method-assign,assignment]

    sentinel = _SpyFuture()
    monkeypatch.setattr(adapter_mod, "torch_dev", _FakeTorchDev)
    monkeypatch.setattr(
        adapter_mod,
        "send_lmcache_request",
        lambda mq_client, request_type, payload: _FakeRaw(sentinel),
    )

    future = conn.store_kv_async(_store_metadata(num_tokens=4 * _CHUNK_SIZE))

    # It returns the daemon's own future, and must NOT have blocked on it.
    assert future is sentinel
    assert sentinel.result_called is False
    # The exporting CUDA event must be pinned to the future so it isn't
    # garbage-collected before the daemon waits on its IPC handle.
    assert hasattr(future, "_export_event")
