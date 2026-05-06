# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the driver bridge (worker->driver plumbing).

These are single-process tests that use ``queue.Queue`` (thread-safe,
same interface as ``torch.multiprocessing.Queue`` for get/put/get_nowait)
to avoid sandbox restrictions on Unix socket creation that
``multiprocessing.Queue`` requires for tensor shared-memory handoff.

They exercise the round-trip through
``_DriverQueueShim`` -> ``_DriverReceiver`` -> consumer callback ->
result queue -> shim's ``get_result``.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Any, ClassVar, Literal

import torch

from vllm.v1.capture.consumer import CaptureConsumer
from vllm.v1.capture.driver_bridge import (
    _DriverQueueShim,
    _DriverReceiver,
    install_driver_consumer,
)
from vllm.v1.capture.types import (
    CaptureChunk,
    CaptureFinalize,
    CaptureKey,
    CaptureResult,
    VllmInternalRequestId,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _key(req_id: str = "req-1", layer: int = 3, hook: str = "post_mlp") -> CaptureKey:
    return (VllmInternalRequestId(req_id), layer, hook)


def _rows(start: int, count: int) -> torch.Tensor:
    """A ``(count, 2)`` float32 tensor whose first column is ``range``."""
    return torch.stack(
        [
            torch.arange(start, start + count, dtype=torch.float32),
            torch.zeros(count, dtype=torch.float32),
        ],
        dim=1,
    )


class _RecordingDriverConsumer(CaptureConsumer):
    """Driver-side consumer that records every on_capture call."""

    location: ClassVar[Literal["worker", "driver"]] = "driver"

    def __init__(self) -> None:
        self.captures: list[tuple[CaptureKey, torch.Tensor, dict[str, Any]]] = []
        self.errors: list[tuple[CaptureKey, str]] = []

    def on_capture(
        self,
        key: CaptureKey,
        tensor: torch.Tensor,
        sidecar: dict[str, Any],
    ) -> None:
        self.captures.append((key, tensor.clone(), dict(sidecar)))

    def on_error(self, key: CaptureKey, error: str) -> None:
        self.errors.append((key, error))


class _RaisingDriverConsumer(CaptureConsumer):
    """Driver-side consumer whose on_capture always raises."""

    location: ClassVar[Literal["worker", "driver"]] = "driver"

    def __init__(self) -> None:
        self.captures: list[tuple[CaptureKey, torch.Tensor, dict[str, Any]]] = []
        self.errors: list[tuple[CaptureKey, str]] = []

    def on_capture(
        self,
        key: CaptureKey,
        tensor: torch.Tensor,
        sidecar: dict[str, Any],
    ) -> None:
        raise RuntimeError("consumer exploded")

    def on_error(self, key: CaptureKey, error: str) -> None:
        self.errors.append((key, error))


def _make_queue_pair(
    event_size: int = 64, result_size: int = 64
) -> tuple[queue.Queue, queue.Queue]:
    """Create a pair of thread-safe queues (same interface as mp.Queue)."""
    return queue.Queue(maxsize=event_size), queue.Queue(maxsize=result_size)


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


def test_round_trip_chunk_and_finalize():
    """Submit chunk + finalize through shim; verify consumer fires and
    result is available."""
    consumer = _RecordingDriverConsumer()
    event_q, result_q = _make_queue_pair()

    receiver = _DriverReceiver(consumer, event_q, result_q)
    receiver.start()

    shim = _DriverQueueShim(event_q, result_q, timeout=5.0)

    key = _key()
    chunk = CaptureChunk(
        key=key,
        tensor=_rows(0, 3),
        dtype=torch.float32,
        row_offset=0,
        step_index=0,
    )
    shim.submit_chunk(chunk)
    shim.submit_finalize(CaptureFinalize(key=key, sidecar={"tag": "test"}))

    result = shim.wait_for_result(key, timeout=5.0)
    assert result is not None, "Timed out waiting for result"
    assert result.status == "ok"
    assert result.error is None

    # Verify consumer received the data.
    assert len(consumer.captures) == 1
    captured_key, tensor, sidecar = consumer.captures[0]
    assert captured_key == key
    assert sidecar == {"tag": "test"}
    assert tensor.shape == (3, 2)

    shim.shutdown()
    receiver.join(timeout=5.0)


def test_multiple_keys_finalize_independently():
    """Two keys submitted interleaved, each finalizes independently."""
    consumer = _RecordingDriverConsumer()
    event_q, result_q = _make_queue_pair()

    receiver = _DriverReceiver(consumer, event_q, result_q)
    receiver.start()

    shim = _DriverQueueShim(event_q, result_q, timeout=5.0)

    key_a = _key("req-a", layer=1, hook="pre_attn")
    key_b = _key("req-b", layer=5, hook="post_mlp")

    shim.submit_chunk(
        CaptureChunk(
            key=key_a,
            tensor=_rows(0, 2),
            dtype=torch.float32,
            row_offset=0,
            step_index=0,
        )
    )
    shim.submit_chunk(
        CaptureChunk(
            key=key_b,
            tensor=_rows(10, 3),
            dtype=torch.float32,
            row_offset=0,
            step_index=0,
        )
    )

    # Finalize only key_a first.
    shim.submit_finalize(CaptureFinalize(key=key_a))

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if shim.get_result(key_a) is not None:
            break
        time.sleep(0.05)

    assert shim.get_result(key_a) is not None
    assert shim.get_result(key_a).status == "ok"  # type: ignore[union-attr]
    assert shim.get_result(key_b) is None  # Not yet finalized.

    # Now finalize key_b.
    shim.submit_finalize(CaptureFinalize(key=key_b))

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if shim.get_result(key_b) is not None:
            break
        time.sleep(0.05)

    assert shim.get_result(key_b) is not None
    assert shim.get_result(key_b).status == "ok"  # type: ignore[union-attr]

    shim.shutdown()
    receiver.join(timeout=5.0)


# ---------------------------------------------------------------------------
# Back-pressure
# ---------------------------------------------------------------------------


def test_back_pressure_marks_partial_error():
    """With a tiny queue (maxsize=1) and a very short timeout, rapid
    submissions eventually fail with partial_error.

    Uses ``queue.Queue`` (not mp.Queue) so ``maxsize`` is enforced
    synchronously on ``put`` — no background feeder thread.
    """
    event_q: queue.Queue = queue.Queue(maxsize=1)
    result_q: queue.Queue = queue.Queue(maxsize=64)

    # Do NOT start a receiver — the event queue will stay full.
    shim = _DriverQueueShim(event_q, result_q, timeout=0.1)

    # First put succeeds (fills the single slot).
    key1 = _key("req-fill")
    shim.submit_chunk(
        CaptureChunk(
            key=key1,
            tensor=_rows(0, 1),
            dtype=torch.float32,
            row_offset=0,
            step_index=0,
        )
    )

    # Second put should time out -> partial_error.
    key2 = _key("req-overflow")
    shim.submit_chunk(
        CaptureChunk(
            key=key2,
            tensor=_rows(0, 1),
            dtype=torch.float32,
            row_offset=0,
            step_index=0,
        )
    )

    result = shim.get_result(key2)
    assert result is not None
    assert result.status == "partial_error"
    assert result.error is not None
    assert "queue full" in result.error


# ---------------------------------------------------------------------------
# Shutdown / sentinel
# ---------------------------------------------------------------------------


def test_shutdown_sends_sentinel_and_receiver_exits():
    """Sending shutdown sentinel causes the receiver thread to exit."""
    consumer = _RecordingDriverConsumer()
    event_q, result_q = _make_queue_pair()

    receiver = _DriverReceiver(consumer, event_q, result_q)
    receiver.start()
    assert receiver.is_alive

    shim = _DriverQueueShim(event_q, result_q, timeout=5.0)
    shim.shutdown(timeout=5.0)

    receiver.join(timeout=5.0)
    assert not receiver.is_alive


# ---------------------------------------------------------------------------
# Exception in on_capture
# ---------------------------------------------------------------------------


def test_exception_in_on_capture_produces_error_result():
    """When consumer.on_capture raises, the result has status='error'
    and other keys continue to work."""
    consumer = _RaisingDriverConsumer()
    event_q, result_q = _make_queue_pair()

    receiver = _DriverReceiver(consumer, event_q, result_q)
    receiver.start()

    shim = _DriverQueueShim(event_q, result_q, timeout=5.0)

    bad_key = _key("req-bad")
    shim.submit_chunk(
        CaptureChunk(
            key=bad_key,
            tensor=_rows(0, 1),
            dtype=torch.float32,
            row_offset=0,
            step_index=0,
        )
    )
    shim.submit_finalize(CaptureFinalize(key=bad_key))

    result = shim.wait_for_result(bad_key, timeout=5.0)
    assert result is not None, "Timed out waiting for error result"
    assert result.status == "error"
    assert result.error is not None
    assert "consumer exploded" in result.error

    # The on_error callback on the consumer should have fired too.
    assert len(consumer.errors) == 1
    assert "consumer exploded" in consumer.errors[0][1]

    shim.shutdown()
    receiver.join(timeout=5.0)


def test_wait_for_result_blocks_until_delayed_result_arrives():
    event_q, result_q = _make_queue_pair()
    shim = _DriverQueueShim(event_q, result_q, timeout=5.0)
    key = _key("req-delayed")

    def _post_result() -> None:
        time.sleep(0.2)
        result_q.put(CaptureResult(key=key, status="error", error="delayed failure"))

    thread = threading.Thread(target=_post_result)
    thread.start()

    start = time.monotonic()
    result = shim.wait_for_result(key, timeout=1.0)
    elapsed = time.monotonic() - start
    thread.join(timeout=1.0)

    assert result is not None
    assert result.status == "error"
    assert result.error == "delayed failure"
    assert elapsed >= 0.15


def test_wait_for_result_times_out_when_no_result_arrives():
    event_q, result_q = _make_queue_pair()
    shim = _DriverQueueShim(event_q, result_q, timeout=0.1)
    key = _key("req-timeout")

    start = time.monotonic()
    result = shim.wait_for_result(key, timeout=0.2)
    elapsed = time.monotonic() - start

    assert result is None
    assert elapsed >= 0.15


# ---------------------------------------------------------------------------
# install_driver_consumer (uses mp.Queue internally)
# ---------------------------------------------------------------------------


def test_install_driver_consumer_returns_shim_with_correct_location():
    """``install_driver_consumer`` creates a ``_DriverQueueShim`` with
    ``location = 'worker'``.

    Note: ``install_driver_consumer`` uses ``torch.multiprocessing.Queue``
    internally.  In sandboxed CI this may fail with a socket permission
    error.  We test the public API contract here (type + location); the
    full round-trip is exercised above using ``queue.Queue``.
    """
    consumer = _RecordingDriverConsumer()
    try:
        shim = install_driver_consumer(consumer, queue_size=32, timeout=5.0)
    except (PermissionError, OSError):
        # Sandbox blocks Unix socket creation needed by mp.Queue.
        import pytest

        pytest.skip("mp.Queue not available in sandboxed environment")

    assert isinstance(shim, _DriverQueueShim)
    assert shim.location == "worker"

    shim.shutdown()
