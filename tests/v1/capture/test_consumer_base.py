# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for ``CaptureConsumer`` and its internal ``_BatchedAdapter``.

Exercises the Phase A surface end-to-end: build a tiny consumer,
wire it through the adapter, verify per-key accumulation, finalize
behavior, exception isolation, and the hello-world path.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
import torch

from vllm.v1.capture import (
    CaptureChunk,
    CaptureConsumer,
    CaptureContext,
    CaptureFinalize,
    CaptureKey,
    CaptureSpec,
    VllmInternalRequestId,
)
from vllm.v1.capture.consumer import _BatchedAdapter

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


class _RecordingConsumer(CaptureConsumer):
    """Records every finalized capture into a list."""

    location = "worker"

    def __init__(self) -> None:
        # Intentionally do not call super().__init__ — we don't have a
        # real ``VllmConfig`` in these tests and the default __init__
        # is a no-op anyway.
        self.captures: list[tuple[CaptureKey, torch.Tensor, dict[str, Any]]] = []
        self.errors: list[tuple[CaptureKey, str]] = []
        self.shutdown_calls: list[float] = []

    def on_capture(
        self,
        key: CaptureKey,
        tensor: torch.Tensor,
        sidecar: dict[str, Any],
    ) -> None:
        self.captures.append((key, tensor.clone(), dict(sidecar)))

    def on_error(self, key: CaptureKey, error: str) -> None:
        self.errors.append((key, error))

    def shutdown(self, timeout: float = 30.0) -> None:
        self.shutdown_calls.append(timeout)


class _RaisingConsumer(_RecordingConsumer):
    def on_capture(
        self,
        key: CaptureKey,
        tensor: torch.Tensor,
        sidecar: dict[str, Any],
    ) -> None:
        raise RuntimeError("boom")


# ---------------------------------------------------------------------------
# _BatchedAdapter
# ---------------------------------------------------------------------------


def test_in_order_chunks_concat_on_finalize():
    consumer = _RecordingConsumer()
    adapter = _BatchedAdapter(consumer)

    key = _key()
    adapter.submit_chunk(
        CaptureChunk(
            key=key,
            tensor=_rows(0, 2),
            dtype=torch.float32,
            row_offset=0,
            step_index=0,
        )
    )
    adapter.submit_chunk(
        CaptureChunk(
            key=key,
            tensor=_rows(2, 2),
            dtype=torch.float32,
            row_offset=2,
            step_index=1,
        )
    )
    assert adapter.get_result(key) is None

    adapter.submit_finalize(CaptureFinalize(key=key, sidecar={"layer": 3}))

    assert len(consumer.captures) == 1
    recorded_key, tensor, sidecar = consumer.captures[0]
    assert recorded_key == key
    assert sidecar == {"layer": 3}
    expected = torch.cat([_rows(0, 2), _rows(2, 2)], dim=0)
    assert torch.equal(tensor, expected)

    result = adapter.get_result(key)
    assert result is not None
    assert result.status == "ok"
    assert result.error is None


def test_out_of_order_chunks_are_sorted_by_row_offset():
    consumer = _RecordingConsumer()
    adapter = _BatchedAdapter(consumer)

    key = _key()
    # Submit row_offset=2 FIRST, then row_offset=0.
    adapter.submit_chunk(
        CaptureChunk(
            key=key,
            tensor=_rows(2, 2),
            dtype=torch.float32,
            row_offset=2,
            step_index=1,
        )
    )
    adapter.submit_chunk(
        CaptureChunk(
            key=key,
            tensor=_rows(0, 2),
            dtype=torch.float32,
            row_offset=0,
            step_index=0,
        )
    )
    adapter.submit_finalize(CaptureFinalize(key=key))

    _, tensor, _ = consumer.captures[0]
    expected = torch.cat([_rows(0, 2), _rows(2, 2)], dim=0)
    assert torch.equal(tensor, expected)


def test_multiple_keys_finalize_independently():
    consumer = _RecordingConsumer()
    adapter = _BatchedAdapter(consumer)

    key_a = _key("req-a", layer=1, hook="pre_attn")
    key_b = _key("req-b", layer=5, hook="post_mlp")

    adapter.submit_chunk(
        CaptureChunk(
            key=key_a,
            tensor=_rows(0, 2),
            dtype=torch.float32,
            row_offset=0,
            step_index=0,
        )
    )
    adapter.submit_chunk(
        CaptureChunk(
            key=key_b,
            tensor=_rows(10, 3),
            dtype=torch.float32,
            row_offset=0,
            step_index=0,
        )
    )

    adapter.submit_finalize(CaptureFinalize(key=key_a))

    assert len(consumer.captures) == 1
    finalized_key, _, _ = consumer.captures[0]
    assert finalized_key == key_a
    assert adapter.get_result(key_a) is not None
    assert adapter.get_result(key_b) is None

    adapter.submit_finalize(CaptureFinalize(key=key_b))
    assert len(consumer.captures) == 2
    assert adapter.get_result(key_b) is not None


def test_on_capture_exception_marks_error_and_isolates_consumer():
    consumer = _RaisingConsumer()
    adapter = _BatchedAdapter(consumer)

    key = _key()
    adapter.submit_chunk(
        CaptureChunk(
            key=key,
            tensor=_rows(0, 1),
            dtype=torch.float32,
            row_offset=0,
            step_index=0,
        )
    )
    # Must not raise — consumer isolation (invariant 9).
    adapter.submit_finalize(CaptureFinalize(key=key))

    result = adapter.get_result(key)
    assert result is not None
    assert result.status == "error"
    assert result.error is not None
    assert "boom" in result.error
    assert consumer.errors == [(key, result.error)]


def test_get_result_is_none_until_finalize():
    consumer = _RecordingConsumer()
    adapter = _BatchedAdapter(consumer)

    key = _key()
    adapter.submit_chunk(
        CaptureChunk(
            key=key,
            tensor=_rows(0, 1),
            dtype=torch.float32,
            row_offset=0,
            step_index=0,
        )
    )
    assert adapter.get_result(key) is None


def test_shutdown_forwards_to_consumer():
    consumer = _RecordingConsumer()
    adapter = _BatchedAdapter(consumer)
    adapter.shutdown(timeout=5.0)
    assert consumer.shutdown_calls == [5.0]


def test_finalize_without_chunks_still_runs_on_capture_with_empty_tensor():
    consumer = _RecordingConsumer()
    adapter = _BatchedAdapter(consumer)

    key = _key()
    adapter.submit_finalize(CaptureFinalize(key=key, sidecar={"note": "empty"}))

    assert len(consumer.captures) == 1
    _, tensor, sidecar = consumer.captures[0]
    assert tensor.numel() == 0
    assert sidecar == {"note": "empty"}
    result = adapter.get_result(key)
    assert result is not None
    assert result.status == "ok"


def test_adapter_location_mirrors_consumer():
    consumer = _RecordingConsumer()
    adapter = _BatchedAdapter(consumer)
    assert adapter.location == "worker"


# ---------------------------------------------------------------------------
# CaptureConsumer defaults
# ---------------------------------------------------------------------------


def test_default_global_capture_spec_is_none():
    consumer = _RecordingConsumer()
    assert consumer.global_capture_spec() is None


def test_default_validate_client_spec_raises_not_implemented():
    consumer = _RecordingConsumer()
    ctx = CaptureContext(
        vllm_internal_request_id=VllmInternalRequestId("req-1"),
        num_prompt_tokens=8,
        num_computed_tokens=0,
        num_hidden_layers=4,
        hidden_size=16,
        element_size_bytes=2,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
    )
    with pytest.raises(NotImplementedError):
        consumer.validate_client_spec({"positions": "all"}, ctx)


def test_default_class_metadata():
    assert CaptureConsumer.location == "worker"
    assert CaptureConsumer.required_sidecar_fields == frozenset()
    assert CaptureConsumer.reads_client_spec is False


def test_capture_consumer_cannot_be_instantiated_without_on_capture():
    class _Missing(CaptureConsumer):
        pass

    with pytest.raises(TypeError):
        _Missing(MagicMock(), {})  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# Hello-world consumer
#
# The roadmap's "Done when" for Phase A requires that the types +
# protocol + registry are sufficient to implement a hello-world consumer
# in a test file. This test proves it: a ``GlobalSumConsumer`` uses only
# the public Phase A surface (CaptureConsumer, CaptureSpec,
# CaptureChunk, CaptureFinalize).
# ---------------------------------------------------------------------------


class GlobalSumConsumer(CaptureConsumer):
    """Toy consumer: records the sum of every finalized tensor."""

    location = "worker"

    def __init__(self) -> None:
        self.sums: dict[CaptureKey, float] = {}

    def global_capture_spec(self) -> CaptureSpec | None:
        return CaptureSpec(hooks={"post_mlp": [0]}, positions="last_prompt")

    def on_capture(
        self,
        key: CaptureKey,
        tensor: torch.Tensor,
        sidecar: dict[str, Any],
    ) -> None:
        self.sums[key] = float(tensor.sum().item())


def test_hello_world_consumer_through_batched_adapter():
    consumer = GlobalSumConsumer()
    adapter = _BatchedAdapter(consumer)

    spec = consumer.global_capture_spec()
    assert spec is not None
    assert spec.hooks == {"post_mlp": [0]}
    assert spec.positions == "last_prompt"

    key = _key()
    adapter.submit_chunk(
        CaptureChunk(
            key=key,
            tensor=_rows(0, 3),  # column 0 has values 0, 1, 2 → sum 3
            dtype=torch.float32,
            row_offset=0,
            step_index=0,
        )
    )
    adapter.submit_finalize(CaptureFinalize(key=key))

    assert consumer.sums == {key: 3.0}
    assert adapter.get_result(key) is not None
    assert adapter.get_result(key).status == "ok"  # type: ignore[union-attr]
