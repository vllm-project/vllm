# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the core types in ``vllm.v1.capture.types``."""

from __future__ import annotations

import dataclasses

import pytest
import torch

from vllm.v1.capture import (
    CaptureChunk,
    CaptureContext,
    CaptureFinalize,
    CaptureKey,
    CaptureResult,
    CaptureSpec,
    VllmInternalRequestId,
)


def _key(req_id: str = "req-1", layer: int = 3, hook: str = "post_mlp") -> CaptureKey:
    return (VllmInternalRequestId(req_id), layer, hook)


def test_capture_key_is_a_three_tuple():
    key = _key()
    assert isinstance(key, tuple)
    assert len(key) == 3
    req_id, layer, hook = key
    assert req_id == "req-1"
    assert layer == 3
    assert hook == "post_mlp"


def test_capture_spec_is_frozen():
    spec = CaptureSpec(
        hooks={"post_mlp": [1, 2, 3]},
        positions="last_prompt",
    )
    assert spec.hooks == {"post_mlp": [1, 2, 3]}
    assert spec.positions == "last_prompt"

    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.positions = "all"  # type: ignore[misc]


def test_capture_spec_accepts_explicit_position_list():
    spec = CaptureSpec(
        hooks={"pre_attn": [0]},
        positions=[0, 5, 10],
    )
    assert spec.positions == [0, 5, 10]


def test_capture_chunk_round_trip():
    tensor = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    chunk = CaptureChunk(
        key=_key(),
        tensor=tensor,
        dtype=torch.float32,
        row_offset=12,
        step_index=3,
        metadata={"note": "test"},
    )
    assert chunk.key == _key()
    assert chunk.row_offset == 12
    assert chunk.step_index == 3
    assert chunk.dtype == torch.float32
    assert chunk.metadata == {"note": "test"}
    assert torch.equal(chunk.tensor, tensor)


def test_capture_chunk_default_metadata_is_empty_dict():
    chunk = CaptureChunk(
        key=_key(),
        tensor=torch.zeros((1, 2)),
        dtype=torch.float32,
        row_offset=0,
        step_index=0,
    )
    assert chunk.metadata == {}
    # Distinct instances must get distinct dicts.
    other = CaptureChunk(
        key=_key("req-2"),
        tensor=torch.zeros((1, 2)),
        dtype=torch.float32,
        row_offset=0,
        step_index=0,
    )
    chunk.metadata["k"] = "v"
    assert "k" not in other.metadata


def test_capture_finalize_round_trip():
    finalize = CaptureFinalize(key=_key(), sidecar={"layer": 3})
    assert finalize.key == _key()
    assert finalize.sidecar == {"layer": 3}


def test_capture_finalize_default_sidecar_is_empty_dict():
    finalize = CaptureFinalize(key=_key())
    assert finalize.sidecar == {}


def test_capture_result_defaults():
    result = CaptureResult(key=_key(), status="ok")
    assert result.status == "ok"
    assert result.error is None
    assert result.payload is None


def test_capture_result_with_error_and_payload():
    result = CaptureResult(
        key=_key(),
        status="error",
        error="boom",
        payload={"written": 0},
    )
    assert result.status == "error"
    assert result.error == "boom"
    assert result.payload == {"written": 0}


def test_capture_context_round_trip():
    ctx = CaptureContext(
        vllm_internal_request_id=VllmInternalRequestId("req-1"),
        num_prompt_tokens=128,
        num_computed_tokens=0,
        num_hidden_layers=32,
        hidden_size=4096,
        element_size_bytes=2,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
    )
    assert ctx.num_prompt_tokens == 128
    assert ctx.num_hidden_layers == 32
    assert ctx.hidden_size == 4096
    assert ctx.element_size_bytes == 2
    assert ctx.tensor_parallel_size == 1
    assert ctx.pipeline_parallel_size == 1
