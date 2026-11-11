# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PCP-aware M-RoPE position preparation."""

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.worker.gpu.mm.rope import RopeState
from vllm.v1.worker.gpu.model_states.default import DefaultModelState


NUM_DIMS = 3
MAX_NUM_REQS = 2
MAX_MODEL_LEN = 8
DEVICE = torch.device("npu")

pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="requires NPU")


def _make_rope() -> RopeState:
    rope = object.__new__(RopeState)
    rope.num_dims = NUM_DIMS
    rope.max_num_reqs = MAX_NUM_REQS
    rope.max_model_len = MAX_MODEL_LEN
    rope.positions = torch.zeros(
        (NUM_DIMS, 2 * MAX_MODEL_LEN + 1), dtype=torch.int64, device=DEVICE
    )

    prefill_positions = torch.zeros(
        (NUM_DIMS * MAX_NUM_REQS, MAX_MODEL_LEN), dtype=torch.int32, device=DEVICE
    )
    for req_idx in range(MAX_NUM_REQS):
        for dim in range(NUM_DIMS):
            base = 100 * (req_idx + 1) + 10 * (dim + 1)
            prefill_positions[
                req_idx * NUM_DIMS + dim, :MAX_MODEL_LEN
            ] = torch.arange(base, base + MAX_MODEL_LEN, device=DEVICE)
    rope.prefill_positions = SimpleNamespace(gpu=prefill_positions)
    rope.prefill_delta = SimpleNamespace(
        gpu=torch.tensor([0, 2], dtype=torch.int32, device=DEVICE)
    )
    return rope


def _make_prefill_input_batch() -> SimpleNamespace:
    return SimpleNamespace(
        num_tokens_after_padding=5,
        num_tokens=4,
        num_reqs=2,
        idx_mapping=torch.tensor([0, 0], dtype=torch.int64, device=DEVICE),
        query_start_loc=torch.tensor([0, 2, 4], dtype=torch.int32, device=DEVICE),
        positions=torch.tensor([0, 1, 3, 4, 0], dtype=torch.int64, device=DEVICE),
    )


def test_prepare_positions_pcp_prefill_slices_global_positions():
    rope = _make_rope()
    input_batch = _make_prefill_input_batch()
    prefill_lens = torch.tensor(
        [MAX_MODEL_LEN, MAX_MODEL_LEN], dtype=torch.int32, device=DEVICE
    )

    result = rope.prepare_positions_pcp(input_batch, prefill_lens)
    torch.npu.synchronize()

    expected = torch.zeros((NUM_DIMS, 5), dtype=torch.int64, device=DEVICE)
    expected[:, :2] = torch.tensor(
        [[110, 111], [120, 121], [130, 131]], dtype=torch.int64, device=DEVICE
    )
    expected[:, 2:4] = torch.tensor(
        [[113, 114], [123, 124], [133, 134]], dtype=torch.int64, device=DEVICE
    )
    assert torch.equal(result, expected)


def test_prepare_positions_pcp_decode_adds_delta():
    rope = _make_rope()
    input_batch = SimpleNamespace(
        num_tokens_after_padding=3,
        num_tokens=2,
        num_reqs=1,
        idx_mapping=torch.tensor([1], dtype=torch.int64, device=DEVICE),
        query_start_loc=torch.tensor([0, 2, 3], dtype=torch.int32, device=DEVICE),
        positions=torch.tensor([4, 5, 0], dtype=torch.int64, device=DEVICE),
    )
    prefill_lens = torch.tensor([MAX_MODEL_LEN, 4], dtype=torch.int32, device=DEVICE)

    result = rope.prepare_positions_pcp(input_batch, prefill_lens)
    torch.npu.synchronize()

    expected = torch.zeros((NUM_DIMS, 3), dtype=torch.int64, device=DEVICE)
    expected[:, :2] = torch.tensor([[6, 7]] * NUM_DIMS, dtype=torch.int64, device=DEVICE)
    assert torch.equal(result, expected)


def test_default_model_state_uses_pcp_positions(monkeypatch):
    rope = _make_rope()
    state = object.__new__(DefaultModelState)
    state.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(prefill_context_parallel_size=2)
    )
    state.rope_state = rope
    input_batch = _make_prefill_input_batch()
    req_states = SimpleNamespace(
        prefill_len=SimpleNamespace(
            gpu=torch.tensor([MAX_MODEL_LEN, MAX_MODEL_LEN], dtype=torch.int32, device=DEVICE)
        )
    )

    result = state.prepare_inputs(input_batch, req_states)
    torch.npu.synchronize()

    expected = torch.zeros((NUM_DIMS, 5), dtype=torch.int64, device=DEVICE)
    expected[:, :2] = torch.tensor(
        [[110, 111], [120, 121], [130, 131]], dtype=torch.int64, device=DEVICE
    )
    expected[:, 2:4] = torch.tensor(
        [[113, 114], [123, 124], [133, 134]], dtype=torch.int64, device=DEVICE
    )
    assert torch.equal(result["positions"], expected)


def test_default_model_state_without_pcp_uses_upstream_positions(monkeypatch):
    rope = _make_rope()
    state = object.__new__(DefaultModelState)
    state.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(prefill_context_parallel_size=1)
    )
    state.rope_state = rope
    sentinel = object()
    monkeypatch.setattr(RopeState, "prepare_positions", lambda self, *args, **kwargs: None)
    monkeypatch.setattr(RopeState, "get_positions", lambda self, num_tokens: sentinel)

    input_batch = SimpleNamespace(idx_mapping=None, query_start_loc=None, num_tokens_after_padding=1)
    req_states = SimpleNamespace(
        prefill_len=SimpleNamespace(gpu=None), num_computed_tokens=SimpleNamespace(gpu=None)
    )
    result = state.prepare_inputs(input_batch, req_states)
    assert result["positions"] is sentinel
