# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PCP-aware M-RoPE position preparation."""

from types import SimpleNamespace

import numpy as np
import torch

from vllm.v1.worker.gpu.mm.rope import RopeState
from vllm.v1.worker.gpu.model_states.default import DefaultModelState


NUM_DIMS = 3
MAX_NUM_REQS = 2
MAX_MODEL_LEN = 8


def _make_rope() -> RopeState:
    rope = object.__new__(RopeState)
    rope.num_dims = NUM_DIMS
    rope.max_num_reqs = MAX_NUM_REQS
    rope.max_model_len = MAX_MODEL_LEN
    rope.positions = torch.zeros(
        (NUM_DIMS, 2 * MAX_MODEL_LEN + 1), dtype=torch.int64
    )

    prefill_positions = torch.zeros(
        (NUM_DIMS * MAX_NUM_REQS, MAX_MODEL_LEN), dtype=torch.int32
    )
    for req_idx in range(MAX_NUM_REQS):
        for dim in range(NUM_DIMS):
            base = 100 * (req_idx + 1) + 10 * (dim + 1)
            prefill_positions[
                req_idx * NUM_DIMS + dim, :MAX_MODEL_LEN
            ] = torch.arange(base, base + MAX_MODEL_LEN)
    rope.prefill_positions = SimpleNamespace(gpu=prefill_positions)
    rope.prefill_delta = SimpleNamespace(np=np.array([0, 2], dtype=np.int32))
    return rope


def _make_prefill_input_batch() -> SimpleNamespace:
    return SimpleNamespace(
        num_tokens_after_padding=5,
        num_reqs=2,
        num_scheduled_tokens=np.array([2, 2], dtype=np.int32),
        idx_mapping_np=np.array([0, 0], dtype=np.int64),
        is_prefilling_np=np.array([True, True], dtype=np.bool_),
        num_computed_prefill_tokens_np=np.array([0, 3], dtype=np.int32),
        positions=torch.tensor([0, 1, 3, 4, 0], dtype=torch.int64),
    )


def test_prepare_positions_pcp_prefill_slices_global_positions():
    rope = _make_rope()
    input_batch = _make_prefill_input_batch()

    result = rope.prepare_positions_pcp(input_batch)

    expected = torch.zeros((NUM_DIMS, 5), dtype=torch.int64)
    expected[:, :2] = torch.tensor(
        [[110, 111], [120, 121], [130, 131]], dtype=torch.int64
    )
    expected[:, 2:4] = torch.tensor(
        [[113, 114], [123, 124], [133, 134]], dtype=torch.int64
    )
    assert torch.equal(result, expected)


def test_prepare_positions_pcp_decode_adds_delta():
    rope = _make_rope()
    input_batch = SimpleNamespace(
        num_tokens_after_padding=3,
        num_reqs=1,
        num_scheduled_tokens=np.array([2], dtype=np.int32),
        idx_mapping_np=np.array([1], dtype=np.int64),
        is_prefilling_np=np.array([False], dtype=np.bool_),
        num_computed_prefill_tokens_np=np.array([MAX_MODEL_LEN], dtype=np.int32),
        positions=torch.tensor([4, 5, 0], dtype=torch.int64),
    )

    result = rope.prepare_positions_pcp(input_batch)

    expected = torch.zeros((NUM_DIMS, 3), dtype=torch.int64)
    expected[:, :2] = torch.tensor([[6, 7]] * NUM_DIMS, dtype=torch.int64)
    assert torch.equal(result, expected)


def test_default_model_state_uses_pcp_positions():
    rope = _make_rope()
    state = object.__new__(DefaultModelState)
    state.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(prefill_context_parallel_size=2)
    )
    state.rope_state = rope
    input_batch = _make_prefill_input_batch()

    result = state.prepare_inputs(input_batch, SimpleNamespace())

    expected = torch.zeros((NUM_DIMS, 5), dtype=torch.int64)
    expected[:, :2] = torch.tensor(
        [[110, 111], [120, 121], [130, 131]], dtype=torch.int64
    )
    expected[:, 2:4] = torch.tensor(
        [[113, 114], [123, 124], [133, 134]], dtype=torch.int64
    )
    assert torch.equal(result["positions"], expected)
