# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the V2 model runner's InputBatch (vllm.v1.worker.gpu.input_batch)."""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.worker.gpu.input_batch import (
    InputBatch,
    InputBuffers,
    prepare_pos_seq_lens,
)
from vllm.v1.worker.gpu.mm.rope import RopeState

DEVICE = current_platform.device_type


@pytest.mark.parametrize(
    "num_reqs,num_tokens",
    [
        (256, 496),  # remainder 240: previously gave the last request 241 tokens
        (128, 512),  # no remainder
        (3, 8),
        (1, 7),
    ],
)
def test_make_dummy_distributes_remainder(num_reqs: int, num_tokens: int):
    """No dummy request may exceed ceil(num_tokens / num_reqs) tokens.

    Dumping the remainder on a single request can produce a dummy request with
    seq_len > max_model_len, which the block tables cannot back; attention
    kernels running on the dummy batch during cudagraph capture then read
    block-table entries out of bounds (https://github.com/vllm-project/vllm/pull/49364
    CI failure).
    """
    buffers = InputBuffers(
        max_num_reqs=num_reqs, max_num_tokens=num_tokens, device=torch.device(DEVICE)
    )
    batch = InputBatch.make_dummy(num_reqs, num_tokens, buffers)

    max_per_req = -(-num_tokens // num_reqs)
    assert batch.num_scheduled_tokens.sum() == num_tokens
    assert batch.num_scheduled_tokens.max() == max_per_req
    assert batch.num_scheduled_tokens.min() >= num_tokens // num_reqs
    # Requests with an extra token are placed at the end of the batch.
    assert (batch.num_scheduled_tokens[:-1] <= batch.num_scheduled_tokens[1:]).all()

    # seq_len == query_len for the dummy prefill-shaped batch, on GPU and CPU.
    query_lens = batch.query_start_loc_np[1:] - batch.query_start_loc_np[:-1]
    assert (query_lens == batch.num_scheduled_tokens).all()
    assert torch.equal(
        batch.seq_lens, torch.from_numpy(batch.num_scheduled_tokens).to(DEVICE)
    )
    assert batch.query_start_loc_np[-1] == num_tokens
    assert torch.equal(
        batch.query_start_loc.cpu(), torch.from_numpy(batch.query_start_loc_np)
    )


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="Requires a CUDA-like device"
)
def test_prepare_pos_seq_lens_separates_physical_and_rope_positions():
    device = torch.device(DEVICE)
    idx_mapping = torch.tensor([1, 0], dtype=torch.int32, device=device)
    query_start_loc = torch.tensor([0, 3, 5], dtype=torch.int32, device=device)
    num_computed_tokens = torch.tensor([20, 10], dtype=torch.int32, device=device)
    rope_offsets = torch.tensor([200, 100], dtype=torch.int64, device=device)
    positions = torch.empty(5, dtype=torch.int64, device=device)
    rope_positions = torch.empty_like(positions)
    seq_lens = torch.empty(4, dtype=torch.int32, device=device)

    prepare_pos_seq_lens(
        idx_mapping,
        query_start_loc,
        num_computed_tokens,
        positions,
        seq_lens,
        rope_positions,
        rope_offsets,
    )
    torch.accelerator.synchronize()

    torch.testing.assert_close(positions.cpu(), torch.tensor([10, 11, 12, 20, 21]))
    torch.testing.assert_close(
        rope_positions.cpu(), torch.tensor([110, 111, 112, 220, 221])
    )
    torch.testing.assert_close(
        seq_lens.cpu(), torch.tensor([13, 22, 0, 0], dtype=torch.int32)
    )


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="Requires a CUDA-like device"
)
def test_rope_state_applies_request_static_offsets():
    device = torch.device(DEVICE)
    state = RopeState(
        num_dims=3,
        has_delta=True,
        max_num_reqs=2,
        max_num_tokens=5,
        max_model_len=32,
        device=device,
    )
    state.prefill_delta.gpu.zero_()
    idx_mapping = torch.tensor([1, 0], dtype=torch.int32, device=device)
    query_start_loc = torch.tensor([0, 3, 5], dtype=torch.int32, device=device)
    prefill_lens = torch.zeros(2, dtype=torch.int32, device=device)
    num_computed_tokens = torch.tensor([20, 10], dtype=torch.int32, device=device)
    rope_offsets = torch.tensor([200, 100], dtype=torch.int64, device=device)

    state.prepare_positions(
        idx_mapping,
        query_start_loc,
        prefill_lens,
        num_computed_tokens,
        rope_offsets,
    )
    torch.accelerator.synchronize()

    expected = torch.tensor([110, 111, 112, 220, 221]).expand(3, -1)
    torch.testing.assert_close(state.get_positions(5).cpu(), expected)
