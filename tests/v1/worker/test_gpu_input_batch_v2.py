# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the V2 model runner's InputBatch (vllm.v1.worker.gpu.input_batch)."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.worker.gpu.input_batch import (
    InputBatch,
    InputBuffers,
    set_dummy_context,
)

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


def test_make_dummy_preserves_mixed_profile_shape_and_context():
    split = np.array([8, 8, 8, 96], dtype=np.int32)
    device = torch.device(DEVICE)
    batch = InputBatch.make_dummy(
        num_reqs=4,
        num_tokens=120,
        input_buffers=InputBuffers(4, 120, device),
        num_scheduled_tokens=split,
    )
    input_block_table = torch.full((4, 8), -1, dtype=torch.int32, device=device)
    block_tables = SimpleNamespace(
        input_block_tables=[input_block_table],
        kernel_block_sizes=[16],
        blocks_per_kv_block=[1],
    )

    set_dummy_context(
        batch,
        block_tables,
        context_len=64,
        num_kv_blocks=32,
        max_model_len=128,
        num_context_reqs=3,
    )

    assert batch.num_scheduled_tokens.tolist() == split.tolist()
    assert batch.seq_lens.tolist() == [72, 72, 72, 96]
    assert batch.num_computed_tokens_np.tolist() == [64, 64, 64, 0]
    assert batch.prefill_len_np.tolist() == [0, 0, 0, 96]
    assert batch.is_prefilling_np.tolist() == [False, False, False, True]
    assert batch.has_prefill
    assert batch.positions.tolist() == list(range(64, 72)) * 3 + list(range(96))
    assert (input_block_table[:, :6] >= 0).all()
    assert (input_block_table[:, 6:] == -1).all()
