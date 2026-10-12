# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""int64 row offsets in the MRV2 staged-write and idx-mapping kernels.

With int32 request-slot indices, `slot * stride` wraps once the product
crosses 2**31 (issue #57030): at max_model_len=1M from row 2148, and from
row 716 for the 3-dim mrope positions. Each test runs a control row below
and an overflow row above the threshold in the same kernel launch.
Requires CUDA/ROCm and ~11 GiB free device memory: the tensors must exceed
2**31 elements by construction.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv
from vllm.v1.worker.gpu.buffer_utils import StagedWriteTensor
from vllm.v1.worker.gpu.input_batch import post_update
from vllm.v1.worker.gpu.mm.rope import RopeState
from vllm.v1.worker.gpu.sample.penalties import bincount

STRIDE = 1_000_000  # max_model_len
ROW = 2559  # ROW * STRIDE > 2**31 (first overflowing row at 1M: 2148)
NUM_ROWS = ROW + 1  # (2560, 1M) int32 = 10.24 GB
NEEDED_BYTES = NUM_ROWS * STRIDE * 4
VOCAB_SIZE = 8192
PROMPT_LEN = 1024
PREFILL_LEN = 2048

# mrope: flat row = NUM_DIMS * req + j and stride0 = NUM_DIMS * STRIDE, so
# the overflow threshold drops to row 716 of a 1M-row buffer.
NUM_DIMS = 3
ROPE_ROW = 716  # NUM_DIMS * ROPE_ROW * STRIDE > 2**31

pytestmark = [
    pytest.mark.skipif(
        not current_platform.is_cuda_alike(), reason="requires CUDA or ROCm"
    ),
    pytest.mark.skipif(
        not current_platform.is_cuda_alike()
        or torch.accelerator.get_memory_info(0)[0]
        < NEEDED_BYTES + 2**30,  # slack for the outputs
        reason=f"requires >= {NEEDED_BYTES / 2**30:.0f} GiB free memory",
    ),
]


def test_staged_write_high_row():
    """Apply-write past the int32 offset limit must land on the target row."""
    device = torch.device("cuda:0")
    content = list(range(9000, 9016))
    staged = StagedWriteTensor((NUM_ROWS, STRIDE), torch.int32, device)
    staged.stage_write(ROW, 0, content)
    staged.apply_write()
    torch.accelerator.synchronize()

    assert staged.gpu[ROW, : len(content)].tolist() == content
    # A wrapped offset must not clobber row 0 (silent-corruption variant).
    assert staged.gpu[0, : len(content)].tolist() == [0] * len(content)


def test_bincount_high_row():
    """penalties.bincount past the int32 offset limit must hit the right row."""
    device = torch.device("cuda:0")
    tokens = torch.arange(PREFILL_LEN, dtype=torch.int32, device=device)

    all_token_ids = torch.zeros(NUM_ROWS, STRIDE, dtype=torch.int32, device=device)
    all_token_ids[0, :PREFILL_LEN] = tokens
    all_token_ids[ROW, :PREFILL_LEN] = tokens

    prompt_len = torch.zeros(NUM_ROWS, dtype=torch.int32, device=device)
    prefill_len = torch.zeros(NUM_ROWS, dtype=torch.int32, device=device)
    prompt_len[[0, ROW]] = PROMPT_LEN
    prefill_len[[0, ROW]] = PREFILL_LEN

    idx_mapping = torch.tensor([0, ROW], dtype=torch.int64, device=device)
    prompt_bin_mask = torch.zeros(
        NUM_ROWS, cdiv(VOCAB_SIZE, 32), dtype=torch.int32, device=device
    )
    output_bin_counts = torch.zeros(
        NUM_ROWS, VOCAB_SIZE, dtype=torch.int32, device=device
    )

    bincount(
        idx_mapping,
        all_token_ids,
        prompt_len,
        prefill_len,
        prompt_bin_mask,
        output_bin_counts,
        max_prefill_len=PREFILL_LEN,
    )
    torch.accelerator.synchronize()

    assert torch.equal(prompt_bin_mask[ROW], prompt_bin_mask[0])
    assert torch.equal(output_bin_counts[ROW], output_bin_counts[0])
    # Sanity: the control row got prompt bits and output counts.
    assert prompt_bin_mask[0, : cdiv(PROMPT_LEN, 32)].any().item()
    assert (
        output_bin_counts[0, PROMPT_LEN:PREFILL_LEN].sum().item()
        == PREFILL_LEN - PROMPT_LEN
    )


def test_post_update_high_row():
    """post_update (runs every sampled token) must store on the mapped row."""
    device = torch.device("cuda:0")
    # Big buffer first so a wrapped (negative) offset cannot land in-bounds.
    all_token_ids = torch.zeros(NUM_ROWS, STRIDE, dtype=torch.int32, device=device)
    num_computed = torch.zeros(NUM_ROWS, dtype=torch.int32, device=device)
    last_sampled = torch.full((NUM_ROWS, 1), -1, dtype=torch.int64, device=device)
    total_len = torch.zeros(NUM_ROWS, dtype=torch.int32, device=device)
    total_len[[0, ROW]] = torch.tensor([50, 100], dtype=torch.int32, device=device)

    idx_mapping = torch.tensor([0, ROW], dtype=torch.int64, device=device)
    sampled_tokens = torch.tensor([[111], [222]], dtype=torch.int64, device=device)
    num_sampled = torch.tensor([1, 1], dtype=torch.int32, device=device)
    num_rejected = torch.tensor([0, 0], dtype=torch.int32, device=device)
    query_start_loc = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)

    post_update(
        idx_mapping,
        num_computed,
        last_sampled,
        None,  # output_bin_counts: a second product of the same idx load
        sampled_tokens,
        num_sampled,
        num_rejected,
        query_start_loc,
        all_token_ids,
        total_len,
    )
    torch.accelerator.synchronize()

    assert int(all_token_ids[0, 50]) == 111
    assert int(all_token_ids[ROW, 100]) == 222
    # Each mapped row holds exactly its own token; unmapped rows stay zero.
    assert int(all_token_ids[0].sum()) == 111
    assert int(all_token_ids[ROW].sum()) == 222
    assert int(all_token_ids[1].sum()) == 0
    assert int(last_sampled[0, 0]) == 111
    assert int(last_sampled[ROW, 0]) == 222
    assert int(total_len[0]) == 51
    assert int(total_len[ROW]) == 101
    assert int(num_computed[0]) == 1
    assert int(num_computed[ROW]) == 1


def test_rope_positions_high_row():
    """M-RoPE prepare_positions must read staged rows past the int32 limit.

    Threshold is 3x lower than the other tests (stride0 = num_dims * 1M), so
    this is the variant that trips at default max_num_seqs on 1M context.
    """
    device = torch.device("cuda:0")
    max_num_reqs = NUM_ROWS // NUM_DIMS
    base = torch.arange(PREFILL_LEN, dtype=torch.int32, device=device)
    # (num_dims * max_num_reqs, max_model_len), logically [req][dim][pos].
    prefill_positions = torch.zeros(NUM_ROWS, STRIDE, dtype=torch.int32, device=device)
    # Distinct per-(dim, request) patterns so mixed-up rows are detectable.
    for j in range(NUM_DIMS):
        prefill_positions[j, :PREFILL_LEN] = 1_000 * (j + 1) + base
        target_row = NUM_DIMS * ROPE_ROW + j
        prefill_positions[target_row, :PREFILL_LEN] = 7_000 * (j + 1) + base

    # Skeletal RopeState: the real __init__ pins num_dims * max_num_reqs host
    # memory, which the stride arithmetic does not depend on.
    rope = object.__new__(RopeState)
    rope.num_dims = NUM_DIMS
    rope.max_num_reqs = max_num_reqs
    rope.max_model_len = STRIDE
    rope.prefill_positions = SimpleNamespace(gpu=prefill_positions)
    rope.positions = torch.zeros(
        (NUM_DIMS, 2 * PREFILL_LEN + 1), dtype=torch.int64, device=device
    )
    rope.prefill_delta = SimpleNamespace(
        gpu=torch.zeros(max_num_reqs, dtype=torch.int32, device=device)
    )

    idx_mapping = torch.tensor([0, ROPE_ROW], dtype=torch.int64, device=device)
    query_start_loc = torch.tensor(
        [0, PREFILL_LEN, 2 * PREFILL_LEN], dtype=torch.int32, device=device
    )
    prefill_len = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)
    prefill_len[[0, ROPE_ROW]] = PREFILL_LEN
    num_computed = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)

    rope.prepare_positions(idx_mapping, query_start_loc, prefill_len, num_computed)
    torch.accelerator.synchronize()

    expect_control = torch.stack([1_000 * (j + 1) + base for j in range(NUM_DIMS)]).to(
        torch.int64
    )
    expect_target = torch.stack([7_000 * (j + 1) + base for j in range(NUM_DIMS)]).to(
        torch.int64
    )
    assert torch.equal(rope.positions[:, :PREFILL_LEN], expect_control)
    assert torch.equal(rope.positions[:, PREFILL_LEN : 2 * PREFILL_LEN], expect_target)
    # Padding column stays zero.
    assert bool((rope.positions[:, 2 * PREFILL_LEN :] == 0).all())
