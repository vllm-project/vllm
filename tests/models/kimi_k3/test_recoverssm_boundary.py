# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Align-mode RecoverSSM state columns at recurrent block boundaries.

A speculative window whose accepted tokens end exactly on a recurrent block
boundary must keep its state in the block holding the last committed token.
The following block can still be unallocated, so selecting it would drop the
commit while the bookkeeping recorded the state as advanced.
"""

import pytest
import torch

from vllm.models.kimi_k3.nvidia.ops.recoverssm import _prepare_commit_plan_kernel
from vllm.platforms import current_platform
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
from vllm.v1.worker.gpu.model_states.recoverssm import (
    _postprocess_recoverssm_align_kernel,
)

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="The RecoverSSM kernels require a CUDA-alike device.",
)

BLOCK_SIZE = 2048
SPEC_QUERY_LEN = 4
BLOCKS = [100, 101, 102, 103, 104, 105]

# (computed before the step, committed tokens, allocated block columns,
#  final column, boundary column or None)
CASES = [
    # 8,188 + 4 ends exactly on the 8,192 boundary; column 4 is unallocated.
    (8188, 4, 4, 3, 3),
    # The window crosses the boundary: state moves on, checkpoint stays behind.
    (8190, 4, 5, 4, 3),
    # The window stays inside column 3.
    (8180, 4, 4, 3, None),
    # The first token after a boundary belongs to the new column.
    (8192, 1, 5, 4, None),
    # Single-token commit ending on the first boundary.
    (2047, 1, 1, 0, 0),
]


@pytest.mark.parametrize(
    "num_computed,commit_len,allocated,final_col,boundary_col", CASES
)
def test_commit_plan_keeps_boundary_state_in_last_computed_block(
    num_computed, commit_len, allocated, final_col, boundary_col
):
    device = current_platform.device_type
    columns = [
        block if col < allocated else NULL_BLOCK_ID for col, block in enumerate(BLOCKS)
    ]
    block_table = torch.tensor([columns], dtype=torch.int32, device=device)
    num_accepted = torch.tensor([commit_len], dtype=torch.int32, device=device)
    state_indices = torch.tensor(
        [BLOCKS[(num_computed - 1) // BLOCK_SIZE]], dtype=torch.int32, device=device
    )
    query_start_loc = torch.tensor(
        [0, SPEC_QUERY_LEN], dtype=torch.int32, device=device
    )
    computed = torch.tensor([num_computed], dtype=torch.int32, device=device)
    commit_lens, final_indices, boundary_indices, recovery_lens = (
        torch.full((1,), -7, dtype=torch.int32, device=device) for _ in range(4)
    )

    _prepare_commit_plan_kernel[(1,)](
        num_accepted,
        None,
        state_indices,
        query_start_loc,
        block_table,
        computed,
        commit_lens,
        final_indices,
        boundary_indices,
        recovery_lens,
        NULL_BLOCK_ID,
        BLOCK_SIZE,
        block_table.shape[1],
        num_accepted.stride(0),
        0,
        state_indices.stride(0),
        query_start_loc.stride(0),
        block_table.stride(0),
        block_table.stride(1),
        computed.stride(0),
        SPEC_QUERY_LEN=SPEC_QUERY_LEN,
        num_warps=1,
    )

    assert commit_lens.item() == commit_len
    assert final_indices.item() == BLOCKS[final_col]
    expected_boundary = NULL_BLOCK_ID if boundary_col is None else BLOCKS[boundary_col]
    assert boundary_indices.item() == expected_boundary
    if boundary_col is not None:
        next_boundary = (num_computed // BLOCK_SIZE + 1) * BLOCK_SIZE
        assert recovery_lens.item() == next_boundary - num_computed


@pytest.mark.parametrize(
    "num_computed,commit_len,allocated,final_col,boundary_col", CASES
)
def test_postprocess_records_the_committed_state_column(
    num_computed, commit_len, allocated, final_col, boundary_col
):
    device = current_platform.device_type
    idx_mapping = torch.tensor([2], dtype=torch.int32, device=device)
    num_sampled = torch.tensor([commit_len], dtype=torch.int32, device=device)
    computed = torch.tensor([num_computed], dtype=torch.int32, device=device)
    state_idx = torch.full((4,), -7, dtype=torch.int32, device=device)
    num_accepted = torch.full((4,), -7, dtype=torch.int32, device=device)

    _postprocess_recoverssm_align_kernel[(1,)](
        idx_mapping,
        num_sampled,
        None,
        computed,
        state_idx,
        num_accepted,
        MAMBA_BLOCK_SIZE=BLOCK_SIZE,
        BLOCK_TABLE_WIDTH=len(BLOCKS),
    )

    assert state_idx[2].item() == final_col
    assert num_accepted[2].item() == 1
    assert state_idx[[0, 1, 3]].tolist() == [-7, -7, -7]
