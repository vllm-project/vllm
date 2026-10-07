# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.triton_utils import tl, triton


def prepare_dcp_local_seq_lens(
    dcp_local_seq_lens: torch.Tensor,
    seq_lens: torch.Tensor,
    num_reqs: int,
    dcp_size: int,
    dcp_rank: int,
    cp_interleave: int,
    *,
    num_reqs_padded: int | None = None,
) -> torch.Tensor | None:
    """Populate caller-owned storage and return its padded view, or None without DCP."""
    assert dcp_size > 1

    max_num_reqs = dcp_local_seq_lens.shape[0]
    BLOCK_SIZE = 128
    num_blocks = triton.cdiv(max_num_reqs, BLOCK_SIZE)
    _dcp_local_seq_lens_kernel[(num_blocks,)](
        dcp_local_seq_lens,
        seq_lens,
        dcp_size,
        dcp_rank,
        cp_interleave,
        num_reqs,
        max_num_reqs,
        BLOCK_SIZE,
    )
    return dcp_local_seq_lens[:num_reqs_padded]


@triton.jit
def _dcp_local_seq_lens_kernel(
    out_ptr,
    seq_lens_ptr,
    dcp_size,
    dcp_rank,
    cp_interleave,
    num_reqs,
    max_num_reqs,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    seq_lens = tl.load(seq_lens_ptr + block, mask=block < num_reqs)

    # Distribute KV cache among different ranks, in a round-robin manner.
    rounds = seq_lens // (dcp_size * cp_interleave)
    remainder = seq_lens % (dcp_size * cp_interleave)

    remainder = tl.maximum(remainder - dcp_rank * cp_interleave, 0)
    remainder = tl.minimum(remainder, cp_interleave)
    local_seq_lens = rounds * cp_interleave + remainder

    # For [num_reqs, max_num_reqs), pad with 0
    local_seq_lens = tl.where(block < num_reqs, local_seq_lens, 0)
    tl.store(out_ptr + block, local_seq_lens, mask=block < max_num_reqs)


@triton.jit
def cp_local_slot(
    positions,
    block_table_row_ptr,
    block_table_stride,
    is_valid,
    block_size,
    kernel_block_size,
    cp_rank,
    CP_SIZE: tl.constexpr,
    CP_INTERLEAVE: tl.constexpr,
    PAD_ID: tl.constexpr,
):
    """Return rank-local KV slots, or PAD_ID for positions not owned by this rank."""
    if CP_SIZE == 1:
        local_positions = positions
        is_local = is_valid
    else:
        virtual_block_size = block_size * CP_SIZE
        virtual_block_indices = positions // virtual_block_size
        virtual_block_offsets = positions % virtual_block_size
        is_local = is_valid & (
            virtual_block_offsets // CP_INTERLEAVE % CP_SIZE == cp_rank
        )
        rounds = virtual_block_offsets // (CP_INTERLEAVE * CP_SIZE)
        remainder = virtual_block_offsets % CP_INTERLEAVE
        local_offsets = rounds * CP_INTERLEAVE + remainder
        local_positions = virtual_block_indices * block_size + local_offsets

    block_num = tl.minimum(local_positions // kernel_block_size, block_table_stride - 1)
    block_id = tl.load(
        block_table_row_ptr + block_num,
        mask=is_local,
        other=0,
    ).to(tl.int64)
    resident = is_local & (block_id != 0)
    block_offset = local_positions % kernel_block_size
    return tl.where(
        resident,
        block_id * kernel_block_size + block_offset,
        PAD_ID,
    )
