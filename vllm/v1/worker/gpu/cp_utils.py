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
) -> None:
    """Populate the persistent DCP local seq_lens buffer (CUDA graph safe)."""
    if dcp_size == 1:
        return

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
    block_numbers,
    block_size,
    cp_rank,
    CP_SIZE: tl.constexpr,
    CP_INTERLEAVE: tl.constexpr,
    PAD_ID: tl.constexpr,
):
    """KV slot for `positions` on this CP rank, or PAD_ID if not owned here.

    DCP distributes a sequence round-robin over the CP ranks in chunks of
    CP_INTERLEAVE, so a token's rank-local offset is not its global offset.
    At CP_SIZE == 1 this reduces to the non-CP expression exactly.

    `block_numbers` must have been loaded with the matching block index,
    `positions // (block_size * CP_SIZE)`.
    """
    block_offsets = positions % (block_size * CP_SIZE)
    if CP_SIZE == 1:
        return block_numbers * block_size + block_offsets
    is_local = block_offsets // CP_INTERLEAVE % CP_SIZE == cp_rank
    rounds = block_offsets // (CP_INTERLEAVE * CP_SIZE)
    remainder = block_offsets % CP_INTERLEAVE
    local_offsets = rounds * CP_INTERLEAVE + remainder
    return tl.where(is_local, block_numbers * block_size + local_offsets, PAD_ID)
