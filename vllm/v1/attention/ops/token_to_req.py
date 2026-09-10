# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kernels for constructing packed-token to request mappings."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _scatter_token_to_req_kernel(
    query_start_loc,
    token_to_req,
    NUM_REQS: tl.constexpr,
    NUM_TOKENS: tl.constexpr,
    NUM_REQ_BLOCKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)

    num_req_programs = NUM_REQS * NUM_REQ_BLOCKS
    is_req_program = pid < num_req_programs
    safe_pid = tl.minimum(pid, num_req_programs - 1)
    req_idx = safe_pid // NUM_REQ_BLOCKS
    req_block = safe_pid % NUM_REQ_BLOCKS
    req_start = tl.load(query_start_loc + req_idx)
    req_end = tl.load(query_start_loc + req_idx + 1)
    req_offsets = req_start + req_block * BLOCK_SIZE + offsets
    tl.store(
        token_to_req + req_offsets,
        req_idx,
        mask=is_req_program & (req_offsets < req_end),
    )

    # FULL cudagraph batches can have fewer logical tokens than physical rows.
    # Tail programs initialize those rows in this same kernel launch. Their
    # writes cannot overlap the request scatters above.
    tail_pid = pid - num_req_programs
    tail_offsets = tail_pid * BLOCK_SIZE + offsets
    mapped_tokens = tl.load(query_start_loc + NUM_REQS)
    tl.store(
        token_to_req + tail_offsets,
        0,
        mask=(~is_req_program)
        & (tail_offsets < NUM_TOKENS)
        & (tail_offsets >= mapped_tokens),
    )


def scatter_token_to_req_indices(
    query_start_loc: torch.Tensor,
    output: torch.Tensor,
    num_reqs: int,
    num_tokens: int,
    max_query_len: int,
) -> torch.Tensor:
    """Build a token-to-request map from device-side query boundaries.

    Unlike ``torch.repeat_interleave``, this is one GPU launch and requires no
    temporary allocation or prefix scan. Zero-length requests are supported.
    The physical padding tail ``[query_start_loc[-1], num_tokens)`` is zeroed.
    """
    assert query_start_loc.dtype == torch.int32
    assert output.dtype == torch.int32
    assert query_start_loc.shape[0] >= num_reqs + 1
    assert output.shape[0] >= num_tokens
    assert num_reqs > 0 and num_tokens > 0 and max_query_len > 0

    block_size = min(256, triton.next_power_of_2(max(32, max_query_len)))
    num_req_blocks = triton.cdiv(max_query_len, block_size)
    grid = (num_reqs * num_req_blocks + triton.cdiv(num_tokens, block_size),)
    _scatter_token_to_req_kernel[grid](
        query_start_loc,
        output,
        NUM_REQS=num_reqs,
        NUM_TOKENS=num_tokens,
        NUM_REQ_BLOCKS=num_req_blocks,
        BLOCK_SIZE=block_size,
    )
    return output[:num_tokens]
