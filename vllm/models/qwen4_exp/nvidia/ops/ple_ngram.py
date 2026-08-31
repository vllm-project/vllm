# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused PLE n-gram embedding-ID kernel.

For each token, the kernel hashes suffixes of orders two through
``context_length + 1``. Tokens before the current request chunk come from its
saved context. Once a suffix crosses EOS, all older positions use EOS. Each
n-gram order produces ``heads_per_ngram`` IDs by applying per-head vocabulary
sizes and offsets.
"""

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

BLOCK_T = 8
NUM_WARPS = 4


@triton.jit(do_not_specialize=["num_tokens", "num_reqs", "binary_search_iters"])
def _ple_ngram_ids_kernel(
    input_ids_ptr,
    qsl_ptr,
    ctx_ptr,
    multipliers_ptr,
    sizes_ptr,
    offsets_ptr,
    out_ptr,
    num_tokens,
    num_reqs,
    eos_token_id,
    binary_search_iters,
    NGRAM_CONTEXT_LEN: tl.constexpr,
    HEADS_PER_NGRAM: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    NGRAM_HEADS: tl.constexpr = NGRAM_CONTEXT_LEN * HEADS_PER_NGRAM
    BLOCK_H: tl.constexpr = triton.next_power_of_2(NGRAM_HEADS)
    pid = tl.program_id(0)
    token_offsets = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    token_mask = token_offsets < num_tokens

    # A flattened token offset does not identify its request when query chunks
    # have different lengths. Binary-search the request boundaries instead.
    p_lo = tl.full([BLOCK_T], 1, tl.int32)
    p_hi = tl.full([BLOCK_T], num_reqs + 1, tl.int32)
    for _ in range(binary_search_iters):
        mid = (p_lo + p_hi) // 2
        qmid = tl.load(qsl_ptr + mid, mask=token_mask & (mid <= num_reqs), other=0)
        pred = qmid <= token_offsets
        p_lo = tl.where(pred, mid + 1, p_lo)
        p_hi = tl.where(pred, p_hi, mid)
    req = tl.minimum(p_lo - 1, num_reqs - 1).to(tl.int64)
    request_start = tl.load(qsl_ptr + req, mask=token_mask, other=0)
    chunk_pos = token_offsets - request_start

    # Initialize every head with the current-token term.
    current_token = tl.load(input_ids_ptr + token_offsets, mask=token_mask, other=0).to(
        tl.int64
    )
    current_multiplier = tl.load(multipliers_ptr)
    mixed = current_token[:, None] * current_multiplier

    g = tl.arange(0, BLOCK_H)
    head_mask = g < NGRAM_HEADS
    ngram_order = g // HEADS_PER_NGRAM + 2

    # Walk predecessors from newest to oldest. At chunk boundaries they come
    # from ngram_context; otherwise they come from this step's input_ids.
    crossed = tl.zeros([BLOCK_T], tl.int1)
    for shift in tl.static_range(1, NGRAM_CONTEXT_LEN + 1):
        in_step = chunk_pos >= shift
        ctx_col = NGRAM_CONTEXT_LEN - shift + chunk_pos
        step_token = tl.load(
            input_ids_ptr + token_offsets - shift,
            mask=token_mask & in_step,
            other=0,
        )
        context_token = tl.load(
            ctx_ptr + req * NGRAM_CONTEXT_LEN + ctx_col,
            mask=token_mask & (~in_step),
            other=0,
        )
        candidate = tl.where(in_step, step_token, context_token).to(tl.int64)
        # Older positions remain behind the first EOS boundary.
        candidate = tl.where(crossed, eos_token_id, candidate)
        crossed = crossed | (candidate == eos_token_id)
        multiplier = tl.load(multipliers_ptr + shift)
        term = candidate[:, None] * multiplier
        mixed = mixed ^ tl.where((ngram_order > shift)[None, :], term, 0)

    # Map each hash into its embedding-table partition.
    sizes = tl.load(sizes_ptr + g, mask=head_mask, other=1)[None, :]
    head_offsets = tl.load(offsets_ptr + g, mask=head_mask, other=0)[None, :]
    ids = mixed % sizes + head_offsets
    tl.store(
        out_ptr + token_offsets[:, None] * NGRAM_HEADS + g[None, :],
        ids,
        mask=token_mask[:, None] & head_mask[None, :],
    )


def _ple_ngram_ids(
    input_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    ngram_context: torch.Tensor,
    layer_multipliers: torch.Tensor,
    ngram_heads_vocab_sizes: torch.Tensor,
    ngram_heads_offsets: torch.Tensor,
    eos_token_id: int,
    heads_per_ngram: int,
) -> torch.Tensor:
    input_ids = input_ids.reshape(-1)
    num_tokens = input_ids.shape[0]
    num_reqs = query_start_loc.numel() - 1
    ctx_len = ngram_context.shape[1]
    ngram_heads = ctx_len * heads_per_ngram
    out = torch.empty(
        (num_tokens, ngram_heads), dtype=torch.int64, device=input_ids.device
    )
    _ple_ngram_ids_kernel[(triton.cdiv(num_tokens, BLOCK_T),)](
        input_ids,
        query_start_loc,
        ngram_context,
        layer_multipliers,
        ngram_heads_vocab_sizes,
        ngram_heads_offsets,
        out,
        num_tokens,
        num_reqs,
        eos_token_id,
        binary_search_iters=num_reqs.bit_length(),
        NGRAM_CONTEXT_LEN=ctx_len,
        HEADS_PER_NGRAM=heads_per_ngram,
        BLOCK_T=BLOCK_T,
        num_warps=NUM_WARPS,
    )
    return out


def _ple_ngram_ids_fake(
    input_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    ngram_context: torch.Tensor,
    layer_multipliers: torch.Tensor,
    ngram_heads_vocab_sizes: torch.Tensor,
    ngram_heads_offsets: torch.Tensor,
    eos_token_id: int,
    heads_per_ngram: int,
) -> torch.Tensor:
    num_tokens = input_ids.reshape(-1).shape[0]
    ngram_heads = ngram_context.shape[1] * heads_per_ngram
    return torch.empty(
        (num_tokens, ngram_heads), dtype=torch.int64, device=input_ids.device
    )


direct_register_custom_op(
    op_name="qwen4_exp_ple_ngram_ids",
    op_func=_ple_ngram_ids,
    mutates_args=[],
    fake_impl=_ple_ngram_ids_fake,
)


def ple_ngram_ids(
    input_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    ngram_context: torch.Tensor,
    layer_multipliers: torch.Tensor,
    ngram_heads_vocab_sizes: torch.Tensor,
    ngram_heads_offsets: torch.Tensor,
    eos_token_id: int,
    heads_per_ngram: int,
) -> torch.Tensor:
    return torch.ops.vllm.qwen4_exp_ple_ngram_ids(
        input_ids,
        query_start_loc,
        ngram_context,
        layer_multipliers,
        ngram_heads_vocab_sizes,
        ngram_heads_offsets,
        eos_token_id,
        heads_per_ngram,
    )
