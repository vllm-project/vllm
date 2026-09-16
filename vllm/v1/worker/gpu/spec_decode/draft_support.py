# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Restrict probabilistic draft sampling to the target's top-k / top-p support.

The drafter runs this on every draft step, usually inside a captured CUDA
graph, so nothing here depends on the batch on the host. Both kernels read
the request's top-k / top-p from the sampler's persistent tensors. Rows
without top-k, and CUDA graph padding, return right away, so batches without
top-k cost two near-empty launches. Top-p is applied within top-k only: on
its own it removes little of a language model's vocabulary, and finding its
cut would cost a full pass over the logits.

1. ``_draft_block_max_kernel`` splits each row's vocabulary into blocks and
   takes the maximum of every block, several programs per row, plus each
   block's sum of exponentials when top-p is in use. Vocabularies of up to
   ``_MAX_NUM_BLOCKS`` tokens get one block per token.
2. ``_draft_support_threshold_kernel`` finds, per row, the k-th largest block
   maximum by bisection on the value, then the top-p cut over the block
   maxima above it, again by bisection, in the order the target sampler
   applies them. The mass is normalised over every token of the blocks
   above the top-k cut.

The threshold never masks a token the target keeps: the k-th largest block
maximum is not above the k-th largest logit, and the mass strictly above a
cut is counted with block maxima only and normalised over at least the
target's top-k set. It is exact when every block holds one token, and
otherwise as tight as the blocks allow. The Gumbel sampling kernel drops
everything below the threshold in its own vocab pass. A wider draft support
only lowers the acceptance gain: rejection sampling keeps the output
distribution for any draft distribution, as long as the cached draft logits
match the sampled ones. Rows whose top-k exceeds the number of blocks are
left unmasked, like rows without top-k.
"""

import torch

from vllm.triton_utils import tl, triton

# The vocabulary is split into at most this many blocks.
_MAX_NUM_BLOCKS = 4096
# Programs that share one row's pass over the vocabulary.
_NUM_SPLITS = 16
_BISECTION_STEPS = 30


def _block_layout(vocab_size: int) -> tuple[int, int]:
    """(tokens per block, number of blocks)."""
    block = triton.next_power_of_2(triton.cdiv(vocab_size, _MAX_NUM_BLOCKS))
    return block, triton.cdiv(vocab_size, block)


def draft_support_num_blocks(vocab_size: int) -> int:
    """Number of block maxima per row; rows with top-k up to it are masked."""
    return _block_layout(vocab_size)[1]


@triton.jit
def _draft_block_max_kernel(
    block_max_ptr,
    block_sum_ptr,
    block_stride,
    logits_ptr,
    logits_stride,
    idx_mapping_ptr,
    top_k_ptr,
    top_p_ptr,
    temperature_ptr,
    vocab_size,
    num_blocks,
    BLOCK: tl.constexpr,
    BLOCKS_PER_SPLIT: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    split = tl.program_id(1)
    req_state_idx = tl.load(idx_mapping_ptr + row).to(tl.int64)
    if req_state_idx < 0:
        # CUDA graph padding.
        return
    top_k = tl.load(top_k_ptr + req_state_idx)
    if (top_k >= vocab_size) | (top_k > num_blocks):
        # No top-k, or a top-k cut the block maxima cannot place.
        return
    top_p = tl.load(top_p_ptr + req_state_idx).to(tl.float32)

    blocks = split * BLOCKS_PER_SPLIT + tl.arange(0, BLOCKS_PER_SPLIT)
    tokens = blocks[:, None] * BLOCK + tl.arange(0, BLOCK)[None, :]
    x = tl.load(
        logits_ptr + row * logits_stride + tokens,
        mask=tokens < vocab_size,
        other=float("-inf"),
    ).to(tl.float32)
    block_max = tl.max(x, axis=1)
    is_block = blocks < num_blocks
    tl.store(block_max_ptr + row * block_stride + blocks, block_max, mask=is_block)

    if top_p < 1.0:
        # Each block's mass relative to its maximum, temperature-scaled; the
        # threshold kernel normalises top-p over the blocks above the top-k cut.
        temperature = tl.load(temperature_ptr + req_state_idx).to(tl.float32)
        temperature = tl.where(temperature > 0.0, temperature, 1.0)
        block_max_safe = tl.where(block_max > float("-inf"), block_max, 0.0)
        block_sum = tl.sum(tl.exp((x - block_max_safe[:, None]) / temperature), axis=1)
        tl.store(block_sum_ptr + row * block_stride + blocks, block_sum, mask=is_block)


@triton.jit
def _draft_support_threshold_kernel(
    threshold_ptr,
    block_max_ptr,
    block_sum_ptr,
    block_stride,
    idx_mapping_ptr,
    top_k_ptr,
    top_p_ptr,
    temperature_ptr,
    vocab_size,
    num_blocks,
    NUM_BLOCKS_PADDED: tl.constexpr,
    BISECTION_STEPS: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    req_state_idx = tl.load(idx_mapping_ptr + row).to(tl.int64)
    if req_state_idx < 0:
        tl.store(threshold_ptr + row, float("-inf"))
        return
    top_k = tl.load(top_k_ptr + req_state_idx)
    if (top_k >= vocab_size) | (top_k > num_blocks):
        tl.store(threshold_ptr + row, float("-inf"))
        return
    top_p = tl.load(top_p_ptr + req_state_idx).to(tl.float32)

    blocks = tl.arange(0, NUM_BLOCKS_PADDED)
    is_block = blocks < num_blocks
    block_max = tl.load(
        block_max_ptr + row * block_stride + blocks,
        mask=is_block,
        other=float("-inf"),
    )
    top = tl.max(block_max)

    # 1. The k-th largest block maximum: the largest value with at least top_k
    # block maxima at or above it.
    lo = tl.min(tl.where(is_block, block_max, float("inf")))
    hi = top
    for _ in range(BISECTION_STEPS):
        mid = lo + (hi - lo) * 0.5
        enough = tl.sum((block_max >= mid).to(tl.int32)) >= top_k
        lo = tl.where(enough, mid, lo)
        hi = tl.where(enough, hi, mid)
    k_cut = tl.min(tl.where(block_max >= lo, block_max, float("inf")))
    threshold = k_cut

    if top_p < 1.0:
        temperature = tl.load(temperature_ptr + req_state_idx).to(tl.float32)
        temperature = tl.where(temperature > 0.0, temperature, 1.0)
        in_top_k = block_max >= k_cut
        ref = top / temperature
        weight = tl.where(in_top_k, tl.exp(block_max / temperature - ref), 0.0)
        # Normalise over every token of the blocks above the top-k cut. The
        # target's top-k set lies within those blocks, so this counts at least
        # its mass, and exactly its mass when each block holds one token.
        block_sum = tl.load(
            block_sum_ptr + row * block_stride + blocks, mask=is_block, other=0.0
        )
        norm = tl.sum(weight * block_sum)
        prob = weight / norm
        # 2. The target keeps a token while the mass strictly above it is below
        # top_p: the lowest such value among the block maxima. When they hold
        # less than top_p of the mass, that is the top-k cut itself.
        lo = k_cut
        hi = top
        for _ in range(BISECTION_STEPS):
            mid = lo + (hi - lo) * 0.5
            below = tl.sum(tl.where(block_max > mid, prob, 0.0)) < top_p
            hi = tl.where(below, mid, hi)
            lo = tl.where(below, lo, mid)
        below_lo = tl.sum(tl.where(block_max > lo, prob, 0.0)) < top_p
        hi = tl.where(below_lo, lo, hi)
        p_cut = tl.max(tl.where(in_top_k & (block_max <= hi), block_max, float("-inf")))
        threshold = tl.maximum(k_cut, p_cut)

    tl.store(threshold_ptr + row, threshold)


def draft_top_k_top_p_threshold(
    # [num_tokens, vocab_size]
    logits: torch.Tensor,
    # [num_tokens]; -1 for CUDA graph padding
    idx_mapping: torch.Tensor,
    # [max_num_reqs]; vocab_size where a request does not use top-k
    top_k: torch.Tensor,
    # [max_num_reqs]; 1.0 where a request does not use top-p
    top_p: torch.Tensor,
    # [max_num_reqs]
    temperature: torch.Tensor,
) -> torch.Tensor:
    """Per-row float32 threshold: logits below it are outside the top-k / top-p
    support, -inf where the row stays unmasked (no top-k, or top-k beyond
    ``draft_support_num_blocks``).

    Shapes and host-side control flow depend only on ``logits.shape``, so this
    can be recorded in a CUDA graph.
    """
    num_tokens, vocab_size = logits.shape
    if logits.stride(1) != 1:
        logits = logits.contiguous()
    threshold = logits.new_empty(num_tokens, dtype=torch.float32)
    if num_tokens == 0:
        return threshold
    block, num_blocks = _block_layout(vocab_size)
    blocks_per_split = triton.next_power_of_2(triton.cdiv(num_blocks, _NUM_SPLITS))
    block_max = logits.new_empty((num_tokens, num_blocks), dtype=torch.float32)
    block_sum = logits.new_empty((num_tokens, num_blocks), dtype=torch.float32)
    _draft_block_max_kernel[(num_tokens, _NUM_SPLITS)](
        block_max,
        block_sum,
        block_max.stride(0),
        logits,
        logits.stride(0),
        idx_mapping,
        top_k,
        top_p,
        temperature,
        vocab_size,
        num_blocks,
        BLOCK=block,
        BLOCKS_PER_SPLIT=blocks_per_split,
        NUM_SPLITS=_NUM_SPLITS,
    )
    _draft_support_threshold_kernel[(num_tokens,)](
        threshold,
        block_max,
        block_sum,
        block_max.stride(0),
        idx_mapping,
        top_k,
        top_p,
        temperature,
        vocab_size,
        num_blocks,
        NUM_BLOCKS_PADDED=triton.next_power_of_2(num_blocks),
        BISECTION_STEPS=_BISECTION_STEPS,
    )
    return threshold


def apply_draft_top_k_top_p(
    logits: torch.Tensor,
    idx_mapping: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    temperature: torch.Tensor,
) -> torch.Tensor:
    """Return the logits with tokens outside the top-k / top-p support at -inf."""
    threshold = draft_top_k_top_p_threshold(
        logits, idx_mapping, top_k, top_p, temperature
    )
    below = logits.to(torch.float32) < threshold.unsqueeze(1)
    return logits.masked_fill(below, float("-inf"))
