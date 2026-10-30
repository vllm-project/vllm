# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark's sequential Markov sampling as the bias GEMM plus two kernels a step.
Tokens are drawn by inverse CDF of softmax(cached logits / temperature), one exp
per logit instead of Gumbel-max's hash and two logs; temperature 0 is argmax."""

import torch

from vllm.triton_utils import LOG2E, tl, triton
from vllm.v1.worker.gpu.sample.gumbel import (
    _DRAFT_NOISE_SALT,
    murmur3_uniform32,
    spec_step_seed,
)

_BLOCK_SIZE = 2048
_LOG2E = tl.constexpr(LOG2E)


@triton.jit(do_not_specialize=["step"])
def _markov_sample_kernel(
    base_logits_ptr,  # [num_reqs * NUM_STEPS, VOCAB], rows ordered (req, step)
    base_logits_stride,
    bias_ptr,  # [num_reqs, VOCAB]
    bias_stride,
    step,
    idx_mapping_ptr,
    temp_ptr,
    cache_ptr,  # [max_num_reqs, NUM_STEPS, VOCAB], contiguous
    partial_max_ptr,  # [max_num_reqs, BLOCK_P]
    partial_sum_ptr,
    NUM_STEPS: tl.constexpr,
    VOCAB: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    req = tl.program_id(0).to(tl.int64)
    block_idx = tl.program_id(1)
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < VOCAB
    row = req * NUM_STEPS + step
    base = tl.load(
        base_logits_ptr + row * base_logits_stride + block,
        mask=mask,
        other=-float("inf"),
    )
    bias = tl.load(bias_ptr + req * bias_stride + block, mask=mask, other=0.0)
    # Rounded once to the cache dtype, as the loop's base_logits + bias.
    logits = (base.to(tl.float32) + bias.to(tl.float32)).to(cache_ptr.dtype.element_ty)

    req_state_idx = tl.load(idx_mapping_ptr + row).to(tl.int64)
    is_valid = req_state_idx >= 0
    cache_row = cache_ptr + (req_state_idx * NUM_STEPS + step) * VOCAB
    tl.store(cache_row + block, logits, mask=mask & is_valid)
    temp = tl.load(temp_ptr + req_state_idx, mask=is_valid, other=0.0).to(tl.float32)
    logits = logits.to(tl.float32)
    out = req * BLOCK_P + block_idx
    if temp != 0.0:
        scaled = logits * (_LOG2E / temp)
        block_max = tl.max(scaled, axis=0)
        tl.store(partial_max_ptr + out, block_max)
        tl.store(partial_sum_ptr + out, tl.sum(tl.exp2(scaled - block_max), axis=0))
    else:
        value, idx = tl.max(logits, axis=0, return_indices=True)
        tl.store(partial_max_ptr + out, value)
        # Temperature 0 keeps the block's argmax in partial_sum (exact below 2^24).
        tl.store(partial_sum_ptr + out, (block_idx * BLOCK_SIZE + idx).to(tl.float32))


@triton.jit(do_not_specialize=["step"])
def _markov_select_kernel(
    step,
    idx_mapping_ptr,
    sample_pos_ptr,
    temp_ptr,
    seeds_ptr,
    cache_ptr,
    draft_tokens_ptr,
    draft_tokens_stride,
    embed_weight_ptr,  # [VOCAB, RANK] markov_w1
    embed_ptr,  # [max_num_reqs, RANK], the next step's bias GEMM input
    partial_max_ptr,
    partial_sum_ptr,
    NUM_STEPS: tl.constexpr,
    VOCAB: tl.constexpr,
    RANK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    req = tl.program_id(0).to(tl.int64)
    row = req * NUM_STEPS + step
    req_state_idx = tl.load(idx_mapping_ptr + row).to(tl.int64)
    is_valid = req_state_idx >= 0
    temp = tl.load(temp_ptr + req_state_idx, mask=is_valid, other=0.0).to(tl.float32)
    p = tl.arange(0, BLOCK_P)
    p_mask = p < tl.cdiv(VOCAB, BLOCK_SIZE)
    values = tl.load(
        partial_max_ptr + req * BLOCK_P + p, mask=p_mask, other=-float("inf")
    )
    sums = tl.load(partial_sum_ptr + req * BLOCK_P + p, mask=p_mask, other=0.0)
    if temp != 0.0:
        seed = spec_step_seed(
            tl.load(seeds_ptr + req_state_idx, mask=is_valid, other=0), step
        )
        # Keyed by the position before the predicted token, as gumbel_sample.
        pos = tl.load(sample_pos_ptr + row) - 1 + _DRAFT_NOISE_SALT
        top = tl.max(values, axis=0)
        mass = tl.where(sums > 0, sums * tl.exp2(values - top), 0.0)
        cum = tl.cumsum(mass, axis=0)
        target = murmur3_uniform32(seed, pos, tl.full((), VOCAB, tl.int32)) * tl.sum(
            mass, axis=0
        )
        # A rounding overshoot past the last block takes the last nonzero one.
        last = tl.max(tl.where(mass > 0, p, 0), axis=0)
        blk = tl.minimum(tl.sum((cum <= target).to(tl.int32)), last)
        at_blk = p == blk
        blk_max = tl.sum(tl.where(at_blk, values, 0.0))
        # The mass left inside the block, in units of its own weights.
        target = (target - tl.sum(tl.where(at_blk, cum - mass, 0.0))) * tl.exp2(
            top - blk_max
        )
        i = tl.arange(0, BLOCK_SIZE)
        offs = blk * BLOCK_SIZE + i
        cache_row = cache_ptr + (req_state_idx * NUM_STEPS + step) * VOCAB
        cached = tl.load(
            cache_row + offs, mask=(offs < VOCAB) & is_valid, other=-float("inf")
        )
        # The same fp32 weights _markov_sample_kernel summed for this block.
        weights = tl.exp2(cached.to(tl.float32) * (_LOG2E / temp) - blk_max)
        last_tok = tl.max(tl.where(weights > 0, i, 0), axis=0)
        idx = tl.sum((tl.cumsum(weights, axis=0) <= target).to(tl.int32))
        token = (blk * BLOCK_SIZE + tl.minimum(idx, last_tok)).to(tl.int64)
    else:
        winner = tl.argmax(values, axis=0, tie_break_left=True)
        token = tl.sum(tl.where(p == winner, sums, 0.0)).to(tl.int64)
    tl.store(draft_tokens_ptr + req * draft_tokens_stride + step, token)
    r = tl.arange(0, RANK)
    tl.store(embed_ptr + req * RANK + r, tl.load(embed_weight_ptr + token * RANK + r))


class FusedMarkovSampler:
    """``DSparkSpeculator._sample_sequential``'s hook. Its scratch is allocated
    at model init, so CUDA graphs keep the addresses."""

    def __init__(
        self, max_num_reqs: int, vocab_size: int, rank: int, dtype: torch.dtype
    ):
        assert rank & (rank - 1) == 0, rank
        self.vocab_size, self.rank = vocab_size, rank
        self.num_blocks = triton.cdiv(vocab_size, _BLOCK_SIZE)
        self.block_p = triton.next_power_of_2(self.num_blocks)
        self.markov_embed = torch.empty(max_num_reqs, rank, dtype=dtype)
        self.partial_max = torch.empty(max_num_reqs, self.block_p, dtype=torch.float32)
        self.partial_sum = torch.empty_like(self.partial_max)

    def __call__(self, spec, num_reqs: int, head_hidden: torch.Tensor) -> bool:
        model, n_spec = spec.model, spec.num_speculative_steps
        num_sample = num_reqs * n_spec
        base_logits = model.compute_draft_logits(
            head_hidden[spec.sample_indices[:num_sample]]
        )
        markov_embed = model.markov_embed(
            spec.input_buffers.input_ids[spec._anchor_idx[:num_reqs]]
        )
        consts = dict(
            NUM_STEPS=n_spec,
            VOCAB=self.vocab_size,
            BLOCK_SIZE=_BLOCK_SIZE,
            BLOCK_P=self.block_p,
        )
        for step in range(n_spec):
            bias = model.markov_bias(markov_embed)
            _markov_sample_kernel[(num_reqs, self.num_blocks)](
                base_logits,
                base_logits.stride(0),
                bias,
                bias.stride(0),
                step,
                spec.sample_idx_mapping,
                spec.temperature,
                spec.draft_logits.values,
                self.partial_max,
                self.partial_sum,
                **consts,
                num_warps=2,
            )
            _markov_select_kernel[(num_reqs,)](
                step,
                spec.sample_idx_mapping,
                spec.sample_pos,
                spec.temperature,
                spec.seeds,
                spec.draft_logits.values,
                spec.draft_tokens,
                spec.draft_tokens.stride(0),
                model.model.markov_head.markov_w1.weight,
                self.markov_embed,
                self.partial_max,
                self.partial_sum,
                RANK=self.rank,
                **consts,
            )
            markov_embed = self.markov_embed[:num_reqs]
        return True
