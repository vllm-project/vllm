# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark top-k Markov walk in one launch.

Equivalent to the ``_sample_sequential_topk`` loop for an unquantized Markov
head: per request, correct the top-k base logits with ``W2[ids] @ W1[prev]``,
cache the corrected candidates and sample one of them, for every draft step.
The draft logits cache keeps ``-inf`` outside the candidates by clearing the
ids written by the previous walk on the same request slot, so no full-vocab
pass remains after the top-k.
"""

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.sample.batch_shard import shard_owner, shard_row
from vllm.v1.worker.gpu.sample.gumbel import gumbel_noised_logits, spec_step_seed


@triton.jit
def _topk_markov_walk_kernel(
    base_values_ptr,
    candidate_ids_ptr,
    anchor_ids_ptr,
    markov_w1_ptr,
    markov_w1_stride,
    markov_w2_ptr,
    markov_w2_stride,
    idx_mapping_ptr,
    sample_pos_ptr,
    temperature_ptr,
    seeds_ptr,
    draft_tokens_ptr,
    draft_tokens_stride,
    draft_logits_ptr,
    draft_logits_stride_0,
    draft_logits_stride_1,
    cached_ids_ptr,
    scale,
    RANK: tl.constexpr,
    NUM_STEPS: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_R: tl.constexpr,
    SAMPLE_PROBABILISTIC: tl.constexpr,
    USE_FP64: tl.constexpr,
    CACHE_NUM_SHARDS: tl.constexpr,
    CACHE_SHARD_IDX: tl.constexpr,
):
    row = tl.program_id(0)
    k = tl.arange(0, BLOCK_K)
    k_mask = k < TOP_K
    r = tl.arange(0, BLOCK_R)
    req_state = tl.load(idx_mapping_ptr + row * NUM_STEPS).to(tl.int64)
    valid = req_state >= 0
    # A batch-sharded cache holds only this rank's requests, packed by row.
    cache_mask = (
        k_mask & valid & (shard_owner(req_state, CACHE_NUM_SHARDS) == CACHE_SHARD_IDX)
    )
    cache_row = shard_row(req_state, CACHE_NUM_SHARDS) * draft_logits_stride_0
    if SAMPLE_PROBABILISTIC:
        temperature = tl.load(temperature_ptr + req_state, mask=valid, other=0.0)
        seed = tl.load(seeds_ptr + req_state, mask=valid, other=0)
        for step in range(NUM_STEPS):
            cached = cached_ids_ptr + (req_state * NUM_STEPS + step) * TOP_K + k
            stale_ids = tl.load(cached, mask=cache_mask, other=0)
            tl.store(
                draft_logits_ptr + cache_row + step * draft_logits_stride_1 + stale_ids,
                float("-inf"),
                mask=cache_mask,
            )
        # A candidate id can be both stale and new; clear before writing.
        tl.debug_barrier()
    else:
        temperature = 0.0
        seed = 0

    prev = tl.load(anchor_ids_ptr + row).to(tl.int64)
    r_mask = r < RANK
    # W2[ids] does not depend on the walk, so each step prefetches the next
    # step's tile; only the W1[prev] load stays on the serial chain.
    next_ids = tl.load(
        candidate_ids_ptr + row * NUM_STEPS * TOP_K + k, mask=k_mask, other=0
    )
    next_w = tl.load(
        markov_w2_ptr + next_ids.to(tl.int64)[:, None] * markov_w2_stride + r[None, :],
        mask=k_mask[:, None] & r_mask[None, :],
        other=0.0,
    )
    for step in range(NUM_STEPS):
        flat = row * NUM_STEPS + step
        ids = next_ids
        weight = next_w
        values = tl.load(base_values_ptr + flat * TOP_K + k, mask=k_mask, other=0.0)
        has_next = step + 1 < NUM_STEPS
        next_ids = tl.load(
            candidate_ids_ptr + (flat + 1) * TOP_K + k, mask=k_mask & has_next, other=0
        )
        next_w = tl.load(
            markov_w2_ptr
            + next_ids.to(tl.int64)[:, None] * markov_w2_stride
            + r[None, :],
            mask=(k_mask & has_next)[:, None] & r_mask[None, :],
            other=0.0,
        )
        embed = tl.load(markov_w1_ptr + prev * markov_w1_stride + r, mask=r_mask)
        bias = tl.sum(weight.to(tl.float32) * embed.to(tl.float32)[None, :], axis=1)
        # Round once to the cache dtype, as baddbmm does, and sample exactly the
        # value the rejection sampler will read back.
        logits = (values.to(tl.float32) + scale * bias).to(values.dtype)

        if SAMPLE_PROBABILISTIC:
            tl.store(
                draft_logits_ptr + cache_row + step * draft_logits_stride_1 + ids,
                logits,
                mask=cache_mask,
            )
            tl.store(
                cached_ids_ptr + (req_state * NUM_STEPS + step) * TOP_K + k,
                ids,
                mask=cache_mask,
            )

        # sample_pos is the predicted token's position P. Sampling keys a draw
        # by the position before the sampled token, P-1.
        pos = tl.load(sample_pos_ptr + flat) - 1
        scores = gumbel_noised_logits(
            logits.to(tl.float32),
            ids,
            k_mask,
            spec_step_seed(seed, step),
            pos,
            temperature,
            IS_DRAFTING=True,
            USE_FP64=USE_FP64,
        )
        scores = tl.where(k_mask, scores, float("-inf"))
        best, best_k = tl.max(scores, axis=0, return_indices=True)
        # Ties go to the smallest token id, as in the dense argmax. A NaN score
        # matches nothing; then keep the max's own candidate so prev stays in
        # range.
        is_best = k_mask & (scores == best)
        tied_min = tl.min(tl.where(is_best, ids, 1 << 62), axis=0)
        best_id = tl.sum(tl.where(k == best_k, ids, 0), axis=0)
        any_best = tl.sum(is_best.to(tl.int32), axis=0) > 0
        prev = tl.where(any_best, tied_min, best_id).to(tl.int64)
        tl.store(draft_tokens_ptr + row * draft_tokens_stride + step, prev)


class TopkMarkovWalk:
    """Runs the DSpark top-k Markov walk and keeps the cached candidate ids."""

    # One program holds two [top_k, rank] W2 tiles; larger ones spill.
    MAX_TILE = 128 * 256

    @classmethod
    def supports(cls, top_k: int, rank: int) -> bool:
        return (
            triton.next_power_of_2(top_k) * triton.next_power_of_2(rank) <= cls.MAX_TILE
        )

    def __init__(
        self,
        markov_w1: torch.nn.Module,
        markov_w2: torch.nn.Module,
        scale: float,
        max_num_reqs: int,
        num_steps: int,
        top_k: int,
        device: torch.device,
    ):
        # Modules, not tensors: weight reloads may rebind their parameters.
        self.markov_w1 = markov_w1
        self.markov_w2 = markov_w2
        self.scale = scale
        self.num_steps = num_steps
        self.top_k = top_k
        # Points at column 0 of every row until the first walk on a slot, which
        # matches the cache's -inf fill.
        self.cached_ids = torch.zeros(
            max_num_reqs, num_steps, top_k, dtype=torch.int64, device=device
        )

    def sample(
        self,
        base_values: torch.Tensor,  # [num_reqs, num_steps, top_k]
        candidate_ids: torch.Tensor,  # [num_reqs, num_steps, top_k]
        anchor_ids: torch.Tensor,  # [num_reqs]
        idx_mapping: torch.Tensor,  # [num_reqs * num_steps]
        sample_pos: torch.Tensor,  # [num_reqs * num_steps]
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        draft_tokens: torch.Tensor,  # [>= num_reqs, num_steps]
        draft_logits: torch.Tensor | None,
        use_fp64: bool,
        draft_logits_num_shards: int = 1,
        draft_logits_shard_idx: int = 0,
    ) -> None:
        num_reqs = base_values.shape[0]
        markov_w1 = self.markov_w1.weight
        markov_w2 = self.markov_w2.weight
        rank = markov_w1.shape[1]
        _topk_markov_walk_kernel[(num_reqs,)](
            base_values.contiguous(),
            candidate_ids.contiguous(),
            anchor_ids.contiguous(),
            markov_w1,
            markov_w1.stride(0),
            markov_w2,
            markov_w2.stride(0),
            idx_mapping.contiguous(),
            sample_pos.contiguous(),
            temperature,
            seeds,
            draft_tokens,
            draft_tokens.stride(0),
            draft_logits,
            draft_logits.stride(0) if draft_logits is not None else 0,
            draft_logits.stride(1) if draft_logits is not None else 0,
            self.cached_ids,
            self.scale,
            RANK=rank,
            NUM_STEPS=self.num_steps,
            TOP_K=self.top_k,
            BLOCK_K=triton.next_power_of_2(self.top_k),
            BLOCK_R=triton.next_power_of_2(rank),
            SAMPLE_PROBABILISTIC=draft_logits is not None,
            USE_FP64=use_fp64,
            CACHE_NUM_SHARDS=draft_logits_num_shards,
            CACHE_SHARD_IDX=draft_logits_shard_idx,
            num_warps=8,
        )
