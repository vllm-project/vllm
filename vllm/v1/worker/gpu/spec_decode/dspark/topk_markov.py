# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Candidate-pruned (top-k) vanilla Markov head for DSpark drafting.

The vanilla DSpark Markov head biases every draft position with the full-vocab
projection ``W1[prev] @ W2^T`` (``[B, r] @ [r, V]``) even though only the few
base-logit candidates that survive pruning can win the position. Evaluating the
head on the top-k candidates instead shrinks that projection to
``[B, k, r] @ [B, r, 1]`` -- for ``k = 16``, ``r = 256`` and a 152k vocabulary
that is ~9500x less linear algebra per position -- and it lets the selection and
the sampling run inside the candidate set (``[B, k]``) rather than over ``V``.

:meth:`markov_walk_topk` fuses the whole sequential block into a single kernel
launch: one program per request walks the ``num_steps`` draft positions, and at
each position it

  * embeds the *previously sampled* token through ``W1`` and gathers the ``W2``
    rows of the current candidates (``rank`` is chunked so the ``[k, r]`` tile
    stays register friendly),
  * adds the scaled correction to the candidates' base logits,
  * picks the winner inside the candidate set only -- plain argmax at temperature
    0, Gumbel-max otherwise, keyed by the *token id* with ``IS_DRAFTING=True``
    so the draft noise stream is disjoint from the target/residual one,
  * stores the resulting real (target-vocab) token id and chains it into the next
    position as the new ``prev``.

The chosen weights are the trained ones; nothing is retrained or re-indexed, so
``markov_topk`` is a pure inference-time knob.

For probabilistic drafting the verifier must see the distribution the drafter
actually sampled from, i.e. the candidate-truncated one renormalized over the
candidates with zero mass outside them. :meth:`cache_markov_candidates` publishes
that: the walk writes its pre-temperature candidate scores and this kernel
scatters them into the dense draft-logit cache, which is pre-filled with ``-inf``
and keeps ``-inf`` for every non-candidate column. Only the ``2 * k`` columns
touched per position are written (the previous candidates are reset first), so
the cache update costs ``O(k)`` instead of ``O(V)`` per step.
"""

import logging
import time

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.sample.gumbel import gumbel_noised_argmax

# Guard against pathological register pressure: the [BLOCK_K, BLOCK_R] gather
# tile is walked in BLOCK_R chunks, and BLOCK_RANK materializes a whole W1 row.
MAX_WALK_RANK = 1024
_WALK_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def walk_is_supported(
    w1: torch.Tensor,
    w2: torch.Tensor,
    *,
    tp_size: int = 1,
) -> bool:
    """Whether the fused candidate walk can read these Markov weights directly.

    The kernel indexes ``W1``/``W2`` rows by token id, so both must be dense,
    replicated (draft TP 1), row-contiguous tensors in a plain floating dtype.
    Quantized or sharded heads keep using the dense full-vocab projection.
    """
    if tp_size != 1 or w1.dim() != 2 or w2.dim() != 2:
        return False
    if w1.stride(1) != 1 or w2.stride(1) != 1:
        return False
    if w1.dtype not in _WALK_DTYPES or w2.dtype not in _WALK_DTYPES:
        return False
    return int(w1.shape[1]) == int(w2.shape[1]) <= MAX_WALK_RANK


@triton.jit
def _markov_walk_kernel(
    # [num_reqs, num_steps, base_top_k] top-k base logits / draft-vocab ids
    cand_values_ptr,
    cand_ids_ptr,
    # [target_vocab, static_m] precomputed bigram top-m draft ids, or nullptr
    static_ids_ptr,
    # [target_vocab, static_m] fp32 precomputed bias values, or nullptr
    static_biases_ptr,
    # [num_reqs * num_steps, draft_vocab] base logits, read by the bigram half
    base_logits_ptr,
    # [num_reqs, num_steps, top_k] int64 out: the union ids actually scored
    union_ids_ptr,
    # [vocab, rank] Markov weights (replicated, row contiguous)
    w1_ptr,
    w2_ptr,
    # [draft_vocab] draft -> target offset table, or nullptr
    d2t_ptr,
    # [num_reqs, num_steps, top_k] fp32 out: pre-temperature candidate scores
    realized_ptr,
    # [num_reqs, num_steps, rank] out: W1 rows of the chained `prev` tokens
    markov_embed_ptr,
    # [num_reqs, num_steps] int64 out: sampled target-vocab token ids
    draft_tokens_ptr,
    input_ids_ptr,
    anchor_indices_ptr,
    sample_pos_ptr,
    req_state_ptr,
    temperature_ptr,
    seeds_ptr,
    draft_tokens_stride,
    base_row_stride,
    w1_stride,
    w2_stride,
    scale,
    rank,
    num_steps: tl.constexpr,
    top_k: tl.constexpr,
    base_top_k: tl.constexpr,
    static_m: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_RANK: tl.constexpr,
    PROBABILISTIC: tl.constexpr,
    STORE_REALIZED: tl.constexpr,
    STORE_EMBED: tl.constexpr,
    STORE_IDS: tl.constexpr,
    HAS_D2T: tl.constexpr,
    HAS_STATIC: tl.constexpr,
    USE_FP64: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, BLOCK_K)
    k_mask = offsets < top_k

    req_state = tl.load(req_state_ptr + row * num_steps).to(tl.int64)
    valid = req_state >= 0

    if valid:
        temperature = tl.load(temperature_ptr + req_state).to(tl.float32)
        seed = tl.load(seeds_ptr + req_state)
        # The anchor (bonus) token seeds the chain: it is the input id of query
        # offset 0 for this request, read through the persistent index buffer so a
        # replayed CUDA graph sees the current batch.
        prev = tl.load(input_ids_ptr + tl.load(anchor_indices_ptr + row)).to(tl.int64)

        for step in range(num_steps):
            flat = row * num_steps + step
            cand_base = flat * top_k

            if HAS_STATIC:
                is_base_col = offsets < base_top_k
                static_col = offsets - base_top_k
                is_static_col = (static_col >= 0) & (static_col < static_m)
                base_idx = tl.where(is_base_col, offsets, 0)
                static_idx = tl.where(is_static_col, static_col, 0)
                base_k_mask = k_mask & is_base_col
                static_k_mask = k_mask & is_static_col
                cand = tl.where(
                    is_base_col,
                    tl.load(
                        cand_ids_ptr + flat * base_top_k + base_idx,
                        mask=base_k_mask,
                        other=0,
                    ).to(tl.int64),
                    tl.load(
                        static_ids_ptr + prev * static_m + static_idx,
                        mask=static_k_mask,
                        other=0,
                    ).to(tl.int64),
                )
                base = tl.where(
                    is_base_col,
                    tl.load(
                        cand_values_ptr + flat * base_top_k + base_idx,
                        mask=base_k_mask,
                        other=0.0,
                    ).to(tl.float32),
                    tl.load(
                        base_logits_ptr + flat * base_row_stride + cand,
                        mask=static_k_mask,
                        other=float("-inf"),
                    ).to(tl.float32),
                ).to(tl.float32)
            else:
                cand = tl.load(cand_ids_ptr + cand_base + offsets, mask=k_mask, other=0)
                base = tl.load(
                    cand_values_ptr + cand_base + offsets,
                    mask=k_mask,
                    other=float("-inf"),
                ).to(tl.float32)
            cand = cand.to(tl.int64)

            # Candidate-only Markov bias.
            if HAS_STATIC:
                # Base candidates: online W1[prev]·W2[cand] projection.
                # Static candidates: precomputed fp32 bias loaded directly from
                # the bigram table, eliminating the W2 row gather and dot
                # product.
                base_bias = tl.zeros([BLOCK_K], dtype=tl.float32)
                for r_start in tl.range(0, rank, BLOCK_R):
                    r_offsets = r_start + tl.arange(0, BLOCK_R)
                    r_mask = r_offsets < rank
                    embed = tl.load(
                        w1_ptr + prev * w1_stride + r_offsets,
                        mask=r_mask,
                        other=0.0,
                    ).to(tl.float32)
                    base_cand_ids = tl.where(is_base_col, cand, 0)
                    weight = tl.load(
                        w2_ptr
                        + base_cand_ids[:, None].to(tl.int64) * w2_stride
                        + r_offsets[None, :],
                        mask=is_base_col[:, None] & k_mask[:, None] & r_mask[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    base_bias += tl.sum(weight * embed[None, :], axis=1)
                static_bias = tl.load(
                    static_biases_ptr + prev * static_m + static_idx,
                    mask=k_mask & is_static_col,
                    other=0.0,
                ).to(tl.float32)
                bias = tl.where(is_base_col, base_bias, static_bias)
            else:
                bias = tl.zeros([BLOCK_K], dtype=tl.float32)
                for r_start in tl.range(0, rank, BLOCK_R):
                    r_offsets = r_start + tl.arange(0, BLOCK_R)
                    r_mask = r_offsets < rank
                    embed = tl.load(
                        w1_ptr + prev * w1_stride + r_offsets,
                        mask=r_mask,
                        other=0.0,
                    ).to(tl.float32)
                    weight = tl.load(
                        w2_ptr
                        + cand[:, None].to(tl.int64) * w2_stride
                        + r_offsets[None, :],
                        mask=k_mask[:, None] & r_mask[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    bias += tl.sum(weight * embed[None, :], axis=1)
            scores = base + bias * scale

            position = tl.load(sample_pos_ptr + flat) - 1
            _, index = gumbel_noised_argmax(
                scores,
                cand,
                k_mask,
                seed,
                position,
                temperature if PROBABILISTIC else 0.0,
                IS_DRAFTING=True,
                USE_FP64=USE_FP64,
            )

            if STORE_IDS:
                tl.store(union_ids_ptr + cand_base + offsets, cand, mask=k_mask)
            if STORE_REALIZED:
                tl.store(realized_ptr + cand_base + offsets, scores, mask=k_mask)
            if STORE_EMBED:
                rank_offsets = tl.arange(0, BLOCK_RANK)
                rank_mask = rank_offsets < rank
                prev_embed = tl.load(
                    w1_ptr + prev * w1_stride + rank_offsets,
                    mask=rank_mask,
                    other=0.0,
                )
                tl.store(
                    markov_embed_ptr + flat * rank + rank_offsets,
                    prev_embed,
                    mask=rank_mask,
                )

            # The winner's id, read out of the register tile the union was
            # built in (dynamic indexing of a vector is a masked reduction in
            # Triton).
            token = tl.sum(tl.where(offsets == index, cand, 0), axis=0)
            if HAS_D2T:
                token = token + tl.load(d2t_ptr + token)
            tl.store(draft_tokens_ptr + row * draft_tokens_stride + step, token)
            prev = token


@triton.jit
def _cache_candidates_kernel(
    # [max_num_reqs, num_steps, vocab] fp32, pre-filled with -inf
    draft_logits_ptr,
    # [max_num_reqs, num_steps, top_k] target ids currently cached per column
    cached_ids_ptr,
    cand_ids_ptr,
    realized_ptr,
    d2t_ptr,
    req_state_ptr,
    draft_logits_stride_0,
    draft_logits_stride_1,
    num_steps: tl.constexpr,
    top_k: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HAS_D2T: tl.constexpr,
):
    flat = tl.program_id(0).to(tl.int64)
    req_state = tl.load(req_state_ptr + flat).to(tl.int64)
    valid = req_state >= 0
    step = flat % num_steps
    offsets = tl.arange(0, BLOCK_K)
    mask = valid & (offsets < top_k)

    # Reset the columns this (request, step) wrote last time, then publish the
    # new candidates: everything else stays at the cache's -inf fill, which is
    # exactly the truncated draft distribution the walk sampled from.
    cache_base = (req_state * num_steps + step) * top_k
    old_ids = tl.load(cached_ids_ptr + cache_base + offsets, mask=mask, other=0)
    logits_base = (
        draft_logits_ptr
        + req_state * draft_logits_stride_0
        + step * draft_logits_stride_1
    )
    tl.store(logits_base + old_ids, float("-inf"), mask=mask)

    cand_base = flat * top_k
    cand = tl.load(cand_ids_ptr + cand_base + offsets, mask=mask, other=0)
    if HAS_D2T:
        cand = cand + tl.load(d2t_ptr + cand, mask=mask, other=0)
    scores = tl.load(realized_ptr + cand_base + offsets, mask=mask, other=float("-inf"))
    tl.store(logits_base + cand, scores, mask=mask)
    tl.store(cached_ids_ptr + cache_base + offsets, cand, mask=mask)


def compute_markov_bias_top_ids(
    w1: torch.Tensor,
    w2: torch.Tensor,
    m: int,
    scale: float = 1.0,
    *,
    chunk: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-``m`` draft ids and fp32 bias values of the bigram bias."""
    if m <= 0:
        raise ValueError("markov_bias_topk must be > 0 to build a bigram table")
    device = w1.device
    target_vocab, rank = w1.shape

    start_time = time.perf_counter()
    # fp32 matmul matches the walk kernel's fp32 accumulation.
    w2t = w2.float().t()
    largest = scale >= 0
    ids = torch.empty((target_vocab, m), dtype=torch.int32, device=device)
    values = torch.empty((target_vocab, m), dtype=torch.float32, device=device)
    for begin in range(0, target_vocab, chunk):
        stop = min(begin + chunk, target_vocab)
        bias = w1[begin:stop].float() @ w2t
        topk = torch.topk(bias, m, dim=-1, largest=largest)
        ids[begin:stop] = topk.indices.to(torch.int32)
        values[begin:stop] = topk.values
        del bias, topk
    elapsed = time.perf_counter() - start_time

    logging.getLogger(__name__).info(
        "DSpark bigram candidate table: built %d x top-%d in %.1fs",
        target_vocab,
        m,
        elapsed,
    )
    return ids, values


def markov_walk_topk(
    *,
    num_reqs: int,
    cand_values: torch.Tensor,  # [num_reqs, num_steps, base_top_k]
    cand_ids: torch.Tensor,  # [num_reqs, num_steps, base_top_k] draft-vocab ids
    w1: torch.Tensor,  # [target_vocab, rank]
    w2: torch.Tensor,  # [draft_vocab, rank]
    scale: float,
    draft_tokens: torch.Tensor,  # [max_num_reqs, num_steps], int64 out
    input_ids: torch.Tensor,  # [max_num_tokens], anchor input ids
    anchor_indices: torch.Tensor,  # [max_num_reqs]
    sample_pos: torch.Tensor,  # [num_reqs * num_steps]
    sample_idx_mapping: torch.Tensor,  # [num_reqs * num_steps], -1 = inert row
    temperature: torch.Tensor,  # [max_num_reqs]
    seeds: torch.Tensor,  # [max_num_reqs]
    static_ids: torch.Tensor | None = None,  # [target_vocab, static_m] int32
    static_biases: torch.Tensor | None = None,  # [target_vocab, static_m] fp32
    base_logits: torch.Tensor | None = None,  # [num_reqs, num_steps, draft_vocab]
    union_ids: torch.Tensor | None = None,  # [num_reqs, num_steps, top_k] int64 out
    d2t: torch.Tensor | None = None,  # [draft_vocab] draft -> target offsets
    realized_scores: torch.Tensor | None = None,  # [num_reqs, num_steps, top_k] fp32
    markov_embeds: torch.Tensor | None = None,  # [num_reqs, num_steps, rank]
    probabilistic: bool = False,
    use_fp64: bool = False,
    num_warps: int = 4,
) -> None:
    """Run the sequential candidate-pruned Markov walk over the whole block.

    Writes the sampled target-vocab token id of every draft position into
    ``draft_tokens[:num_reqs]``; the walk is fully chained, so position ``i``
    reads the token position ``i - 1`` produced. When ``static_ids`` is given the
    candidate set of every position is the union of the base-logit top-k and the
    precomputed bigram top-m of the chained ``prev`` token.

    When ``static_biases`` is provided alongside ``static_ids``, the walk kernel
    loads the precomputed fp32 bigram bias directly for static candidates
    instead of computing ``W1[prev] @ W2[candidate]`` online, eliminating
    scattered ``W2`` row reads for the static half of the candidate set.
    """
    base_top_k = cand_ids.shape[-1]
    num_steps = cand_ids.shape[-2]
    static_m = 0 if static_ids is None else int(static_ids.shape[1])
    top_k = base_top_k + static_m
    rank = int(w1.shape[1])
    if w1.stride(1) != 1 or w2.stride(1) != 1:
        raise ValueError("Markov head weights must be row contiguous.")
    if static_m > 0 and base_logits is not None and not base_logits.is_contiguous():
        raise ValueError("base_logits must be contiguous for the bigram gather.")

    # Warmup/CUDA-graph capture execute padded rows; `sample_idx_mapping` marks
    # them with -1 so they neither sample nor write.
    _markov_walk_kernel[(num_reqs,)](
        cand_values.contiguous(),
        cand_ids.contiguous(),
        static_ids if static_ids is not None else cand_ids,
        static_biases if static_biases is not None else cand_values,
        base_logits if base_logits is not None else cand_values,
        union_ids if union_ids is not None else cand_ids,
        w1,
        w2,
        d2t if d2t is not None else cand_ids,
        realized_scores if realized_scores is not None else cand_values,
        markov_embeds if markov_embeds is not None else cand_values,
        draft_tokens,
        input_ids,
        anchor_indices,
        sample_pos.contiguous(),
        sample_idx_mapping.contiguous(),
        temperature,
        seeds,
        draft_tokens.stride(0),
        base_logits.shape[-1] if base_logits is not None else 1,
        w1.stride(0),
        w2.stride(0),
        float(scale),
        rank,
        num_steps=num_steps,
        top_k=top_k,
        base_top_k=base_top_k,
        static_m=static_m,
        BLOCK_K=triton.next_power_of_2(top_k),
        BLOCK_R=min(triton.next_power_of_2(rank), 64),
        BLOCK_RANK=triton.next_power_of_2(rank),
        PROBABILISTIC=probabilistic,
        STORE_REALIZED=realized_scores is not None,
        STORE_EMBED=markov_embeds is not None,
        STORE_IDS=union_ids is not None,
        HAS_D2T=d2t is not None,
        HAS_STATIC=static_m > 0,
        USE_FP64=use_fp64,
        num_warps=num_warps,
    )


def cache_markov_candidates(
    *,
    draft_logits: torch.Tensor,  # [max_num_reqs, num_steps, vocab], -inf filled
    cached_ids: torch.Tensor,  # [max_num_reqs, num_steps, top_k], int64 in/out
    cand_ids: torch.Tensor,  # [num_reqs, num_steps, top_k]
    realized_scores: torch.Tensor,  # [num_reqs, num_steps, top_k] fp32
    sample_idx_mapping: torch.Tensor,  # [num_reqs * num_steps]
    d2t: torch.Tensor | None = None,
    num_warps: int = 1,
) -> None:
    """Publish the truncated draft distribution for probabilistic verification."""
    top_k = cand_ids.shape[-1]
    num_steps = cand_ids.shape[-2]
    _cache_candidates_kernel[(cand_ids.shape[0] * num_steps,)](
        draft_logits,
        cached_ids,
        cand_ids.contiguous(),
        realized_scores.contiguous(),
        d2t if d2t is not None else cand_ids,
        sample_idx_mapping.contiguous(),
        draft_logits.stride(0),
        draft_logits.stride(1),
        num_steps=num_steps,
        top_k=top_k,
        BLOCK_K=triton.next_power_of_2(top_k),
        HAS_D2T=d2t is not None,
        num_warps=num_warps,
    )
