# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused DSpark draft sampling epilogue.

The stock per-draft-position sampling tail (``DSparkSpeculator.
_sample_logits`` probabilistic path, full-vocab drafts such as
DeepSeek-V4.1-Flash: ``draft_id_to_target_id is None``) runs, for every
one of the ``num_speculative_steps`` (=5) positions per decode step:

    sample_pos[:, i] - 1          # tensor subtract  -> 1 kernel
    idx_map[:, i].contiguous()    # strided-view copy (inside gumbel_sample)
    gumbel_sample(...)            # the 127-block noise+argmax kernel
                                  #   + full-row logits-cache store
    local_max.argmax(dim=-1)      # phase-2 reduce   -> 1 kernel
    local_argmax.gather(...)      # winner gather    -> 1 kernel
    draft_tokens[:, i] = sampled  # slice copy       -> 1 kernel

~5 kernel launches per position (~25 per decode step). This module fuses
the whole tail into ONE Triton kernel per draft position:

  * grid ``(num_tokens, num_blocks)`` with ``BLOCK_SIZE=1024`` — the
    SAME block decomposition as the stock ``_gumbel_sample_kernel``, so
    every per-block ``tl.max(..., return_indices=True)`` is the identical
    operation on the identical values (the within-block tie behavior is
    inherited bit-for-bit);
  * the per-position ``sample_pos - 1`` subtract and the
    ``idx_map``/``sample_pos`` strided-view contiguity copies disappear
    (the kernel indexes both with their row strides);
  * the noise/temperature/cache-store arithmetic is the stock code path
    verbatim — the kernel calls the SAME ``@triton.jit`` helpers from
    ``vllm.v1.worker.gpu.sample.gumbel`` (``spec_step_seed``,
    ``murmur3_uniform32/64``, ``_log1p_neg_stable``, the draft noise
    salt), applies the temperature division on the fp32-converted logits
    in the same order, and stores the RAW pre-temperature values into
    the logits cache at the same addresses;
  * phase 2 (``local_max.argmax`` + gather + the ``draft_tokens`` slice
    copy) folds into the same launch: each program writes its
    (value, token_id) partial, then bumps a per-row arrival counter with
    an acq_rel atomic; the LAST program to arrive for a row performs the
    final reduction — first-block-wins over the block maxima, exactly
    ``torch.argmax(dim=-1)`` + ``gather`` semantics — and writes both the
    sampled id buffer (the returned ``prev`` for the next Markov step)
    and ``draft_tokens[row, step]`` directly.

BIT-EXACTNESS CONTRACT: sampled ids, the full logits-cache rows, and
``draft_tokens`` are bit-identical to the stock chain (same ops, same
order, same tie-breaks; the atomic only selects WHICH program runs the
fixed deterministic reduction, never the result). The arrival counter is
self-cleaning (the last program resets it to 0), so the kernel is
CUDA-graph capture/replay safe (fixed scratch addresses, state returns
to zero after every call). Fixed-seed maxdiff==0 is pinned by
``tests/v1/spec_decode/test_dspark_sample_epilogue_fusion.py``.

Eligibility (decided once at ``load_draft_model``): probabilistic
drafting (``draft_logits`` allocated), a FULL-VOCAB draft
(``_d2t_scatter_index is None`` — reduced-vocab models keep the stock
scatter chain), and no draft watermarker (its sampler is a different
code path). The greedy no-cache path and the nucleus mask are
untouched: the mask still applies in place BEFORE this kernel (the
source contract "estimator read -> mask -> epilogue" is preserved; the
kernel consumes the row as given).

Kill switch: ``VLLM_DISABLE_DRAFT_SAMPLE_EPILOGUE_FUSION=1`` (read per
call via os.environ, so eager in-process A/B works and the choice is
baked at CUDA-graph capture). Log lines:
"DSpark draft sample-epilogue fusion READY/engaged/is DISABLED/gated
off".

Draft-batch gate: like the fused markov GEMV it pairs with, the fusion
wins by launch elimination at tiny draft batches and loses to the stock
sampler at large ones (measured crossover on MI355X between 2 and 4
concurrent requests; the ship shape is 1). It engages at num draft rows
<= VLLM_DRAFT_SAMPLING_FUSION_MAX_TOKENS (default 2, shared with the
markov GEMV) and falls back to the stock chain above. Read per call;
under CUDA graphs the choice is baked per captured size.
"""

import os

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.sample.gumbel import (
    _DRAFT_NOISE_SALT,
    _log1p_neg_stable,
    murmur3_uniform32,
    murmur3_uniform64,
    spec_step_seed,
)

logger = init_logger(__name__)


def sample_epilogue_fusion_disabled() -> bool:
    return os.environ.get("VLLM_DISABLE_DRAFT_SAMPLE_EPILOGUE_FUSION", "0") == "1"


def draft_sampling_fusion_max_tokens() -> int:
    """Draft-batch gate shared by the fused draft-sampling levers (this
    sample epilogue and the markov GEMV): engage at num draft rows <=
    this value, fall back to the stock chain above."""
    return int(os.environ.get("VLLM_DRAFT_SAMPLING_FUSION_MAX_TOKENS", "2"))


# Must equal gumbel.py's BLOCK_SIZE: the per-block tl.max reduction tree
# (and therefore its exact-tie behavior) is inherited from the stock
# kernel only at the same block width.
_BLOCK_SIZE = 1024


@triton.jit
def _dspark_sample_epilogue_kernel(
    logits_ptr,  # [T, V] raw input (post-markov, post-mask), any row stride
    logits_stride,
    idx_map_ptr,  # [T] -> request slot (int64, -1 = inert), any row stride
    idx_map_stride,
    temp_ptr,  # [max_num_reqs] fp32
    seeds_ptr,  # [max_num_reqs] int64
    pos_ptr,  # [T] int64 sample positions (UN-offset; kernel applies - 1)
    pos_stride,
    logits_cache_ptr,  # [max_num_reqs, num_cols, V]
    logits_cache_stride_0,
    logits_cache_stride_1,
    logits_cache_col_ptr,  # 0-d int tensor (the step column)
    partials_val_ptr,  # [T, num_blocks] block maxima
    partials_idx_ptr,  # [T, num_blocks] block winner token ids (int64)
    partials_stride,
    counter_ptr,  # [T] int32 arrival counters (self-cleaning)
    sampled_ptr,  # [T] int64 out: the returned per-row sampled ids
    draft_tokens_ptr,  # [max_num_reqs, num_cols] int64 (positional rows)
    draft_tokens_stride,
    vocab_size,
    num_blocks,
    APPLY_TEMPERATURE: tl.constexpr,
    USE_FP64: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    block_idx = tl.program_id(1)
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size

    req_state_idx = tl.load(idx_map_ptr + token_idx * idx_map_stride)
    is_valid_req = req_state_idx >= 0
    req64 = req_state_idx.to(tl.int64)
    temp = tl.load(temp_ptr + req64, mask=is_valid_req, other=0.0).to(tl.float32)
    seed = tl.load(seeds_ptr + req64, mask=is_valid_req, other=0)
    col = tl.load(logits_cache_col_ptr)
    seed = spec_step_seed(seed, col)
    pos = tl.load(pos_ptr + token_idx * pos_stride) - 1

    raw = tl.load(
        logits_ptr + token_idx * logits_stride + block,
        mask=mask,
        other=float("-inf"),
    )
    # Cache store: the RAW pre-temperature values, same addresses/mask as
    # the stock kernel's logits_cache_source store.
    tl.store(
        logits_cache_ptr
        + req64 * logits_cache_stride_0
        + col * logits_cache_stride_1
        + block,
        raw,
        mask=mask & is_valid_req,
    )

    # gumbel_noised_argmax, verbatim order: fp32 convert -> temperature
    # divide -> fp64 upcast (optional) -> noise -> -inf outside vocab.
    x = raw.to(tl.float32)
    if temp != 0.0 and APPLY_TEMPERATURE:
        x = x / temp
    if USE_FP64:
        x = x.to(tl.float64)
    if temp != 0.0:
        pos = pos + _DRAFT_NOISE_SALT
        if USE_FP64:
            u = murmur3_uniform64(seed, pos, block)
            gumbel_noise = -tl.log(-tl.log(u))
        else:
            u = murmur3_uniform32(seed, pos, block)
            gumbel_noise = -tl.log(-_log1p_neg_stable(u))
        x = tl.where(mask, x + gumbel_noise, float("-inf"))
    value, idx = tl.max(x, axis=0, return_indices=True)
    token_id = (block_idx * BLOCK_SIZE + idx).to(tl.int64)

    tl.store(partials_val_ptr + token_idx * partials_stride + block_idx, value)
    tl.store(partials_idx_ptr + token_idx * partials_stride + block_idx, token_id)

    # Block barrier BEFORE the counter atomic: the partial stores are
    # spread across every lane of the program, but the scalar atomic
    # executes in a single lane -- the bar.sync gives those stores a
    # happens-before edge into the releasing lane, so the acq_rel's
    # cumulativity covers them (the canonical last-block-reduce fence
    # pattern; without it, cross-lane store visibility is codegen luck).
    tl.debug_barrier()
    # Arrival counter: the release half of the acq_rel makes every prior
    # partial store visible; the last arriver's acquire sees them all.
    arrived = tl.atomic_add(counter_ptr + token_idx, 1, sem="acq_rel", scope="gpu")
    if arrived == num_blocks - 1:
        # Self-clean for the next launch (no other program touches this
        # counter in this call; kernel boundaries order the reset).
        tl.store(counter_ptr + token_idx, 0)
        # Phase 2, exactly local_max.argmax(dim=-1) + gather: the FIRST
        # block index achieving the row max wins (torch.argmax semantics),
        # then that block's per-block argmax id (stock tl.max tie behavior).
        offs = tl.arange(0, BLOCK_P)
        pmask = offs < num_blocks
        vals = tl.load(
            partials_val_ptr + token_idx * partials_stride + offs,
            mask=pmask,
            other=float("-inf"),
        )
        mx = tl.max(vals, axis=0)
        winners = tl.where((vals == mx) & pmask, offs, 1 << 30)
        winner_block = tl.minimum(tl.min(winners, axis=0), num_blocks - 1)
        # NaN inputs are out of contract (a healthy draft row has none), but
        # the clamp keeps even an all-NaN row's gather in bounds -- divergence
        # there is preferred to an out-of-bounds read.
        winner_id = tl.load(
            partials_idx_ptr + token_idx * partials_stride + winner_block
        )
        tl.store(sampled_ptr + token_idx, winner_id)
        tl.store(draft_tokens_ptr + token_idx * draft_tokens_stride + col, winner_id)


class SampleEpilogueScratch:
    """Persistent scratch for the fused epilogue.

    Fixed addresses for CUDA-graph capture; the counters are zeroed at
    allocation and self-clean to zero after every launch.
    """

    def __init__(
        self,
        max_rows: int,
        vocab_size: int,
        device: torch.device,
        use_fp64: bool,
    ):
        val_dtype = torch.float64 if use_fp64 else torch.float32
        num_blocks = triton.cdiv(vocab_size, _BLOCK_SIZE)
        self.num_blocks = num_blocks
        self.block_p = triton.next_power_of_2(num_blocks)
        self.partials_val = torch.empty(
            (max_rows, num_blocks), dtype=val_dtype, device=device
        )
        self.partials_idx = torch.empty(
            (max_rows, num_blocks), dtype=torch.int64, device=device
        )
        self.counters = torch.zeros((max_rows,), dtype=torch.int32, device=device)
        self.sampled = torch.full((max_rows,), -1, dtype=torch.int64, device=device)


def dspark_sample_epilogue_fused(
    logits: torch.Tensor,  # [T, V] (any row stride, last dim contiguous)
    idx_map: torch.Tensor,  # [T] request-slot mapping (any row stride)
    temperature: torch.Tensor,  # [max_num_reqs]
    seeds: torch.Tensor,  # [max_num_reqs]
    sample_pos: torch.Tensor,  # [T] UN-offset positions (kernel applies -1)
    logits_cache: torch.Tensor,  # [max_num_reqs, num_cols, V]
    logits_cache_col: torch.Tensor,  # 0-d int tensor (the step column)
    draft_tokens: torch.Tensor,  # [max_num_reqs, num_cols] int64
    scratch: SampleEpilogueScratch,
    use_fp64: bool = False,
) -> torch.Tensor | None:
    """One-launch draft sampling tail; returns the [T] sampled-id tensor
    (a view of persistent scratch), or ``None`` when the kill switch is
    set or the draft batch exceeds the fusion gate (the caller must then
    run the stock chain)."""
    if sample_epilogue_fusion_disabled():
        logger.warning_once(
            "DSpark draft sample-epilogue fusion is DISABLED (env override "
            "VLLM_DISABLE_DRAFT_SAMPLE_EPILOGUE_FUSION=1); using the stock "
            "gumbel sampling chain"
        )
        return None
    max_tokens = draft_sampling_fusion_max_tokens()
    if logits.shape[0] > max_tokens:
        logger.warning_once(
            "DSpark draft sample-epilogue fusion gated off (draft rows "
            "%d > %d, VLLM_DRAFT_SAMPLING_FUSION_MAX_TOKENS; large draft "
            "batches keep the stock sampler); using the stock gumbel "
            "sampling chain",
            logits.shape[0],
            max_tokens,
        )
        return None
    num_tokens, vocab_size = logits.shape
    num_blocks = triton.cdiv(vocab_size, _BLOCK_SIZE)
    assert logits.stride(-1) == 1, "logits must be contiguous in the vocab dim"
    assert logits.dtype == logits_cache.dtype, (
        "logits and the draft logits cache must share a dtype"
    )
    assert logits_cache.size(-1) >= vocab_size, (
        "draft logits cache vocab dim is narrower than the sampled logits"
    )
    assert num_blocks == scratch.num_blocks, (
        f"scratch was built for {scratch.num_blocks} blocks; logits need {num_blocks}"
    )
    assert num_tokens <= scratch.counters.numel(), "scratch rows exhausted"
    assert num_tokens <= draft_tokens.shape[0]
    _dspark_sample_epilogue_kernel[(num_tokens, num_blocks)](
        logits,
        logits.stride(0),
        idx_map,
        idx_map.stride(0),
        temperature,
        seeds,
        sample_pos,
        sample_pos.stride(0),
        logits_cache,
        logits_cache.stride(0),
        logits_cache.stride(1),
        logits_cache_col,
        scratch.partials_val,
        scratch.partials_idx,
        scratch.partials_val.stride(0),
        scratch.counters,
        scratch.sampled,
        draft_tokens,
        draft_tokens.stride(0),
        vocab_size,
        num_blocks,
        APPLY_TEMPERATURE=True,
        USE_FP64=use_fp64,
        BLOCK_SIZE=_BLOCK_SIZE,
        BLOCK_P=scratch.block_p,
        num_warps=4,
    )
    logger.warning_once(
        "DSpark draft sample-epilogue fusion engaged: one Triton kernel per "
        "draft position replaces the stock sample_pos-offset + gumbel_sample "
        "+ argmax + gather + draft_tokens-copy chain (bit-exact; disable var "
        "VLLM_DISABLE_DRAFT_SAMPLE_EPILOGUE_FUSION)"
    )
    return scratch.sampled[:num_tokens]
