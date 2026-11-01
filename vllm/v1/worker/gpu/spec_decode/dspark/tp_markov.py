# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Vocab-parallel DSpark Markov sampling over the TP group.

The stock Markov loop runs replicated on every TP rank. Each of its
``num_speculative_steps`` steps reads the whole ``markov_w2`` projection
([draft_vocab, rank] bf16, about 100 MB for a 200k vocabulary) for a few rows
of bias, then Gumbel-samples the full vocabulary. Both are memory bound, so
every rank pays the full cost.

Here rank ``r`` owns a contiguous range of the sampler's 1024-column blocks.
Per step it computes the bias of its columns only (a row slice of
``markov_w2``), Gumbel-samples its blocks with the stock noise keys, and
publishes one 8-byte word per block, ``[block max (fp32) | tag | token]``, to
every rank through symmetric memory. Each word carries its own tag, so no
fence or flag is needed (NCCL's LL protocol). Every rank then picks the
winner the way stock does: the first NaN block, else the first maximum.
Block maxima are the values stock computes, so the drafted tokens are
identical.

The draft-logits cache, which the rejection sampler reads at full vocabulary
on every rank, is completed after the loop in one of two ways:

* ``push`` (small batches): during the loop each rank also pushes its biased
  logits, two bf16 per tagged 8-byte word, into its peers' staging buffers;
  one kernel unpacks them after the loop.
* ``recompute`` (larger batches, where pushing would cost more NVLink time
  than it saves): every rank recomputes the peers' columns for all steps at
  once from the sampled tokens, with one projection over all steps. That
  reads ``markov_w2`` once instead of once per step.

Either way the cache equals the stock cache bitwise, provided the projection
slices compute the same values as the full projection (checked by the tests).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

import vllm.envs as envs
from vllm.distributed.parallel_state import get_tp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.sample.gumbel import gumbel_noised_argmax, spec_step_seed

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

logger = init_logger(__name__)

# The stock sampler's block size: block maxima are only comparable to stock
# at this granularity.
_BLOCK_SIZE = 1024
_TOKEN_BITS = 18
_TAG_BITS = 14
# Push the cache slices while sampling up to this many requests; recompute
# them after the loop above it. Also sizes the push staging buffer.
_PUSH_MAX_REQS = 4
# Larger batches run the replicated loop (bounds the recompute's memory and
# the load-time check).
_MAX_REQS = 128
# Bounded spin (about a second) so a missing peer cannot hang the GPU; a
# timeout is counted in `errors`.
_SPIN_LIMIT = 1 << 24


@triton.jit
def _peer_ptr(ptrs, rank, dtype: tl.constexpr):
    return tl.load(ptrs + rank).to(tl.pointer_type(dtype))


@triton.jit
def _tp_markov_sample_kernel(
    word_ptrs,  # int64 [WORLD]: each rank's words, int64 [2, max_reqs, num_blocks]
    stage_ptrs,  # int64 [WORLD]: each rank's push staging (PUSH only)
    cache_ptr,  # [max_reqs, n_spec, vocab]: draft logits cache
    cache_stride_0,
    cache_stride_1,
    logits_ptr,  # [num_reqs, vocab]: base logits of this step
    logits_stride,
    bias_ptr,  # [num_reqs, >= PER * BLOCK_SIZE]: bias of this rank's columns
    bias_stride,
    idx_map_ptr,  # [num_reqs]: request state index (-1: padding)
    seeds_ptr,
    pos_ptr,  # [num_reqs]: draw position
    temp_ptr,
    epoch_ptr,
    step,
    n_spec,
    max_reqs,
    push_max_reqs,
    vocab_size,
    num_blocks,
    RANK: tl.constexpr,
    WORLD: tl.constexpr,
    PER: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    PUSH: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    j = tl.program_id(1)
    block_idx = RANK * PER + j
    if block_idx >= num_blocks:
        return
    e = tl.load(epoch_ptr).to(tl.int64) + step + 1
    cols = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = cols < vocab_size
    logits = tl.load(
        logits_ptr + row * logits_stride + cols, mask=mask, other=float("-inf")
    )
    bias = tl.load(
        bias_ptr + row * bias_stride + j * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE),
        mask=mask,
        other=0.0,
    )
    # Rounded to the logits dtype, as `logits + bias` is in the stock loop.
    logits = (logits.to(tl.float32) + bias.to(tl.float32)).to(logits.dtype)

    req = tl.load(idx_map_ptr + row).to(tl.int64)
    valid = req >= 0
    tl.store(
        cache_ptr + req * cache_stride_0 + step * cache_stride_1 + cols,
        logits,
        mask=mask & valid,
    )
    if PUSH:
        # Pairs of bf16 with the 32-bit epoch as the tag, into every peer's
        # staging slot [parity, source rank, row, step, column pair].
        HALF: tl.constexpr = BLOCK_SIZE // 2
        bits = logits.to(tl.int16, bitcast=True).to(tl.int64) & 0xFFFF
        lo, hi = tl.split(tl.reshape(bits, (HALF, 2)))
        cword = ((e & 0xFFFFFFFF) << 32) | (hi << 16) | lo
        draft_step = (e - 1) // n_spec
        par = draft_step & 1
        soff = (
            (((par * WORLD + RANK) * push_max_reqs + row) * n_spec + step)
            * (PER * HALF)
            + j * HALF
            + tl.arange(0, HALF)
        )
        for d in tl.static_range(WORLD):
            if d != RANK:
                tl.store(_peer_ptr(stage_ptrs, d, tl.int64) + soff, cword, mask=valid)

    temp = tl.load(temp_ptr + req, mask=valid, other=0.0).to(tl.float32)
    seed = tl.load(seeds_ptr + req, mask=valid, other=0)
    seed = spec_step_seed(seed, step)
    pos = tl.load(pos_ptr + row)
    value, idx = gumbel_noised_argmax(
        logits.to(tl.float32),
        cols,
        mask,
        seed,
        pos,
        temp,
        IS_DRAFTING=True,
        USE_FP64=False,
        APPLY_TEMPERATURE=True,
    )
    token = (block_idx * BLOCK_SIZE + idx).to(tl.int64)
    tag = e & ((1 << 14) - 1)
    vbits = value.to(tl.float32).to(tl.int32, bitcast=True).to(tl.int64) & 0xFFFFFFFF
    word = (vbits << 32) | (tag << 18) | token
    woff = ((e & 1) * max_reqs + row) * num_blocks + block_idx
    for d in tl.static_range(WORLD):
        tl.store(_peer_ptr(word_ptrs, d, tl.int64) + woff, word)


@triton.jit
def _tp_markov_combine_kernel(
    word_ptr,  # local int64 [2, max_reqs, num_blocks]
    out_ptr,  # int64 [num_reqs] (strided): sampled token
    out_stride,
    epoch_ptr,
    err_ptr,
    step,
    max_reqs,
    num_blocks,
    PADDED_NUM_BLOCKS: tl.constexpr,
    SPIN_LIMIT: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    e = tl.load(epoch_ptr).to(tl.int64) + step + 1
    tag = e & ((1 << 14) - 1)
    block = tl.arange(0, PADDED_NUM_BLOCKS)
    mask = block < num_blocks
    ptrs = word_ptr + ((e & 1) * max_reqs + row) * num_blocks + block
    w = tl.load(ptrs, mask=mask, other=0, volatile=True)
    bad = tl.sum((mask & (((w >> 18) & ((1 << 14) - 1)) != tag)).to(tl.int32), axis=0)
    it = 0
    while (bad > 0) & (it < SPIN_LIMIT):
        w = tl.load(ptrs, mask=mask, other=0, volatile=True)
        bad = tl.sum(
            (mask & (((w >> 18) & ((1 << 14) - 1)) != tag)).to(tl.int32), axis=0
        )
        it += 1
    if bad > 0:
        tl.atomic_add(err_ptr, 1)
    values = tl.where(
        mask, (w >> 32).to(tl.int32).to(tl.float32, bitcast=True), float("-inf")
    )
    tokens = w & ((1 << 18) - 1)
    # As torch.argmax over the stock per-block maxima: the first NaN, else the
    # first maximum.
    nan_block = tl.min(tl.where(values != values, block, PADDED_NUM_BLOCKS), axis=0)
    if nan_block < PADDED_NUM_BLOCKS:
        winner = nan_block
    else:
        maximum = tl.max(values, axis=0)
        winner = tl.min(
            tl.where(mask & (values == maximum), block, PADDED_NUM_BLOCKS), axis=0
        )
    sampled = tl.sum(tl.where(block == winner, tokens, 0), axis=0)
    tl.store(out_ptr + row * out_stride, sampled)


@triton.jit
def _tp_markov_unpack_kernel(
    cache_ptr,
    cache_stride_0,
    cache_stride_1,
    stage_ptr,  # local int64 [2, WORLD, push_max_reqs, n_spec, PER * HALF]
    idx_map_ptr,  # [num_reqs] (strided): request state index
    idx_map_stride,
    epoch_ptr,
    err_ptr,
    n_spec,
    push_max_reqs,
    vocab_size,
    RANK: tl.constexpr,
    WORLD: tl.constexpr,
    PER: tl.constexpr,
    HALF: tl.constexpr,
    SPIN_LIMIT: tl.constexpr,
):
    # grid: (num_reqs, n_spec, WORLD * PER), one source block each.
    row = tl.program_id(0).to(tl.int64)
    step = tl.program_id(1).to(tl.int64)
    sb = tl.program_id(2)
    src = sb // PER
    jb = sb % PER
    if src == RANK:
        return
    req = tl.load(idx_map_ptr + row * idx_map_stride).to(tl.int64)
    if req < 0:
        return
    base = tl.load(epoch_ptr).to(tl.int64)
    e = base + step + 1
    par = ((e - 1) // n_spec) & 1
    w = tl.arange(0, HALF)
    c_lo = (src * PER + jb) * (2 * HALF) + 2 * w
    m = c_lo < vocab_size
    ptrs = (
        stage_ptr
        + (((par * WORLD + src) * push_max_reqs + row) * n_spec + step) * (PER * HALF)
        + jb * HALF
        + w
    )
    want = e & 0xFFFFFFFF
    x = tl.load(ptrs, mask=m, other=0, volatile=True)
    bad = tl.sum((m & (((x >> 32) & 0xFFFFFFFF) != want)).to(tl.int32), axis=0)
    it = 0
    while (bad > 0) & (it < SPIN_LIMIT):
        x = tl.load(ptrs, mask=m, other=0, volatile=True)
        bad = tl.sum((m & (((x >> 32) & 0xFFFFFFFF) != want)).to(tl.int32), axis=0)
        it += 1
    if bad > 0:
        tl.atomic_add(err_ptr, 1)
    lo = (x & 0xFFFF).to(tl.int16).to(tl.bfloat16, bitcast=True)
    hi = ((x >> 16) & 0xFFFF).to(tl.int16).to(tl.bfloat16, bitcast=True)
    dst = cache_ptr + req * cache_stride_0 + step * cache_stride_1 + c_lo
    tl.store(dst, lo, mask=m)
    tl.store(dst + 1, hi, mask=m & (c_lo + 1 < vocab_size))


@triton.jit
def _tp_markov_fill_kernel(
    cache_ptr,
    cache_stride_0,
    cache_stride_1,
    logits_ptr,  # [num_reqs * n_spec, vocab]: base logits, (request, step) rows
    logits_stride,
    bias_ptr,  # [num_reqs * n_spec, vocab]: bias, same rows
    bias_stride,
    idx_map_ptr,  # [num_reqs] (strided)
    idx_map_stride,
    n_spec,
    vocab_size,
    own_start,
    own_end,
    BLOCK_SIZE: tl.constexpr,
):
    # grid: (num_reqs * n_spec, cdiv(vocab, BLOCK_SIZE))
    r = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if (tl.program_id(1) * BLOCK_SIZE >= own_start) & (
        tl.program_id(1) * BLOCK_SIZE < own_end
    ):
        return  # this rank's own blocks were cached while sampling
    req = tl.load(idx_map_ptr + (r // n_spec) * idx_map_stride).to(tl.int64)
    if req < 0:
        return
    mask = cols < vocab_size
    logits = tl.load(logits_ptr + r * logits_stride + cols, mask=mask)
    bias = tl.load(bias_ptr + r * bias_stride + cols, mask=mask)
    logits = (logits.to(tl.float32) + bias.to(tl.float32)).to(logits.dtype)
    tl.store(
        cache_ptr + req * cache_stride_0 + (r % n_spec) * cache_stride_1 + cols,
        logits,
        mask=mask,
    )


class TPMarkovSampler:
    """Runs a DSpark speculator's Markov loop vocab-parallel over TP."""

    def __init__(self, speculator: DSparkSpeculator):
        import torch.distributed._symmetric_memory as symm_mem

        group = get_tp_group()
        self.world = group.world_size
        self.rank = group.rank_in_group
        model = speculator.model
        w2 = model.model.markov_head.markov_w2.weight
        # Vocab padding rows (if any) are cut off, as the logits processor does.
        self.vocab_size = min(w2.shape[0], model.logits_processor.org_vocab_size)
        self.w2 = w2[: self.vocab_size]
        self.num_blocks = triton.cdiv(self.vocab_size, _BLOCK_SIZE)
        self.per = triton.cdiv(self.num_blocks, self.world)
        self.col_start = min(self.vocab_size, self.rank * self.per * _BLOCK_SIZE)
        self.col_end = min(self.vocab_size, (self.rank + 1) * self.per * _BLOCK_SIZE)
        # Contiguous rows: F.linear on the slice computes those columns.
        self.w2_shard = w2[self.col_start : self.col_end]
        self.max_reqs = min(speculator.max_num_reqs, _MAX_REQS)
        self.n_spec = speculator.num_speculative_steps
        self.push_max_reqs = min(_PUSH_MAX_REQS, self.max_reqs)
        device = w2.device

        name = group.device_group.group_name
        self.words = symm_mem.empty(
            2 * self.max_reqs * self.num_blocks, dtype=torch.int64, device=device
        )
        half = _BLOCK_SIZE // 2
        self.stage = symm_mem.empty(
            2 * self.world * self.push_max_reqs * self.n_spec * self.per * half,
            dtype=torch.int64,
            device=device,
        )
        # Tags never match a zeroed word: the first epoch is 1.
        self.words.zero_()
        self.stage.zero_()
        handles = [symm_mem.rendezvous(t, name) for t in (self.words, self.stage)]
        self._handles = handles
        self.word_ptrs, self.stage_ptrs = (
            torch.tensor(list(h.buffer_ptrs), dtype=torch.int64, device=device)
            for h in handles
        )
        # Device-side step counter: tags must change on every graph replay.
        self.epoch = torch.zeros(1, dtype=torch.int32, device=device)
        # Exchange timeouts (expected to stay 0).
        self.errors = torch.zeros(1, dtype=torch.int32, device=device)
        torch.accelerator.synchronize()
        torch.distributed.barrier(group=group.cpu_group)

    @staticmethod
    def unsupported_reason(speculator: DSparkSpeculator) -> str | None:
        """Why the speculator's Markov loop cannot run vocab-parallel."""
        model = speculator.model
        tp_size = get_tp_group().world_size
        head = getattr(getattr(model, "model", None), "markov_head", None)
        w2 = getattr(getattr(head, "markov_w2", None), "weight", None)
        processor = getattr(model, "logits_processor", None)
        if not current_platform.is_cuda():
            return "it needs CUDA"
        if tp_size == 1:
            return "tensor parallelism is 1"
        if w2 is None or w2.dtype != torch.bfloat16 or w2.dim() != 2:
            return "markov_w2 is not an unquantized bf16 projection"
        if not isinstance(
            getattr(head.markov_w2, "quant_method", None),
            (UnquantizedEmbeddingMethod, UnquantizedLinearMethod),
        ):
            return "markov_w2 is quantized"
        if processor is None or processor.scale != 1.0 or processor.soft_cap:
            return "the logits processor scales or caps the Markov bias"
        if processor.head_dtype not in (None, torch.bfloat16):
            return "the logits processor changes the head dtype"
        if envs.VLLM_BATCH_INVARIANT:
            return "batch-invariant mode is enabled"
        if processor.org_vocab_size > (1 << _TOKEN_BITS):
            return "the draft vocabulary is too large"
        if speculator._draft_topk is not None:
            return "top-k drafting is enabled"
        if speculator.draft_logits is None:
            return "drafting is greedy"
        if speculator.draft_logits.row_mapping is not None:
            return "the draft logits are batch-sharded"
        if speculator._d2t_scatter_index is not None:
            return "the draft vocabulary is reduced"
        if speculator.draft_watermarker is not None:
            return "draft watermarking is enabled"
        if speculator.acceptance_estimator is not None or (
            speculator.use_confidence_head
        ):
            return "adaptive verification is enabled"
        if speculator.use_fp64_gumbel:
            return "fp64 Gumbel sampling is enabled"
        if hasattr(model, "sample_sequential") or hasattr(
            model, "apply_output_logit_transform"
        ):
            return "the draft model has its own Markov sampling"
        if speculator.draft_logits.values.dtype != torch.bfloat16:
            return "the draft logits cache is not bf16"
        return None

    @staticmethod
    @torch.inference_mode()
    def inexact_projection_reason(speculator: DSparkSpeculator) -> str | None:
        """Why this rank's sliced and batched Markov projections would differ
        from the replicated loop's, or None if they match bitwise.

        The loop computes its columns' bias with ``markov_w2``'s row slice, and
        the recompute path all steps' bias at once, while the replicated loop
        projects every step separately onto the full vocabulary. Bitwise equal
        results depend on the GEMM library picking kernels with the same
        reduction order (it does for the usual, 64-aligned vocabularies), so
        check them on the real weight for every batch size; all TP ranks agree
        on the result.
        """
        group = get_tp_group()
        w2 = speculator.model.model.markov_head.markov_w2.weight
        vocab = min(w2.shape[0], speculator.model.logits_processor.org_vocab_size)
        w2 = w2[:vocab]
        num_blocks = triton.cdiv(vocab, _BLOCK_SIZE)
        per = triton.cdiv(num_blocks, group.world_size)
        c0 = min(vocab, group.rank_in_group * per * _BLOCK_SIZE)
        c1 = min(vocab, (group.rank_in_group + 1) * per * _BLOCK_SIZE)
        n_spec = speculator.num_speculative_steps
        gen = torch.Generator(device=w2.device).manual_seed(0)
        bad = None
        for num_reqs in range(1, min(speculator.max_num_reqs, _MAX_REQS) + 1):
            x = torch.randn(
                n_spec * num_reqs, w2.shape[1], device=w2.device, generator=gen
            ).to(w2.dtype)
            steps = [F.linear(x[i::n_spec].contiguous(), w2) for i in range(n_spec)]
            batched = F.linear(x, w2)
            sliced = F.linear(x[0::n_spec].contiguous(), w2[c0:c1])
            if not torch.equal(sliced, steps[0][:, c0:c1]) or not all(
                torch.equal(batched[i::n_spec], steps[i]) for i in range(n_spec)
            ):
                bad = num_reqs
                break
        flag = torch.tensor([bad or 0], dtype=torch.int64)
        torch.distributed.all_reduce(
            flag, op=torch.distributed.ReduceOp.MAX, group=group.cpu_group
        )
        if flag.item():
            return (
                "the sliced Markov projection is not bitwise equal to the full "
                f"one (batch of {flag.item()} requests)"
            )
        return None

    def sample(
        self,
        speculator: DSparkSpeculator,
        num_reqs: int,
        base_logits: torch.Tensor,  # [num_reqs, n_spec, vocab]
        idx_map: torch.Tensor,  # [num_reqs, n_spec]
        sample_pos: torch.Tensor,  # [num_reqs, n_spec]
    ) -> None:
        model = speculator.model
        n_spec = self.n_spec
        cache = speculator.draft_logits.values
        assert cache is not None
        vocab_size = base_logits.shape[-1]
        push = num_reqs <= self.push_max_reqs
        idx_map_steps = idx_map.t().contiguous()
        draw_pos_steps = (sample_pos.t() - 1).contiguous()
        draft_tokens = speculator.draft_tokens
        anchor = speculator.input_buffers.input_ids[speculator._anchor_idx[:num_reqs]]
        prev = anchor
        padded = triton.next_power_of_2(self.num_blocks)
        for i in range(n_spec):
            bias = F.linear(model.markov_embed(prev), self.w2_shard)
            logits = base_logits[:, i]
            _tp_markov_sample_kernel[(num_reqs, self.per)](
                self.word_ptrs,
                self.stage_ptrs,
                cache,
                cache.stride(0),
                cache.stride(1),
                logits,
                logits.stride(0),
                bias,
                bias.stride(0),
                idx_map_steps[i],
                speculator.seeds,
                draw_pos_steps[i],
                speculator.temperature,
                self.epoch,
                i,
                n_spec,
                self.max_reqs,
                self.push_max_reqs,
                vocab_size,
                self.num_blocks,
                RANK=self.rank,
                WORLD=self.world,
                PER=self.per,
                BLOCK_SIZE=_BLOCK_SIZE,
                PUSH=push,
                num_warps=8,
            )
            out = draft_tokens[:num_reqs, i]
            _tp_markov_combine_kernel[(num_reqs,)](
                self.words,
                out,
                out.stride(0),
                self.epoch,
                self.errors,
                i,
                self.max_reqs,
                self.num_blocks,
                PADDED_NUM_BLOCKS=padded,
                SPIN_LIMIT=_SPIN_LIMIT,
                num_warps=4,
            )
            prev = out

        rows = idx_map[:, 0]
        if push:
            _tp_markov_unpack_kernel[(num_reqs, n_spec, self.world * self.per)](
                cache,
                cache.stride(0),
                cache.stride(1),
                self.stage,
                rows,
                rows.stride(0),
                self.epoch,
                self.errors,
                n_spec,
                self.push_max_reqs,
                vocab_size,
                RANK=self.rank,
                WORLD=self.world,
                PER=self.per,
                HALF=_BLOCK_SIZE // 2,
                SPIN_LIMIT=_SPIN_LIMIT,
                num_warps=4,
            )
        else:
            # Each step's Markov input is the previous step's token.
            prevs = torch.cat(
                [anchor.unsqueeze(1), draft_tokens[:num_reqs, : n_spec - 1]], dim=1
            )
            bias = F.linear(model.markov_embed(prevs.reshape(-1)), self.w2)
            flat = base_logits.reshape(num_reqs * n_spec, vocab_size)
            _tp_markov_fill_kernel[
                (num_reqs * n_spec, triton.cdiv(vocab_size, _BLOCK_SIZE))
            ](
                cache,
                cache.stride(0),
                cache.stride(1),
                flat,
                flat.stride(0),
                bias,
                bias.stride(0),
                rows,
                rows.stride(0),
                n_spec,
                vocab_size,
                self.col_start,
                self.col_end,
                BLOCK_SIZE=_BLOCK_SIZE,
                num_warps=4,
            )
        self.epoch.add_(n_spec)
