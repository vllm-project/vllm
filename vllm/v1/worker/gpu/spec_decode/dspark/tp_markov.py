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
publishes one 8-byte word per block, ``[block max (fp32) | tag | index in the
block]``, to every rank through symmetric memory. Each word carries its own
tag, so no fence or flag is needed (NCCL's LL protocol), and its reader
clears it, so a slot never holds a stale word. Every rank then picks the
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
  reads the peers' rows of ``markov_w2`` once instead of once per step.

Either way the cache equals the stock cache bitwise, provided the projection
slices compute the same values as the stock projection, which is checked on
the real weight at load time.
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
from vllm.utils.math_utils import cdiv
from vllm.v1.worker.gpu.sample.gumbel import gumbel_noised_argmax, spec_step_seed

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

logger = init_logger(__name__)

# The stock sampler's block size: block maxima are only comparable to stock
# at this granularity.
_BLOCK_SIZE = 1024
# An exchange word is [block max (fp32) | tag | index in the block].
_INDEX_BITS = tl.constexpr(10)
_TAG_BITS = tl.constexpr(22)
# Push the cache slices while sampling up to this many requests; recompute
# them after the loop above it. Also sizes the push staging buffer.
_PUSH_MAX_REQS = 4
# Larger batches run the replicated loop (bounds the recompute's memory and
# the load-time check).
_MAX_REQS = 128
# Bounded spin, about 2 s per kernel on GB300 (0.24 us per iteration), so that
# a missing peer cannot hang the GPU. A timeout is counted in `errors`, and the
# next `propose` raises.
_SPIN_LIMIT = 1 << 23


@triton.jit
def _peer_ptr(ptrs, rank, dtype: tl.constexpr):
    return tl.load(ptrs + rank).to(tl.pointer_type(dtype))


@triton.jit
def _tag(e, BITS: tl.constexpr):
    # Never 0, so a cleared (or never written) word does not match.
    return e % ((1 << BITS) - 1) + 1


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
        cword = (_tag(e, 32) << 32) | (hi << 16) | lo
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
    tl.static_assert(BLOCK_SIZE == 1 << _INDEX_BITS)
    vbits = value.to(tl.float32).to(tl.int32, bitcast=True).to(tl.int64) & 0xFFFFFFFF
    word = (vbits << 32) | (_tag(e, _TAG_BITS) << _INDEX_BITS) | idx.to(tl.int64)
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
    tag = _tag(e, _TAG_BITS)
    block = tl.arange(0, PADDED_NUM_BLOCKS)
    mask = block < num_blocks
    ptrs = word_ptr + ((e & 1) * max_reqs + row) * num_blocks + block
    tag_mask: tl.constexpr = (1 << _TAG_BITS) - 1
    w = tl.load(ptrs, mask=mask, other=0, volatile=True)
    bad = tl.sum((mask & (((w >> _INDEX_BITS) & tag_mask) != tag)).to(tl.int32), axis=0)
    it = 0
    while (bad > 0) & (it < SPIN_LIMIT):
        w = tl.load(ptrs, mask=mask, other=0, volatile=True)
        bad = tl.sum(
            (mask & (((w >> _INDEX_BITS) & tag_mask) != tag)).to(tl.int32), axis=0
        )
        it += 1
    if bad > 0:
        tl.atomic_add(err_ptr, 1)
    # Consumed. The slot's next word, two epochs on, is written after this.
    tl.store(ptrs, 0, mask=mask)
    values = tl.where(
        mask, (w >> 32).to(tl.int32).to(tl.float32, bitcast=True), float("-inf")
    )
    tokens = (block.to(tl.int64) << _INDEX_BITS) | (w & ((1 << _INDEX_BITS) - 1))
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
    want = _tag(e, 32)
    x = tl.load(ptrs, mask=m, other=0, volatile=True)
    bad = tl.sum((m & (((x >> 32) & 0xFFFFFFFF) != want)).to(tl.int32), axis=0)
    it = 0
    while (bad > 0) & (it < SPIN_LIMIT):
        x = tl.load(ptrs, mask=m, other=0, volatile=True)
        bad = tl.sum((m & (((x >> 32) & 0xFFFFFFFF) != want)).to(tl.int32), axis=0)
        it += 1
    if bad > 0:
        tl.atomic_add(err_ptr, 1)
    tl.store(ptrs, 0, mask=m)  # consumed
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
    bias_lo_ptr,  # [num_reqs * n_spec, own_start]: bias left of this rank's
    bias_lo_stride,
    bias_hi_ptr,  # [num_reqs * n_spec, vocab - own_end]: and right of it
    bias_hi_stride,
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
    start = tl.program_id(1) * BLOCK_SIZE
    if (start >= own_start) & (start < own_end):
        return  # this rank's own blocks were cached while sampling
    req = tl.load(idx_map_ptr + (r // n_spec) * idx_map_stride).to(tl.int64)
    if req < 0:
        return
    cols = start + tl.arange(0, BLOCK_SIZE)
    mask = cols < vocab_size
    logits = tl.load(logits_ptr + r * logits_stride + cols, mask=mask)
    if start < own_start:
        bias = tl.load(bias_lo_ptr + r * bias_lo_stride + cols, mask=mask)
    else:
        bias = tl.load(bias_hi_ptr + r * bias_hi_stride + cols - own_end, mask=mask)
    logits = (logits.to(tl.float32) + bias.to(tl.float32)).to(logits.dtype)
    tl.store(
        cache_ptr + req * cache_stride_0 + (r % n_spec) * cache_stride_1 + cols,
        logits,
        mask=mask,
    )


class TPMarkovSampler:
    """Runs a DSpark speculator's Markov loop vocab-parallel over TP.

    Build it with `create`, which falls back (returns no sampler) on every TP
    rank alike.
    """

    def __init__(self, speculator: DSparkSpeculator):
        self.speculator = speculator
        self.group = get_tp_group()
        self.world = self.group.world_size
        self.rank = self.group.rank_in_group
        model = speculator.model
        w2 = model.model.markov_head.markov_w2.weight
        # Vocab padding rows (if any) are cut off, as the logits processor does.
        self.vocab_size = min(w2.shape[0], model.logits_processor.org_vocab_size)
        self.w2 = w2[: self.vocab_size]
        self.num_blocks = cdiv(self.vocab_size, _BLOCK_SIZE)
        self.per = cdiv(self.num_blocks, self.world)
        self.col_start = min(self.vocab_size, self.rank * self.per * _BLOCK_SIZE)
        self.col_end = min(self.vocab_size, (self.rank + 1) * self.per * _BLOCK_SIZE)
        # Contiguous rows: F.linear on a row slice computes those columns.
        self.w2_shard = self.w2[self.col_start : self.col_end]
        self.w2_peers = (self.w2[: self.col_start], self.w2[self.col_end :])
        self.max_reqs = min(speculator.max_num_reqs, _MAX_REQS)
        self.n_spec = speculator.num_speculative_steps
        self.push_max_reqs = min(_PUSH_MAX_REQS, self.max_reqs)

    @classmethod
    def create(
        cls, speculator: DSparkSpeculator
    ) -> tuple[TPMarkovSampler | None, str | None]:
        """The speculator's vocab-parallel sampler, or None and the reason it
        cannot run. Collective over the TP group; every rank gets the same
        outcome."""
        reason = cls.unsupported_reason(speculator)
        if reason is not None:
            return None, reason
        sampler = cls(speculator)
        reason = sampler._inexact_projection_reason() or sampler._init_exchange()
        return (None, reason) if reason is not None else (sampler, None)

    @staticmethod
    def unsupported_reason(speculator: DSparkSpeculator) -> str | None:
        """Why the speculator's Markov loop cannot run vocab-parallel."""
        model = speculator.model
        tp_size = get_tp_group().world_size
        head = getattr(getattr(model, "model", None), "markov_head", None)
        w2_layer = getattr(head, "markov_w2", None)
        w2 = getattr(w2_layer, "weight", None)
        processor = getattr(model, "logits_processor", None)
        if not current_platform.is_cuda():
            return "it needs CUDA"
        if tp_size == 1:
            return "tensor parallelism is 1"
        if w2 is None or w2.dtype != torch.bfloat16 or w2.dim() != 2:
            return "markov_w2 is not an unquantized bf16 projection"
        if not isinstance(
            getattr(w2_layer, "quant_method", None),
            (UnquantizedEmbeddingMethod, UnquantizedLinearMethod),
        ):
            return "markov_w2 is quantized"
        if getattr(w2_layer, "use_skinny_gemm", False):
            return "markov_w2 uses the skinny GEMM"
        if processor is None or processor.scale != 1.0 or processor.soft_cap:
            return "the logits processor scales or caps the Markov bias"
        if processor.head_dtype not in (None, torch.bfloat16):
            return "the logits processor changes the head dtype"
        if envs.VLLM_BATCH_INVARIANT:
            return "batch-invariant mode is enabled"
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

    def _any_rank(self, failed: bool) -> bool:
        flag = torch.tensor([int(failed)], dtype=torch.int64)
        torch.distributed.all_reduce(
            flag, op=torch.distributed.ReduceOp.MAX, group=self.group.cpu_group
        )
        return bool(flag.item())

    def _project_peers(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The peers' columns of the bias of x, left and right of this rank's."""
        lo, hi = (F.linear(x, w) for w in self.w2_peers)
        return lo, hi

    @torch.inference_mode()
    def _inexact_projection_reason(self) -> str | None:
        """Why the sliced and batched projections would differ from the
        replicated loop's ``model.markov_bias``, or None if they match bitwise
        (agreed across the TP group).

        A step's own columns come from a row slice of ``markov_w2``, and the
        recompute path projects the peers' rows for all steps at once. They
        match the stock projection only if the GEMM library reduces in the
        same order for these shapes (it does for 64-aligned vocabularies), so
        check on the real weight, for every batch size the sampler runs.
        """
        model = self.speculator.model
        n_spec = self.n_spec
        c0, c1 = self.col_start, self.col_end
        gen = torch.Generator(device=self.w2.device).manual_seed(0)
        x = torch.randn(
            n_spec * self.max_reqs,
            self.w2.shape[1],
            device=self.w2.device,
            generator=gen,
        ).to(self.w2.dtype)
        bad = torch.zeros((), dtype=torch.bool, device=self.w2.device)
        for num_reqs in range(1, self.max_reqs + 1):
            # Rows are (request, step), as in the recompute path.
            xs = x[: num_reqs * n_spec]
            lo, hi = self._project_peers(xs)
            for i in range(n_spec):
                step_x = xs[i::n_spec].contiguous()
                ref = model.markov_bias(step_x)
                bad |= (lo[i::n_spec] != ref[:, :c0]).any()
                bad |= (hi[i::n_spec] != ref[:, c1:]).any()
                if i == 0:
                    own = F.linear(step_x, self.w2_shard)
                    bad |= (own != ref[:, c0:c1]).any()
            del lo, hi, ref
        failed = bool(bad.item())
        del x
        torch.accelerator.empty_cache()
        if self._any_rank(failed):
            return (
                "the sliced Markov projection is not bitwise equal to the "
                "full one on some TP rank"
            )
        return None

    def _init_exchange(self) -> str | None:
        """Allocate the exchange buffers in symmetric memory, or return why
        that failed on some TP rank."""
        device = self.w2.device
        half = _BLOCK_SIZE // 2
        error: Exception | None = None
        try:
            import torch.distributed._symmetric_memory as symm_mem

            buffers = (
                symm_mem.empty(
                    2 * self.max_reqs * self.num_blocks,
                    dtype=torch.int64,
                    device=device,
                ),
                symm_mem.empty(
                    2 * self.world * self.push_max_reqs * self.n_spec * self.per * half,
                    dtype=torch.int64,
                    device=device,
                ),
            )
        except Exception as e:  # e.g. no symmetric memory on this platform
            error = e
        if not self._any_rank(error is not None):
            # Collective; agreed on above so that no rank waits for a peer
            # that failed to allocate.
            try:
                name = self.group.device_group.group_name
                handles = [symm_mem.rendezvous(t, name) for t in buffers]
            except Exception as e:  # e.g. the peers are not P2P reachable
                error = e
        if self._any_rank(error is not None):
            return f"symmetric memory is unavailable ({error or 'on a peer rank'})"

        self.words, self.stage = buffers
        # A zeroed word matches no tag.
        self.words.zero_()
        self.stage.zero_()
        self._handles = handles
        self.word_ptrs, self.stage_ptrs = (
            torch.tensor(list(h.buffer_ptrs), dtype=torch.int64, device=device)
            for h in handles
        )
        # Device-side step counter: tags must change on every graph replay.
        self.epoch = torch.zeros(1, dtype=torch.int64, device=device)
        # Exchange timeouts, copied to the host after every sample.
        self.errors = torch.zeros(1, dtype=torch.int32, device=device)
        self._errors_host = torch.zeros(
            1, dtype=torch.int32, device="cpu", pin_memory=True
        )
        torch.accelerator.synchronize()
        torch.distributed.barrier(group=self.group.cpu_group)
        return None

    def check_exchange(self) -> None:
        """Raise if an exchange of an earlier step timed out waiting for a
        peer. Reads the host copy, so it does not synchronize."""
        if self._errors_host.item():
            raise RuntimeError(
                "DSpark vocab-parallel Markov sampling timed out waiting for a "
                "TP peer; the drafts of that step are invalid. Disable "
                "speculative_config.dspark_tp_markov_sampling if this recurs."
            )

    def sample(
        self,
        num_reqs: int,
        base_logits: torch.Tensor,  # [num_reqs, n_spec, vocab]
        idx_map: torch.Tensor,  # [num_reqs, n_spec]
        sample_pos: torch.Tensor,  # [num_reqs, n_spec]
    ) -> None:
        spc = self.speculator
        model = spc.model
        n_spec = self.n_spec
        vocab_size = self.vocab_size
        assert base_logits.shape[-1] == vocab_size
        assert spc.draft_logits is not None
        cache = spc.draft_logits.values
        push = num_reqs <= self.push_max_reqs
        idx_map_steps = idx_map.t().contiguous()
        draw_pos_steps = (sample_pos.t() - 1).contiguous()
        draft_tokens = spc.draft_tokens
        anchor = spc.input_buffers.input_ids[spc._anchor_idx[:num_reqs]]
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
                spc.seeds,
                draw_pos_steps[i],
                spc.temperature,
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
            lo, hi = self._project_peers(model.markov_embed(prevs.reshape(-1)))
            # An empty side is never read; pass a valid pointer.
            lo = lo if lo.numel() else hi
            hi = hi if hi.numel() else lo
            flat = base_logits.reshape(num_reqs * n_spec, vocab_size)
            _tp_markov_fill_kernel[(num_reqs * n_spec, self.num_blocks)](
                cache,
                cache.stride(0),
                cache.stride(1),
                flat,
                flat.stride(0),
                lo,
                lo.stride(0),
                hi,
                hi.stride(0),
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
        self._errors_host.copy_(self.errors, non_blocking=True)
