# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Muse Spark DSpark sequential Markov sampling, fused for ROCm gfx950.

`DSparkSpeculator._sample_markov_chain` runs many small kernels for each of
its draft steps: the Markov embedding of the previous token, the bias GEMM over
the draft vocabulary, the bias add, the output logit transform (Muse Spark:
fp32 cast, metap multiplier, tanh soft cap), the cast to the draft logits cache
dtype and its index_copy into the target-vocabulary scratch row, the position /
index-map copies, the Gumbel kernel, the argmax over its block maxima, the
gather and the copy into draft_tokens. Apart from the GEMM, each is a short
launch on a few MB, so the steps are launch-bound.

A step here is the unchanged bias GEMM (the model's
own `markov_bias`, so the bias is bitwise the same) plus one Triton launch
over (row, 1024-token block) that

  * adds the bias to the base logits (bf16 add, as torch's),
  * applies the model's output transform in fp32 with torch's op sequence
    (`x * m`; `c * tanh(x * fl32(1/c))`: torch divides by a CPU scalar as a
    multiply by its fp32 reciprocal) and rounds to the cache dtype,
  * stores the row into the draft logits cache (`draft_logits_row` mapping),
  * takes the block's Gumbel-max with vLLM's own `gumbel_noised_argmax`
    (same seed mixing `spec_step_seed(seed, step)`, position `pos - 1`,
    keys = token id, fp32 or fp64 noise as the speculator's
    `use_fp64_gumbel`), or its plain argmax for greedy drafts, and
  * reduces the block winners of the row: max score, then the lowest token
    id, i.e. the same winner as Gumbel's per-block `tl.max` (first of equal
    maxima) followed by `torch.argmax` over blocks (first of equal maxima).
    With fp32 scores this is one 64-bit atomic max keyed by (order-preserving
    score bits, ~token id); fp64 scores do not fit that key, so each program
    publishes its winner (two 64-bit exchanges) and the row's last program
    reduces them in block order.

The row's last program (an atomic ticket) writes the token into the output
tokens and gathers `markov_w1[token]` into a persistent buffer that is the
next step's GEMM input.

Against the loop (same RNG stream), each token of a row without NaN is sampled
from exactly the value stored in the draft logits cache, so verification sees
the draft's distribution. The values can differ from the loop's in the last fp32
bits (the fp32 tanh is not torch's instruction sequence); a token can then
differ from the loop's only on a near-tie, or in a row with NaN (below).
Greedy drafts (no cache) take the same argmax over the same fp32 logits as the
loop's `greedy_argmax`. On both paths a NaN logit is stored as is but samples
as -inf, as `greedy_argmax` and the rejection sampler do: a row draws the
Gumbel-max (or argmax) of its other tokens, and an all-NaN row draws token 0.
This does not depend on the block's layout or the row strides, unlike `tl.max`
over NaN, so every rank and run draws the same token from the same logits. The
cache keeps the loop's NaNs, and the rejection sampler rejects any sampled draft
from a row with a NaN logit (its draft log-sum-exp is NaN), so the drafted token
of such a row cannot change the output.

The scratch buffers are allocated once for `max_num_reqs` rows, so the launches
are CUDA-graph safe; the ticket counters and slots are reset by the last program
of each row. The draft installs `MuseSparkFusedMarkov.sample` as its
`sample_sequential` hook where `use_fused_markov_sampling` holds.
"""

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.triton_utils import tldevice as libdevice
from vllm.v1.worker.gpu.sample.gumbel import gumbel_noised_argmax, spec_step_seed
from vllm.v1.worker.gpu.spec_decode.draft_logits import draft_logits_row

logger = init_logger(__name__)

BLOCK = 1024
NUM_WARPS = 4


def use_fused_markov_sampling(vllm_config) -> bool:
    """Whether the draft takes this module's launches: on gfx950, except with
    --monokernel (`ops.fused_markov`) or batch-sharded sampling (its own
    sharded loop)."""
    if (
        vllm_config.kernel_config.monokernel
        or vllm_config.parallel_config.enable_batch_sharded_sampling
    ):
        return False
    from vllm.platforms.rocm import on_gfx950

    return on_gfx950()


@triton.jit
def _markov_round_kernel(
    base_ptr,  # base logits of row r at base_ptr + r * base_stride + v
    base_stride,
    bias_ptr,  # [rows, vocab] Markov bias of this step
    bias_stride,
    idx_map_ptr,  # [rows, n_spec] request state index (or -1)
    idx_stride,
    pos_ptr,  # [rows, n_spec] sampled token position
    pos_stride,
    temp_ptr,
    seeds_ptr,
    cache_ptr,  # draft logits cache [max_rows, n_spec, >= vocab] (HAS_CACHE)
    cache_stride0,
    cache_stride1,
    row_map_ptr,  # DraftLogitsCache.row_mapping or None
    tokens_ptr,  # [rows, n_spec] output draft tokens
    tokens_stride,
    w1_ptr,  # markov_w1 [vocab_in, RANK]
    w1_stride,
    emb_ptr,  # [max_rows, RANK] next step's Markov embedding
    emb_stride,
    slot_ptr,  # [max_rows] uint64 running (score, ~id) max (fp32 scores)
    count_ptr,  # [max_rows] int32 ticket
    blk_max_ptr,  # [max_rows, num_progs] int64 bits of fp64 maxima (USE_FP64)
    blk_arg_ptr,  # [max_rows, num_progs] int64 their token ids (USE_FP64)
    step,
    mult,
    inv_cap,
    cap,
    vocab,
    num_progs,
    RANK: tl.constexpr,
    BLOCK: tl.constexpr,
    PROGS_P2: tl.constexpr,
    HAS_TRANSFORM: tl.constexpr,
    HAS_CAP: tl.constexpr,
    HAS_CACHE: tl.constexpr,
    USE_FP64: tl.constexpr,
    WRITE_EMB: tl.constexpr,
):
    r = tl.program_id(0)
    p = tl.program_id(1)
    pos = tl.load(pos_ptr + r * pos_stride + step) - 1
    if HAS_CACHE:
        req = tl.load(idx_map_ptr + r * idx_stride + step).to(tl.int64)
        valid = req >= 0
        row = draft_logits_row(req, row_map_ptr)
        temp = tl.load(temp_ptr + req, mask=valid, other=0.0).to(tl.float32)
        seed = tl.load(seeds_ptr + req, mask=valid, other=0)
        seed = spec_step_seed(seed, step)
    else:
        # Greedy drafts: the loop's greedy_argmax of the step's logits.
        temp = 0.0
        seed = 0
    v = p * BLOCK + tl.arange(0, BLOCK)
    mask = v < vocab
    base = tl.load(base_ptr + r * base_stride + v, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + r * bias_stride + v, mask=mask, other=0.0)
    # torch's bf16 add: fp32 sum, one rounding to the result dtype.
    x = (base.to(tl.float32) + bias.to(tl.float32)).to(base.dtype)
    if HAS_TRANSFORM:
        x = x.to(tl.float32) * mult
        if HAS_CAP:
            x = cap * libdevice.tanh(x * inv_cap)
    if HAS_CACHE:
        # The loop casts to the cache dtype before Gumbel reads it back as fp32.
        x = x.to(cache_ptr.dtype.element_ty)
        tl.store(
            cache_ptr + row * cache_stride0 + step * cache_stride1 + v,
            x,
            mask=mask & (row >= 0),
        )
    # Gumbel reads the logits as fp32. NaN counts as -inf, as in
    # greedy_argmax: tl.max's result over NaN depends on its reduction order.
    lf = x.to(tl.float32)
    lf = tl.where(mask & (lf == lf), lf, float("-inf"))
    value, idx = gumbel_noised_argmax(
        lf,
        v,
        mask,
        seed,
        pos,
        temp,
        IS_DRAFTING=True,
        USE_FP64=USE_FP64,
        APPLY_TEMPERATURE=True,
    )
    token = (p * BLOCK + idx).to(tl.int64)
    # Agent-scope atomics are performed at the device coherence point, so
    # relaxed ones suffice: the ticket increment depends on the values the
    # program's atomics return, so it issues only after they are performed,
    # and the row's last program then reads every program's result with
    # atomics. acq_rel would add an L2 writeback/invalidate per atomic on
    # multi-XCD parts.
    if USE_FP64:
        # A 64-bit score and a token id do not fit one 64-bit atomic: each
        # program publishes its winner in its own slots. The previous step's
        # values are never -1 (a NaN payload that no score has; ids >= 0).
        vbits = value.to(tl.int64, bitcast=True)
        old_v = tl.atomic_xchg(blk_max_ptr + r * num_progs + p, vbits, sem="relaxed")
        old_t = tl.atomic_xchg(blk_arg_ptr + r * num_progs + p, token, sem="relaxed")
        inc = ((old_v != -1) & (old_t != -1)).to(tl.int32)
        ticket = tl.atomic_add(count_ptr + r * 32, inc, sem="relaxed")
    else:
        # +0.0 maps -0.0 to +0.0 (tl.max / torch.argmax treat them as equal).
        bits = (value + 0.0).to(tl.uint32, bitcast=True)
        ordered = tl.where((bits >> 31) != 0, ~bits, bits | 0x80000000)
        # Max score first, then the lowest token id: Gumbel's per-block tl.max
        # and the combine over the blocks both keep the first of equal maxima.
        low = (0xFFFFFFFF - token.to(tl.uint32)).to(tl.uint64)
        key = (ordered.to(tl.uint64) << 32) | low
        # One cache line per row for the slot and the ticket; key 1 never
        # occurs.
        old = tl.atomic_max(slot_ptr + r * 16, key, sem="relaxed")
        inc = (old != 1).to(tl.int32)
        ticket = tl.atomic_add(count_ptr + r * 32, inc, sem="relaxed")
    if ticket == num_progs - 1:
        if USE_FP64:
            # torch.argmax over the program maxima: the first of equal maxima.
            k = tl.arange(0, PROGS_P2)
            km = k < num_progs
            vb = tl.atomic_add(
                blk_max_ptr + r * num_progs + k,
                tl.zeros([PROGS_P2], dtype=tl.int64),
                mask=km,
                sem="relaxed",
            )
            vals = tl.where(km, vb.to(tl.float64, bitcast=True), float("-inf"))
            top = tl.max(vals, axis=0)
            win_p = tl.min(tl.where(km & (vals == top), k, PROGS_P2), axis=0)
            win_p = tl.minimum(win_p, num_progs - 1)
            tok = tl.atomic_add(blk_arg_ptr + r * num_progs + win_p, 0, sem="relaxed")
        else:
            win = tl.atomic_xchg(slot_ptr + r * 16, tl.zeros_like(key), sem="relaxed")
            tok = (0xFFFFFFFF - (win & 0xFFFFFFFF)).to(tl.int64)
        # Keep the token in the vocabulary for the store and the gather,
        # whatever the scores.
        tok = tl.minimum(tl.maximum(tok, 0), vocab - 1)
        tl.atomic_xchg(count_ptr + r * 32, 0, sem="relaxed")
        tl.store(tokens_ptr + r * tokens_stride + step, tok)
        if WRITE_EMB:
            k = tl.arange(0, RANK)
            e = tl.load(w1_ptr + tok * w1_stride + k)
            tl.store(emb_ptr + r * emb_stride + k, e)


class MuseSparkFusedMarkov:
    """The Markov steps of `DSparkSpeculator._sample_sequential` as one launch
    each. Installed as the draft model's `sample_sequential` hook."""

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self._reason: str | None = None
        self._checked = False
        self._scratch: tuple[torch.Tensor, ...] | None = None
        self._fp64_scratch: tuple[torch.Tensor, torch.Tensor] | None = None

    def why_not(self, spec) -> str | None:
        """Why the speculator's Markov loop cannot take the fused launches."""
        model = self.model
        if spec.draft_watermarker is not None:
            return "a draft watermarker"
        if spec.use_confidence_head or spec.use_acceptance_estimator:
            return "adaptive verification"
        if spec._draft_topk is not None:
            return "top-k Markov sampling"
        if getattr(model, "has_recurrent_markov_head", False):
            return "a recurrent Markov head"
        # Set by the draft model after loading: the kernel samples draft ids
        # and writes them as target ids.
        if not getattr(model, "d2t_is_identity", False):
            return "a non-identity draft->target map"
        head = model.model.markov_head
        # The kernel writes the draft vocabulary's columns of the cache row.
        cache = spec.draft_logits
        if (
            cache is not None
            and cache.values.shape[-1] != head.markov_w2.org_vocab_size
        ):
            return "the draft logits cache width"
        w1 = head.markov_w1.weight
        if w1.stride(1) != 1 or (w1.shape[1] & (w1.shape[1] - 1)) != 0:
            return "the Markov embedding layout"
        return None

    def sample(self, spec, num_reqs: int, head_hidden: torch.Tensor) -> bool:
        """`DSparkSpeculator._sample_sequential` with the fused launches.
        Returns False, having done nothing, when the loop must run."""
        has_cache = spec.draft_logits is not None
        use_fp64 = has_cache and bool(spec.use_fp64_gumbel)
        if not self._checked:
            # The first call is the memory-profiling run, outside graph capture.
            self._reason = self.why_not(spec)
            self._checked = True
            if self._reason is None:
                mode = (
                    "greedy drafts"
                    if not has_cache
                    else ("fp64 Gumbel" if use_fp64 else "fp32 Gumbel")
                )
                logger.info_once(
                    "DSpark fused Markov sampling: %d steps, block %d, %s.",
                    spec.num_speculative_steps,
                    BLOCK,
                    mode,
                )
            else:
                logger.info_once("DSpark fused Markov sampling off: %s.", self._reason)
        if self._reason is not None:
            return False

        model = self.model
        n_spec = spec.num_speculative_steps
        num_sample = num_reqs * n_spec
        sample_hidden = head_hidden[spec.sample_indices[:num_sample]]
        base_logits = model.compute_draft_logits(sample_hidden)
        vocab = base_logits.shape[-1]
        base_logits = base_logits.view(num_reqs, n_spec, vocab)
        if base_logits.stride(-1) != 1:
            base_logits = base_logits.contiguous()
        idx_map = spec.sample_idx_mapping[:num_sample].view(num_reqs, n_spec)
        sample_pos = spec.sample_pos[:num_sample].view(num_reqs, n_spec)
        out_tokens = spec.draft_tokens[:num_reqs]
        prev = spec.input_buffers.input_ids[spec._anchor_idx[:num_reqs]]

        w1 = model.model.markov_head.markov_w1.weight
        rank = w1.shape[1]
        has_transform = hasattr(model, "apply_output_logit_transform")
        soft_cap = model.output_soft_cap_temp if has_transform else None
        mult = float(model.output_multiplier) if has_transform else 1.0
        cap = float(soft_cap) if soft_cap is not None else 1.0
        # torch divides by a CPU scalar as a multiply by its fp32 reciprocal.
        inv_cap = float(
            torch.tensor(1.0, dtype=torch.float32)
            / torch.tensor(cap, dtype=torch.float32)
        )
        if has_cache:
            cache = spec.draft_logits.values
            cache_strides = (cache.stride(0), cache.stride(1))
            row_mapping = spec.draft_logits.row_mapping
        else:
            # Unused (HAS_CACHE=False): any tensor of the base logits' dtype.
            cache, cache_strides, row_mapping = base_logits, (0, 0), None
        num_progs = triton.cdiv(vocab, BLOCK)
        device = base_logits.device
        if self._scratch is None:
            n = spec.max_num_reqs
            self._scratch = (
                torch.zeros(n * 16, dtype=torch.uint64, device=device),
                torch.zeros(n * 32, dtype=torch.int32, device=device),
                torch.zeros(n, rank, dtype=w1.dtype, device=device),
            )
        slot, count, emb = self._scratch
        # Unused without fp64 Gumbel (USE_FP64=False): any tensors.
        blk_max, blk_arg = slot, slot
        if use_fp64:
            if self._fp64_scratch is None:
                n = spec.max_num_reqs
                self._fp64_scratch = (
                    torch.zeros(n, num_progs, dtype=torch.int64, device=device),
                    torch.zeros(n, num_progs, dtype=torch.int64, device=device),
                )
            blk_max, blk_arg = self._fp64_scratch
        markov_embed = model.markov_embed(prev)
        for i in range(n_spec):
            bias = model.markov_bias(markov_embed)
            _markov_round_kernel[(num_reqs, num_progs)](
                base_logits[:, i],
                base_logits.stride(0),
                bias,
                bias.stride(0),
                idx_map,
                idx_map.stride(0),
                sample_pos,
                sample_pos.stride(0),
                spec.temperature,
                spec.seeds,
                cache,
                cache_strides[0],
                cache_strides[1],
                row_mapping,
                out_tokens,
                out_tokens.stride(0),
                w1,
                w1.stride(0),
                emb,
                emb.stride(0),
                slot,
                count,
                blk_max,
                blk_arg,
                i,
                mult,
                inv_cap,
                cap,
                vocab,
                num_progs,
                RANK=rank,
                BLOCK=BLOCK,
                PROGS_P2=triton.next_power_of_2(num_progs),
                HAS_TRANSFORM=has_transform,
                HAS_CAP=soft_cap is not None,
                HAS_CACHE=has_cache,
                USE_FP64=use_fp64,
                WRITE_EMB=i + 1 < n_spec,
                num_warps=NUM_WARPS,
            )
            markov_embed = emb[:num_reqs]
        return True
