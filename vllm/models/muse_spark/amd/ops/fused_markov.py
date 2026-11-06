# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Muse Spark DSpark Markov rounds with one launch a round (--monokernel).

DSparkSpeculator._sample_sequential runs many small kernels in each of the 7
draft rounds: the Markov embedding of the previous token, the bias GEMV over
the 202048-token draft vocabulary (207 MB of markov_w2), the add to the base
logits, Muse's output transform (multiplier and soft cap), the copy into the
d2t scatter buffer, the Gumbel sample (its kernel, the argmax over its blocks
and a gather) and the copy into draft_tokens.

Here a round is one launch over the vocabulary in blocks of BLOCK tokens. A
program gets the previous token of each row (the argmax over the previous
round's block maxima), computes its block's bias for all rows with one read
of markov_w2, and for each row adds the base logits, applies the transform,
stores the round's logits into the draft logits cache and takes the
Gumbel-max of its block with vLLM's own ``gumbel_noised_argmax``. One small
launch then writes the last round's tokens. The noise depends only on the
request's seed, the round, the position and the token id, and both paths keep
the first of equal maxima, so the sampled token does not depend on the block
size.

The bias is an fp32 sum of 512 products rounded to bf16, as vLLM's GEMV
output, but in another order. So a bias value can differ in its last bf16
bit. This changes only which tokens the draft proposes: the rejection sampler
verifies every draft token against the target's distribution. The other steps
are bitwise the same as vLLM's loop.

MuseSparkDSparkForCausalLM sets this module's ``sample_sequential`` as its
own with --monokernel on ROCm. tests/kernels/models/amd/
test_muse_spark_fused_markov.py compares it with vLLM's loop.
"""

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.sample.gumbel import gumbel_noised_argmax, spec_step_seed

logger = init_logger(__name__)

# Vocabulary tokens a program: 395 programs for 202048 tokens, about 1.5 for
# each of MI355X's 256 CUs. A program reads 512 KB of markov_w2.
BLOCK = 512
# The Markov rank's columns that a program multiplies at a time.
K_CHUNK = 64
NUM_WARPS = 4
# The bias is one tl.dot with this many columns, one for each row.
MAX_ROWS = 16


@triton.jit
def _markov_round_kernel(
    prev_max_ptr,  # [MAX_ROWS, num_blocks] the previous round's block maxima
    prev_arg_ptr,  # [MAX_ROWS, num_blocks] the token ids of those maxima
    input_ids_ptr,
    anchor_idx_ptr,  # [rows] round 0's previous token: input_ids[anchor_idx[r]]
    tokens_ptr,  # draft_tokens [max_num_reqs, n_spec]
    tokens_stride,
    w1_ptr,  # markov_w1 [vocab, RANK] bf16, the Markov embedding
    w2_ptr,  # markov_w2 [vocab, RANK] bf16, the bias head
    base_ptr,  # bf16 base logits of row r and this round at r * base_stride + v
    base_stride,
    cache_ptr,  # draft_logits [max_num_reqs, n_spec, vocab]
    cache_stride0,
    cache_stride1,
    idx_map_ptr,  # [rows, n_spec] the row's request state index
    idx_stride,
    pos_ptr,  # [rows, n_spec] the sampled token's position
    pos_stride,
    temp_ptr,
    seeds_ptr,
    out_max_ptr,  # [MAX_ROWS, num_blocks] this round's block maxima
    out_arg_ptr,
    step,
    num_rows,
    vocab,
    num_blocks,
    out_mult,
    soft_cap,
    inv_soft_cap,
    RANK: tl.constexpr,
    BLOCK: tl.constexpr,
    K_CHUNK: tl.constexpr,
    ROWS_P2: tl.constexpr,
    DOT_N: tl.constexpr,
    BLOCKS_P2: tl.constexpr,
    HAS_CACHE: tl.constexpr,
    SOFT_CAP: tl.constexpr,
    USE_FP64: tl.constexpr,
):
    b = tl.program_id(0)
    v = b * BLOCK + tl.arange(0, BLOCK)
    vmask = v < vocab
    rows = tl.arange(0, ROWS_P2)
    rmask = rows < num_rows
    # Each row's previous token. Round 0 takes the request's anchor token. A
    # later round takes the first of the largest block maxima of the previous
    # round, as vLLM's argmax over its block maxima does, and program 0 also
    # writes that token into draft_tokens.
    if step == 0:
        anchor = tl.load(anchor_idx_ptr + rows, mask=rmask, other=0)
        prev = tl.load(input_ids_ptr + anchor, mask=rmask, other=0).to(tl.int64)
    else:
        j = tl.arange(0, BLOCKS_P2)
        m = tl.load(
            prev_max_ptr + rows[:, None] * num_blocks + j[None, :],
            mask=rmask[:, None] & (j[None, :] < num_blocks),
            other=float("-inf"),
        )
        best = tl.argmax(m, axis=1, tie_break_left=True)
        prev = tl.load(prev_arg_ptr + rows * num_blocks + best, mask=rmask, other=0)
        tl.store(
            tokens_ptr + rows * tokens_stride + (step - 1),
            prev,
            mask=rmask & (b == 0),
        )
    # The tl.dot has DOT_N columns: column n is row n, the others read token 0.
    n = tl.arange(0, DOT_N)
    sel = n[:, None] == rows[None, :]
    prev_n = tl.sum(tl.where(sel, prev[None, :], 0), axis=1)
    acc = tl.zeros([BLOCK, DOT_N], dtype=tl.float32)
    for k0 in tl.static_range(0, RANK, K_CHUNK):
        k = k0 + tl.arange(0, K_CHUNK)
        w = tl.load(
            w2_ptr + v[:, None].to(tl.int64) * RANK + k[None, :],
            mask=vmask[:, None],
            other=0.0,
        )
        e = tl.load(w1_ptr + prev_n[None, :] * RANK + k[:, None])
        acc = tl.dot(w, e, acc)
    for r in range(num_rows):
        # Row r's bias, rounded to bf16 as vLLM's GEMV output. Then the bf16
        # add, which vLLM computes in fp32 and rounds once.
        bias = tl.max(tl.where(n[None, :] == r, acc, float("-inf")), axis=1)
        bias = bias.to(tl.bfloat16).to(tl.float32)
        base = tl.load(base_ptr + r * base_stride + v, mask=vmask, other=0.0)
        x = (base.to(tl.float32) + bias).to(tl.bfloat16).to(tl.float32)
        # apply_output_logit_transform in fp32. torch divides by a scalar as
        # a multiply by its fp32 reciprocal.
        x = x * out_mult
        if SOFT_CAP:
            x = soft_cap * tl.extra.hip.libdevice.tanh(x * inv_soft_cap)
        if HAS_CACHE:
            # vLLM copies the logits into the d2t scatter buffer (the cache's
            # dtype), and gumbel_sample stores and samples that copy.
            req = tl.load(idx_map_ptr + r * idx_stride + step).to(tl.int64)
            valid = req >= 0
            xc = x.to(cache_ptr.dtype.element_ty)
            tl.store(
                cache_ptr + req * cache_stride0 + step * cache_stride1 + v,
                xc,
                mask=vmask & valid,
            )
            x = xc.to(tl.float32)
            temp = tl.load(temp_ptr + req, mask=valid, other=0.0).to(tl.float32)
            # gumbel_sample mixes the cache column (the round) into the seed.
            seed = spec_step_seed(tl.load(seeds_ptr + req, mask=valid, other=0), step)
        else:
            # Greedy drafts: greedy_argmax of the fp32 logits.
            temp = 0.0
            seed = 0
        # gumbel_sample keys a draw by the position before the sampled token.
        pos = tl.load(pos_ptr + r * pos_stride + step) - 1
        value, idx = gumbel_noised_argmax(
            tl.where(vmask, x, float("-inf")),
            v,
            vmask,
            seed,
            pos,
            temp,
            IS_DRAFTING=True,
            USE_FP64=USE_FP64,
            APPLY_TEMPERATURE=True,
        )
        tl.store(out_max_ptr + r * num_blocks + b, value)
        tl.store(out_arg_ptr + r * num_blocks + b, (b * BLOCK + idx).to(tl.int64))


@triton.jit
def _markov_last_kernel(
    prev_max_ptr,
    prev_arg_ptr,
    tokens_ptr,
    tokens_stride,
    num_blocks,
    STEP: tl.constexpr,
    BLOCKS_P2: tl.constexpr,
):
    # The last round's token of row r, as a round takes its previous token.
    r = tl.program_id(0)
    j = tl.arange(0, BLOCKS_P2)
    m = tl.load(
        prev_max_ptr + r * num_blocks + j, mask=j < num_blocks, other=float("-inf")
    )
    best = tl.argmax(m, axis=0, tie_break_left=True)
    tl.store(
        tokens_ptr + r * tokens_stride + STEP,
        tl.load(prev_arg_ptr + r * num_blocks + best),
    )


_buffers: dict = {}


def _scratch(device, vocab, use_fp64):
    """Two sets of block maxima and their token ids, one written by the
    current round and one read from the previous round. They are allocated
    once, so a CUDA graph keeps their addresses."""
    key = (device, vocab, use_fp64)
    if key not in _buffers:
        nb = triton.cdiv(vocab, BLOCK)
        dt = torch.float64 if use_fp64 else torch.float32
        _buffers[key] = [
            (
                torch.empty(MAX_ROWS, nb, dtype=dt, device=device),
                torch.empty(MAX_ROWS, nb, dtype=torch.int64, device=device),
            )
            for _ in range(2)
        ]
    return _buffers[key]


def _why_not(spec, num_reqs: int) -> str | None:
    """Why this step takes vLLM's loop, or None when the fused launches
    compute the same sampling. Every check is the same on all ranks."""
    model = spec.model
    head = model.model.markov_head
    w1, w2 = head.markov_w1.weight, head.markov_w2.weight
    if not getattr(model, "d2t_is_identity", False):
        return "a d2t mapping that is not the identity"
    if spec.draft_watermarker is not None:
        return "a draft watermarker"
    if spec.use_confidence_head:
        return "the confidence head"
    if spec.acceptance_estimator is not None:
        return "the acceptance estimator"
    if not 1 <= num_reqs <= MAX_ROWS:
        return f"{num_reqs} requests"
    if w1.dtype != torch.bfloat16 or w2.dtype != torch.bfloat16:
        return "Markov weights not bf16"
    if w1.shape[1] != w2.shape[1] or not (w1.is_contiguous() and w2.is_contiguous()):
        return "Markov weight layout"
    if head.markov_w2.quant_method.__class__.__name__ != "UnquantizedEmbeddingMethod":
        return "a quantized Markov head"
    # The base logits are bf16 when the lm_head runs in bf16.
    bf16 = (None, torch.bfloat16)
    if (
        model.lm_head.weight.dtype not in bf16
        or model.logits_processor.head_dtype not in bf16
    ):
        return "base logits not bf16"
    cache = spec.draft_logits
    if cache is not None and (
        cache.row_mapping is not None
        or cache.values.dtype != torch.bfloat16
        or cache.values.shape[-1] != w2.shape[0]
    ):
        return "the draft logits cache layout"
    return None


def sample_sequential(spec, num_reqs: int, head_hidden: torch.Tensor) -> bool:
    """DSparkSpeculator._sample_sequential with one launch a Markov round.
    Returns False, having done nothing, when this step takes vLLM's loop."""
    why = _why_not(spec, num_reqs)
    if why is not None:
        logger.debug_once("Muse Spark fused Markov sampling off for a step: %s", why)
        return False
    model = spec.model
    n_spec = spec.num_speculative_steps
    num_sample = num_reqs * n_spec
    sample_hidden = head_hidden[spec.sample_indices[:num_sample]]
    base_logits = model.compute_draft_logits(sample_hidden)
    assert base_logits.dtype == torch.bfloat16 and base_logits.stride(-1) == 1
    vocab = base_logits.shape[-1]
    head = model.model.markov_head
    w1, w2 = head.markov_w1.weight, head.markov_w2.weight
    assert w2.shape[0] == vocab
    idx_map = spec.sample_idx_mapping[:num_sample].view(num_reqs, n_spec)
    sample_pos = spec.sample_pos[:num_sample].view(num_reqs, n_spec)
    has_cache = spec.draft_logits is not None
    cache = spec.draft_logits.values if has_cache else base_logits
    tokens = spec.draft_tokens
    use_fp64 = bool(spec.use_fp64_gumbel)
    soft_cap = model.output_soft_cap_temp
    bufs = _scratch(base_logits.device, vocab, use_fp64)
    nb = triton.cdiv(vocab, BLOCK)
    nb2 = triton.next_power_of_2(nb)
    rows_p2 = triton.next_power_of_2(num_reqs)
    logger.info_once(
        "Muse Spark fused Markov sampling: %d rounds, vocabulary %d, rank %d",
        n_spec, vocab, w1.shape[1],
    )  # fmt: skip
    for step in range(n_spec):
        prev_max, prev_arg = bufs[(step + 1) % 2]
        out_max, out_arg = bufs[step % 2]
        _markov_round_kernel[(nb,)](
            prev_max,
            prev_arg,
            spec.input_buffers.input_ids,
            spec._anchor_idx,
            tokens,
            tokens.stride(0),
            w1,
            w2,
            base_logits[step::n_spec],
            base_logits.stride(0) * n_spec,
            cache,
            cache.stride(0),
            cache.stride(1) if has_cache else 0,
            idx_map,
            idx_map.stride(0),
            sample_pos,
            sample_pos.stride(0),
            spec.temperature,
            spec.seeds,
            out_max,
            out_arg,
            step,
            num_reqs,
            vocab,
            nb,
            float(np.float32(model.output_multiplier)),
            float(soft_cap or 1.0),
            float(np.float32(1.0) / np.float32(soft_cap or 1.0)),
            RANK=w1.shape[1],
            BLOCK=BLOCK,
            K_CHUNK=K_CHUNK,
            ROWS_P2=rows_p2,
            DOT_N=max(16, rows_p2),
            BLOCKS_P2=nb2,
            HAS_CACHE=has_cache,
            SOFT_CAP=soft_cap is not None,
            USE_FP64=use_fp64,
            num_warps=NUM_WARPS,
        )
    last_max, last_arg = bufs[(n_spec - 1) % 2]
    _markov_last_kernel[(num_reqs,)](
        last_max,
        last_arg,
        tokens,
        tokens.stride(0),
        nb,
        STEP=n_spec - 1,
        BLOCKS_P2=nb2,
    )
    return True
