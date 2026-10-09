# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark's sequential Markov sampling with one launch a draft step.

vLLM's ``DSparkSpeculator._sample_sequential`` runs about 9 kernels for each
of the 5 draft steps: the Markov embedding of the previous token, the bias
GEMV over the whole vocabulary (66 MB of markov_w2 at 129280 x 256 bf16),
the add to the base logits, the Gumbel sample (its kernel, the argmax over
its blocks and a gather) and the copy into draft_tokens. On MI325X at TP 4
that is about 220 us a decode step.

Here each step is one launch over the vocabulary in blocks of BLOCK tokens.
For each row, a program computes its block's bias from markov_w2, adds the
base logits, stores the step's logits into the draft logits cache and takes
the Gumbel-max of its block with vLLM's own ``gumbel_noised_argmax``. The
next step's programs first take the argmax over the previous step's block
maxima themselves, which gives the previous token, and the last step's
tokens get one small launch. The noise depends only on the request's seed,
the position and the token id, and both paths keep the first of equal
maxima, so the sampled token does not depend on the block size.

The bias is an fp32 sum of 256 products rounded to bf16, as vLLM's GEMV
output. The products are added in another order than in vLLM's GEMV, so a
bias value can differ in its last bf16 bit. That changes only which tokens
the draft proposes, not the target's output, because the rejection sampler
verifies every draft token against the target's own distribution.

It runs on gfx942 with VLLM_ROCM_MONO_DECODE=1 (``dsv41_gfx942.enabled()``).
tests/kernels/test_dsv41_gfx942_fused_markov.py compares it with vLLM's loop.
"""

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.dsv41_gfx942 import enabled
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.sample.gumbel import gumbel_noised_argmax

logger = init_logger(__name__)

# Vocabulary tokens a program. 129280 / 256 gives 505 programs, about 2 for
# each of MI325X's 304 CUs, and each program reads 128 KB of markov_w2.
BLOCK = 256
# The Markov rank's columns that a program multiplies at a time.
K_CHUNK = 64
# Warps of a step program.
NUM_WARPS = 4
# The step launch unrolls its loop over the rows (requests) of a step, so it
# is built for at most this many. A larger step takes vLLM's loop.
MAX_ROWS = 8


@triton.jit
def _markov_step_kernel(
    prev_max_ptr,  # [MAX_ROWS, num_blocks] the previous step's block maxima
    prev_arg_ptr,  # [MAX_ROWS, num_blocks] the token ids of those maxima
    input_ids_ptr,
    anchor_idx_ptr,  # [rows] step 0's previous token: input_ids[anchor_idx[r]]
    tokens_ptr,  # draft_tokens [max_num_reqs, n_spec]
    tokens_stride,
    w1_ptr,  # markov_w1 [vocab, RANK] bf16, the Markov embedding
    w2_ptr,  # markov_w2 [vocab, RANK] bf16, the bias head
    base_ptr,  # base logits of row r and this step at r * base_stride + v
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
    out_max_ptr,  # [MAX_ROWS, num_blocks] this step's block maxima
    out_arg_ptr,
    vocab,
    num_blocks,
    STEP: tl.constexpr,
    ROWS: tl.constexpr,
    RANK: tl.constexpr,
    BLOCK: tl.constexpr,
    K_CHUNK: tl.constexpr,
    BLOCKS_POW2: tl.constexpr,
    USE_FP64: tl.constexpr,
):
    b = tl.program_id(0)
    v = b * BLOCK + tl.arange(0, BLOCK)
    vmask = v < vocab
    j = tl.arange(0, BLOCKS_POW2)
    for r in tl.static_range(ROWS):
        # The row's previous token. Step 0 takes the request's anchor token.
        # A later step takes the first of the largest block maxima of the
        # previous step, as vLLM's argmax over its block maxima does, and
        # program 0 also writes that token into draft_tokens.
        if STEP == 0:
            prev = tl.load(input_ids_ptr + tl.load(anchor_idx_ptr + r)).to(tl.int64)
        else:
            m = tl.load(
                prev_max_ptr + r * num_blocks + j,
                mask=j < num_blocks,
                other=float("-inf"),
            )
            best = tl.argmax(m, axis=0, tie_break_left=True)
            prev = tl.load(prev_arg_ptr + r * num_blocks + best)
            if b == 0:
                tl.store(tokens_ptr + r * tokens_stride + (STEP - 1), prev)
        # The block's Markov bias of the row. A second row reads the same
        # markov_w2 rows again, which the caches mostly hold by then.
        acc = tl.zeros([BLOCK], dtype=tl.float32)
        for k0 in tl.static_range(0, RANK, K_CHUNK):
            k = k0 + tl.arange(0, K_CHUNK)
            w = tl.load(
                w2_ptr + v[:, None].to(tl.int64) * RANK + k[None, :],
                mask=vmask[:, None],
                other=0.0,
            ).to(tl.float32)
            e = tl.load(w1_ptr + prev * RANK + k).to(tl.float32)
            acc += tl.sum(w * e[None, :], axis=1)
        # vLLM rounds the GEMV's bias to bf16 and adds the bf16 base logits
        # in fp32 with one rounding to bf16, as a bf16 tensor add does.
        bias = acc.to(tl.bfloat16).to(tl.float32)
        base = tl.load(base_ptr + r * base_stride + v, mask=vmask, other=0.0)
        logits = (base.to(tl.float32) + bias).to(tl.bfloat16)
        req = tl.load(idx_map_ptr + r * idx_stride + STEP).to(tl.int64)
        valid = req >= 0
        # The cache holds the logits before temperature, as gumbel_sample
        # stores them for the rejection sampler.
        tl.store(
            cache_ptr + req * cache_stride0 + STEP * cache_stride1 + v,
            logits,
            mask=vmask & valid,
        )
        temp = tl.load(temp_ptr + req, mask=valid, other=0.0).to(tl.float32)
        seed = tl.load(seeds_ptr + req, mask=valid, other=0)
        # gumbel_sample keys a draw by the position before the sampled token.
        pos = tl.load(pos_ptr + r * pos_stride + STEP) - 1
        lf = tl.where(vmask, logits.to(tl.float32), float("-inf"))
        value, idx = gumbel_noised_argmax(
            lf,
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
    BLOCKS_POW2: tl.constexpr,
):
    # The last step's token of row r, as the step kernel takes the previous
    # token at the start of a step.
    r = tl.program_id(0)
    j = tl.arange(0, BLOCKS_POW2)
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
    current step and one read from the previous step. They are allocated
    once, so a CUDA graph keeps their addresses."""
    key = (device, vocab, use_fp64, BLOCK)
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
    """Why this speculator's step takes vLLM's loop, or None when the fused
    launches compute the same sampling."""
    model = spec.model
    head = getattr(getattr(model, "model", None), "markov_head", None)
    if head is None:
        return "no Markov head"
    if spec.draft_logits is None:
        return "greedy drafts"
    if spec.draft_watermarker is not None:
        return "a draft watermarker"
    if spec._d2t_scatter_index is not None:
        return "a reduced draft vocabulary"
    if spec.use_confidence_head:
        return "the confidence head"
    if spec.acceptance_estimator is not None:
        return "the acceptance estimator"
    if not 1 <= num_reqs <= MAX_ROWS:
        return f"{num_reqs} requests"
    w1 = head.markov_w1.weight
    w2 = head.markov_w2.weight
    if w1.dtype != torch.bfloat16 or w2.dtype != torch.bfloat16:
        return "Markov weights not bf16"
    if w1.shape[1] != w2.shape[1] or not (w1.is_contiguous() and w2.is_contiguous()):
        return "Markov weight layout"
    if spec.draft_logits.dtype != torch.bfloat16:
        return f"a {spec.draft_logits.dtype} draft logits cache"
    if head.markov_w2.quant_method.__class__.__name__ != "UnquantizedEmbeddingMethod":
        return "a quantized Markov head"
    return None


def sample_sequential(spec, num_reqs: int, head_hidden: torch.Tensor) -> bool:
    """DSparkSpeculator._sample_sequential with the fused launches. Returns
    False, having done nothing, when this step takes vLLM's loop."""
    if not enabled():
        return False
    why = _why_not(spec, num_reqs)
    if why is not None:
        logger.debug_once("DSpark fused Markov sampling off for a step: %s", why)
        return False
    n_spec = spec.num_speculative_steps
    num_sample = num_reqs * n_spec
    sample_hidden = head_hidden[spec.sample_indices[:num_sample]]
    base_logits = spec.model.compute_draft_logits(sample_hidden)
    if base_logits.dtype != torch.bfloat16 or base_logits.stride(-1) != 1:
        # The base logits come from the lm_head, which may differ from the
        # Markov head. vLLM's loop then adds them in their own dtype.
        logger.debug_once(
            "DSpark fused Markov sampling off: %s base logits", base_logits.dtype
        )
        return _loop_rest(spec, num_reqs, base_logits, sample_hidden)
    vocab = base_logits.shape[-1]
    base_logits = base_logits.view(num_reqs, n_spec, vocab)
    head = spec.model.model.markov_head
    w1, w2 = head.markov_w1.weight, head.markov_w2.weight
    rank = w1.shape[1]
    idx_map = spec.sample_idx_mapping[:num_sample].view(num_reqs, n_spec)
    sample_pos = spec.sample_pos[:num_sample].view(num_reqs, n_spec)
    assert idx_map.stride(1) == 1 and sample_pos.stride(1) == 1
    cache = spec.draft_logits
    tokens = spec.draft_tokens
    use_fp64 = bool(spec.use_fp64_gumbel)
    bufs = _scratch(base_logits.device, vocab, use_fp64)
    nb = triton.cdiv(vocab, BLOCK)
    nb2 = triton.next_power_of_2(nb)
    logger.info_once(
        "DSpark fused Markov sampling: %d steps, vocabulary %d, rank %d, %d blocks",
        n_spec, vocab, rank, nb,
    )  # fmt: skip
    for step in range(n_spec):
        prev_max, prev_arg = bufs[(step + 1) % 2]
        out_max, out_arg = bufs[step % 2]
        _markov_step_kernel[(nb,)](
            prev_max,
            prev_arg,
            spec.input_buffers.input_ids,
            spec._anchor_idx,
            tokens,
            tokens.stride(0),
            w1,
            w2,
            base_logits[:, step],
            base_logits.stride(0),
            cache,
            cache.stride(0),
            cache.stride(1),
            idx_map,
            idx_map.stride(0),
            sample_pos,
            sample_pos.stride(0),
            spec.temperature,
            spec.seeds,
            out_max,
            out_arg,
            vocab,
            nb,
            STEP=step,
            ROWS=num_reqs,
            RANK=rank,
            BLOCK=BLOCK,
            K_CHUNK=K_CHUNK,
            BLOCKS_POW2=nb2,
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
        BLOCKS_POW2=nb2,
    )
    return True


def _loop_rest(spec, num_reqs, base_logits, sample_hidden) -> bool:
    """VLLM's loop from its base logits on, for a step whose base logits the
    fused launch does not take. It is the loop of _sample_sequential after
    compute_draft_logits, so the step computes them only once."""
    n_spec = spec.num_speculative_steps
    num_sample = num_reqs * n_spec
    vocab_size = base_logits.shape[-1]
    base_logits = base_logits.view(num_reqs, n_spec, vocab_size)
    idx_map = spec.sample_idx_mapping[:num_sample].view(num_reqs, n_spec)
    sample_pos = spec.sample_pos[:num_sample].view(num_reqs, n_spec)
    prev = spec.input_buffers.input_ids[spec._anchor_idx[:num_reqs]]
    for i in range(n_spec):
        bias = spec.model.markov_bias(spec.model.markov_embed(prev))
        draft = spec._sample_logits(
            base_logits[:, i] + bias, idx_map[:, i], sample_pos[:, i], i
        )
        spec.draft_tokens[:num_reqs, i] = draft
        prev = draft
    return True
