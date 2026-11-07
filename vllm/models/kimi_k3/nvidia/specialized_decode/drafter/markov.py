# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark's sequential Markov sampling, one launch per draft step.

``DSparkSpeculator._sample_sequential`` runs, for each of the ``k`` draft steps,
the Markov embedding of the previous token, the bias GEMV over the whole
vocabulary (``markov_w2``: 163840 x 256 BF16, 84 MB, replicated), the add to the
base logits ([R, 163840] written twice), the Gumbel sample (its kernel, a torch
argmax over the block maxima and a gather) and the copy into ``draft_tokens``:
about nine launches and three full-vocabulary [R, V] tensors per step.

Here each step is two launches. The step kernel's programs each own 256
vocabulary entries and 16 request rows: they gather the rows' Markov embeddings
(``markov_w1[prev]``), stream their ``markov_w2`` range in [64, rank] tiles and
multiply all rows at once on the tensor cores (FP32 accumulation), add the base
logits, write the step's logits into the draft logits cache and keep each row's
Gumbel-max over their range. A select launch (one program per row) takes the
first of the largest per-program maxima, which is the step's token: it writes
``draft_tokens`` and the next step's previous token.

Exactness: the same sampling as vLLM's loop. The bias is the FP32 dot product of
the embedding and the ``markov_w2`` row rounded once to the logits dtype (the GEMV
output dtype), the add is one FP32 add rounded to the logits dtype (a BF16 tensor
add), the cache holds the logits before temperature, and the Gumbel noise is
vLLM's own draw (``murmur3`` keyed by the request seed mixed with the draft step,
the position before the sampled token plus the draft salt, and the token id),
divided by the temperature exactly as ``gumbel_noised_argmax`` does. Only the
summation order of the 256-term bias dot product differs from cuBLAS's GEMV, as
any GEMM tiling may; ties keep the first token as vLLM's two-level argmax does.
"""

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.sample.gumbel import (
    _log1p_neg_stable,
    murmur3_uniform32,
    murmur3_uniform64,
    spec_step_seed,
)
from vllm.v1.worker.gpu.spec_decode.draft_logits import draft_logits_row

logger = init_logger(__name__)

# Vocabulary rows of markov_w2 a tile multiplies at a time (tensor-core N).
VT = 64
_DRAFT_NOISE_SALT = 1 << 30


@triton.jit
def _add_rn(a, b):
    # One rounded FP32 add, kept apart from the dot: the compiler would otherwise
    # fold the base logits into the dot's accumulator (another summation order).
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        "=r,r,r",
        [a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _gumbel_key(lf, temp, seed, pos, v, USE_FP64: tl.constexpr):
    """``gumbel_noised_argmax``'s key of FP32 logits ``lf`` (temperature applied
    in FP32, then the draft-salted noise; plain logits at temperature 0)."""
    noisy = temp != 0.0
    lf = tl.where(noisy, lf / temp, lf)
    if USE_FP64:
        lf = lf.to(tl.float64)
        u = murmur3_uniform64(seed, pos, v)
        g = -tl.log(-tl.log(u))
    else:
        u = murmur3_uniform32(seed, pos, v)
        g = -tl.log(-_log1p_neg_stable(u))
    return tl.where(noisy, lf + g, lf)


@triton.jit
def _gumbel_log_arg32(seed, pos, v):
    """The argument of the outer log of ``_gumbel_key``'s FP32 noise
    ``-log(-log1p(-u))``: the noise is ``-log`` of it. The step kernel takes that
    last log itself, so its add to the logits compiles as in
    ``gumbel_noised_argmax`` (a stored noise, added in another kernel, rounds
    differently: 1 ulp on about a fifth of the keys)."""
    u = murmur3_uniform32(seed, pos, v)
    return -_log1p_neg_stable(u)


@triton.jit
def _markov_noise_kernel(
    noise_ptr,  # [rows_cap, k, V] FP32
    noise_stride0,
    noise_stride1,
    idx_map_ptr,
    idx_stride,
    pos_ptr,
    pos_stride,
    seeds_ptr,
    vocab,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
    SALT: tl.constexpr,
):
    # Program (vocabulary block, row * K + step): the step's noise (up to its
    # last log) of one row, which depends only on the request seed, the position
    # and the token id.
    v = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    flat = tl.program_id(1)
    row = (flat // K).to(tl.int64)
    step = flat % K
    req = tl.load(idx_map_ptr + row * idx_stride + step).to(tl.int64)
    seed = spec_step_seed(tl.load(seeds_ptr + req, mask=req >= 0, other=0), step)
    pos = tl.load(pos_ptr + row * pos_stride + step) - 1
    pos = pos.to(tl.int64) + SALT
    arg = _gumbel_log_arg32(seed, pos, v)
    tl.store(
        noise_ptr + row * noise_stride0 + step * noise_stride1 + v, arg, mask=v < vocab
    )


@triton.jit
def _markov_step_kernel(
    prev_tok_ptr,  # [rows_cap] the previous step's tokens (select kernel)
    input_ids_ptr,
    anchor_idx_ptr,  # [rows] step 0's previous token: input_ids[anchor_idx[r]]
    w1_ptr,  # markov_w1 [V1, RANK]
    w2_ptr,  # markov_w2 [V, RANK]
    base_ptr,  # this step's base logits of row r at r * base_stride + v
    base_stride,
    cache_ptr,  # draft logits [rows, k, V]
    cache_stride0,
    cache_stride1,
    row_map_ptr,  # DraftLogitsCache.row_mapping or None
    idx_map_ptr,  # [rows, k] the row's request state index
    idx_stride,
    pos_ptr,  # [rows, k] the sampled token's position
    pos_stride,
    temp_ptr,
    seeds_ptr,
    out_max_ptr,  # [rows_cap, NB] this step's per-program maxima
    out_arg_ptr,
    noise_ptr,  # [rows_cap, k, V] FP32 ``_gumbel_log_arg32`` (``_markov_noise_kernel``)
    noise_stride0,
    noise_stride1,
    num_rows,
    vocab,
    NB,
    STEP: tl.constexpr,
    RANK: tl.constexpr,
    VB: tl.constexpr,  # vocabulary entries a program owns (multiple of VT)
    VT: tl.constexpr,
    RB: tl.constexpr,  # rows a program holds (tensor-core M)
    USE_FP64: tl.constexpr,
    SALT: tl.constexpr,
    NOISE: tl.constexpr,  # read the precomputed noise instead of hashing
):
    # Row chunks of one vocabulary range run as neighbouring programs, so the
    # range's markov_w2 tiles come from L2 for all but the first.
    pid = tl.program_id(1)
    rows = tl.program_id(0) * RB + tl.arange(0, RB)
    rmask = rows < num_rows
    rows64 = rows.to(tl.int64)
    # The previous token of each row: the anchor at step 0, else the select
    # launch's token of the previous step.
    if STEP == 0:
        anchor = tl.load(anchor_idx_ptr + rows, mask=rmask, other=0)
        prev = tl.load(input_ids_ptr + anchor, mask=rmask, other=0).to(tl.int64)
    else:
        prev = tl.load(prev_tok_ptr + rows64, mask=rmask, other=0)
    k = tl.arange(0, RANK)
    # [RB, RANK] Markov embeddings of the previous tokens.
    emb = tl.load(
        w1_ptr + prev[:, None] * RANK + k[None, :], mask=rmask[:, None], other=0.0
    )
    req = tl.load(idx_map_ptr + rows64 * idx_stride + STEP, mask=rmask, other=-1)
    req = req.to(tl.int64)
    valid = req >= 0
    crow = draft_logits_row(req, row_map_ptr)
    temp = tl.load(temp_ptr + req, mask=valid, other=0.0).to(tl.float32)
    seed = spec_step_seed(tl.load(seeds_ptr + req, mask=valid, other=0), STEP)
    pos = tl.load(pos_ptr + rows64 * pos_stride + STEP, mask=rmask, other=1) - 1
    pos = pos.to(tl.int64) + SALT
    cmask = rmask & valid & (crow >= 0)

    best_val = tl.full([RB], float("-inf"), tl.float64 if USE_FP64 else tl.float32)
    best_idx = tl.zeros([RB], tl.int64)
    v0 = pid * VB
    for t in range(0, VB, VT):
        v = v0 + t + tl.arange(0, VT)
        vmask = v < vocab
        w = tl.load(
            w2_ptr + v[:, None].to(tl.int64) * RANK + k[None, :],
            mask=vmask[:, None],
            other=0.0,
            eviction_policy="evict_last",
        )
        # The bias: one FP32 dot product per (row, token), rounded to the logits
        # dtype like the GEMV output; the add rounds once like a tensor add.
        acc = tl.dot(emb, tl.trans(w))
        bmask = rmask[:, None] & vmask[None, :]
        base = tl.load(
            base_ptr + rows64[:, None] * base_stride + v[None, :],
            mask=bmask,
            other=0.0,
            eviction_policy="evict_first",
        )
        dt = base.dtype
        logits = _add_rn(base.to(tl.float32), acc.to(dt).to(tl.float32)).to(dt)
        tl.store(
            cache_ptr
            + crow[:, None] * cache_stride0
            + STEP * cache_stride1
            + v[None, :],
            logits,
            mask=cmask[:, None] & vmask[None, :],
        )
        if NOISE:
            # ``_gumbel_key``'s FP32 path with the hash and the inner log read
            # back: the same operations, in the same kernel, from the outer log on.
            noisy = temp[:, None] != 0.0
            arg = tl.load(
                noise_ptr
                + rows64[:, None] * noise_stride0
                + STEP * noise_stride1
                + v[None, :],
                mask=bmask & noisy,
                other=1.0,
            )
            lf = logits.to(tl.float32)
            lf = tl.where(noisy, lf / temp[:, None], lf)
            x = tl.where(noisy, lf + -tl.log(arg), lf)
        else:
            x = _gumbel_key(
                logits.to(tl.float32),
                temp[:, None],
                seed[:, None],
                pos[:, None],
                v[None, :],
                USE_FP64,
            )
        x = tl.where(vmask[None, :], x, float("-inf"))
        bv, bi = tl.max(
            x, axis=1, return_indices=True, return_indices_tie_break_left=True
        )
        take = bv > best_val
        best_val = tl.where(take, bv, best_val)
        best_idx = tl.where(take, (v0 + t + bi).to(tl.int64), best_idx)
    tl.store(out_max_ptr + rows64 * NB + pid, best_val, mask=rmask)
    tl.store(out_arg_ptr + rows64 * NB + pid, best_idx, mask=rmask)


@triton.jit
def _markov_select_kernel(
    max_ptr,
    arg_ptr,
    prev_tok_ptr,
    tokens_ptr,
    tokens_stride,
    num_rows,
    NB,
    STEP: tl.constexpr,
    NB_POW2: tl.constexpr,
):
    # Row r's token of step STEP: the first of its largest per-program maxima.
    r = tl.program_id(0).to(tl.int64)
    j = tl.arange(0, NB_POW2)
    m = tl.load(max_ptr + r * NB + j, mask=j < NB, other=float("-inf"))
    best = tl.argmax(m, axis=0, tie_break_left=True)
    tok = tl.load(arg_ptr + r * NB + best)
    tl.store(prev_tok_ptr + r, tok)
    tl.store(tokens_ptr + r * tokens_stride + STEP, tok)


# Vocabulary entries a program owns: 640 programs at 163840, about four per SM
# (GB300, one cold step, us: VB 256 -> 22, 1024 -> 40; a bare markov_w2 read
# streams at 15 us, bytes / 7.5 TB/s + the ~3.5 us kernel boundary).
VB = 256
# Request rows a program multiplies at once (the tensor-core M; more rows run as
# further programs over the same, then L2-resident, markov_w2 range).
RB = 16
# Sampled rows (requests x draft steps) up to which the Gumbel noise of all the
# steps is computed ahead, up to its last log, beside the lm_head
# (``_markov_noise_kernel``): the step kernels then read it instead of hashing
# [RB, VT] tiles whose rows are mostly padding (one request at c1: 16x the needed
# hashes and logs; GB300 ncu of a warm step: 41K cycles at 0.65 issued warps per
# scheduler, L2 at 17 %). They keep the last log, so that the keys round as in
# ``gumbel_noised_argmax`` (bit-equal; a fully precomputed noise is 1 ulp off on
# about a fifth of the keys). R1 k=4 graph with the noise kernel, us: 94 -> 82.
NOISE_ROWS = 32
NOISE_BLOCK = 1024


class FusedMarkov:
    """The fused Markov steps of one speculator (scratch sized for its largest
    batch, allocated once so captured graphs keep their addresses)."""

    def __init__(self, vocab, rank, max_rows, device, use_fp64):
        self.vocab, self.rank, self.use_fp64 = vocab, rank, use_fp64
        self.vb = VB
        self.nb = triton.cdiv(vocab, self.vb)
        self.nb_pow2 = triton.next_power_of_2(self.nb)
        dt = torch.float64 if use_fp64 else torch.float32
        self.bufs = (
            torch.empty(max_rows, self.nb, dtype=dt, device=device),
            torch.empty(max_rows, self.nb, dtype=torch.int64, device=device),
        )
        self.prev = torch.zeros(max_rows, dtype=torch.int64, device=device)
        self.max_rows = max_rows
        self.device = device
        self.noise = None  # [NOISE_ROWS * V] FP32, at the first noise launch
        self.stream = None

    def noise_rows(self, num_reqs, n_spec):
        """Whether this batch's Gumbel noise is precomputed (FP32 noise, at most
        ``NOISE_ROWS`` sampled rows)."""
        return not self.use_fp64 and 1 <= num_reqs * n_spec <= NOISE_ROWS

    def fork_noise(self, spec, num_reqs, n_spec):
        """Launch this draft step's Gumbel noise on a side stream (it depends only
        on the seeds, the positions and the token ids, not on any logits), so it
        runs beside the lm_head; ``join`` before the steps read it."""
        if self.noise is None:
            self.noise = torch.empty(
                NOISE_ROWS * self.vocab, dtype=torch.float32, device=self.device
            )
            self.stream = torch.cuda.Stream(device=self.device)
        main = torch.cuda.current_stream(self.device)
        assert self.stream is not None
        self.stream.wait_stream(main)
        noise = self.noise[: num_reqs * n_spec * self.vocab].view(
            num_reqs, n_spec, self.vocab
        )
        idx_map = spec.sample_idx_mapping[: num_reqs * n_spec].view(num_reqs, n_spec)
        sample_pos = spec.sample_pos[: num_reqs * n_spec].view(num_reqs, n_spec)
        with torch.cuda.stream(self.stream):
            grid = (triton.cdiv(self.vocab, NOISE_BLOCK), num_reqs * n_spec)
            _markov_noise_kernel[grid](
                noise,
                noise.stride(0),
                noise.stride(1),
                idx_map,
                idx_map.stride(0),
                sample_pos,
                sample_pos.stride(0),
                spec.seeds,
                self.vocab,
                K=n_spec,
                BLOCK=NOISE_BLOCK,
                SALT=_DRAFT_NOISE_SALT,
                num_warps=4,
            )
        return noise

    def join(self):
        torch.cuda.current_stream(self.device).wait_stream(self.stream)

    def run(self, spec, num_reqs, base_logits, w1, w2, noise=None):
        """``base_logits``: [num_reqs, k, V]; writes the draft logits cache and
        ``spec.draft_tokens[:num_reqs]``. ``noise``: ``fork_noise``'s (joined), or
        None to hash the noise in the steps."""
        n_spec = base_logits.shape[1]
        use_noise = noise is not None
        if noise is None:
            noise = self.prev  # unused pointer (NOISE=False)
            noise_strides = (0, 0)
        else:
            noise_strides = (noise.stride(0), noise.stride(1))
        rb = RB
        grid = (triton.cdiv(num_reqs, rb), self.nb)
        idx_map = spec.sample_idx_mapping[: num_reqs * n_spec].view(num_reqs, n_spec)
        sample_pos = spec.sample_pos[: num_reqs * n_spec].view(num_reqs, n_spec)
        cache = spec.draft_logits.values
        tokens = spec.draft_tokens
        for step in range(n_spec):
            out_max, out_arg = self.bufs
            _markov_step_kernel[grid](
                self.prev,
                spec.input_buffers.input_ids,
                spec._anchor_idx,
                w1,
                w2,
                base_logits[:, step],
                base_logits.stride(0),
                cache,
                cache.stride(0),
                cache.stride(1),
                spec.draft_logits.row_mapping,
                idx_map,
                idx_map.stride(0),
                sample_pos,
                sample_pos.stride(0),
                spec.temperature,
                spec.seeds,
                out_max,
                out_arg,
                noise,
                *noise_strides,
                num_reqs,
                self.vocab,
                self.nb,
                STEP=step,
                RANK=self.rank,
                VB=self.vb,
                VT=VT,
                RB=rb,
                USE_FP64=self.use_fp64,
                SALT=_DRAFT_NOISE_SALT,
                NOISE=use_noise,
                num_warps=4,
                num_stages=2,
            )
            _markov_select_kernel[(num_reqs,)](
                out_max,
                out_arg,
                self.prev,
                tokens,
                tokens.stride(0),
                num_reqs,
                self.nb,
                STEP=step,
                NB_POW2=self.nb_pow2,
            )


def why_not(spec, model, num_reqs):
    """Why this draft step takes vLLM's loop, or None when the fused launches
    compute the same sampling."""
    head = model.model.markov_head
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
    if hasattr(model, "apply_output_logit_transform"):
        return "an output logit transform"
    lp = model.logits_processor
    if lp.scale != 1.0 or lp.soft_cap is not None:
        return "scaled logits"
    w1, w2 = head.markov_w1.weight, head.markov_w2.weight
    if w1.dtype != torch.bfloat16 or w2.dtype != torch.bfloat16:
        return "Markov weights not bf16"
    rank = w1.shape[1]
    if w2.shape[1] != rank or rank % 16 or rank & (rank - 1) or rank > 512:
        return f"Markov rank {rank}"
    if not (w1.is_contiguous() and w2.is_contiguous()):
        return "Markov weight layout"
    if head.markov_w2.quant_method.__class__.__name__ != "UnquantizedEmbeddingMethod":
        return "a quantized Markov head"
    if num_reqs < 1:
        return "no requests"
    return None


_FUSED: dict = {}


def sample_sequential(spec, model, num_reqs: int, head_hidden: torch.Tensor) -> bool:
    """``DSparkSpeculator._sample_sequential`` with the fused launches. Returns
    False, having done nothing, when this step takes vLLM's loop."""
    why = why_not(spec, model, num_reqs)
    if why is not None:
        logger.info_once("DSpark fused Markov sampling off: %s", why)
        return False
    n_spec = spec.num_speculative_steps
    num_sample = num_reqs * n_spec
    head = model.model.markov_head
    w1, w2 = head.markov_w1.weight, head.markov_w2.weight
    fused = _fused(spec, w1, w2, head_hidden.device)
    noise = None
    if fused.noise_rows(num_reqs, n_spec):
        noise = fused.fork_noise(spec, num_reqs, n_spec)
    sample_hidden = head_hidden[spec.sample_indices[:num_sample]]
    base_logits = model.compute_draft_logits(sample_hidden)
    if noise is not None:
        fused.join()
    cache = spec.draft_logits.values
    vocab = base_logits.shape[-1]
    if (
        base_logits.dtype not in (torch.bfloat16, torch.float32)
        or base_logits.dtype != cache.dtype
        or base_logits.stride(-1) != 1
        or w2.shape[0] != vocab
        or cache.shape[-1] < vocab
    ):
        logger.info_once(
            "DSpark fused Markov sampling off: %s base logits, %s cache",
            base_logits.dtype,
            cache.dtype,
        )
        return _loop_rest(spec, model, num_reqs, base_logits)
    fused.run(spec, num_reqs, base_logits.view(num_reqs, n_spec, vocab), w1, w2, noise)
    return True


def _fused(spec, w1, w2, device):
    """The speculator's FusedMarkov (markov_w2's vocabulary; ``sample_sequential``
    checks it against the base logits')."""
    vocab, rank = w2.shape[0], w1.shape[1]
    key = (device, vocab, rank, bool(spec.use_fp64_gumbel))
    fused = _FUSED.get(key)
    if fused is None or fused.max_rows < spec.max_num_reqs:
        fused = FusedMarkov(vocab, rank, spec.max_num_reqs, device, key[-1])
        _FUSED[key] = fused
        logger.info_once(
            "DSpark fused Markov sampling: vocabulary %d, rank %d, %d programs",
            vocab,
            rank,
            fused.nb,
        )
    return fused


def _loop_rest(spec, model, num_reqs, base_logits) -> bool:
    """The loop of vLLM from its base logits on (base logits computed once)."""
    n_spec = spec.num_speculative_steps
    num_sample = num_reqs * n_spec
    base_logits = base_logits.view(num_reqs, n_spec, -1)
    idx_map = spec.sample_idx_mapping[:num_sample].view(num_reqs, n_spec)
    sample_pos = spec.sample_pos[:num_sample].view(num_reqs, n_spec)
    prev = spec.input_buffers.input_ids[spec._anchor_idx[:num_reqs]]
    for i in range(n_spec):
        bias = model.markov_bias(model.markov_embed(prev))
        draft = spec._sample_logits(
            base_logits[:, i] + bias, idx_map[:, i], sample_pos[:, i], i
        )
        spec.draft_tokens[:num_reqs, i] = draft
        prev = draft
    return True
