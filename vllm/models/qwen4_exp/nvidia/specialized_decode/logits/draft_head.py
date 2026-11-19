# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The MTP draft passes' greedy token at TP1 from an FP8 copy of the LM head,
exactly equal to the BF16 head's.

The problem
-----------
Each decode step runs three MTP draft passes, and each ends with the LM head
over the whole vocabulary: `[rows, 2560] x [248320, 2560]^T` at TP1. The
drafts are greedy, so a pass needs only every row's argmax, but the stock head
streams the 1.27 GB BF16 weight: about 182 us a pass on GB300, the largest
single cost of the draft passes. `DraftHead` reads an FP8 copy of the
weight (half the bytes) to find the few vocabulary entries that could be the
argmax, then decides among them with the BF16 weight. The FP8 copy only prunes
entries that provably cannot win, so the token is the BF16 head's.

What "the BF16 head's token" is
-------------------------------
At TP1 on GB300 the draft head is `F.linear` (cuBLAS): Qwen4Exp's skinny
GEMM has no plan for the head's shape on SM103 (`low_latency_gemm.py`). Its
kernel adds each logit's 2560 products with tensor-core MMAs in k order, and
the tensor cores add 16 consecutive products at a time to the FP32
accumulator in one fused step: the products (exact) and the accumulator are
aligned to the largest of their nominal exponents `E` (a product's is the
sum of its factors' exponents), cut toward zero to multiples of
`2^(E - 25)`, added exactly, and the sum is rounded toward zero to FP32. A
warp's chain of `mma.sync.m16n8k16` over the same 16-column groups in k
order does the same, bit for bit: on GB300, 1.77 M FP32 sums of real draft
rows all equal cuBLAS's. Call `R_v` that sum for entry `v` and `l_v =
rn(R_v)` its BF16 logit (the GEMM's epilogue rounds to nearest). The draft
token is the first index of the largest BF16 logit, NaN as -inf
(`choice.py`). The rescoring computes `R_v` with that MMA chain, so it
reproduces `l_v` bit for bit; the tests compare every logit of full
rescoring with `F.linear`. The head is used only where the stock GEMM is
that cuBLAS call (`processor.bind_draft_head`).

The prefilter
-------------
* Weight (once, `draft_head_weights`): every vocabulary row `w_v`
  gets a power-of-two scale `b_v` (the smallest with `max|w_v| / b_v <=
  448`) and E4M3 codes `p_v`; `w^_v = b_v p_v` and `dw_v = w_v - w^_v`.
  `wstat` holds `b_v` and upper bounds of `|w_v|`, `|w^_v|` and of
  `|dw_vb|` for the 20 blocks `b` of 128 columns, in BF16.
* Activation (`HeadQuantKernel`): `h` becomes two code rows, `hi` with
  scale `a_hi` (as above) and `lo`, the codes of the remainder `d = h -
  a_hi hi` with scale `a_lo = a_hi / 16`; `h^ = a_hi hi + a_lo lo` and
  `dh = h - h^`. Every step is exact in FP32: the operands have at most
  eight significant bits and the scales are powers of two.
* GEMM (cuBLASLt through `torch._scaled_mm`): `S = [hi; lo] p^T` in FP32,
  and `s~_v = a_hi b_v S_hi[v] + a_lo b_v S_lo[v]`.

The bound
---------
With `z_v = h . w_v` exact, `z_v - h^ . w^_v = dh . w_v + h^ . dw_v`, so by
Cauchy-Schwarz (the second term block by block, `|h^_b| <= |h_b| +
|dh_b|`)::

    |z_v - h^ . w^_v| <= |dh| |w_v| + sum_b |h^_b| |dw_vb|              (1)

The FP8 GEMM adds 32 consecutive code products at a time in the same fused
way (on GB300 every FP32 output for real code rows matches that model). Each
of its 80 steps cuts at most 33 terms by less than `2^(E - 25)` and rounds
once toward zero, and `2^E` is at most the largest term's magnitude (`2^-6`
for an E4M3 subnormal factor). Summed over the steps, with `|a_hi hi| +
|a_lo lo| <= |h| + 3 |d|`, the GEMM is within `9e-5 (|h| + 3 |d|) |w^_v|`
of `h^ . w^_v`, plus less than `3e-10 |h| |w^_v|`. The bound allows
**8 gamma_K** (`gamma_K = 1.5262e-4`)::

    |s~_v - h^ . w^_v| <= 8 gamma_K (|h| + 3 |d|) |w^_v| + 2^-22 |s~_v|     (2)

(the last term covers the rounding of `s~_v`'s sum). The same count for the
stock sum's 160 steps of 17 terms gives `|R_v - z_v| <= 1.002e-4 sum_k |h_k
w_vk|`, so `|R_v - z_v| <= gamma_K |h| |w_v|` (3). Both need magnitudes
far from FP32's subnormals, which the kernels ensure: rows with `max|h| <
2^-20` rescore every entry, and the build refuses weight rows that are not
zero and have `max|w_v| < 2^-40`. Then `|h| |w^_v| >= 2^-62` for every
non-zero row, so (2)'s 8 gamma_K, more than ten times the GEMM's `9e-5`,
also covers the kernels' own FP32 roundings of `s~_v`'s products (`2^-24`
of `(|h| + 3 |d|) |w^_v|` each, or `2^-150` below the normal range), and
in (3) BF16 subnormals add less than the margin. With `E_v` the sum of
(1)-(3), every factor rounded up, `|s~_v - R_v| <= E_v`. (2) and (3) rest on
the accumulation the hardware
performs, measured on GB300, not on a PTX guarantee; were they to fail on
another GPU or library version, only a draft token could change, never the
target's output: drafts are verified.

The candidates
--------------
Let `L* = max_u (s~_u - E_u)`, the largest lower bound, at `u*`. The stock
winner `w` has `l_w >= l_u* = rn(R_u*) >= rn(L*)` (rounding to nearest
BF16 is monotone) and `l_w <= rn(s~_w + E_w)`. So every `v` with
`rn(s~_v + E_v) < rn(L*)` has `l_v < l_w`: it can neither win nor tie, and
the **candidates** `{v : rn(s~_v + E_v) >= rn(L*)}` contain the stock winner.
Comparing rounded bounds keeps every entry that could tie with the winner in
BF16 and win on a lower index. The kernels round lower bounds down and upper
bounds up.

A row with a bound that is not finite (`h` with NaN or infinity), or whose
largest `|h|` is below `2^-20` (no real draft row; above, the quantization
steps and the bound stay far from FP32's subnormals), rescores every entry
the same way: the answer never depends on
the path, only its time does. A row of zeros has `R_v = +0` for every `v`,
so its token is index 0.

Kernels
-------
::

    HeadQuantKernel   h                     -> codes [2 rows, 2560], hstat;
                                               resets the rows' state words
    torch._scaled_mm  codes, FP8 weight     -> S [2 rows, vocabulary] FP32
    HeadBoundKernel   S, hstat, wstat       -> per 512 entries the largest upper
                                               bound; per row L* (atomic max)
    HeadSelectKernel  S, hstat, wstat       -> per 512 entries and every row: the
                                               candidates (if the block's upper
                                               bound reaches L*), appended to
                                               the pass's list
    HeadRescoreKernel list, h, w            -> the candidates' BF16 logits, 16
                                               a tile, the rows' best keys; the
                                               last CTA writes the tokens

They run on the caller's stream with programmatic dependent launch. Inputs
already complete when a grid starts (the weight statistics; for the select
and rescore kernels also `hstat` and the draft rows) are read before its
grid-dependency wait, and the bound and select kernels request every row's
sums at once, so more rows add little latency. A
rescore CTA takes tiles of 16 listed candidates (any rows): its threads copy
the 16 weight rows to shared memory in four groups along k, and one warp runs
the 160 MMAs as the groups land, with the 16 rows as A and the draft rows (at
most 8, in L1) as B; each candidate takes its own row's column. A lane feeds
an MMA four consecutive columns of each row (one 8-byte load) instead of the
PTX fragment's pairs: the 16 columns of a step are the same, and the fused
step does not depend on their order (the tests check the logits bit for
bit). The rescore grid is two CTAs a multiprocessor, so the tiles of a
typical pass run at once and a pass with many candidates (a flat draft
distribution, or the full rescoring) takes more rounds. The CTAs fold their
best keys into the rows' words with GPU-scope atomic maxima and count
themselves; in the last to arrive, one thread a row writes the row's token,
and the words and the list are reset (`choice.py`'s pattern). No CTA waits
for another, so any grid is live. The quant kernel resets the row words the
next kernels accumulate into after its grid-dependency wait, when every
kernel of the previous pass has completed: like the rest of the package, this
assumes that every kernel between two calls on the stream completes only
after its predecessor (it waits for its grid dependency, or is launched
without programmatic dependent launch), and that one `DraftHead` serves one
stream. Broken invariants trap rather than corrupt silently: an arrival count
or a list longer than its capacity (unreset), or a row that reaches the last CTA
without a key (its token would leave the vocabulary).

Cost: an FP8 copy and statistics of 0.65 GB at TP1, built from the head's
weight with the executor (before memory profiling, which counts them), only
when the deployment asks for it (`draft_head_fp8_prefilter`, `..`).
"""

from functools import cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint32
from cutlass.cute.runtime import make_fake_stream

from ..boundary._ptx import store_u32
from ..hc._ptx import (
    copy_async_16,
    evict_first_policy,
    load_weight_x4,
    load_x4,
    mma_bf16_16816,
)
from ._head_ptx import (
    abs_or_inf,
    add_rn,
    add_ru,
    atomic_add_u32,
    atomic_max_u32,
    bf16_bits,
    code_scale,
    copy_async_commit,
    copy_async_wait_group,
    fma_ru,
    from_ordered_f32,
    load_f32,
    load_f32_l2,
    load_f32x2,
    load_shared_f32,
    load_shared_s32,
    load_shared_u64,
    load_shared_x2,
    load_u32,
    load_u32x2,
    logit_key,
    mul_rn,
    mul_ru,
    not_finite,
    ordered_f32,
    prefetch_l1,
    quantize_e4m3x2,
    round_bf16,
    shared_add_s32,
    sqrt_ru,
    store_f32,
    store_shared_f32,
    store_shared_s32,
    store_shared_u64,
    store_u16,
    store_x2,
    sub_rd,
    sub_rn,
    trap,
    widen_bf16x2,
)
from ._ptx import (
    arrive_acq_rel_u32,
    atomic_max_u64,
    exchange_u64,
    max_u64,
    min_u32,
    store_u64,
)

HIDDEN = 2560
BLOCK_COLUMNS = 128
BLOCKS = HIDDEN // BLOCK_COLUMNS  # 20 column blocks of the bound
WSTAT = 24
"""BF16 words of `wstat` per vocabulary entry: `b_v`, `|w_v|`,
`|w^_v|`, 0, then `|dw_vb|` for the 20 column blocks."""
HSTAT = 32
"""FP32 words of `hstat` per row: `a_hi`, `a_lo`, the factor of
`|w_v|` (`|dh| + gamma_K |h|`), the factor of `|w^_v|` (`8 gamma_K (|h|
+ 3 |d|)`), `|h|`, three unused, then `|h_b| + |dh_b|` for the 20
blocks."""
STATE = 4
"""32-bit words of per-row state: rescore every entry, zero row, `L*` (the
ordered bits of `ordered_f32`, 0 below every value), candidates rescored."""
MAX_ROWS = 8
"""Draft rows the kernels serve; larger passes keep the stock head."""
E4M3_MAX = 448.0
FLOAT_MAX = 3.4028234663852886e38
QUANT_THREADS = HIDDEN // 8  # 320: eight columns a thread
BOUND_CHUNK = 512  # vocabulary entries per bound CTA, two a thread
BOUND_THREADS = BOUND_CHUNK // 2
RESCORE_THREADS = 128
TILE = 16  # candidates a rescoring MMA chain takes (A rows)
STEPS = HIDDEN // 16  # MMAs of a chain
ROW_BYTES = HIDDEN * 2 + 32  # a BF16 row in shared memory (padded: no bank conflicts)
ROW_CHUNKS = HIDDEN // 8  # 16-byte pieces of a row
STAGES = 4  # copy groups of a tile along k (the chain starts on the first)
STAGE_CHUNKS = ROW_CHUNKS // STAGES
STAGE_STEPS = STEPS // STAGES
ENTRY_BITS = 20  # a listed candidate is row << ENTRY_BITS | entry
RESCORE_SHARED = (
    TILE * ROW_BYTES  # a tile of weight rows
    + TILE * 4  # its entries
    + TILE * 8 * 4  # the chain's sums
    + MAX_ROWS * 8  # best keys
    + 1024  # alignment
)
# gamma_K = K u / (1 - K u) for K = 2560, u = 2^-24, rounded up (module
# docstring, "The bound"). The FP8 GEMM's term is 8 gamma_K.
GAMMA_K = 1.5262e-4
GAMMA_FP8 = 8.0 * GAMMA_K
SUM_SLACK = 2.0**-22
# Rows whose largest |h| is below this rescore every entry; weight rows must
# be zero or have a largest |w| of at least the second. Together they keep the
# quantization and every FP32 product of the bound far above the subnormals
# (module docstring, "The bound").
MIN_AMAX = 2.0**-20
MIN_WEIGHT_AMAX = 2.0**-40


def _round_up_bf16(values: torch.Tensor) -> torch.Tensor:
    """Non-negative FP64 values rounded up to BF16."""
    rounded = values.to(torch.bfloat16)
    low = rounded.double() < values
    bits = rounded.view(torch.int16)
    return torch.where(low, bits + 1, bits).view(torch.bfloat16)


def draft_head_weights(
    weight: torch.Tensor, rows_per_pass: int = 2048
) -> tuple[torch.Tensor, torch.Tensor]:
    """The E4M3 codes and statistics of a BF16 LM head weight (module
    docstring, "The prefilter").

    Args:
        weight: `[vocabulary, 2560]` BF16, finite, every row zero or with a
            largest `|w|` of at least `2^-100`.
        rows_per_pass: Rows converted at a time (bounds the FP64
            temporaries, about 40 MB each at 2048 rows).

    Returns:
        `codes` `[vocabulary, 2560]` E4M3 (`w^_v = b_v codes_v`) and
        `wstat` `[vocabulary, 24]` BF16.

    """
    vocab, hidden = weight.shape
    if hidden != HIDDEN or weight.dtype != torch.bfloat16:
        raise ValueError(f"the draft head needs a [V, {HIDDEN}] BF16 weight")
    codes = torch.empty(vocab, hidden, dtype=torch.float8_e4m3fn, device=weight.device)
    wstat = torch.zeros(vocab, WSTAT, dtype=torch.bfloat16, device=weight.device)
    for start in range(0, vocab, rows_per_pass):
        w = weight[start : start + rows_per_pass].double()
        if not bool(torch.isfinite(w).all()):
            raise ValueError("the draft head needs a finite LM head weight")
        amax = w.abs().amax(-1)
        if bool(((amax > 0) & (amax < MIN_WEIGHT_AMAX)).any()):
            raise ValueError(
                "the draft head needs LM head rows that are zero or have a "
                "largest |w| of at least 2^-40"
            )
        scale = torch.exp2(torch.ceil(torch.log2(amax / E4M3_MAX)))
        scale = torch.where(amax > 0, scale, torch.ones_like(scale))
        # A power of two chosen so that max |w| / scale <= 448; recheck it.
        scale = torch.where(amax / scale > E4M3_MAX, scale * 2, scale)
        code = (w / scale[:, None]).float().to(torch.float8_e4m3fn)
        hat = code.double() * scale[:, None]
        delta = w - hat
        stat = torch.zeros(w.shape[0], WSTAT, dtype=torch.float64, device=w.device)
        stat[:, 0] = scale
        stat[:, 1] = w.square().sum(-1).sqrt()
        stat[:, 2] = hat.square().sum(-1).sqrt()
        stat[:, 4 : 4 + BLOCKS] = (
            delta.square().unflatten(-1, (BLOCKS, BLOCK_COLUMNS)).sum(-1).sqrt()
        )
        # FP64 sums of squares of exact values are within ~1e-13 relative;
        # 2^-40 of slack before rounding up keeps every norm an upper bound.
        stat[:, 1:] = stat[:, 1:] * (1 + 2.0**-40)
        codes[start : start + rows_per_pass] = code
        wstat[start : start + rows_per_pass] = _round_up_bf16(stat)
    return codes, wstat


class HeadQuantKernel:
    """`h` into its hi and lo E4M3 code rows and `hstat`, one CTA a row
    (module docstring, "The prefilter"); resets the row's state words."""

    @cute.jit
    def __call__(
        self,
        h: Int64,
        h_stride: Int32,
        codes: Int64,
        hstat: Int64,
        state: Int64,
        rows: Int32,
        stream: cuda.CUstream,
    ):
        """Launch for `rows` rows of `h` (BF16, row stride `h_stride`
        values): codes `[2 rows, 2560]` (the hi rows, then the lo rows),
        `hstat` and `state` per row."""
        self.kernel(h, h_stride, codes, hstat, state, rows).launch(
            grid=(rows, 1, 1),
            block=(QUANT_THREADS, 1, 1),
            stream=stream,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self,
        h: Int64,
        h_stride: Int32,
        codes: Int64,
        hstat: Int64,
        state: Int64,
        rows: Int32,
    ):
        """One row, eight columns a thread."""
        tid, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(tid // 32)
        lane = tid % 32
        warps = QUANT_THREADS // 32
        allocator = cutlass.memory.SmemAllocator()
        warp_max = allocator.allocate_array(Float32, warps, byte_alignment=16).toint()
        warp_sums = allocator.allocate_array(
            Float32, 3 * warps, byte_alignment=16
        ).toint()

        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()

        words = load_x4(h + (Int64(row) * Int64(h_stride) + Int64(tid * 8)) * 2)
        values = []
        for i in cutlass.range_constexpr(4):
            low, high = widen_bf16x2(words[i])
            values.append(low)
            values.append(high)
        # max |h| over the row (NaN and infinities count as +inf).
        m = Float32(0.0)
        for i in cutlass.range_constexpr(8):
            m = cute.arch.fmax(m, abs_or_inf(values[i]))
        for offset in (16, 8, 4, 2, 1):
            m = cute.arch.fmax(m, cute.arch.shuffle_sync_bfly(m, offset=offset))
        if lane == 0:
            store_shared_f32(warp_max + warp * 4, m)
        cute.arch.sync_threads()
        amax = load_shared_f32(warp_max)
        for w in cutlass.range_constexpr(1, warps):
            amax = cute.arch.fmax(amax, load_shared_f32(warp_max + w * 4))
        # A usable row: finite, and its largest value far above the subnormals.
        a_hi = Float32(1.0)
        full = Uint32(1)
        zero = Uint32(0)
        if amax == Float32(0.0):
            full = Uint32(0)
            zero = Uint32(1)
        if (amax >= Float32(MIN_AMAX)) & (amax <= Float32(FLOAT_MAX)):
            a_hi = code_scale(amax)
            full = Uint32(0)
        a_lo = mul_rn(a_hi, Float32(0.0625))

        # hi codes, the remainder d, lo codes, the residual dh (all exact), and
        # the squared norms of h, d and dh, rounded up.
        hi_codes = []
        lo_codes = []
        h2 = Float32(0.0)
        d2 = Float32(0.0)
        r2 = Float32(0.0)
        for i in cutlass.range_constexpr(4):
            a = values[2 * i]
            b = values[2 * i + 1]
            code, qa, qb = quantize_e4m3x2(a, b, a_hi)
            da = sub_rn(a, qa)
            db = sub_rn(b, qb)
            code_lo, la, lb = quantize_e4m3x2(da, db, a_lo)
            ra = sub_rn(da, la)
            rb = sub_rn(db, lb)
            hi_codes.append(code)
            lo_codes.append(code_lo)
            h2 = fma_ru(a, a, h2)
            h2 = fma_ru(b, b, h2)
            d2 = fma_ru(da, da, d2)
            d2 = fma_ru(db, db, d2)
            r2 = fma_ru(ra, ra, r2)
            r2 = fma_ru(rb, rb, r2)
        sixteen = Uint32(16)
        store_x2(
            codes + Int64(row) * HIDDEN + Int64(tid * 8),
            hi_codes[0] | (hi_codes[1] << sixteen),
            hi_codes[2] | (hi_codes[3] << sixteen),
        )
        store_x2(
            codes + Int64(rows + row) * HIDDEN + Int64(tid * 8),
            lo_codes[0] | (lo_codes[1] << sixteen),
            lo_codes[2] | (lo_codes[3] << sixteen),
        )

        # Column block tid // 16 (16 threads of eight columns): |h_b| + |dh_b|.
        for offset in (8, 4, 2, 1):
            h2 = add_ru(h2, cute.arch.shuffle_sync_bfly(h2, offset=offset))
            d2 = add_ru(d2, cute.arch.shuffle_sync_bfly(d2, offset=offset))
            r2 = add_ru(r2, cute.arch.shuffle_sync_bfly(r2, offset=offset))
        out = hstat + Int64(row) * (HSTAT * 4)
        if lane % 16 == 0:
            store_f32(
                out + Int64((8 + tid // 16) * 4), add_ru(sqrt_ru(h2), sqrt_ru(r2))
            )
        h2 = add_ru(h2, cute.arch.shuffle_sync_bfly(h2, offset=16))
        d2 = add_ru(d2, cute.arch.shuffle_sync_bfly(d2, offset=16))
        r2 = add_ru(r2, cute.arch.shuffle_sync_bfly(r2, offset=16))
        if lane == 0:
            store_shared_f32(warp_sums + warp * 4, h2)
            store_shared_f32(warp_sums + (warps + warp) * 4, d2)
            store_shared_f32(warp_sums + (2 * warps + warp) * 4, r2)
        cute.arch.sync_threads()
        if tid == 0:
            total_h = Float32(0.0)
            total_d = Float32(0.0)
            total_r = Float32(0.0)
            for w in cutlass.range_constexpr(warps):
                total_h = add_ru(total_h, load_shared_f32(warp_sums + w * 4))
                total_d = add_ru(total_d, load_shared_f32(warp_sums + (warps + w) * 4))
                total_r = add_ru(
                    total_r, load_shared_f32(warp_sums + (2 * warps + w) * 4)
                )
            norm_h = sqrt_ru(total_h)
            norm_d = sqrt_ru(total_d)
            norm_r = sqrt_ru(total_r)
            store_f32(out, a_hi)
            store_f32(out + 4, a_lo)
            store_f32(out + 8, fma_ru(Float32(GAMMA_K), norm_h, norm_r))
            store_f32(
                out + 12,
                mul_ru(Float32(GAMMA_FP8), fma_ru(Float32(3.0), norm_d, norm_h)),
            )
            store_f32(out + 16, norm_h)
            row_state = state + Int64(row) * (STATE * 4)
            store_u32(row_state, full)
            store_u32(row_state + 4, zero)
            store_u32(row_state + 8, Uint32(0))
            store_u32(row_state + 12, Uint32(0))


def _entry_stats(wstat: Int64, entry: Int64, policy: Int64) -> list:
    """The 24 statistics of vocabulary entry `entry`, as FP32 values."""
    out = []
    for part in range(3):
        words = load_weight_x4(wstat + entry * (WSTAT * 2) + part * 16, policy)
        for i in range(4):
            low, high = widen_bf16x2(words[i])
            out.append(low)
            out.append(high)
    return out


def _row_stats(hs: Int32) -> list:
    """The 24 `hstat` values the bounds use (`a_hi`, `a_lo`, the two
    norm factors, the 20 block terms) from shared memory at `hs`, loaded
    together into registers."""
    return [load_shared_f32(hs + i * 4) for i in range(4)] + [
        load_shared_f32(hs + (8 + b) * 4) for b in range(BLOCKS)
    ]


def _bounds(hv: list, stats: list, s_hi: Float32, s_lo: Float32):
    """The lower and upper bound of `R_v` (module docstring, "The bound")
    from the entry's two FP8 sums, the row's `_row_stats` and the entry's
    statistics."""
    scale = stats[0]
    approx = add_rn(
        mul_rn(mul_rn(hv[0], scale), s_hi), mul_rn(mul_rn(hv[1], scale), s_lo)
    )
    e = mul_ru(hv[2], stats[1])
    e = fma_ru(hv[3], stats[2], e)
    for b in range(BLOCKS):
        e = fma_ru(hv[4 + b], stats[4 + b], e)
    e = fma_ru(cute.arch.fmax(approx, -approx), Float32(SUM_SLACK), e)
    return sub_rd(approx, e), add_ru(approx, e)


class HeadBoundKernel:
    """Per 512 vocabulary entries and row, the largest upper bound; per row,
    the largest lower bound `L*` (an atomic max of ordered bits), and full
    rescoring when a bound is not finite.

    Args:
        vocab: Entries of the weight (columns of the GEMM output), even.
        valid: The first `valid` are vocabulary entries; the rest never win.

    """

    def __init__(self, vocab: int, valid: int) -> None:
        assert vocab % 2 == 0 and 0 < valid <= vocab
        self.vocab = vocab
        self.valid = valid
        self.chunks = -(-vocab // BOUND_CHUNK)

    @cute.jit
    def __call__(
        self,
        scores: Int64,
        hstat: Int64,
        wstat: Int64,
        upper: Int64,
        state: Int64,
        rows: Int32,
        stream: cuda.CUstream,
    ):
        """Launch for `rows` rows: the GEMM output (FP32 `[2 rows,
        vocab]`), `hstat` and `wstat`; writes `upper [rows, chunks]`
        and the rows' state words."""
        self.kernel(scores, hstat, wstat, upper, state, rows).launch(
            grid=(self.chunks, 1, 1),
            block=(BOUND_THREADS, 1, 1),
            stream=stream,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self,
        scores: Int64,
        hstat: Int64,
        wstat: Int64,
        upper: Int64,
        state: Int64,
        rows: Int32,
    ):
        """Two entries a thread, every row."""
        tid, _, _ = cute.arch.thread_idx()
        chunk, _, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(tid // 32)
        lane = tid % 32
        warps = BOUND_THREADS // 32
        allocator = cutlass.memory.SmemAllocator()
        hs_all = allocator.allocate_array(
            Float32, MAX_ROWS * HSTAT, byte_alignment=16
        ).toint()
        reduced = allocator.allocate_array(
            Float32, MAX_ROWS * 3 * warps, byte_alignment=16
        ).toint()
        policy = evict_first_policy()
        first = chunk * BOUND_CHUNK + 2 * tid
        # The entries' statistics stream in while the GEMM finishes (an entry
        # past the weight reads the last pair and counts nothing).
        pair = Int64(min_u32(Uint32(first), Uint32(self.vocab - 2)))
        stats0 = _entry_stats(wstat, pair, policy)
        stats1 = _entry_stats(wstat, pair + 1, policy)

        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()

        # Every row's sums are requested at once (a row past ``rows`` reads
        # the last row's and is skipped), so the rows cost one memory latency.
        sums_hi = []
        sums_lo = []
        for r in cutlass.range_constexpr(MAX_ROWS):
            row = Int64(cute.min(Int32(r), rows - 1))
            sums_hi.append(load_f32x2(scores + (row * self.vocab + pair) * 4))
            sums_lo.append(
                load_f32x2(scores + ((Int64(rows) + row) * self.vocab + pair) * 4)
            )
        # ``hstat`` only after the wait: when this grid starts, the GEMM
        # (launched by cuBLAS) may not have waited for the quant kernel yet.
        i = tid
        while i < rows * HSTAT:
            store_shared_f32(hs_all + i * 4, load_f32(hstat + Int64(i) * 4))
            i = i + BOUND_THREADS
        cute.arch.sync_threads()
        neg_inf = Float32(float("-inf"))
        for r in cutlass.range_constexpr(MAX_ROWS):
            if Int32(r) < rows:
                hv = _row_stats(hs_all + r * (HSTAT * 4))
                lower = neg_inf
                up = neg_inf
                bad = Float32(0.0)
                if first < self.valid:
                    low0, up0 = _bounds(hv, stats0, sums_hi[r][0], sums_lo[r][0])
                    bad = cute.arch.fmax(bad, not_finite(low0))
                    bad = cute.arch.fmax(bad, not_finite(up0))
                    lower = cute.arch.fmax(lower, low0)
                    up = cute.arch.fmax(up, up0)
                if first + 1 < self.valid:
                    low1, up1 = _bounds(hv, stats1, sums_hi[r][1], sums_lo[r][1])
                    bad = cute.arch.fmax(bad, not_finite(low1))
                    bad = cute.arch.fmax(bad, not_finite(up1))
                    lower = cute.arch.fmax(lower, low1)
                    up = cute.arch.fmax(up, up1)
                for offset in (16, 8, 4, 2, 1):
                    lower = cute.arch.fmax(
                        lower, cute.arch.shuffle_sync_bfly(lower, offset=offset)
                    )
                    up = cute.arch.fmax(
                        up, cute.arch.shuffle_sync_bfly(up, offset=offset)
                    )
                    bad = cute.arch.fmax(
                        bad, cute.arch.shuffle_sync_bfly(bad, offset=offset)
                    )
                if lane == 0:
                    base = reduced + (r * 3 * warps + warp) * 4
                    store_shared_f32(base, lower)
                    store_shared_f32(base + warps * 4, up)
                    store_shared_f32(base + 2 * warps * 4, bad)
        cute.arch.sync_threads()
        if tid < rows:
            base = reduced + tid * (3 * warps * 4)
            lower = load_shared_f32(base)
            up = load_shared_f32(base + warps * 4)
            bad = load_shared_f32(base + 2 * warps * 4)
            for w in cutlass.range_constexpr(1, warps):
                lower = cute.arch.fmax(lower, load_shared_f32(base + w * 4))
                up = cute.arch.fmax(up, load_shared_f32(base + (warps + w) * 4))
                bad = cute.arch.fmax(bad, load_shared_f32(base + (2 * warps + w) * 4))
            store_f32(upper + (Int64(tid) * self.chunks + Int64(chunk)) * 4, up)
            row_state = state + Int64(tid) * (STATE * 4)
            if bad > Float32(0.0):
                atomic_max_u32(row_state, Uint32(1))
            else:
                if lower > neg_inf:
                    atomic_max_u32(row_state + 8, ordered_f32(lower))


class HeadSelectKernel:
    """The candidates of 512 vocabulary entries for every row, appended to the
    pass's candidate list (module docstring, "Kernels").

    Args:
        vocab: Entries of the weight (columns of the GEMM output), even,
            below `2^20`.
        valid: The first `valid` are vocabulary entries; the rest never win.
        debug: List every entry of every non-zero row (tests).

    """

    def __init__(self, vocab: int, valid: int, debug: bool = False) -> None:
        assert vocab % 2 == 0 and 0 < valid <= vocab < 1 << ENTRY_BITS
        self.vocab = vocab
        self.valid = valid
        self.debug = debug
        self.chunks = -(-vocab // BOUND_CHUNK)

    @cute.jit
    def __call__(
        self,
        scores: Int64,
        hstat: Int64,
        wstat: Int64,
        upper: Int64,
        state: Int64,
        listed: Int64,
        count: Int64,
        best: Int64,
        rows: Int32,
        stream: cuda.CUstream,
    ):
        """Launch for `rows` rows: the bound kernel's inputs and outputs;
        appends to the INT32 candidate list `listed` (`rows << 20 | entry`)
        and its length `count`; a zero row's key goes to `best`."""
        self.kernel(
            scores, hstat, wstat, upper, state, listed, count, best, rows
        ).launch(
            grid=(self.chunks, 1, 1),
            block=(BOUND_THREADS, 1, 1),
            stream=stream,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self,
        scores: Int64,
        hstat: Int64,
        wstat: Int64,
        upper: Int64,
        state: Int64,
        listed: Int64,
        count: Int64,
        best: Int64,
        rows: Int32,
    ):
        """One CTA: 512 entries, every row; two entries a thread."""
        tid, _, _ = cute.arch.thread_idx()
        chunk, _, _ = cute.arch.block_idx()
        allocator = cutlass.memory.SmemAllocator()
        hs_all = allocator.allocate_array(
            Float32, MAX_ROWS * HSTAT, byte_alignment=16
        ).toint()
        local = allocator.allocate_array(
            Int32, MAX_ROWS * BOUND_CHUNK, byte_alignment=16
        ).toint()
        # The CTA's candidates per row, then their total and their base.
        counts = allocator.allocate_array(
            Int32, MAX_ROWS + 4, byte_alignment=16
        ).toint()
        policy = evict_first_policy()
        # ``hstat`` is complete: this grid starts after the bound kernel's
        # wait, so after the GEMM, which read the quant kernel's codes. Read
        # from L2: this SM's L1 may still hold an earlier pass's lines.
        i = tid
        while i < rows * HSTAT:
            store_shared_f32(hs_all + i * 4, load_f32_l2(hstat + Int64(i) * 4))
            i = i + BOUND_THREADS
        if tid < MAX_ROWS + 4:
            store_shared_s32(counts + tid * 4, Int32(0))

        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()

        # Every row's state words and block bound are requested at once (a
        # row past ``rows`` reads the last row's and is skipped).
        start = chunk * BOUND_CHUNK
        words = []
        for r in cutlass.range_constexpr(MAX_ROWS):
            row = Int64(cute.min(Int32(r), rows - 1))
            row_state = state + row * (STATE * 4)
            words.append(
                (
                    load_u32(row_state),
                    load_u32(row_state + 4),
                    load_u32(row_state + 8),
                    load_f32(upper + (row * self.chunks + Int64(chunk)) * 4),
                )
            )
        cute.arch.sync_threads()

        # Per row: rescore every entry (``full``), filter by the bound, or
        # skip the block (a zero row, or a block bound below the threshold).
        fulls = []
        thresholds = []
        lives = []
        any_live = Int32(0)
        for r in cutlass.range_constexpr(MAX_ROWS):
            full, zero, floor, block_up = words[r]
            if cutlass.const_expr(self.debug):
                full = Uint32(1)
            in_rows = Int32(r) < rows
            if in_rows & (zero != Uint32(0)) & (chunk == 0) & (tid == 0):
                # Every logit is +0: the lowest index wins.
                atomic_max_u64(best + Int64(r) * 8, logit_key(Float32(0.0), Uint32(0)))
            threshold = round_bf16(from_ordered_f32(floor))
            live = Int32(0)
            if in_rows & (zero == Uint32(0)) & (start < self.valid):
                if full != Uint32(0):
                    live = Int32(1)
                if round_bf16(block_up) >= threshold:
                    live = Int32(1)
            fulls.append(full)
            thresholds.append(threshold)
            lives.append(live)
            any_live = any_live | live
        if any_live != Int32(0):
            # The two entries' statistics and every row's sums at once.
            entries = []
            for e in cutlass.range_constexpr(2):
                v = start + e * BOUND_THREADS + tid
                clamped = Int64(cute.min(v, Int32(self.vocab - 1)))
                sums = []
                for q in cutlass.range_constexpr(MAX_ROWS):
                    src = Int64(cute.min(Int32(q), rows - 1))
                    sums.append(
                        (
                            load_f32(scores + (src * self.vocab + clamped) * 4),
                            load_f32(
                                scores
                                + ((Int64(rows) + src) * self.vocab + clamped) * 4
                            ),
                        )
                    )
                entries.append((v, _entry_stats(wstat, clamped, policy), sums))
            for q in cutlass.range_constexpr(MAX_ROWS):
                if lives[q] != Int32(0):
                    hv = _row_stats(hs_all + q * (HSTAT * 4))
                    for k in cutlass.range_constexpr(2):
                        entry, stats, entry_sums = entries[k]
                        if entry < self.valid:
                            keep = Int32(1)
                            if fulls[q] == Uint32(0):
                                bound = _bounds(
                                    hv, stats, entry_sums[q][0], entry_sums[q][1]
                                )[1]
                                if round_bf16(bound) < thresholds[q]:
                                    keep = Int32(0)
                            if keep != Int32(0):
                                slot = shared_add_s32(counts + MAX_ROWS * 4, Int32(1))
                                store_shared_s32(
                                    local + slot * 4, Int32(q << ENTRY_BITS) | entry
                                )
                                shared_add_s32(counts + q * 4, Int32(1))
        cute.arch.sync_threads()
        found = load_shared_s32(counts + MAX_ROWS * 4)
        if found > 0:
            if tid == 0:
                base = atomic_add_u32(count, Uint32(found))
                # The list holds every entry of every row once; more means
                # the previous call did not reset its length.
                if base + Uint32(found) > Uint32(MAX_ROWS * self.vocab):
                    trap()
                store_shared_s32(counts + (MAX_ROWS + 1) * 4, Int32(base))
            if tid < rows:
                n = load_shared_s32(counts + tid * 4)
                if n > 0:
                    atomic_add_u32(state + Int64(tid) * (STATE * 4) + 12, Uint32(n))
            cute.arch.sync_threads()
            base = load_shared_s32(counts + (MAX_ROWS + 1) * 4)
            i = tid
            while i < found:
                store_u32(
                    listed + Int64(base + i) * 4,
                    Uint32(load_shared_s32(local + i * 4)),
                )
                i = i + BOUND_THREADS


class HeadRescoreKernel:
    """The listed candidates' exact BF16 logits (the stock head's MMA chain),
    16 a tile, tiles spread over the CTAs; the last CTA writes the tokens and
    resets the list (module docstring, "Kernels").

    Args:
        vocab: Entries of the weight, below `2^20`.
        ctas: CTAs of the grid (two a streaming multiprocessor).
        debug: Also write every rescored entry's BF16 logit (tests).

    """

    def __init__(self, vocab: int, ctas: int, debug: bool = False) -> None:
        assert vocab < 1 << ENTRY_BITS and ctas > 0
        self.vocab = vocab
        self.ctas = ctas
        self.debug = debug

    @cute.jit
    def __call__(
        self,
        h: Int64,
        h_stride: Int32,
        weight: Int64,
        listed: Int64,
        count: Int64,
        best: Int64,
        arrivals: Int64,
        tokens: Int64,
        logits: Int64,
        rows: Int32,
        stream: cuda.CUstream,
    ):
        """Launch for `rows` rows: `h` (BF16, row stride `h_stride`),
        the BF16 head `weight`, the candidate list and its length, the
        per-row best keys and the arrival counter (zero between launches), the
        INT64 `tokens [rows]` output and, with `debug`, the BF16
        `logits [rows, vocab]` of the rescored entries."""
        self.kernel(
            h, h_stride, weight, listed, count, best, arrivals, tokens, logits, rows
        ).launch(
            grid=(self.ctas, 1, 1),
            block=(RESCORE_THREADS, 1, 1),
            smem=RESCORE_SHARED,
            stream=stream,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self,
        h: Int64,
        h_stride: Int32,
        weight: Int64,
        listed: Int64,
        count: Int64,
        best: Int64,
        arrivals: Int64,
        tokens: Int64,
        logits: Int64,
        rows: Int32,
    ):
        """One CTA: tiles `cta, cta + ctas, ...` of the list."""
        tid, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(tid // 32)
        lane = tid % 32
        allocator = cutlass.memory.SmemAllocator()
        tile = allocator.allocate_array(
            Uint32, TILE * ROW_BYTES // 4, byte_alignment=16
        ).toint()
        entries = allocator.allocate_array(Int32, TILE, byte_alignment=16).toint()
        sums = allocator.allocate_array(Float32, TILE * 8, byte_alignment=16).toint()
        keys = allocator.allocate_array(Int64, MAX_ROWS, byte_alignment=16).toint()
        flag = allocator.allocate_array(Int32, 4, byte_alignment=16).toint()
        mask = (1 << ENTRY_BITS) - 1
        # The draft rows are complete (this grid starts after the select
        # kernel's wait, so after the GEMM and the quant kernel that read
        # them): pull them into L1 (the chain's B operand) while the select
        # kernel runs. A prefetch only: nothing reads them before the wait.
        lines = HIDDEN * 2 // 128 + 1
        i = tid
        while i < rows * lines:
            r = i // lines
            offset = cute.min((i - r * lines) * 128, Int32(HIDDEN * 2 - 2))
            prefetch_l1(h + (Int64(r) * Int64(h_stride)) * 2 + Int64(offset))
            i = i + RESCORE_THREADS

        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()

        found = Int32(load_u32(count))
        # At most every entry of every row (the select kernel lists each once).
        if found > rows * self.vocab:
            trap()
        if tid < MAX_ROWS:
            store_shared_u64(keys + tid * 8, Int64(0))
        first = cta * TILE
        while first < found:
            # The previous tile's chain is done with the tile and entries.
            cute.arch.sync_threads()
            if tid < TILE:
                # A short last tile repeats its last candidate.
                p = cute.min(first + tid, found - 1)
                store_shared_s32(
                    entries + tid * 4, Int32(load_u32(listed + Int64(p) * 4))
                )
            cute.arch.sync_threads()
            # The rows arrive in STAGES groups along k; the chain starts on
            # the first group while the others are in flight.
            for s in cutlass.range_constexpr(STAGES):
                i = tid
                while i < TILE * STAGE_CHUNKS:
                    c = i // STAGE_CHUNKS
                    part = s * STAGE_CHUNKS + i - c * STAGE_CHUNKS
                    v = load_shared_s32(entries + c * 4) & mask
                    copy_async_16(
                        tile + c * ROW_BYTES + part * 16,
                        weight + (Int64(v) * HIDDEN + Int64(part * 8)) * 2,
                    )
                    i = i + RESCORE_THREADS
                copy_async_commit()
            # Lane (g, t) of warp 0: A rows g and g + 8, B column (draft row)
            # g, columns 4 t .. 4 t + 3 of every 16 (module docstring).
            g = lane // 4
            t = lane % 4
            a_lo = tile + g * ROW_BYTES + t * 8
            a_hi = a_lo + 8 * ROW_BYTES
            b_at = (
                h + (Int64(cute.min(g, rows - 1)) * Int64(h_stride) + Int64(t * 4)) * 2
            )
            c0 = Float32(0.0)
            c1 = Float32(0.0)
            c2 = Float32(0.0)
            c3 = Float32(0.0)
            for s in cutlass.range_constexpr(STAGES):
                copy_async_wait_group(STAGES - 1 - s)
                cute.arch.sync_threads()
                if warp == 0:
                    for step in cutlass.range(
                        s * STAGE_STEPS, (s + 1) * STAGE_STEPS, unroll=8
                    ):
                        lo = load_shared_x2(a_lo + step * 32)
                        hi = load_shared_x2(a_hi + step * 32)
                        x = load_u32x2(b_at + Int64(step * 32))
                        c0, c1, c2, c3 = mma_bf16_16816(
                            (c0, c1, c2, c3),
                            (lo[0], hi[0], lo[1], hi[1]),
                            (x[0], x[1]),
                        )
            if warp == 0:
                # D[candidate][draft row]: lane (g, t) holds D[g][2t],
                # D[g][2t + 1], D[g + 8][2t], D[g + 8][2t + 1].
                store_shared_f32(sums + (g * 8 + 2 * t) * 4, c0)
                store_shared_f32(sums + (g * 8 + 2 * t + 1) * 4, c1)
                store_shared_f32(sums + ((g + 8) * 8 + 2 * t) * 4, c2)
                store_shared_f32(sums + ((g + 8) * 8 + 2 * t + 1) * 4, c3)
                cute.arch.sync_warp()
                if lane == 0:
                    c = Int32(0)
                    last = cute.min(Int32(TILE), found - first)
                    while c < last:
                        entry = load_shared_s32(entries + c * 4)
                        rr = entry >> ENTRY_BITS
                        v = entry & mask
                        total = load_shared_f32(sums + (c * 8 + rr) * 4)
                        old = load_shared_u64(keys + rr * 8)
                        store_shared_u64(
                            keys + rr * 8, max_u64(old, logit_key(total, Uint32(v)))
                        )
                        if cutlass.const_expr(self.debug):
                            store_u16(
                                logits + (Int64(rr) * self.vocab + Int64(v)) * 2,
                                bf16_bits(total),
                            )
                        c = c + 1
            first = first + self.ctas * TILE
        cute.arch.sync_threads()

        # Thread 0 (lane 0 of warp 0, which folded every key) publishes the
        # CTA's keys, then arrives; in the last CTA to arrive, thread r writes
        # row r's token.
        if tid == 0:
            r = Int32(0)
            while r < rows:
                key = load_shared_u64(keys + r * 8)
                if key != Int64(0):
                    atomic_max_u64(best + Int64(r) * 8, key)
                r = r + 1
            arrived = arrive_acq_rel_u32(arrivals)
            # The counter starts each launch at 0 (the last CTA resets it).
            if arrived >= Uint32(self.ctas):
                trap()
            is_last = Int32(0)
            if arrived == Uint32(self.ctas - 1):
                is_last = Int32(1)
            store_shared_s32(flag, is_last)
        cute.arch.sync_threads()
        if load_shared_s32(flag) != Int32(0):
            # Every other CTA's max happened before its arrival, which thread
            # 0 acquired before the barrier, and so did the select kernel's
            # (an earlier grid).
            if tid < rows:
                row_key = exchange_u64(best + Int64(tid) * 8, Int64(0))
                # Every row has a key: its candidates' (one at least, the
                # entry of L*) or the zero row's.
                if row_key == Int64(0):
                    trap()
                store_u64(
                    tokens + Int64(tid) * 8,
                    Int64(0xFFFFFF) - (row_key & Int64(0xFFFFFF)),
                )
            if tid == 0:
                store_u32(count, Uint32(0))
                store_u32(arrivals, Uint32(0))


@cache
def _quant(device_index: int):
    """`HeadQuantKernel` compiled for a device."""
    with torch.accelerator.device_index(device_index):
        return cute.compile(
            HeadQuantKernel(),
            Int64(0),
            Int32(HIDDEN),
            *(Int64(0),) * 3,
            Int32(1),
            make_fake_stream(),
        )


@cache
def _bound(device_index: int, vocab: int, valid: int):
    """`HeadBoundKernel` compiled for a device and vocabulary."""
    with torch.accelerator.device_index(device_index):
        return cute.compile(
            HeadBoundKernel(vocab, valid),
            *(Int64(0),) * 5,
            Int32(1),
            make_fake_stream(),
        )


@cache
def _select(device_index: int, vocab: int, valid: int, debug: bool):
    """`HeadSelectKernel` compiled for a device and vocabulary."""
    with torch.accelerator.device_index(device_index):
        return cute.compile(
            HeadSelectKernel(vocab, valid, debug),
            *(Int64(0),) * 8,
            Int32(1),
            make_fake_stream(),
        )


@cache
def _rescore(device_index: int, vocab: int, ctas: int, debug: bool):
    """`HeadRescoreKernel` compiled for a device, vocabulary and grid."""
    with torch.accelerator.device_index(device_index):
        return cute.compile(
            HeadRescoreKernel(vocab, ctas, debug),
            Int64(0),
            Int32(HIDDEN),
            *(Int64(0),) * 7,
            Int32(1),
            make_fake_stream(),
        )


class DraftHead:
    """The exact FP8-prefiltered greedy draft head over an LM head's BF16
    weight (module docstring).

    Builds the FP8 copy and statistics of `weight` (0.65 GB at TP1) and
    compiles the kernels; launches may be captured into CUDA graphs. Its
    tokens are those of `F.linear` on GB300 (module docstring, "What the
    BF16 head's token is").

    Args:
        weight: The head's BF16 weight `[vocab, 2560]` (kept, not copied).
        valid: The first `valid` entries are vocabulary; the rest (padding)
            never win.

    """

    def __init__(self, weight: torch.Tensor, valid: int) -> None:
        vocab, hidden = weight.shape
        assert hidden == HIDDEN and weight.dtype == torch.bfloat16
        assert weight.is_contiguous() and vocab % 2 == 0
        assert 0 < valid <= vocab < 1 << ENTRY_BITS
        self.weight = weight
        self.vocab = vocab
        self.valid = valid
        self.device = weight.device
        self.codes, self.wstat = draft_head_weights(weight)
        device = self.device
        self._codes_in = torch.zeros(
            2 * MAX_ROWS, HIDDEN, dtype=torch.float8_e4m3fn, device=device
        )
        self._hstat = torch.zeros(MAX_ROWS, HSTAT, dtype=torch.float32, device=device)
        self._state = torch.zeros(MAX_ROWS, STATE, dtype=torch.int32, device=device)
        self._upper = torch.zeros(
            MAX_ROWS, -(-vocab // BOUND_CHUNK), dtype=torch.float32, device=device
        )
        # The candidate list (every entry of every row fits) and its length.
        self._listed = torch.zeros(MAX_ROWS * vocab, dtype=torch.int32, device=device)
        self._count = torch.zeros(1, dtype=torch.int32, device=device)
        self._best = torch.zeros(MAX_ROWS, dtype=torch.int64, device=device)
        self._arrivals = torch.zeros(1, dtype=torch.int32, device=device)
        self._one = torch.ones((), dtype=torch.float32, device=device)
        index = device.index
        self._ctas = 2 * torch.cuda.get_device_properties(device).multi_processor_count
        self._quant = _quant(index)
        self._bound = _bound(index, vocab, valid)
        self._select = _select(index, vocab, valid, False)
        self._rescore = _rescore(index, vocab, self._ctas, False)

    @property
    def nbytes(self) -> int:
        """Bytes of the FP8 copy, its statistics and the candidate list."""
        return self.codes.nbytes + self.wstat.nbytes + self._listed.nbytes

    def takes(self, hidden: torch.Tensor) -> bool:
        """Whether a call on `hidden` fits the kernels (1 to `MAX_ROWS`
        BF16 rows of 2560 values, 16-byte aligned): shapes, strides and dtypes
        only."""
        return (
            hidden.dim() == 2
            and 0 < hidden.shape[0] <= MAX_ROWS
            and hidden.shape[1] == HIDDEN
            and hidden.dtype == torch.bfloat16
            and hidden.device == self.device
            and hidden.stride(1) == 1
            and hidden.stride(0) % 8 == 0
            and hidden.data_ptr() % 16 == 0
        )

    def _launch(
        self, hidden: torch.Tensor, select, rescore, logits: torch.Tensor | None = None
    ) -> torch.Tensor:
        """The five launches of one call (the debug variants of `select` and
        `rescore` rescore everything and write `logits`)."""
        rows = hidden.shape[0]
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        tokens = torch.empty(rows, dtype=torch.int64, device=self.device)
        self._quant(
            Int64(hidden.data_ptr()),
            Int32(hidden.stride(0)),
            Int64(self._codes_in.data_ptr()),
            Int64(self._hstat.data_ptr()),
            Int64(self._state.data_ptr()),
            Int32(rows),
            stream,
        )
        scores = torch._scaled_mm(
            self._codes_in[: 2 * rows],
            self.codes.t(),
            self._one,
            self._one,
            out_dtype=torch.float32,
        )
        self._bound(
            Int64(scores.data_ptr()),
            Int64(self._hstat.data_ptr()),
            Int64(self.wstat.data_ptr()),
            Int64(self._upper.data_ptr()),
            Int64(self._state.data_ptr()),
            Int32(rows),
            stream,
        )
        select(
            Int64(scores.data_ptr()),
            Int64(self._hstat.data_ptr()),
            Int64(self.wstat.data_ptr()),
            Int64(self._upper.data_ptr()),
            Int64(self._state.data_ptr()),
            Int64(self._listed.data_ptr()),
            Int64(self._count.data_ptr()),
            Int64(self._best.data_ptr()),
            Int32(rows),
            stream,
        )
        rescore(
            Int64(hidden.data_ptr()),
            Int32(hidden.stride(0)),
            Int64(self.weight.data_ptr()),
            Int64(self._listed.data_ptr()),
            Int64(self._count.data_ptr()),
            Int64(self._best.data_ptr()),
            Int64(self._arrivals.data_ptr()),
            Int64(tokens.data_ptr()),
            Int64(0 if logits is None else logits.data_ptr()),
            Int32(rows),
            stream,
        )
        return tokens

    def __call__(self, hidden: torch.Tensor) -> torch.Tensor:
        """The greedy token of every row, INT64 `[rows]`: the first index of
        the largest BF16 logit of the stock head (module docstring).

        Args:
            hidden: The draft rows, BF16 `[rows, 2560]` (`takes`).

        """
        assert self.takes(hidden)
        return self._launch(hidden, self._select, self._rescore)

    def rescore_all(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Tests: the tokens and BF16 logits `[rows, vocab]` of rescoring
        every entry of every non-zero row (entries not rescored hold NaN).

        Args:
            hidden: The draft rows, BF16 `[rows, 2560]` (`takes`).

        """
        assert self.takes(hidden)
        logits = torch.full(
            (hidden.shape[0], self.vocab),
            float("nan"),
            dtype=torch.bfloat16,
            device=self.device,
        )
        index = self.device.index
        select = _select(index, self.vocab, self.valid, True)
        rescore = _rescore(index, self.vocab, self._ctas, True)
        return self._launch(hidden, select, rescore, logits), logits

    def candidates(self) -> torch.Tensor:
        """The entries the last call rescored per row (diagnostics)."""
        return self._state[:, 3].clone()

    def warm_up(self) -> None:
        """Run every row count once (rows of ones), outside any capture: the
        GEMM's first calls choose their algorithms."""
        for rows in range(1, MAX_ROWS + 1):
            self(torch.ones(rows, HIDDEN, dtype=torch.bfloat16, device=self.device))
        torch.accelerator.synchronize(self.device)
