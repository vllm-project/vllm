# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The GLM-5.3-Flash MoE router of up to 32 decode tokens as one SM90 kernel:
the gate GEMM, the top-8 and the FP8 quantization of the FFN input, in place
of the stock gate GEMM, `single_group_topk` and the MoE input quant.

18 clusters of 8 CTAs (128 threads). Cluster g owns experts [16g, 16g + 16),
cluster rank r owns K [512r, 512r + 512) and warp w of it K [512r + 128w, +128).
The gate weight (BF16 `[288, 4096]`) and the tokens (BF16 `[M, 4096]`) are read
once per CTA with 16-byte loads straight into `mma.sync m16n8k16` fragments
(experts as M, tokens as N, FP32 accumulation). Within each 16-wide K step A
and B use the same K permutation (thread c of a quad holds [16c + 4s, +4) of a
64-wide chunk), which leaves the dot products unchanged. Before
`griddepcontrol.wait` the kernel only prefetches the gate weight and the bias
into L2 (no kernel of the step writes them); all register loads follow the
wait. The tokens are loaded from L2 (`ld.global.cg`): the kernel before this
one may release its dependents before it writes them, so they are not
read-only for this kernel's lifetime and the non-coherent path is not safe
(ptxas hoisted those `.nc` loads above the wait).

The split-K sum runs in a fixed order (the 4 warps in shared memory, then rank
0 adds ranks 0..7 over DSMEM), so the FP32 logits are deterministic. Every rank
0 publishes its logits with a release add; the last cluster waits for all 18,
then runs the top-k with one warp per token: each lane sorts its 9 experts with
a 25-comparator network, and 8 rounds of `redux.sync.max` pop the heads. The
semantics are stock `single_group_topk` with sigmoid scoring: the bias only
selects, ties go to the lower expert id, non-finite scores never win, the
weights are the unbiased sigmoids renormalized and times the routed scaling
factor, and the ids are int32. The sigmoid is `1 / (1 + 2^(-x log2 e))`, which
can differ from stock's `0.5 tanh(0.5 x) + 0.5` in the last bit.

Cluster 0 also quantizes the tokens to FP8 E4M3 in 128-wide blocks (power-of-two
scales with `ue8m0`, as the stock input quant) in the layout the FFN kernel
reads with TMA (`moe_experts`): `[M / 8][256][8][16]` bytes and `[MP][32]` FP32
scales.

The kernel triggers its programmatic dependents (the FFN kernel) right after
its MMAs.
"""

from functools import cache

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint32

from ._moe_ptx import (
    add_rn,
    bf16x2_to_f32x2,
    bits_f32,
    cluster_rank,
    cluster_sync_all,
    cvt_e4m3x4,
    div_rn,
    eq_u32,
    f32_bits,
    fabs,
    fence_proxy_async_global,
    fmax,
    griddep_launch,
    griddep_wait,
    ld_acquire_i32,
    ld_cluster_f32,
    ldg128_cg,
    ldg128_na,
    ldg_cg_f32,
    ldg_f32,
    lds32,
    lds_f32,
    mapa,
    minmax_key,
    mma_bf16,
    mul_rn,
    nanosleep,
    prefetch_l2,
    red_release_add,
    redux_max_u32,
    score_key,
    sel_f32,
    sel_u32,
    shfl_xor_f32,
    smem_addr,
    stg_f32,
    stg_i32,
    stg_u32x4,
    sts32,
    sts_f32,
)
from .moe import EXPERTS, HIDDEN, MAX_TOKENS, TOP_K

# CTAs per cluster: the K splits.
SPLITS = 8
# Experts per cluster.
GROUP = 16
CLUSTERS = EXPERTS // GROUP
WARPS = 4
THREADS = 32 * WARPS
# Twiddled selection key of -inf.
NEG_INF_KEY = 0x007FFFFF

# Compare-exchange network sorting 9 keys (checked by the 0-1 principle).
SORT9 = (
    (0, 1), (3, 4), (6, 7), (1, 2), (4, 5), (7, 8), (0, 1), (3, 4), (6, 7),
    (0, 3), (3, 6), (0, 3), (1, 4), (4, 7), (1, 4), (2, 5), (5, 8), (2, 5),
    (1, 3), (5, 7), (2, 6), (4, 6), (2, 4), (2, 3), (5, 6),
)  # fmt: skip


def _sort9_desc(hi, lo):
    """Descending lexicographic sort of 9 (hi, lo) keys (traced, not rewritten)."""
    hi, lo = list(hi), list(lo)
    for i, j in SORT9:
        hi[i], lo[i], hi[j], lo[j] = minmax_key(hi[i], lo[i], hi[j], lo[j])
    return hi, lo


@cute.jit
def pow2_ceil(x: Float32):
    """(2^ceil(log2 x), its reciprocal) of a positive normal `x`."""
    b = f32_bits(x)
    e = Int32(b >> Uint32(23))
    if (b & Uint32(0x7FFFFF)) != Uint32(0):
        e = e + 1
    return bits_f32(Uint32(e) << Uint32(23)), bits_f32(Uint32(254 - e) << Uint32(23))


class RouterKernel:
    """Top-8 ids (int32 `[M, 8]`) and weights (FP32 `[M, 8]`) of BF16 `[M, 4096]`
    tokens, their FP32 logits and their FP8 copy for the FFN kernel."""

    def __init__(self, groups: int, scale: float, ue8m0: bool):
        assert 1 <= groups <= MAX_TOKENS // 8
        self.groups = groups
        self.rows = 8 * groups
        self.scale = scale
        self.ue8m0 = ue8m0
        self.outputs = self.rows * GROUP
        self.per_thread = (self.outputs + THREADS - 1) // THREADS
        self.smem = max(WARPS * self.per_thread * THREADS * 4, WARPS * EXPERTS * 4)

    @cute.jit
    def __call__(
        self, x: Int64, w: Int64, bias: Int64, logits: Int64, ids: Int64,
        weights: Int64, xq: Int64, xs: Int64, counter: Int64, m: Int32, stream,
    ):  # fmt: skip
        self.kernel(x, w, bias, logits, ids, weights, xq, xs, counter, m).launch(
            grid=[CLUSTERS * SPLITS, 1, 1], block=[THREADS, 1, 1],
            cluster=[SPLITS, 1, 1], stream=stream, use_pdl=True,
            min_blocks_per_mp=4, preferred_smem_carveout=100,
        )  # fmt: skip

    @cute.jit
    def top_k(
        self, t: Int32, lane: Int32, scores: Int32, logits: Int64, bias: Int64,
        ids: Int64, weights: Int64,
    ):  # fmt: skip
        """One warp: the top-8 of token t, ids and weights to global memory."""
        hi = []
        lo = []
        bv = [ldg_f32(bias + Int64(lane + 32 * j) * 4) for j in range(9)]
        lv = [ldg_cg_f32(logits + (Int64(t) * EXPERTS + lane + 32 * j) * 4)
              for j in range(9)]  # fmt: skip
        for j in cutlass.range_constexpr(9):
            e = lane + 32 * j
            unbiased, key = score_key(lv[j], bv[j])
            sts32(scores + e * 4, unbiased)
            hi.append(key)
            lo.append(Uint32(65535) - Uint32(e))
        cute.arch.sync_warp()
        expert = Uint32(0)
        head = Uint32(0)
        hs, ls = _sort9_desc(hi, lo)
        for k in cutlass.range_constexpr(TOP_K):
            mh = redux_max_u32(hs[0])
            ml = redux_max_u32(sel_u32(eq_u32(hs[0], mh), ls[0], Uint32(0)))
            mine = eq_u32(Uint32(lane), Uint32(k))
            expert = sel_u32(mine, Uint32(65535) - ml, expert)
            head = sel_u32(mine, mh, head)
            # The one lane whose head was taken pops it.
            won = eq_u32(ls[0], ml)
            for j in cutlass.range_constexpr(8 - k):
                hs[j] = sel_u32(won, hs[j + 1], hs[j])
                ls[j] = sel_u32(won, ls[j + 1], ls[j])
            hs[8 - k] = sel_u32(won, Uint32(0), hs[8 - k])
        finite = Int32(0)
        if lane < TOP_K and head != Uint32(NEG_INF_KEY):
            finite = Int32(1)
        score = sel_f32(
            finite, bits_f32(lds32(scores + Int32(expert) * 4)), Float32(0.0)
        )
        any_finite = redux_max_u32(Uint32(finite))
        total = score
        for i in cutlass.range_constexpr(5):
            total = add_rn(total, shfl_xor_f32(total, 16 >> i))
        scale = div_rn(Float32(self.scale), add_rn(total, Float32(1e-20)))
        weight = sel_f32(finite, mul_rn(score, scale), Float32(0.0))
        if any_finite == Uint32(0):
            weight = Float32(1.0 / TOP_K)
        if lane < TOP_K:
            stg_i32(ids + (Int64(t) * TOP_K + lane) * 4, Int32(expert))
            stg_f32(weights + (Int64(t) * TOP_K + lane) * 4, weight)

    @cute.kernel
    def kernel(
        self, x: Int64, w: Int64, bias: Int64, logits: Int64, ids: Int64,
        weights: Int64, xq: Int64, xs: Int64, counter: Int64, m: Int32,
    ):  # fmt: skip
        groups = self.groups
        per_thread = self.per_thread
        stride = per_thread * THREADS  # of a warp's partial sums
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        warp = tid // 32
        lane = tid % 32
        smem = cute.arch.alloc_smem(cutlass.Uint8, self.smem, alignment=128)
        sb = smem_addr(smem)
        rank = cluster_rank()
        if tid < 9:  # the bias (1152 B) into L2 for the top-k cluster
            prefetch_l2(bias + Int64(tid) * 128)
        cluster = bid // SPLITS
        c = lane % 4
        r8 = lane // 4
        k0 = rank * 512 + warp * 128
        # Gate weight rows 8h + r8 of the cluster's 16 experts: into L2 before
        # the wait (one 128-byte line per row and j per quad), into registers
        # after it (ptxas may move a non-coherent load across the wait either
        # way, so only the prefetch is sure to overlap the previous kernel).
        if c == 0:
            for h in cutlass.range_constexpr(2):
                for j in cutlass.range_constexpr(2):
                    row = Int64(cluster * GROUP + 8 * h + r8)
                    prefetch_l2(w + (row * HIDDEN + k0 + 64 * j) * 2)
        griddep_wait()
        wr = []
        for h in cutlass.range_constexpr(2):
            rows = []
            for j in cutlass.range_constexpr(2):
                kb = k0 + 64 * j + 16 * c
                address = w + (Int64(cluster * GROUP + 8 * h + r8) * HIDDEN + kb) * 2
                rows.append(ldg128_na(address) + ldg128_na(address + 16))
            wr.append(rows)
        xr = []
        for j in cutlass.range_constexpr(2):
            kb = k0 + 64 * j + 16 * c
            row = []
            for i in cutlass.range_constexpr(groups):
                n = r8 + 8 * i
                if n > m - 1:
                    n = m - 1
                address = x + (Int64(n) * HIDDEN + kb) * 2
                row.append(ldg128_cg(address) + ldg128_cg(address + 16))
            xr.append(row)
        if cluster == 0:
            # FP8 block-128 copy of the tokens: block 4 rank + warp; this thread
            # holds [16c, 16c + 16) and [64 + 16c, 64 + 16c + 16) of the block.
            b = rank * 4 + warp
            for i in cutlass.range_constexpr(groups):
                n = r8 + 8 * i
                f = []
                for j in cutlass.range_constexpr(2):
                    for q in cutlass.range_constexpr(8):
                        low, high = bf16x2_to_f32x2(xr[j][i][q])
                        f.append(low)
                        f.append(high)
                amax = fabs(f[0])
                for u in cutlass.range_constexpr(1, 32):
                    amax = fmax(amax, fabs(f[u]))
                for u in cutlass.range_constexpr(2):
                    amax = fmax(amax, shfl_xor_f32(amax, 1 << u))
                amax = fmax(amax, Float32(1e-12))
                scale = amax * Float32(1.0 / 448.0)
                inverse = Float32(448.0) / amax
                if cutlass.const_expr(self.ue8m0):
                    scale, inverse = pow2_ceil(scale)
                if n < m:
                    if c == 0:
                        stg_f32(xs + (Int64(n) * 32 + b) * 4, scale)
                    for j in cutlass.range_constexpr(2):
                        words = [
                            cvt_e4m3x4(
                                f[16 * j + 4 * u] * inverse,
                                f[16 * j + 4 * u + 1] * inverse,
                                f[16 * j + 4 * u + 2] * inverse,
                                f[16 * j + 4 * u + 3] * inverse,
                            )
                            for u in range(4)
                        ]
                        chunk = 8 * b + 4 * j + c
                        offset = (Int64(n // 8) * 256 + chunk) * 8 + n % 8
                        stg_u32x4(xq + offset * 16, *words)
            fence_proxy_async_global()
        acc = [[Float32(0.0)] * 4 for _ in range(groups)]
        for j in cutlass.range_constexpr(2):
            for s in cutlass.range_constexpr(4):
                a = [wr[0][j][2 * s], wr[1][j][2 * s], wr[0][j][2 * s + 1],
                     wr[1][j][2 * s + 1]]  # fmt: skip
                for i in cutlass.range_constexpr(groups):
                    acc[i] = mma_bf16(acc[i], a, [xr[j][i][2 * s], xr[j][i][2 * s + 1]])
        griddep_launch()
        # Partial sums [warp][token * GROUP + expert].
        for h in cutlass.range_constexpr(2):
            el = 8 * h + r8
            for i in cutlass.range_constexpr(groups):
                n0 = 8 * i + 2 * c
                sts_f32(sb + (warp * stride + n0 * GROUP + el) * 4, acc[i][2 * h])
                sts_f32(
                    sb + (warp * stride + (n0 + 1) * GROUP + el) * 4, acc[i][2 * h + 1]
                )
        cute.arch.sync_threads()
        # Thread tid owns sums tid + 128 q. All loads issue before the adds.
        pw = []
        for q in cutlass.range_constexpr(per_thread):
            pw.append(
                [
                    lds_f32(sb + (v * stride + tid + THREADS * q) * 4)
                    for v in range(WARPS)
                ]
            )
        for q in cutlass.range_constexpr(per_thread):
            v = pw[q][0]
            for u in cutlass.range_constexpr(1, WARPS):
                v = add_rn(v, pw[q][u])
            sts_f32(sb + (tid + THREADS * q) * 4, v)
        cute.arch.sync_threads()
        cluster_sync_all()
        if rank == 0:
            parts = []
            for q in cutlass.range_constexpr(per_thread):
                a0 = sb + (tid + THREADS * q) * 4
                parts.append(
                    [lds_f32(a0)]
                    + [ld_cluster_f32(mapa(a0, Int32(r))) for r in range(1, SPLITS)]
                )
            for q in cutlass.range_constexpr(per_thread):
                index = tid + THREADS * q
                n = index // GROUP
                el = index % GROUP
                v = parts[q][0]
                for r in cutlass.range_constexpr(1, SPLITS):
                    v = add_rn(v, parts[q][r])
                if n < m:
                    address = logits + (Int64(n) * EXPERTS + cluster * GROUP + el) * 4
                    stg_f32(address, v)
            # The release below covers the CTA's stores (bar.sync orders them).
            cute.arch.sync_threads()
            if tid == 0:
                red_release_add(counter, 1)
                if cluster == CLUSTERS - 1:  # the top-k cluster
                    while ld_acquire_i32(counter) < CLUSTERS:
                        nanosleep(20)
                    stg_i32(counter, 0)
        cluster_sync_all()
        if cluster == CLUSTERS - 1:
            t = rank * WARPS + warp
            if t < m:
                self.top_k(t, lane, sb + warp * EXPERTS * 4, logits, bias, ids, weights)


@cache
def compiled(groups: int, scale: float, ue8m0: bool):
    from cutlass.cute.runtime import make_fake_stream

    zero = Int64(0)
    kernel = RouterKernel(groups, scale, ue8m0)
    return cute.compile(kernel, *([zero] * 9), Int32(1), make_fake_stream())
