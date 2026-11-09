# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The split-K mHC mixes of the Hopper executor, in CuTe DSL (`MixesKernel`):
`residual @ fn.T` and the sums of squares of the streams, the prenorm GEMM of
the mHC pre that the stock path runs on DeepGEMM.

`residual` is BF16 [M, 4 x 5120] (exact in TF32), `fn` FP32 [24, 20480]
rounded to TF32 as the TF32 prenorm GEMMs round it; the products run on TF32
`mma.sync`. Each CTA takes 256 of K with its 24 KB slice of `fn`, loaded
before the grid wait (`fn` does not depend on the previous kernel), and a
chunk of up to 64 rows. The CTAs of one chunk come in clusters of eight along
K, whose partials are summed in a fixed order through distributed shared
memory into ten splits, which the mHC epilogue
(`mhc_pre_big_fuse_with_norm_tilelang`) reduces. The kernel triggers its
dependent launch at the end.
"""

import functools

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint32

from .ptx import (
    cluster_arrive_relaxed,
    cluster_rank,
    cluster_sync,
    cluster_wait,
    cp_async_16,
    cp_async_commit,
    cp_async_wait_all,
    griddep_launch,
    griddep_wait,
    lds128,
    lds_f32,
    mapa,
    mma_tf32_1688,
    smem_addr,
    st_cluster_f32,
    stg_f32,
    sts128_f32,
    tf32_rna,
    u32_as_f32,
)

HIDDEN = 5120
HC = 4
MIXES = HC * (HC + 2)
THREADS = 128
WARPS = THREADS // 32
# A cluster of K_SPLIT CTAs along K; each CTA one split of MIX_K.
K_SPLIT = 8
MIX_K = 256
MIX_CTAS = HC * HIDDEN // MIX_K  # 80
MIX_SPLITS = MIX_CTAS // K_SPLIT  # 10
MIX_CHUNKS = MIX_K // 32
FN_BYTES = 2 * 32 * 64 * MIX_CHUNKS  # [chunk][m-tile][lane][64 B]
# Row pitch of the staged rows: 64 B past a multiple of 128 B, so the eight
# rows of a B fragment load fall in distinct bank halves.
MIX_PITCH = MIX_K * 2 + 64
MAX_TILES = 8  # 8-row tiles a CTA takes: rows per chunk up to 64
CHUNK_ROWS = 8 * MAX_TILES


class MixesKernel:
    """The mix splits [MIX_SPLITS, M, 24] and sum-of-squares splits
    [MIX_SPLITS, M] of `M` rows, in chunks of up to `8 * tiles` rows."""

    def __init__(self, tiles: int) -> None:
        assert 0 < tiles <= MAX_TILES
        self.tiles = tiles
        staged = 8 * tiles * MIX_PITCH
        # Cluster receive buffer: the 32 * tiles partials of each of K_SPLIT
        # CTAs, then the tiles sums of squares of each.
        recv = K_SPLIT * 33 * tiles * 4
        assert WARPS * 2 * tiles * 32 * 16 <= staged
        self.smem = FN_BYTES + staged + recv

    @cute.jit
    def __call__(
        self, fn: Int64, residual: Int64, mix: Int64, sqr: Int64, M: Int32, stream
    ):
        chunks = (M + 8 * self.tiles - 1) // (8 * self.tiles)
        self.kernel(fn, residual, mix, sqr, M).launch(
            grid=[MIX_CTAS, chunks, 1],
            block=[THREADS, 1, 1],
            cluster=[K_SPLIT, 1, 1],
            stream=stream,
            use_pdl=True,
        )

    @cute.jit
    def _stage_rows(self, dst: Int32, src: Int64, m: Int32, tid: Int32):
        """cp.async `m` rows of this CTA's K into the staged rows."""
        chunks = MIX_K // 8
        for q in cutlass.range(tid, m * chunks, THREADS, unroll=1):
            r = q // chunks
            c = q % chunks
            cp_async_16(
                dst + r * MIX_PITCH + c * 16,
                src + Int64(r) * Int64(HC * HIDDEN * 2) + c * 16,
            )
        cp_async_commit()
        cp_async_wait_all()
        cute.arch.sync_threads()

    @cute.kernel
    def kernel(self, fn: Int64, residual: Int64, mix: Int64, sqr: Int64, M: Int32):
        tid, _, _ = cute.arch.thread_idx()
        split, chunk, _ = cute.arch.block_idx()
        warp = tid // 32
        lane = tid % 32
        smem = cute.arch.alloc_smem(cutlass.Uint8, self.smem, alignment=128)
        nt = self.tiles
        r0 = chunk * (8 * nt)
        m = cutlass.min(M - r0, 8 * nt)
        s_fn = smem_addr(smem)
        s_x = s_fn + FN_BYTES
        s_recv = s_x + 8 * nt * MIX_PITCH
        rank = cluster_rank()
        cluster_arrive_relaxed()
        k0 = split * MIX_K
        # The fn slice (24 rows of FP32) in tf32 A-fragment order: [k32 chunk]
        # [m-tile][lane][row g, row g + 8][32 B]; rows 24..31 stay unused.
        for i in cutlass.range_constexpr(MIXES * MIX_K // 4 // THREADS):
            q = tid + THREADS * i
            row = q // (MIX_K // 4)
            cq = q % (MIX_K // 4)
            rr = row % 16
            k8 = cq % 8
            dst = (
                s_fn
                + (((cq // 8) * 2 + row // 16) * 32 + (rr % 8) * 4 + k8 // 2) * 64
                + (rr // 8) * 32
                + (k8 % 2) * 16
            )
            cp_async_16(dst, fn + (Int64(row) * (HC * HIDDEN) + k0 + 4 * cq) * 4)
        cp_async_commit()
        griddep_wait()
        self._stage_rows(s_x, residual + (Int64(r0) * (HC * HIDDEN) + k0) * 2, m, tid)
        g = lane // 4
        t = lane % 4
        acc = cute.make_rmem_tensor((2, nt, 4), Float32)
        acc.fill(Float32(0.0))
        cpw = MIX_CHUNKS // WARPS
        for cc in cutlass.range_constexpr(cpw):
            c = warp * cpw + cc
            a = cute.make_rmem_tensor((2, 16), Uint32)
            for mt in cutlass.range_constexpr(2):
                base = s_fn + ((c * 2 + mt) * 32 + lane) * 64
                for part in cutlass.range_constexpr(4):
                    if cutlass.const_expr(mt == 1 and part >= 2):
                        for w in cutlass.range_constexpr(4):
                            a[mt, 4 * part + w] = Uint32(0)  # rows 24..31
                    else:
                        v = lds128(base + 16 * part)
                        for w in cutlass.range_constexpr(4):
                            a[mt, 4 * part + w] = tf32_rna(v[w])
            for j in cutlass.range_constexpr(nt):
                b = lds128(s_x + (8 * j + g) * MIX_PITCH + (32 * c + 8 * t) * 2)
                for mt in cutlass.range_constexpr(2):
                    for q in cutlass.range_constexpr(4):
                        # k8 step q: logical k t / t + 4 = physical 8t + 2q / + 1
                        d = mma_tf32_1688(
                            (
                                acc[mt, j, 0],
                                acc[mt, j, 1],
                                acc[mt, j, 2],
                                acc[mt, j, 3],
                            ),
                            a[mt, 2 * q],
                            a[mt, 8 + 2 * q],
                            a[mt, 2 * q + 1],
                            a[mt, 8 + 2 * q + 1],
                            b[q] << Uint32(16),
                            b[q] & Uint32(0xFFFF0000),
                        )
                        for i in cutlass.range_constexpr(4):
                            acc[mt, j, i] = d[i]
        # Sums of squares of this CTA's K, a warp per row; row r's partials go
        # to cluster rank r % 8.
        cluster_wait()
        n = 2 * nt * 32 * 4
        s_sqr = s_recv + K_SPLIT * (n // K_SPLIT) * 4
        for r in cutlass.range(warp, m, WARPS, unroll=1):
            v = lds128(s_x + r * MIX_PITCH + lane * 16)
            s = Float32(0.0)
            for w in cutlass.range_constexpr(4):
                lo = u32_as_f32(v[w] << Uint32(16))
                hi = u32_as_f32(v[w] & Uint32(0xFFFF0000))
                s = s + lo * lo + hi * hi
            for sh in cutlass.range_constexpr(5):
                s = s + cute.arch.shuffle_sync_bfly(s, offset=1 << sh)
            if lane == 0:
                st_cluster_f32(
                    mapa(s_sqr + (rank * nt + r // K_SPLIT) * 4, r % K_SPLIT), s
                )
        cute.arch.sync_threads()
        # This warp's accumulators [m-tile][n-tile][4], then the CTA sum.
        red = s_x
        for mt in cutlass.range_constexpr(2):
            for j in cutlass.range_constexpr(nt):
                sts128_f32(
                    red + (((warp * 2 + mt) * nt + j) * 32 + lane) * 16,
                    acc[mt, j, 0],
                    acc[mt, j, 1],
                    acc[mt, j, 2],
                    acc[mt, j, 3],
                )
        cute.arch.sync_threads()
        for it in cutlass.range_constexpr(n // THREADS):
            idx = tid + THREADS * it
            v = lds_f32(red + idx * 4)
            for w in cutlass.range_constexpr(1, WARPS):
                v = v + lds_f32(red + (w * n + idx) * 4)
            st_cluster_f32(
                mapa(
                    s_recv + ((rank * (n // K_SPLIT)) + idx // K_SPLIT) * 4,
                    idx % K_SPLIT,
                ),
                v,
            )
        cluster_sync()
        out = split // K_SPLIT
        for r in cutlass.range(rank + K_SPLIT * tid, m, K_SPLIT * THREADS, unroll=1):
            v = lds_f32(s_sqr + (r // K_SPLIT) * 4)
            for src in cutlass.range_constexpr(1, K_SPLIT):
                v = v + lds_f32(s_sqr + (src * nt + r // K_SPLIT) * 4)
            stg_f32(sqr + (Int64(out) * M + r0 + r) * 4, v)
        for q in cutlass.range(tid, n // K_SPLIT, THREADS, unroll=1):
            v = lds_f32(s_recv + q * 4)
            for src in cutlass.range_constexpr(1, K_SPLIT):
                v = v + lds_f32(s_recv + (src * (n // K_SPLIT) + q) * 4)
            idx = q * K_SPLIT + rank
            i = idx % 4
            ln = (idx // 4) % 32
            j = (idx // 128) % nt
            mt = idx // (128 * nt)
            row = 16 * mt + ln // 4 + 8 * (i // 2)
            token = 8 * j + 2 * (ln % 4) + i % 2
            if row < MIXES and token < m:
                stg_f32(mix + ((Int64(out) * M + r0 + token) * MIXES + row) * 4, v)
        griddep_launch()


@functools.cache
def compiled(tiles: int):
    """The kernel of chunks of up to `8 * tiles` rows."""
    from cutlass.cute.runtime import make_fake_stream

    zero = Int64(0)
    return cute.compile(
        MixesKernel(tiles), zero, zero, zero, zero, Int32(1), make_fake_stream()
    )


def tiles_of(rows: int) -> int:
    return min(MAX_TILES, (rows + 7) // 8)
