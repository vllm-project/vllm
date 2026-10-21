# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The sparse-MLA decode of small batches, in CuTe DSL.

The problem. In a decode step every query token (DeepSeek-V4's MLA after the
query projection: 16 heads a rank at TP4, FP8 E4M3, head dim 512) attends to
its own sparse set of KV rows. A row is 512 FP8 bytes and serves as both K and
V. The set holds up to 640 columns: the 128 sliding-window rows of one KV pool,
then the indexer's top-k rows of the compressed pool, all addressed as storage
offsets into their pool, with -1 for an unused column and a per-token count of
active columns. TRTLLM-gen splits a token's 640 rows over at most 3 CTAs, so at
one or two requests (6 or 12 tokens with DSpark) its kernel runs about 10 µs
on a GPU of 152 SMs; most of that is the latency of one CTA's serial work.

What this module computes. ``sparse_mla_decode`` writes, for each token ``t``
and head ``h``, the BF16 attention output over the token's active columns ``c``
(``c < lens[t]`` and ``indices[t, c] >= 0``)::

    s_c = bmm1_scale * q[t, h] . kv[c]
    p_c = exp(s_c - m)
    out[t, h] = bmm2_scale * sum_c p_c kv[c] / (sum_c p_c + exp(sink_h - m))

where ``kv[c]`` is row ``indices[t, c]`` of the sliding-window pool for the
first 128 columns and of the compressed pool for the others, ``m`` is the
maximum of the scores and the sink, and the sink term is omitted without sinks.
A token with no active column gets zeros. This is what
``trtllm_batch_decode_sparse_mla_dsv4`` computes for these shapes.

How. One CTA of 128 threads takes one token and one 128-column chunk of its
sparse set (chunk 0 = the sliding-window columns); the ``n = ceil(width / 128)``
chunks of a token form a thread block cluster, so a 640-wide layer runs 5 CTAs
a token in parallel::

    gather the chunk's KV rows and the query (cp.async, one group per 128-byte K
    block) into 128B-swizzled tiles
      └► S^T = KV x Q^T on tcgen05 (M 128 rows, N 16 heads, K 512, FP8 -> FP32 TMEM)
           └► chunk softmax per head in base 2: max m_c, sum l_c, and
              P^T = 256 * exp2(S - m_c) as FP8 (scaled: small weights stay normal)
                └► O^T = V^T x P^T on tcgen05, V the same tile read MN-major
                   (four MMAs of 128 head dims) ...
                └► ... while the cluster exchanges (m_c, l_c) per head (st.async)
                     └► each CTA scales its partial O by exp2(m_c - M) / L
                        (M, L over the cluster's chunks and the sink)
                          └► block j of the 512 head dims goes to CTA j % n, which
                             sums the n partials and stores BF16

The partial blocks travel by bulk copies between the CTAs' shared memory into
the receiving CTA's KV tile, whose MMAs are done by then: a cluster barrier
after each CTA's second MMA separates the two uses. A last cluster barrier keeps
every CTA resident until the bulk copies reading its shared memory are complete.

Synchronization. The kernel reads the sinks (weights) before its grid
dependency wait and everything else after it, and lets its dependents launch
once its S MMAs are issued (programmatic dependent launch). Masked columns
gather zeros and score -inf.

Scope. Two CTAs fit an SM (about 78 KB of shared memory each), so a launch is
one wave only while every token's cluster can be resident: ``_max_tokens`` allows
2 x the device's resident clusters of ``n`` CTAs at one CTA an SM (on a 152-SM
GB300, 8 requests with DSpark at width 640).
Larger batches keep TRTLLM-gen, which is faster once the clusters run in waves.
"""

import math
from functools import cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch
from cutlass import Float32, Int32, Int64, Uint32, Uint64
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import make_fake_stream
from cutlass.cutlass_dsl import T, dsl_user_op

from vllm.cute_utils import _tcgen05, recast_val
from vllm.cute_utils import mbarrier as mbar_ops

from ...ops.cute_dsl.primitives import current_cu_stream

HEADS = 16
HEAD_DIM = 512
ROWS = 128  # KV rows (columns of the sparse set) a CTA
SWA_COLUMNS = ROWS  # chunk 0: the sliding-window columns, from the SWA pool
THREADS = 128
MAX_CHUNKS = 5  # widths up to 640
BLOCK_BYTES = ROWS * 128  # one 128-byte K block of the KV tile
Q_BLOCK_BYTES = HEADS * 128  # one 128-byte K block of the query tile
PART_BYTES = ROWS * HEADS * 4  # one 128-dim block of a partial O, FP32
TMEM_COLS = 128  # S: 16 columns, O: 4 x 16
P_SCALE = 256.0
LOG2E = 1.4426950408889634
CTAS_PER_SM = 2
# tcgen05 instruction descriptors, kind::f8f6f4: D FP32, A and B E4M3, M = 128,
# N = 16; bit 15 makes A MN-major (V^T read from the K-major KV tile).
IDESC_QK = (1 << 4) | ((HEADS >> 3) << 17) | ((ROWS >> 4) << 24)
IDESC_PV = IDESC_QK | (1 << 15)


def _sdesc(lbo: int, sbo: int) -> int:
    """A 128B-swizzled shared-memory matrix descriptor without its start
    address: leading and stride byte offsets, version 1."""
    return ((lbo >> 4) << 16) | ((sbo >> 4) << 32) | (1 << 46) | (2 << 61)


SDESC_K = _sdesc(0, 1024)  # K-major operands: 8-row groups 1024 bytes apart
SDESC_MN = _sdesc(BLOCK_BYTES, 1024)  # V^T: 128-row K blocks a tile block apart


def _asm(result, operands, text, constraints, side_effects, loc, ip):
    """One PTX inline assembly operation."""
    return llvm.inline_asm(
        result,
        operands,
        text,
        constraints,
        has_side_effects=side_effects,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def _v(value, loc, ip):
    """IR value of a DSL scalar."""
    return value.ir_value(loc=loc, ip=ip)


@dsl_user_op
def cp_async_16(smem: Int32, gmem: Int64, *, loc=None, ip=None) -> None:
    """Copy 16 bytes from global to this CTA's shared memory (cp.async)."""
    _asm(
        None,
        [_v(Int32(smem), loc, ip), _v(Int64(gmem), loc, ip)],
        "cp.async.cg.shared.global [$0], [$1], 16;",
        "r,l",
        True,
        loc,
        ip,
    )


@dsl_user_op
def st_shared_zero16(smem: Int32, *, loc=None, ip=None) -> None:
    """Store 16 zero bytes at a 16-byte aligned shared address."""
    _asm(
        None,
        [_v(Int32(smem), loc, ip), _v(Int32(0), loc, ip)],
        "st.shared.v4.b32 [$0], {$1, $1, $1, $1};",
        "r,r",
        True,
        loc,
        ip,
    )


@dsl_user_op
def st_shared_v4(
    smem: Int32, w0: Uint32, w1: Uint32, w2: Uint32, w3: Uint32, *, loc=None, ip=None
) -> None:
    """Store four 32-bit words at a 16-byte aligned shared address."""
    _asm(
        None,
        [_v(Int32(smem), loc, ip)] + [_v(Uint32(w), loc, ip) for w in (w0, w1, w2, w3)],
        "st.shared.v4.b32 [$0], {$1, $2, $3, $4};",
        "r,r,r,r,r",
        True,
        loc,
        ip,
    )


@dsl_user_op
def st_shared_bf16(smem: Int32, value: Float32, *, loc=None, ip=None) -> None:
    """Store ``value`` rounded to BF16 at a shared address."""
    _asm(
        None,
        [_v(Int32(smem), loc, ip), _v(Float32(value), loc, ip)],
        "{\n\t.reg .b16 b;\n\tcvt.rn.bf16.f32 b, $1;\n\tst.shared.b16 [$0], b;\n\t}",
        "r,f",
        True,
        loc,
        ip,
    )


@dsl_user_op
def ld_global_s32(address: Int64, *, loc=None, ip=None) -> Int32:
    """A 32-bit integer from global memory an earlier grid wrote (side-effecting,
    so it stays after the grid dependency wait)."""
    return Int32(
        _asm(
            T.i32(),
            [_v(Int64(address), loc, ip)],
            "ld.global.b32 $0, [$1];",
            "=r,l",
            True,
            loc,
            ip,
        )
    )


@dsl_user_op
def ld_global_f32(address: Int64, *, loc=None, ip=None) -> Float32:
    """A float from read-only global memory."""
    return Float32(
        _asm(
            T.f32(),
            [_v(Int64(address), loc, ip)],
            "ld.global.nc.f32 $0, [$1];",
            "=f,l",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def copy16_shared_to_global(smem: Int32, gmem: Int64, *, loc=None, ip=None) -> None:
    """Copy 16 bytes from this CTA's shared memory to global memory."""
    _asm(
        None,
        [_v(Int32(smem), loc, ip), _v(Int64(gmem), loc, ip)],
        "{\n\t.reg .b32 a, b, c, d;\n\tld.shared.v4.b32 {a, b, c, d}, [$0];\n\t"
        "st.global.v4.b32 [$1], {a, b, c, d};\n\t}",
        "r,l",
        True,
        loc,
        ip,
    )


@dsl_user_op
def pack_e4m3x4(
    a: Float32, b: Float32, c: Float32, d: Float32, *, loc=None, ip=None
) -> Uint32:
    """Four floats as E4M3 bytes, ``a`` in the lowest byte (round to nearest,
    saturating)."""
    return Uint32(
        _asm(
            T.i32(),
            [_v(Float32(v), loc, ip) for v in (a, b, c, d)],
            "{\n\t.reg .b16 lo, hi;\n\tcvt.rn.satfinite.e4m3x2.f32 lo, $2, $1;\n\t"
            "cvt.rn.satfinite.e4m3x2.f32 hi, $4, $3;\n\tmov.b32 $0, {lo, hi};\n\t}",
            "=r,f,f,f,f",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def mma_f8(
    d_tmem: Int32,
    adesc: Uint64,
    bdesc: Uint64,
    idesc: Uint32,
    accumulate: Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    """One tcgen05 FP8 MMA of this CTA: D (TMEM) = A x B (+ D if
    ``accumulate``), A and B from shared memory descriptors."""
    _asm(
        None,
        [
            _v(Int32(d_tmem), loc, ip),
            _v(Uint64(adesc), loc, ip),
            _v(Uint64(bdesc), loc, ip),
            _v(Uint32(idesc), loc, ip),
            _v(Int32(accumulate), loc, ip),
        ],
        "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, $4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::f8f6f4 [$0], $1, $2, $3, p;\n\t}",
        "r,l,l,r,r",
        True,
        loc,
        ip,
    )


@dsl_user_op
def st_async_peer_b32(
    smem: Int32, mbar: Int32, peer: Int32, value: Uint32, *, loc=None, ip=None
) -> None:
    """Store a word at ``smem`` of cluster CTA ``peer`` (the same offset as a
    local address), completing 4 bytes on the peer's ``mbar``."""
    _asm(
        None,
        [_v(Int32(v), loc, ip) for v in (smem, mbar, peer)]
        + [_v(Uint32(value), loc, ip)],
        "{\n\t.reg .b32 ra, rb;\n\tmapa.shared::cluster.u32 ra, $0, $2;\n\t"
        "mapa.shared::cluster.u32 rb, $1, $2;\n\t"
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 "
        "[ra], $3, [rb];\n\t}",
        "r,r,r,r",
        True,
        loc,
        ip,
    )


@dsl_user_op
def bulk_copy_to_peer(
    dest: Int32,
    source: Int32,
    size: Int32,
    mbar: Int32,
    peer: Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    """Bulk-copy ``size`` bytes of this CTA's shared memory at ``source`` to
    ``dest`` of cluster CTA ``peer`` (the same offset as a local address),
    completing on the peer's ``mbar``."""
    _asm(
        None,
        [_v(Int32(v), loc, ip) for v in (dest, source, size, mbar, peer)],
        "{\n\t.reg .b32 rd, rb;\n\tmapa.shared::cluster.u32 rd, $0, $4;\n\t"
        "mapa.shared::cluster.u32 rb, $3, $4;\n\t"
        "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes "
        "[rd], [$1], $2, [rb];\n\t}",
        "r,r,r,r,r",
        True,
        loc,
        ip,
    )


class SparseMlaDecodeCute:
    """The sparse-MLA decode (module docstring) of tokens whose sparse sets
    are at most ``chunks`` x 128 columns wide; ``has_sinks``: add each head's
    attention sink to the softmax denominator."""

    def __init__(self, chunks: int, has_sinks: bool) -> None:
        assert 1 <= chunks <= MAX_CHUNKS
        self.n = chunks
        self.has_sinks = has_sinks
        # Received partial blocks a CTA owns at most (block j -> CTA j % n).
        self.slots = (4 + chunks - 1) // chunks
        # Partials go out from the KV tile's upper half, part 4 + j for block j,
        # and arrive in its lower half (at most 4 parts: slots x (n - 1) <= 4).
        self.stage_part = 4
        assert self.slots * (chunks - 1) <= self.stage_part

    @cute.jit
    def __call__(
        self,
        q_ptr: Int64,
        q_stride: Int32,
        swa_ptr: Int64,
        swa_stride: Int32,
        comp_ptr: Int64,
        comp_stride: Int32,
        idx_ptr: Int64,
        idx_stride: Int32,
        idx_cols: Int32,
        lens_ptr: Int64,
        sinks_ptr: Int64,
        out_ptr: Int64,
        out_stride: Int32,
        scale_log2: Float32,
        out_scale: Float32,
        tokens: Int32,
        stream: cuda.CUstream,
    ):
        """Launch on ``stream``: one cluster of ``n`` CTAs a token. Pointers
        are addresses; strides are in elements (bytes for the FP8 query and
        pools, BF16 elements for ``out``); ``scale_log2`` is
        ``bmm1_scale * log2(e)``."""
        self.kernel(
            q_ptr,
            q_stride,
            swa_ptr,
            swa_stride,
            comp_ptr,
            comp_stride,
            idx_ptr,
            idx_stride,
            idx_cols,
            lens_ptr,
            sinks_ptr,
            out_ptr,
            out_stride,
            scale_log2,
            out_scale,
        ).launch(
            grid=(self.n, tokens, 1),
            block=(THREADS, 1, 1),
            cluster=(self.n, 1, 1),
            stream=stream,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self,
        q_ptr: Int64,
        q_stride: Int32,
        swa_ptr: Int64,
        swa_stride: Int32,
        comp_ptr: Int64,
        comp_stride: Int32,
        idx_ptr: Int64,
        idx_stride: Int32,
        idx_cols: Int32,
        lens_ptr: Int64,
        sinks_ptr: Int64,
        out_ptr: Int64,
        out_stride: Int32,
        scale_log2: Float32,
        out_scale: Float32,
    ):
        n = self.n
        tid, _, _ = cute.arch.thread_idx()
        chunk, token, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(tid // 32)
        lane = tid % 32
        # Softmax groups: 8 threads a head, 16 rows each.
        head = tid // 8
        part = tid % 8
        neg_inf = Float32(-math.inf)

        smem = utils.SmemAllocator()
        kv = smem.allocate_array(cutlass.Uint8, 4 * BLOCK_BYTES, byte_alignment=1024)
        sq = smem.allocate_array(cutlass.Uint8, 4 * Q_BLOCK_BYTES, byte_alignment=1024)
        sp = smem.allocate_array(cutlass.Uint8, HEADS * 128, byte_alignment=1024)
        s_addr = smem.allocate_tensor(Int64, cute.make_layout(ROWS), 16)
        # Per head: chunk max, chunk sum, merge factor, sink (base 2).
        s_stat = smem.allocate_tensor(Float32, cute.make_layout((HEADS, 4)), 16)
        # Per head and chunk of the cluster: max (rows 0-15), sum (rows 16-31).
        s_ml = smem.allocate_tensor(Float32, cute.make_layout((2 * HEADS, n)), 16)
        mbars = smem.allocate_array(Int64, 3, byte_alignment=8)
        taddr = smem.allocate_array(Int32, 4, byte_alignment=16)
        mbar_mma = mbars
        mbar_ml = mbars + 1
        mbar_o = mbars + 2
        if tid == 0:
            cute.arch.mbarrier_init(mbar_mma, 1)
            cute.arch.mbarrier_init(mbar_ml, 1)
            cute.arch.mbarrier_init(mbar_o, 1)
            cute.arch.mbarrier_init_fence()
        # The sinks are weights: read them before the grid wait.
        if tid < HEADS:
            sink = neg_inf
            if cutlass.const_expr(self.has_sinks):
                sink = ld_global_f32(sinks_ptr + Int64(tid * 4)) * Float32(LOG2E)
            s_stat[tid, 3] = sink
        cute.arch.sync_threads()
        if cutlass.const_expr(n > 1):
            # Peers must see these barriers initialized before their st.async.
            cute.arch.cluster_arrive_relaxed()
        # Indices, query and KV all come from earlier kernels.
        cute.arch.griddepcontrol_wait()
        if warp == 0:
            cute.arch.alloc_tmem(TMEM_COLS, taddr)
            cute.arch.relinquish_tmem_alloc_permit()

        # 1. This thread's KV row: its address, or 0 when masked. The length
        # and the index load independently.
        col = chunk * ROWS + tid
        length = ld_global_s32(lens_ptr + Int64(token) * 4)
        index = ld_global_s32(
            idx_ptr
            + (Int64(token) * Int64(idx_stride) + Int64(cute.min(col, idx_cols - 1)))
            * 4
        )
        address = Int64(0)
        if (col < length) & (col < idx_cols):  # noqa: SIM102
            if index >= 0:
                if chunk == 0:
                    address = swa_ptr + Int64(index) * Int64(swa_stride)
                else:
                    address = comp_ptr + Int64(index) * Int64(comp_stride)
        masked = address == Int64(0)
        s_addr[tid] = address
        cute.arch.sync_threads()

        # Gather in four cp.async groups, one per 128-byte K block of the rows (with
        # the query's K block): in a group a warp copies 4 rows a step, 8 lanes a
        # row, each 16-byte piece to its 128B-swizzled slot. The MMAs of a K block
        # start as soon as its group has landed, while the later groups load.
        kv_base = kv.toint()
        sq_base = sq.toint()
        sp_base = sp.toint()
        q_row = q_ptr + Int64(token) * Int64(q_stride)
        q_head = tid // 8
        q_piece = tid % 8
        piece = lane % 8
        for kb in cutlass.range_constexpr(4):
            cp_async_16(
                sq_base
                + kb * Q_BLOCK_BYTES
                + q_head * 128
                + ((q_piece ^ (q_head % 8)) * 16),
                q_row + Int64(q_head * HEAD_DIM + kb * 128 + q_piece * 16),
            )
            for i in cutlass.range_constexpr(ROWS // 16):
                row = (i * 4 + warp) * 4 + lane // 8
                source = s_addr[row]
                dest = (
                    kv_base + kb * BLOCK_BYTES + row * 128 + ((piece ^ (row % 8)) * 16)
                )
                if source != 0:
                    cp_async_16(dest, source + Int64(kb * 128 + piece * 16))
                else:
                    st_shared_zero16(dest)
            cute.arch.cp_async_commit_group()

        # 2. S^T = KV x Q^T, 4 MMAs of K = 32 a K block.
        tmem = cute.make_tensor(taddr, cute.make_layout(1))[0]
        for kb in cutlass.range_constexpr(4):
            cute.arch.cp_async_wait_group(3 - kb)
            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.sync_threads()
            if warp == 0:
                _tcgen05.fence_after_thread_sync()
                for k4 in cutlass.range_constexpr(4):
                    adesc = Uint64(SDESC_K) | Uint64(
                        (kv_base + kb * BLOCK_BYTES + k4 * 32) >> 4
                    )
                    bdesc = Uint64(SDESC_K) | Uint64(
                        (sq_base + kb * Q_BLOCK_BYTES + k4 * 32) >> 4
                    )
                    with cute.arch.elect_one():
                        mma_f8(
                            tmem,
                            adesc,
                            bdesc,
                            Uint32(IDESC_QK),
                            Int32(1 if kb + k4 > 0 else 0),
                        )
                if cutlass.const_expr(kb == 3):
                    _tcgen05.commit(mbar_mma)
                    with cute.arch.elect_one():
                        cute.arch.griddepcontrol_launch_dependents()
        cute.arch.mbarrier_wait(mbar_mma, 0)
        _tcgen05.fence_after_thread_sync()

        # 3. Chunk softmax. Each thread's TMEM lane is one KV row; the scores are
        # transposed through the (now free) query tile so that 8 threads reduce
        # one head's 128 rows.
        s = cute.make_rmem_tensor((HEADS,), Float32)
        s.store(_tcgen05.ld(warp * 32, tmem, "32x32b", HEADS))
        _tcgen05.wait_ld()
        s_t = cute.make_tensor(
            cute.recast_ptr(sq, dtype=Float32),
            cute.make_layout((HEADS, ROWS), stride=(ROWS, 1)),
        )
        for h in cutlass.range_constexpr(HEADS):
            value = s[h] * scale_log2
            if masked:
                value = neg_inf
            s_t[h, tid] = value
        cute.arch.sync_threads()
        vals = cute.make_rmem_tensor((16,), Float32)
        for i in cutlass.range_constexpr(16):
            vals[i] = s_t[head, part * 16 + i]
        peak = vals[0]
        for i in cutlass.range_constexpr(1, 16):
            peak = cute.arch.fmax(peak, vals[i])
        peak = cute.arch.warp_reduction_max(peak, threads_in_group=8)
        base = peak
        if peak == neg_inf:
            base = Float32(0.0)
        total = Float32(0.0)
        for i in cutlass.range_constexpr(16):
            vals[i] = cute.math.exp2(vals[i] - base, fastmath=True)
            total = total + vals[i]
        total = cute.arch.warp_reduction_sum(total, threads_in_group=8)
        # P^T as the K-major B operand: head h's 128 rows are one 128-byte row,
        # this thread's 16 at its swizzled 16-byte slot.
        scale_p = Float32(P_SCALE)
        st_shared_v4(
            sp_base + head * 128 + ((part ^ (head % 8)) * 16),
            pack_e4m3x4(
                vals[0] * scale_p,
                vals[1] * scale_p,
                vals[2] * scale_p,
                vals[3] * scale_p,
            ),
            pack_e4m3x4(
                vals[4] * scale_p,
                vals[5] * scale_p,
                vals[6] * scale_p,
                vals[7] * scale_p,
            ),
            pack_e4m3x4(
                vals[8] * scale_p,
                vals[9] * scale_p,
                vals[10] * scale_p,
                vals[11] * scale_p,
            ),
            pack_e4m3x4(
                vals[12] * scale_p,
                vals[13] * scale_p,
                vals[14] * scale_p,
                vals[15] * scale_p,
            ),
        )
        if part == 0:
            s_stat[head, 0] = peak
            s_stat[head, 1] = total
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.sync_threads()

        # 4. O^T = V^T x P^T, four MMAs of 128 head dims (K = 128 rows each);
        # the statistics exchange and the merge factors overlap them.
        if warp == 0:
            _tcgen05.fence_after_thread_sync()
            for j in cutlass.range_constexpr(4):
                for kk in cutlass.range_constexpr(ROWS // 32):
                    adesc = Uint64(SDESC_MN) | Uint64(
                        (kv_base + j * BLOCK_BYTES + kk * 32 * 128) >> 4
                    )
                    bdesc = Uint64(SDESC_K) | Uint64((sp_base + kk * 32) >> 4)
                    with cute.arch.elect_one():
                        mma_f8(
                            tmem + HEADS + HEADS * j,
                            adesc,
                            bdesc,
                            Uint32(IDESC_PV),
                            Int32(1 if kk > 0 else 0),
                        )
            _tcgen05.commit(mbar_mma)
        if cutlass.const_expr(n > 1):
            mbar_ml_addr = mbar_ml.toint()
            s_ml_base = s_ml.iterator.toint()
            cute.arch.cluster_wait()
            if tid == 0:
                mbar_ops.arrive_expect_tx(mbar_ml, n * 2 * HEADS * 4)
            if tid < 2 * HEADS:
                stat = s_stat[tid % HEADS, tid // HEADS]
                for peer in cutlass.range_constexpr(n):
                    st_async_peer_b32(
                        s_ml_base + (chunk * 2 * HEADS + tid) * 4,
                        mbar_ml_addr,
                        Int32(peer),
                        recast_val(stat, Uint32),
                    )
            cute.arch.mbarrier_wait(mbar_ml, 0)
        # 5. Merge factors, one thread a head: exp2(m_c - M) / L, L with the sink.
        if tid < HEADS:
            own = s_stat[tid, 0]
            sink = s_stat[tid, 3]
            top = sink
            denom = Float32(0.0)
            if cutlass.const_expr(n > 1):
                for c in cutlass.range_constexpr(n):
                    top = cute.arch.fmax(top, s_ml[tid, c])
                for c in cutlass.range_constexpr(n):
                    peak_c = s_ml[tid, c]
                    if peak_c != neg_inf:
                        denom = (
                            denom
                            + cute.math.exp2(peak_c - top, fastmath=True)
                            * s_ml[HEADS + tid, c]
                        )
            else:
                top = cute.arch.fmax(top, own)
                if own != neg_inf:
                    denom = cute.math.exp2(own - top, fastmath=True) * s_stat[tid, 1]
            if cutlass.const_expr(self.has_sinks):
                denom = denom + cute.math.exp2(sink - top, fastmath=True)
            f = Float32(0.0)
            if own != neg_inf:  # noqa: SIM102
                if denom > Float32(0.0):
                    f = (
                        cute.math.exp2(own - top, fastmath=True)
                        / denom
                        * out_scale
                        * Float32(1.0 / P_SCALE)
                    )
            s_stat[tid, 2] = f
        cute.arch.mbarrier_wait(mbar_mma, 1)
        _tcgen05.fence_after_thread_sync()
        cute.arch.sync_threads()
        if cutlass.const_expr(n > 1):
            # This CTA's MMAs are done: peers may now write into its tile.
            cute.arch.cluster_arrive()
        o = cute.make_rmem_tensor((4 * HEADS,), Float32)
        o.store(_tcgen05.ld(warp * 32, tmem + HEADS, "32x32b", 4 * HEADS))
        _tcgen05.wait_ld()
        factor = cute.make_rmem_tensor((HEADS,), Float32)
        for h in cutlass.range_constexpr(HEADS):
            factor[h] = s_stat[h, 2]
        for j in cutlass.range_constexpr(4):
            for h in cutlass.range_constexpr(HEADS):
                o[j * HEADS + h] = o[j * HEADS + h] * factor[h]
        # Thread tid holds head dim (128 j + tid) of every head.
        out_row = out_ptr + Int64(token) * Int64(out_stride) * 2
        if cutlass.const_expr(n == 1):
            # Stage [block][head][dim] BF16 in the tile, then 16-byte stores.
            for j in cutlass.range_constexpr(4):
                for h in cutlass.range_constexpr(HEADS):
                    st_shared_bf16(
                        kv_base + ((j * HEADS + h) * 128 + tid) * 2, o[j * HEADS + h]
                    )
            cute.arch.sync_threads()
            for k in cutlass.range_constexpr(4 * HEADS * 16 // THREADS):
                piece = k * THREADS + tid
                rowid = piece // 16
                d16 = piece % 16
                blk = rowid // HEADS
                h = rowid % HEADS
                copy16_shared_to_global(
                    kv_base + piece * 16,
                    out_row + Int64((h * HEAD_DIM + blk * 128 + d16 * 8) * 2),
                )
        else:
            mbar_o_addr = mbar_o.toint()
            owned = Int32(0)
            for j in cutlass.range_constexpr(4):
                if chunk == j % n:
                    owned = owned + 1
            # The tile as 8 FP32 parts of [head][row]: head-major, so a warp's
            # 32 rows of one head are 32 consecutive words (no bank conflicts
            # on either side).
            parts = cute.make_tensor(
                cute.recast_ptr(kv, dtype=Float32),
                cute.make_layout((HEADS, ROWS, 8), stride=(ROWS, 1, ROWS * HEADS)),
            )
            for j in cutlass.range_constexpr(4):
                if chunk != j % n:
                    for h in cutlass.range_constexpr(HEADS):
                        parts[h, tid, self.stage_part + j] = o[j * HEADS + h]
            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.sync_threads()
            if tid == 0:  # noqa: SIM102
                if owned > 0:
                    mbar_ops.arrive_expect_tx(mbar_o, owned * (n - 1) * PART_BYTES)
            # Every peer's MMAs are done, so its tile's lower half can take
            # partials: sender s of block j lands in part (j // n) (n - 1) + s,
            # the senders numbered without the owner.
            cute.arch.cluster_wait()
            if tid == 0:
                for j in cutlass.range_constexpr(4):
                    owner = j % n
                    if chunk != owner:
                        sender = chunk
                        if chunk > owner:
                            sender = chunk - 1
                        bulk_copy_to_peer(
                            kv_base + ((j // n) * (n - 1) + sender) * PART_BYTES,
                            kv_base + (self.stage_part + j) * PART_BYTES,
                            Int32(PART_BYTES),
                            mbar_o_addr,
                            Int32(owner),
                        )
            if owned > 0:
                cute.arch.mbarrier_wait(mbar_o, 0)
                # Owned block j (slot j // n): the sum, BF16 [slot][head][dim]
                # in the query tile, then 16-byte stores.
                for j in cutlass.range_constexpr(4):
                    if chunk == j % n:
                        for h in cutlass.range_constexpr(HEADS):
                            acc = o[j * HEADS + h]
                            for c in cutlass.range_constexpr(n - 1):
                                acc = acc + parts[h, tid, (j // n) * (n - 1) + c]
                            st_shared_bf16(
                                sq_base + (((j // n) * HEADS + h) * 128 + tid) * 2, acc
                            )
                cute.arch.sync_threads()
                for slot in cutlass.range_constexpr(self.slots):
                    if chunk + slot * n < 4:
                        for k in cutlass.range_constexpr(HEADS * 16 // THREADS):
                            piece = k * THREADS + tid
                            h = piece // 16
                            d16 = piece % 16
                            copy16_shared_to_global(
                                sq_base + (slot * HEADS * 128) * 2 + piece * 16,
                                out_row
                                + Int64(
                                    (h * HEAD_DIM + (chunk + slot * n) * 128 + d16 * 8)
                                    * 2
                                ),
                            )
            # The bulk copies read this CTA's staged partials asynchronously:
            # stay resident until every owner has received (its arrive follows
            # its mbarrier wait).
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()
        _tcgen05.fence_before_thread_sync()
        cute.arch.sync_threads()
        if warp == 0:
            _tcgen05.fence_after_thread_sync()
            cute.arch.dealloc_tmem(
                cute.make_ptr(Float32, tmem, cute.AddressSpace.tmem), TMEM_COLS
            )


@cache
def _kernel(device_index: int, chunks: int, has_sinks: bool):
    """``SparseMlaDecodeCute`` compiled for a device, chunk count and sinks."""
    with torch.accelerator.device_index(device_index):
        return cute.compile(
            SparseMlaDecodeCute(chunks, has_sinks),
            Int64(0),
            Int32(0),
            Int64(0),
            Int32(0),
            Int64(0),
            Int32(0),
            Int64(0),
            Int32(0),
            Int32(1),
            Int64(0),
            Int64(0),
            Int64(0),
            Int32(0),
            Float32(0),
            Float32(0),
            Int32(1),
            make_fake_stream(),
        )


@cache
def _max_tokens(device_index: int, chunks: int) -> int:
    """Tokens whose clusters are all resident at once: ``CTAS_PER_SM`` times
    the resident clusters of ``chunks`` CTAs at one CTA an SM."""
    with torch.accelerator.device_index(device_index):
        clusters = cutlass.utils.HardwareInfo(device_index).get_max_active_clusters(
            chunks
        )
    return CTAS_PER_SM * clusters


def precompile(device_index: int, has_sinks: bool) -> None:
    """Compile the kernels of DeepSeek-V4's two widths (128 sliding-window
    columns alone, and 640) and count the resident clusters, outside any CUDA
    graph capture; other widths compile at their first launch.

    Args:
        device_index: The CUDA device to compile for.
        has_sinks: Whether the layers add attention sinks.

    """
    for chunks in (1, MAX_CHUNKS):
        _kernel(device_index, chunks, has_sinks)
        _max_tokens(device_index, chunks)


def _is_pool(cache: torch.Tensor) -> bool:
    """An FP8 KV pool of 512-byte rows ``stride(-2)`` apart, 16-byte aligned."""
    return (
        cache.dtype == torch.float8_e4m3fn
        and cache.dim() >= 2
        and cache.shape[-1] == HEAD_DIM
        and cache.stride(-1) == 1
        and cache.stride(-2) % 16 == 0
        and cache.data_ptr() % 16 == 0
    )


def supports_sparse_mla_cute(
    query: torch.Tensor,
    swa_kv_cache: torch.Tensor,
    compressed_kv_cache: torch.Tensor,
    sparse_indices: torch.Tensor,
    sparse_topk_lens: torch.Tensor,
    out: torch.Tensor,
    bmm1_scale: float | torch.Tensor,
    bmm2_scale: float | torch.Tensor,
    sinks: torch.Tensor | None,
) -> bool:
    """Whether ``sparse_mla_decode`` takes these arguments: 16 FP8 query heads
    of 512 (BF16 output), FP8 pools, int32 index rows of at most 640 columns,
    scalar scales, FP32 sinks, and a batch whose clusters are all resident."""
    if (
        query.dtype != torch.float8_e4m3fn
        or query.dim() != 3
        or query.shape[1:] != (HEADS, HEAD_DIM)
        or query.stride(2) != 1
        or query.stride(1) != HEAD_DIM
        or query.stride(0) % 16 != 0
        or query.data_ptr() % 16 != 0
    ):
        return False
    tokens = query.shape[0]
    if (
        out.dtype != torch.bfloat16
        or out.shape != query.shape
        or out.stride(2) != 1
        or out.stride(1) != HEAD_DIM
        or out.stride(0) % 8 != 0
        or out.data_ptr() % 16 != 0
    ):
        return False
    if (
        sparse_indices.dtype != torch.int32
        or sparse_indices.dim() != 2
        or sparse_indices.shape[0] != tokens
        or sparse_indices.stride(1) != 1
        or not 0 < sparse_indices.shape[1] <= MAX_CHUNKS * ROWS
    ):
        return False
    if (
        sparse_topk_lens.dtype != torch.int32
        or not sparse_topk_lens.is_contiguous()
        or sparse_topk_lens.numel() != tokens
    ):
        return False
    if sinks is not None and (
        sinks.dtype != torch.float32
        or not sinks.is_contiguous()
        or sinks.numel() < HEADS
    ):
        return False
    if not (
        isinstance(bmm1_scale, float | int) and isinstance(bmm2_scale, float | int)
    ):
        return False
    if not (_is_pool(swa_kv_cache) and _is_pool(compressed_kv_cache)):
        return False
    chunks = (sparse_indices.shape[1] + ROWS - 1) // ROWS
    return 0 < tokens <= _max_tokens(query.device.index, chunks)


def sparse_mla_decode(
    query: torch.Tensor,
    swa_kv_cache: torch.Tensor,
    compressed_kv_cache: torch.Tensor,
    sparse_indices: torch.Tensor,
    sparse_topk_lens: torch.Tensor,
    out: torch.Tensor,
    bmm1_scale: float,
    bmm2_scale: float,
    sinks: torch.Tensor | None,
) -> None:
    """Launch `SparseMlaDecodeCute` on the current stream (see
    `supports_sparse_mla_cute` for the accepted layouts).

    Args:
        query: [tokens, 16, 512] FP8 E4M3.
        swa_kv_cache: FP8 pool of 512-byte rows ``stride(-2)`` apart; the first
            128 columns of ``sparse_indices`` are its row offsets.
        compressed_kv_cache: The pool of the other columns, likewise.
        sparse_indices: [tokens, width] int32 row offsets, -1 for none
            (``width`` <= 640).
        sparse_topk_lens: [tokens] int32 active columns of each token.
        out: [tokens, 16, 512] BF16 attention output.
        bmm1_scale: Score scale (the softmax scale times the FP8 scales).
        bmm2_scale: Output scale.
        sinks: [16] FP32 attention sinks (natural log), or None.

    """
    assert supports_sparse_mla_cute(
        query,
        swa_kv_cache,
        compressed_kv_cache,
        sparse_indices,
        sparse_topk_lens,
        out,
        bmm1_scale,
        bmm2_scale,
        sinks,
    )
    tokens = query.shape[0]
    width = sparse_indices.shape[1]
    chunks = (width + ROWS - 1) // ROWS
    _kernel(query.device.index, chunks, sinks is not None)(
        Int64(query.data_ptr()),
        Int32(query.stride(0)),
        Int64(swa_kv_cache.data_ptr()),
        Int32(swa_kv_cache.stride(-2)),
        Int64(compressed_kv_cache.data_ptr()),
        Int32(compressed_kv_cache.stride(-2)),
        Int64(sparse_indices.data_ptr()),
        Int32(sparse_indices.stride(0)),
        Int32(width),
        Int64(sparse_topk_lens.data_ptr()),
        Int64(sinks.data_ptr() if sinks is not None else 0),
        Int64(out.data_ptr()),
        Int32(out.stride(0)),
        Float32(bmm1_scale * LOG2E),
        Float32(bmm2_scale),
        Int32(tokens),
        current_cu_stream(),
    )
