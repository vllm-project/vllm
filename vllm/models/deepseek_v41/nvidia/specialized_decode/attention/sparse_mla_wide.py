# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline TP1 prototype: reuse each KV tile across all 64 query heads.

Unlike four independent 16-head clusters, each CTA reads its 128 KV rows
once. FP32 partials and softmax statistics are merged by a second kernel.
The experimental DEP C8 option can select it; default dispatch is unchanged.
"""

import math
from functools import cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch
from cutlass import Float32, Int32, Int64, Uint32, Uint64
from cutlass.cute.runtime import make_fake_stream

from vllm.cute_utils import _tcgen05
from vllm.models.deepseek_v41.nvidia.specialized_decode.attention import (
    sparse_mla_cute as base,
)
from vllm.triton_utils import tl, triton

HEADS = 64
DIM = 512
ROWS = 128
Q_BLOCK_BYTES = HEADS * 128
IDESC_QK = (1 << 4) | ((HEADS >> 3) << 17) | ((ROWS >> 4) << 24)
IDESC_PV = IDESC_QK | (1 << 15)


class SparseMlaWide:
    def __init__(self, chunks: int):
        assert 1 <= chunks <= 5
        self.chunks = chunks

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
        width: Int32,
        lens_ptr: Int64,
        partial_ptr: Int64,
        stats_ptr: Int64,
        scale_log2: Float32,
        tokens: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            q_ptr,
            q_stride,
            swa_ptr,
            swa_stride,
            comp_ptr,
            comp_stride,
            idx_ptr,
            idx_stride,
            width,
            lens_ptr,
            partial_ptr,
            stats_ptr,
            scale_log2,
        ).launch(
            grid=(self.chunks, tokens, 1),
            block=(128, 1, 1),
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
        width: Int32,
        lens_ptr: Int64,
        partial_ptr: Int64,
        stats_ptr: Int64,
        scale_log2: Float32,
    ):
        tid, _, _ = cute.arch.thread_idx()
        chunk, token, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(tid // 32)
        lane = tid % 32
        head0, part = tid // 8, tid % 8
        smem = utils.SmemAllocator()
        kv = smem.allocate_array(
            cutlass.Uint8, 4 * base.BLOCK_BYTES, byte_alignment=1024
        )
        sq = smem.allocate_array(cutlass.Uint8, 4 * Q_BLOCK_BYTES, byte_alignment=1024)
        sp = smem.allocate_array(cutlass.Uint8, HEADS * ROWS, byte_alignment=1024)
        addresses = smem.allocate_tensor(Int64, cute.make_layout(ROWS), 16)
        stats = smem.allocate_tensor(Float32, cute.make_layout((HEADS, 2)), 16)
        barrier = smem.allocate_array(Int64, 1, byte_alignment=8)
        taddr = smem.allocate_array(Int32, 4, byte_alignment=16)
        if tid == 0:
            cute.arch.mbarrier_init(barrier, 1)
            cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        cute.arch.griddepcontrol_wait()
        if warp == 0:
            cute.arch.alloc_tmem(256, taddr)
            cute.arch.relinquish_tmem_alloc_permit()
        col = chunk * ROWS + tid
        length = base.ld_global_s32(lens_ptr + Int64(token) * 4)
        index = base.ld_global_s32(
            idx_ptr + (Int64(token) * idx_stride + cute.min(col, width - 1)) * 4
        )
        address = Int64(0)
        if (col < length) & (col < width) & (index >= 0):
            if chunk == 0:
                address = swa_ptr + Int64(index) * swa_stride
            else:
                address = comp_ptr + Int64(index) * comp_stride
        addresses[tid] = address
        masked = address == Int64(0)
        cute.arch.sync_threads()
        kv_base, sq_base, sp_base = kv.toint(), sq.toint(), sp.toint()
        q_row = q_ptr + Int64(token) * q_stride
        for kb in cutlass.range_constexpr(4):
            for hg in cutlass.range_constexpr(4):
                head = head0 + hg * 16
                base.cp_async_16(
                    sq_base
                    + kb * Q_BLOCK_BYTES
                    + head * 128
                    + ((part ^ (head % 8)) * 16),
                    q_row + Int64(head * DIM + kb * 128 + part * 16),
                )
            for i in cutlass.range_constexpr(ROWS // 16):
                row = (i * 4 + warp) * 4 + lane // 8
                source = addresses[row]
                dest = (
                    kv_base
                    + kb * base.BLOCK_BYTES
                    + row * 128
                    + ((lane % 8 ^ (row % 8)) * 16)
                )
                if source != 0:
                    base.cp_async_16(dest, source + Int64(kb * 128 + lane % 8 * 16))
                else:
                    base.st_shared_zero16(dest)
            cute.arch.cp_async_commit_group()

        tmem = cute.make_tensor(taddr, cute.make_layout(1))[0]
        for kb in cutlass.range_constexpr(4):
            cute.arch.cp_async_wait_group(3 - kb)
            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.sync_threads()
            if warp == 0:
                _tcgen05.fence_after_thread_sync()
                for k4 in cutlass.range_constexpr(4):
                    adesc = Uint64(base.SDESC_K) | Uint64(
                        (kv_base + kb * base.BLOCK_BYTES + k4 * 32) >> 4
                    )
                    bdesc = Uint64(base.SDESC_K) | Uint64(
                        (sq_base + kb * Q_BLOCK_BYTES + k4 * 32) >> 4
                    )
                    with cute.arch.elect_one():
                        base.mma_f8(
                            tmem,
                            adesc,
                            bdesc,
                            Uint32(IDESC_QK),
                            Int32(1 if kb + k4 > 0 else 0),
                        )
                if cutlass.const_expr(kb == 3):
                    _tcgen05.commit(barrier)
        cute.arch.mbarrier_wait(barrier, 0)
        _tcgen05.fence_after_thread_sync()
        scores = cute.make_rmem_tensor((HEADS,), Float32)
        scores.store(_tcgen05.ld(warp * 32, tmem, "32x32b", HEADS))
        _tcgen05.wait_ld()
        transposed = cute.make_tensor(
            cute.recast_ptr(sq, dtype=Float32),
            cute.make_layout((HEADS, ROWS), stride=(ROWS, 1)),
        )
        for h in cutlass.range_constexpr(HEADS):
            value = scores[h] * scale_log2
            if masked:
                value = Float32(-math.inf)
            transposed[h, tid] = value
        cute.arch.sync_threads()
        for hg in cutlass.range_constexpr(4):
            head = head0 + hg * 16
            vals = cute.make_rmem_tensor((16,), Float32)
            for i in cutlass.range_constexpr(16):
                vals[i] = transposed[head, part * 16 + i]
            peak = vals[0]
            for i in cutlass.range_constexpr(1, 16):
                peak = cute.arch.fmax(peak, vals[i])
            peak = cute.arch.warp_reduction_max(peak, threads_in_group=8)
            center = peak
            if peak == Float32(-math.inf):
                center = Float32(0.0)
            total = Float32(0.0)
            for i in cutlass.range_constexpr(16):
                vals[i] = cute.math.exp2(vals[i] - center, fastmath=True)
                total += vals[i]
            total = cute.arch.warp_reduction_sum(total, threads_in_group=8)
            base.st_shared_v4(
                sp_base + head * 128 + ((part ^ (head % 8)) * 16),
                base.pack_e4m3x4(
                    vals[0] * 256.0, vals[1] * 256.0, vals[2] * 256.0, vals[3] * 256.0
                ),
                base.pack_e4m3x4(
                    vals[4] * 256.0, vals[5] * 256.0, vals[6] * 256.0, vals[7] * 256.0
                ),
                base.pack_e4m3x4(
                    vals[8] * 256.0, vals[9] * 256.0, vals[10] * 256.0, vals[11] * 256.0
                ),
                base.pack_e4m3x4(
                    vals[12] * 256.0,
                    vals[13] * 256.0,
                    vals[14] * 256.0,
                    vals[15] * 256.0,
                ),
            )
            if part == 0:
                stats[head, 0] = peak
                stats[head, 1] = total
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.sync_threads()
        # All scores have been consumed; reuse their TMEM for the four PV tiles.
        if warp == 0:
            _tcgen05.fence_after_thread_sync()
            for j in cutlass.range_constexpr(4):
                for kk in cutlass.range_constexpr(4):
                    adesc = Uint64(base.SDESC_MN) | Uint64(
                        (kv_base + j * base.BLOCK_BYTES + kk * 32 * 128) >> 4
                    )
                    bdesc = Uint64(base.SDESC_K) | Uint64((sp_base + kk * 32) >> 4)
                    with cute.arch.elect_one():
                        base.mma_f8(
                            tmem + HEADS * j,
                            adesc,
                            bdesc,
                            Uint32(IDESC_PV),
                            Int32(1 if kk else 0),
                        )
            _tcgen05.commit(barrier)
        out_stats = cute.make_tensor(
            cute.make_ptr(Float32, stats_ptr, cute.AddressSpace.gmem),
            cute.make_layout((Int32(0x7FFFFFFF),)),
        )
        if tid < HEADS:
            stat_offset = ((token * self.chunks + chunk) * HEADS + tid) * 2
            out_stats[stat_offset] = stats[tid, 0]
            out_stats[stat_offset + 1] = stats[tid, 1]
        cute.arch.mbarrier_wait(barrier, 1)
        _tcgen05.fence_after_thread_sync()
        partials = cute.make_tensor(
            cute.make_ptr(Float32, partial_ptr, cute.AddressSpace.gmem),
            cute.make_layout((Int32(0x7FFFFFFF),)),
        )
        for j in cutlass.range_constexpr(4):
            values = cute.make_rmem_tensor((HEADS,), Float32)
            values.store(_tcgen05.ld(warp * 32, tmem + HEADS * j, "32x32b", HEADS))
            _tcgen05.wait_ld()
            for h in cutlass.range_constexpr(HEADS):
                offset = (
                    ((token * self.chunks + chunk) * HEADS + h) * DIM + j * ROWS + tid
                )
                partials[offset] = values[h] * Float32(1.0 / 256.0)
        _tcgen05.fence_before_thread_sync()
        cute.arch.sync_threads()
        if warp == 0:
            _tcgen05.fence_after_thread_sync()
            cute.arch.dealloc_tmem(
                cute.make_ptr(Float32, tmem, cute.AddressSpace.tmem), 256
            )


@triton.jit
def _merge(
    Partial,
    Stats,
    Sinks,
    Out,
    STRIDE: tl.constexpr,
    CHUNKS: tl.constexpr,
    HAS_SINKS: tl.constexpr,
    SCALE: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
):
    token, head = tl.program_id(0), tl.program_id(1)
    chunks = tl.arange(0, C)
    dims = tl.arange(0, D)
    stat = ((token * CHUNKS + chunks) * H + head) * 2
    peak = tl.load(Stats + stat, chunks < CHUNKS, float("-inf"))
    sums = tl.load(Stats + stat + 1, chunks < CHUNKS, 0.0)
    sink = tl.full((), float("-inf"), tl.float32)
    if HAS_SINKS:
        sink = tl.load(Sinks + head) * 1.4426950408889634
    maximum = tl.maximum(tl.max(peak, 0), sink)
    maximum = tl.where(maximum == float("-inf"), 0.0, maximum)
    factors = tl.exp2(peak - maximum)
    denom = tl.sum(sums * factors, 0)
    if HAS_SINKS:
        denom += tl.exp2(sink - maximum)
    offset = ((token * CHUNKS + chunks[:, None]) * H + head) * D + dims[None, :]
    partial = tl.load(Partial + offset, chunks[:, None] < CHUNKS, 0.0)
    value = (
        tl.sum(partial * factors[:, None], 0) * SCALE / tl.where(denom > 0, denom, 1.0)
    )
    tl.store(Out + token * STRIDE + head * D + dims, value)


@cache
def _compiled(device, chunks):
    with torch.accelerator.device_index(device):
        return cute.compile(
            SparseMlaWide(chunks),
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
            Float32(0.0),
            Int32(1),
            make_fake_stream(),
        )


class WideWorkspace:
    def __init__(self, tokens, chunks, device):
        if not 0 < tokens <= 384 or not 0 < chunks <= 5:
            raise ValueError("wide prototype capacity exceeds its 384-row limit")
        self.tokens, self.chunks = tokens, chunks
        self.partials = torch.empty(
            (tokens, chunks, HEADS, DIM), device=device, dtype=torch.float32
        )
        self.stats = torch.empty(
            (tokens, chunks, HEADS, 2), device=device, dtype=torch.float32
        )


def supports_sparse_mla_wide(
    query, swa, comp, indices, lens, out, bmm1_scale, bmm2_scale, sinks
):
    """Metadata-only eligibility; the caller supplies the performance gate."""
    if (
        query.ndim != 3
        or query.shape[1:] != (HEADS, DIM)
        or not 0 < query.shape[0] <= 384
        or query.dtype != torch.float8_e4m3fn
        or query.stride(1) != DIM
        or query.stride(2) != 1
        or query.stride(0) % 16
        or query.data_ptr() % 16
    ):
        return False
    if (
        out.shape != query.shape
        or out.dtype != torch.bfloat16
        or out.stride(1) != DIM
        or out.stride(2) != 1
        or out.stride(0) % 8
        or out.data_ptr() % 16
    ):
        return False
    if (
        indices.ndim != 2
        or indices.shape[0] != query.shape[0]
        or not 0 < indices.shape[1] <= 640
        or indices.dtype != torch.int32
        or indices.stride(1) != 1
        or lens.dtype != torch.int32
        or not lens.is_contiguous()
        or lens.numel() != query.shape[0]
    ):
        return False
    if not base._is_pool(swa) or not base._is_pool(comp):
        return False
    if not isinstance(bmm1_scale, float | int) or not isinstance(
        bmm2_scale, float | int
    ):
        return False
    if query.device.type != "cuda" or any(
        t.device != query.device for t in (swa, comp, indices, lens, out)
    ):
        return False
    return sinks is None or (
        sinks.device == query.device
        and sinks.dtype == torch.float32
        and sinks.is_contiguous()
        and sinks.numel() >= HEADS
    )


def warmup_sparse_mla_wide(workspace):
    """Compile and launch on dummy storage before FULL graph capture."""
    device = workspace.partials.device
    q = torch.zeros(
        workspace.tokens, HEADS, DIM, device=device, dtype=torch.float8_e4m3fn
    )
    pool = torch.zeros(1, 64, DIM, device=device, dtype=q.dtype)
    indices = torch.full(
        (workspace.tokens, workspace.chunks * ROWS),
        -1,
        device=device,
        dtype=torch.int32,
    )
    lens = torch.zeros(workspace.tokens, device=device, dtype=torch.int32)
    out = torch.empty(q.shape, device=device, dtype=torch.bfloat16)
    sinks = torch.zeros(HEADS, device=device, dtype=torch.float32)
    for value in (sinks, None):
        sparse_mla_wide(
            q, pool, pool, indices, lens, out, DIM**-0.5, 1.0, value, workspace
        )


def sparse_mla_wide(
    query, swa, comp, indices, lens, out, bmm1_scale, bmm2_scale, sinks, workspace
):
    tokens, heads, dim = query.shape
    chunks = (indices.shape[1] + ROWS - 1) // ROWS
    if (
        heads != HEADS
        or dim != DIM
        or not 0 < tokens <= workspace.tokens
        or workspace.chunks != chunks
    ):
        raise ValueError("wide prototype requires a matching fixed-capacity workspace")
    if not supports_sparse_mla_wide(
        query, swa, comp, indices, lens, out, bmm1_scale, bmm2_scale, sinks
    ):
        raise ValueError("unsupported wide attention layout")
    if workspace.partials.device != query.device:
        raise ValueError("workspace is on a different device")
    _compiled(query.device.index, chunks)(
        Int64(query.data_ptr()),
        Int32(query.stride(0)),
        Int64(swa.data_ptr()),
        Int32(swa.stride(-2)),
        Int64(comp.data_ptr()),
        Int32(comp.stride(-2)),
        Int64(indices.data_ptr()),
        Int32(indices.stride(0)),
        Int32(indices.shape[1]),
        Int64(lens.data_ptr()),
        Int64(workspace.partials.data_ptr()),
        Int64(workspace.stats.data_ptr()),
        Float32(bmm1_scale * base.LOG2E),
        Int32(tokens),
        base.current_cu_stream(),
    )
    _merge[(tokens, heads)](
        workspace.partials,
        workspace.stats,
        sinks if sinks is not None else workspace.stats,
        out,
        out.stride(0),
        chunks,
        sinks is not None,
        bmm2_scale,
        triton.next_power_of_2(chunks),
        HEADS,
        DIM,
        num_warps=4,
    )
