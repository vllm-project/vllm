# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused local shared-expert addition and mHC for the DEP executor.

The caller must establish producer completion and protect peer buffers until
both consumers finish. This kernel does not implement a communication protocol.
Keep the BF16 rounding after the two-owner sum and after the shared expert add.
"""

from dataclasses import dataclass

import cuda.bindings.driver as cuda
import cutlass
import torch
from cutlass import BFloat16, Float32, Int32, Int64, Uint8, Uint32, cute

from vllm.models.common.ops.moe_tail_fusion.primitives import (
    VEC_BF16,
    bf16x8_to_packed_u32x4,
    current_cu_stream,
    group_leader_block_sum,
    load_global_u32x4,
    map_shared_to_peer,
    packed_u32x4_to_bf16x8,
    store_global_u32,
    store_global_u32x4,
    store_shared_cluster_f32,
    to_cute,
)
from vllm.models.deepseek_v41.nvidia.ops.cute_dsl.all_reduce_mhc import (
    FRAGMENTS_PER_MXFP8_BLOCK,
    SCALE_TILE_COLS,
    SCALE_TILE_ROWS,
    _LamportMHCDeviceKernel,
)

from . import MAX_ROWS


@dataclass(frozen=True)
class LocalMoEOutput:
    routed: torch.Tensor
    shared: torch.Tensor


class _FusedMHCDeviceKernel(_LamportMHCDeviceKernel):
    def __init__(self, hidden, hc, quantize, capture_aux, reduce_peer, add_shared):
        super().__init__(
            hidden=hidden,
            hc=hc,
            tp=1,
            capacity_m=MAX_ROWS,
            cluster_size=5,
            rank_lanes=1,
            threads=128,
            enable_pdl=True,
            quantize=quantize,
            reduced_input=True,
        )
        self.capture_aux = capture_aux
        self.reduce_peer = reduce_peer
        self.add_shared = add_shared

    @cute.jit
    def begin_collect(self, peer):
        return Uint32(0)

    @cute.jit
    def load_contribution(self, x, peer, offset, stage):
        value = packed_u32x4_to_bf16x8(load_global_u32x4(x.iterator + offset)).to(
            Float32
        )
        if cutlass.const_expr(self.reduce_peer):
            remote = packed_u32x4_to_bf16x8(
                load_global_u32x4(peer.iterator + offset)
            ).to(Float32)
            value = (value + remote).to(BFloat16).to(Float32)
        return value

    @cute.jit
    def end_collect(self, peer, stage):
        pass

    @cute.jit
    def __call__(
        self,
        x,
        peer,
        shared,
        residual,
        post,
        comb,
        pre,
        gamma,
        out,
        norm_out,
        quant_out,
        quant_scales,
        aux_out,
        eps: Float32,
        m: Int32,
        row_start: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            x,
            peer,
            shared,
            residual,
            post,
            comb,
            pre,
            gamma,
            out,
            norm_out,
            quant_out,
            quant_scales,
            aux_out,
            eps,
            row_start,
        ).launch(
            grid=(m, self.cluster_size, 1),
            block=(self.threads, 1, 1),
            cluster=(1, self.cluster_size, 1),
            smem=self.smem_size_in_bytes(),
            stream=stream,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        peer: cute.Tensor,
        shared: cute.Tensor,
        residual: cute.Tensor,
        post: cute.Tensor,
        comb: cute.Tensor,
        pre: cute.Tensor,
        gamma: cute.Tensor,
        out: cute.Tensor,
        norm_out: cute.Tensor,
        quant_out: cute.Tensor,
        quant_scales: cute.Tensor,
        aux_out: cute.Tensor,
        eps: Float32,
        row_start: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        token, _, _ = cute.arch.block_idx()
        cluster_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        base_fragment = cluster_rank * self.threads + tidx
        hc = self.hc
        prenorm = cute.make_rmem_tensor(
            cute.make_layout((self.trips, VEC_BF16), stride=(VEC_BF16, 1)),
            BFloat16,
        )
        prenorm.fill(BFloat16(0.0))
        gammas = cute.make_rmem_tensor(prenorm.layout, BFloat16)
        gammas.fill(BFloat16(0.0))
        old = cute.make_rmem_tensor(
            cute.make_layout(
                (self.trips, hc, VEC_BF16), stride=(hc * VEC_BF16, VEC_BF16, 1)
            ),
            BFloat16,
        )
        old.fill(BFloat16(0.0))
        for trip in cutlass.range_constexpr(self.trips):
            fragment = base_fragment + trip * self.fragment_stride
            if fragment < self.fragments:
                offset = Int64(fragment) * VEC_BF16
                gammas[trip, None].store(
                    packed_u32x4_to_bf16x8(load_global_u32x4(gamma.iterator + offset))
                )
                for source in cutlass.range_constexpr(hc):
                    offset_r = (Int64(token) * hc + source) * self.hidden + offset
                    old[trip, source, None].store(
                        packed_u32x4_to_bf16x8(
                            load_global_u32x4(residual.iterator + offset_r)
                        )
                    )
        post_mix = cute.make_rmem_tensor(cute.make_layout((hc,)), Float32)
        pre_mix = cute.make_rmem_tensor(cute.make_layout((hc,)), Float32)
        comb_mix = cute.make_rmem_tensor(
            cute.make_layout((hc, hc), stride=(hc, 1)), Float32
        )
        for target in cutlass.range_constexpr(hc):
            mix_index = Int64(token) * hc + target
            post_mix[target] = cute.arch.load(
                (post.iterator + mix_index).llvm_ptr, Float32
            )
            pre_mix[target] = cute.arch.load(
                (pre.iterator + mix_index).llvm_ptr, Float32
            )
            for source in cutlass.range_constexpr(hc):
                offset_c = (Int64(token) * hc + source) * hc + target
                comb_mix[source, target] = cute.arch.load(
                    (comb.iterator + offset_c).llvm_ptr, Float32
                )
        cute.arch.griddepcontrol_wait()
        stage = self.begin_collect(peer)
        for trip in cutlass.range_constexpr(self.trips):
            fragment = base_fragment + trip * self.fragment_stride
            if fragment < self.fragments:
                offset = Int64(token) * self.hidden + Int64(fragment) * VEC_BF16
                input_offset = offset + Int64(row_start) * self.hidden
                value = self.load_contribution(x, peer, input_offset, stage)
                if cutlass.const_expr(self.add_shared):
                    shared_value = packed_u32x4_to_bf16x8(
                        load_global_u32x4(shared.iterator + offset)
                    ).to(Float32)
                    value = (value + shared_value).to(BFloat16).to(Float32)
                collapse = cute.make_rmem_tensor(cute.make_layout((VEC_BF16,)), Float32)
                collapse.fill(Float32(0.0))
                if cutlass.const_expr(self.capture_aux):
                    mean = cute.make_rmem_tensor(cute.make_layout((VEC_BF16,)), Float32)
                    mean.fill(Float32(0.0))
                for target in cutlass.range_constexpr(hc):
                    mixed = value * post_mix[target]
                    for source in cutlass.range_constexpr(hc):
                        mixed = (
                            mixed
                            + old[trip, source, None].load().to(Float32)
                            * comb_mix[source, target]
                        )
                    rounded = mixed.to(BFloat16)
                    if cutlass.const_expr(self.capture_aux):
                        mean.store(mean.load() + rounded.to(Float32))
                    offset_r = (Int64(token) * hc + target) * self.hidden + Int64(
                        fragment
                    ) * VEC_BF16
                    store_global_u32x4(
                        Int64((out.iterator + offset_r).toint()),
                        bf16x8_to_packed_u32x4(rounded),
                    )
                    collapse.store(
                        collapse.load() + rounded.to(Float32) * pre_mix[target]
                    )
                prenorm[trip, None].store(collapse.load().to(BFloat16))
                if cutlass.const_expr(self.capture_aux):
                    store_global_u32x4(
                        Int64((aux_out.iterator + offset).toint()),
                        bf16x8_to_packed_u32x4(
                            (mean.load() / Float32(hc)).to(BFloat16)
                        ),
                    )
        self.end_collect(peer, stage)
        cute.arch.griddepcontrol_launch_dependents()
        thread_sum = Float32(0.0)
        for trip in cutlass.range_constexpr(self.trips):
            fragment = base_fragment + trip * self.fragment_stride
            if fragment < self.fragments:
                values = prenorm[trip, None].load().to(Float32)
                thread_sum = thread_sum + (values * values).reduce(
                    cute.ReductionOp.ADD,
                    init_val=Float32(0.0),
                    reduction_profile=0,
                )
        smem = cutlass.utils.SmemAllocator()
        warp_sums = smem.allocate_array(Float32, self.warps)
        cluster_sums = smem.allocate_array(Float32, self.cluster_size)
        cta_sum = group_leader_block_sum(thread_sum, warp_sums, self.warps, 1)
        # Every peer CTA must exist before accessing its distributed shared memory.
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
        if tidx < self.cluster_size:
            remote = map_shared_to_peer(cluster_sums + cluster_rank, Int32(tidx))
            store_shared_cluster_f32(remote, cta_sum)
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
        full_sum = Float32(0.0)
        for peer in cutlass.range_constexpr(self.cluster_size):
            full_sum = full_sum + cute.arch.load(
                (cluster_sums + peer).llvm_ptr, Float32
            )
        inv_rms = cute.math.rsqrt(full_sum / Float32(self.hidden) + eps, fastmath=True)
        for trip in cutlass.range_constexpr(self.trips):
            fragment = base_fragment + trip * self.fragment_stride
            if cutlass.const_expr(self.quantize):
                normalized = cute.make_rmem_tensor(
                    cute.make_layout((VEC_BF16,)), BFloat16
                )
                normalized.fill(BFloat16(0.0))
            if fragment < self.fragments:
                result = (
                    prenorm[trip, None].load().to(Float32)
                    * inv_rms
                    * gammas[trip, None].load().to(Float32)
                ).to(BFloat16)
                offset = Int64(token) * self.hidden + Int64(fragment) * VEC_BF16
                store_global_u32x4(
                    Int64((norm_out.iterator + offset).toint()),
                    bf16x8_to_packed_u32x4(result),
                )
                if cutlass.const_expr(self.quantize):
                    normalized.store(result)
            if cutlass.const_expr(self.quantize):
                words = bf16x8_to_packed_u32x4(normalized.load())
                block_max = self.mxfp8_ops.reduce_max_4threads(
                    self.mxfp8_ops.bfloat2_hmax_reduce_to_f32(
                        self.mxfp8_ops.bfloat2_max_abs_4(
                            words[0], words[1], words[2], words[3]
                        )
                    )
                )
                scale = self.mxfp8_ops.float_to_ue8m0_fast(
                    block_max * Float32(self.mxfp8_ops.INV_FLOAT8_E4M3_MAX)
                )
                inv_scale = self.mxfp8_ops.ue8m0_to_inv_scale_fast(scale)
                if fragment < self.fragments:
                    self.fp4_ops.st_global_u64(
                        self.fp4_ops.get_ptr_as_int64(
                            quant_out,
                            Int64(token) * self.hidden + Int64(fragment) * VEC_BF16,
                        ),
                        self.mxfp8_ops.bfloat2x4_to_fp8x8_packed(
                            words[0], words[1], words[2], words[3], inv_scale
                        ),
                    )
                    if fragment % FRAGMENTS_PER_MXFP8_BLOCK == 0:
                        quant_scales[
                            self.mxfp8_ops.compute_sf_index_swizzled_128x4_gpu(
                                token,
                                fragment // FRAGMENTS_PER_MXFP8_BLOCK,
                                Int32(self.scale_cols),
                            )
                        ] = scale.to(Uint8)
        if cutlass.const_expr(self.quantize):
            m, _, _ = cute.arch.grid_dim()
            words_per_row = self.scale_cols // SCALE_TILE_COLS
            padded_m = (m + SCALE_TILE_ROWS - 1) // SCALE_TILE_ROWS * SCALE_TILE_ROWS
            pad_words = (padded_m - m) * words_per_row
            word = (token * self.cluster_size + cluster_rank) * self.threads + tidx
            while word < pad_words:
                row = m + word // words_per_row
                offset = self.mxfp8_ops.compute_sf_index_swizzled_128x4_gpu(
                    row, word % words_per_row * SCALE_TILE_COLS, Int32(self.scale_cols)
                )
                store_global_u32((quant_scales.iterator + offset).toint(), Uint32(0))
                word = word + m * self.cluster_size * self.threads


class FusedPostPre:
    def __init__(self):
        self.compiled = {}

    def __call__(
        self,
        x,
        residual,
        post,
        comb,
        pre,
        norm,
        mxfp8_out,
        capture_aux=False,
        *,
        peer=None,
        shared=None,
        row_start=0,
    ):
        rows, hc, hidden = residual.shape
        if rows == 0:
            return (
                residual.clone(),
                torch.empty_like(residual[:, 0]),
                (torch.empty_like(residual[:, 0]) if capture_aux else None),
            )
        assert row_start >= 0 and row_start + rows <= x.shape[0]
        assert x.shape[1] == hidden
        if peer is not None:
            assert peer.shape == x.shape and peer.dtype == x.dtype
            assert peer.is_contiguous()
        if shared is not None:
            assert shared.shape == (rows, hidden) and shared.dtype == x.dtype
            assert shared.is_contiguous()
        if not (
            0 < rows <= MAX_ROWS
            and hidden == 5120
            and hc == 4
            and x.dtype == residual.dtype == torch.bfloat16
            and all(
                t.is_contiguous() for t in (x, residual, post, comb, pre, norm.weight)
            )
        ):
            raise ValueError("prototype supports contiguous BF16 H5120 HC4 only")
        key = (
            rows,
            hidden,
            hc,
            mxfp8_out is not None,
            capture_aux,
            peer is not None,
            shared is not None,
            x.device,
        )
        if key not in self.compiled and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("compile this shape before capture")
        out = torch.empty_like(residual)
        norm_out = torch.empty_like(residual[:, 0])
        aux = torch.empty_like(norm_out) if capture_aux else None
        quant_out, scales = (
            mxfp8_out
            if mxfp8_out is not None
            else (norm_out.view(torch.uint8), norm_out.view(torch.uint8))
        )
        tensors = (
            x,
            x if peer is None else peer,
            x if shared is None else shared,
            residual,
            post,
            comb,
            pre,
            norm.weight,
            out,
            norm_out,
            quant_out,
            scales,
            norm_out if aux is None else aux,
        )
        alignments = (16, 16, 16, 16, 4, 4, 4, 16, 16, 16, 16, 16, 16)
        args = tuple(to_cute(t.reshape(-1), a) for t, a in zip(tensors, alignments))
        args += (
            Float32(norm.variance_epsilon),
            Int32(rows),
            Int32(row_start),
            current_cu_stream(),
        )
        if key not in self.compiled:
            self.compiled[key] = cute.compile(
                _FusedMHCDeviceKernel(
                    hidden,
                    hc,
                    mxfp8_out is not None,
                    capture_aux,
                    peer is not None,
                    shared is not None,
                ),
                *args,
            )
        self.compiled[key](*args)
        return out, norm_out, aux
