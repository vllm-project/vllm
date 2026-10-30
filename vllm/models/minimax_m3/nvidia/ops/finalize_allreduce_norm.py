# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# The consumer is adapted from FlashInfer's low-latency MNNVL CuTe DSL
# all-reduce (flashinfer/comm/mnnvl_cutedsl/kernel_ll,
# _LamportResidualRMSNormDeviceKernel), as DSV4.1's AllReduceMHC is.

"""MoE top-k finalize + TP all-reduce + Gemma RMSNorm + MXFP8 quantization.

FlashInfer's Lamport all-reduce: the finalize publish kernel sums each token's
routed expert rows and its shared-expert row, then multicasts that TP-partial
row into a three-generation mailbox; a PDL-chained consumer spins until every
rank's fragment lands, reduces them, adds the residual and applies the Gemma
RMSNorm. Instead of a BF16 row, the consumer writes the normalized row as MXFP8
(E4M3 values and UE8M0 scales of 32-element blocks in the 128x4 swizzled
layout), so the MXFP8 linear that follows skips its own input quantization.
"""

import math

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from cutlass import BFloat16, Float32, Int32, Int64, Uint8, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import make_fake_compact_tensor
from cutlass.cutlass_dsl import T, dsl_user_op

from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size, get_tp_group
from vllm.model_executor.layers.fused_moe.moe_output import (
    MoEOutput,
    UnfinalizedMoEOutput,
)
from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp8Dynamic
from vllm.models.common.ops.moe_tail_fusion.primitives import (
    ACTIVE_STAGE,
    LAMPORT_GENERATIONS,
    NEXT_STAGE,
    VEC_BF16,
    WARP_SIZE,
    bf16x8_to_packed_u32x4,
    current_cu_stream,
    fragment_has_negative_zero,
    group_leader_block_sum,
    load_global_u32x4,
    load_volatile_u32,
    make_fake_dynamic_compact_tensor,
    map_shared_to_peer,
    packed_u32x4_to_bf16x8,
    store_global_u32,
    store_global_u32x4,
    store_lamport_sentinel_u32x4,
    store_shared_cluster_f32,
    to_cute,
    to_cute_dynamic,
)
from vllm.models.deepseek_v41.nvidia.ops.cute_dsl.all_reduce_mhc import (
    _QuadFinalizePublishDeviceKernel,
)
from vllm.platforms import current_platform

MXFP8_BLOCK = 32
# Fragments (of VEC_BF16 values) per MXFP8 block, held by adjacent lanes.
_FRAGMENTS_PER_BLOCK = MXFP8_BLOCK // VEC_BF16
# The swizzled scale layout tiles 128 rows by 4 block columns.
_SCALE_ROW_TILE = 128
_SCALE_COL_TILE = 4


@dsl_user_op
def _ue8m0_ceil(value: Float32, *, loc=None, ip=None) -> Uint32:
    """FlashInfer's float_to_ue8m0_fast: the biased exponent rounded up, 0 for
    non-positive input, as its MXFP8 quantization kernels compute it."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Float32(value).ir_value(loc=loc, ip=ip)],
            "{\n\t"
            ".reg .pred p_zero, p_has_mant, p_exp_zero, p_tiny_sub, p_ovf;\n\t"
            ".reg .u32 bits, exp_biased, mantissa, bump, result;\n\t"
            "setp.le.f32 p_zero, $1, 0f00000000;\n\t"
            "mov.b32 bits, $1;\n\t"
            "shr.b32 exp_biased, bits, 23;\n\t"
            "and.b32 exp_biased, exp_biased, 255;\n\t"
            "and.b32 mantissa, bits, 0x7FFFFF;\n\t"
            "setp.ne.u32 p_has_mant, mantissa, 0;\n\t"
            "selp.u32 bump, 1, 0, p_has_mant;\n\t"
            "setp.eq.u32 p_exp_zero, exp_biased, 0;\n\t"
            "setp.le.u32 p_tiny_sub, mantissa, 0x400000;\n\t"
            "and.pred p_tiny_sub, p_exp_zero, p_tiny_sub;\n\t"
            "@p_tiny_sub mov.u32 bump, 0;\n\t"
            "add.u32 result, exp_biased, bump;\n\t"
            "setp.gt.u32 p_ovf, result, 254;\n\t"
            "selp.u32 result, 254, result, p_ovf;\n\t"
            "selp.u32 $0, 0, result, p_zero;\n\t"
            "}",
            "=r,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ue8m0_inverse(exponent: Uint32, *, loc=None, ip=None) -> Float32:
    """2^(127 - exponent), or 0 for exponent 0."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Uint32(exponent).ir_value(loc=loc, ip=ip)],
            "{\n\t"
            ".reg .s32 new_exp;\n\t"
            ".reg .pred p_zero;\n\t"
            "setp.eq.u32 p_zero, $1, 0;\n\t"
            "sub.s32 new_exp, 254, $1;\n\t"
            "max.s32 new_exp, new_exp, 0;\n\t"
            "shl.b32 $0, new_exp, 23;\n\t"
            "@p_zero mov.b32 $0, 0;\n\t"
            "}",
            "=f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _f32x4_to_e4m3x4(
    a: Float32, b: Float32, c: Float32, d: Float32, *, loc=None, ip=None
) -> Uint32:
    """Four values to E4M3, ``a`` in the low byte, clamped to +-448 first so
    that inf saturates like FlashInfer's MXFP8 kernels and torch."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Float32(v).ir_value(loc=loc, ip=ip) for v in (a, b, c, d)],
            "{\n\t"
            ".reg .f32 f0, f1, f2, f3;\n\t"
            ".reg .b16 lo, hi;\n\t"
            "min.f32 f0, $1, 0f43E00000;\n\t"
            "max.f32 f0, f0, 0fC3E00000;\n\t"
            "min.f32 f1, $2, 0f43E00000;\n\t"
            "max.f32 f1, f1, 0fC3E00000;\n\t"
            "min.f32 f2, $3, 0f43E00000;\n\t"
            "max.f32 f2, f2, 0fC3E00000;\n\t"
            "min.f32 f3, $4, 0f43E00000;\n\t"
            "max.f32 f3, f3, 0fC3E00000;\n\t"
            "cvt.rn.satfinite.e4m3x2.f32 lo, f1, f0;\n\t"
            "cvt.rn.satfinite.e4m3x2.f32 hi, f3, f2;\n\t"
            "mov.b32 $0, {lo, hi};\n\t"
            "}",
            "=r,f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _store_global_u32x2(
    address: Int64, low: Uint32, high: Uint32, *, loc=None, ip=None
) -> None:
    llvm.inline_asm(
        None,
        [
            address.ir_value(loc=loc, ip=ip),
            low.ir_value(loc=loc, ip=ip),
            high.ir_value(loc=loc, ip=ip),
        ],
        "st.global.v2.u32 [$0], {$1, $2};",
        "l,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@cute.jit
def _swizzled_scale_offset(row: Int32, col: Int32, padded_cols: Int32) -> Int32:
    """Byte offset of block ``col`` of ``row`` in the 128x4 swizzled layout."""
    return (
        col % Int32(4)
        + (col // Int32(4)) * Int32(512)
        + (row % Int32(32)) * Int32(16)
        + ((row % Int32(128)) // Int32(32)) * Int32(4)
        + (row // Int32(128)) * Int32(128) * padded_cols
    )


class _LamportGemmaRMSNormMxfp8Kernel:
    """FlashInfer's ``_LamportResidualRMSNormDeviceKernel`` (residual add and
    copy, one rank lane) with a Gemma RMSNorm whose BF16 result is quantized to
    MXFP8 instead of stored.

    A 32-element scale block is four consecutive fragments, which adjacent lanes
    hold, so two shuffles reduce its amax. The scales of the padding rows below
    the next multiple of 128 are zeroed, as FlashInfer's quantization does.
    """

    def __init__(
        self,
        *,
        hidden: int,
        tp: int,
        capacity_m: int,
        cluster_size: int,
        threads: int,
        enable_pdl: bool,
    ) -> None:
        if hidden <= 0 or hidden % (MXFP8_BLOCK * _SCALE_COL_TILE):
            raise ValueError("hidden must be a positive multiple of 128")
        if threads <= 0 or threads % WARP_SIZE:
            raise ValueError("threads must be a positive multiple of the warp size")
        self.hidden = hidden
        self.tp = tp
        self.capacity_m = capacity_m
        self.cluster_size = cluster_size
        self.threads = threads
        self.enable_pdl = enable_pdl
        self.fragments = hidden // VEC_BF16
        self.fragment_stride = cluster_size * threads
        self.trips = math.ceil(self.fragments / self.fragment_stride)
        self.warps = threads // WARP_SIZE
        self.scale_cols = hidden // MXFP8_BLOCK

    def smem_size_in_bytes(self) -> int:
        return (self.warps + self.cluster_size) * 4

    @cute.jit
    def __call__(
        self,
        contribution_mailbox: cute.Tensor,
        residual_source: cute.Tensor,
        gamma: cute.Tensor,
        residual_output: cute.Tensor,
        quant_output: cute.Tensor,
        scale_output: cute.Tensor,
        stage_state: cute.Tensor,
        eps: Float32,
        m: Int32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(
            contribution_mailbox,
            residual_source,
            gamma,
            residual_output,
            quant_output,
            scale_output,
            stage_state,
            eps,
            m,
        ).launch(
            grid=(m, self.cluster_size, 1),
            block=(self.threads, 1, 1),
            cluster=(1, self.cluster_size, 1),
            smem=self.smem_size_in_bytes(),
            stream=stream,
            use_pdl=self.enable_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        contribution_mailbox: cute.Tensor,
        residual_source: cute.Tensor,
        gamma: cute.Tensor,
        residual_output: cute.Tensor,
        quant_output: cute.Tensor,
        scale_output: cute.Tensor,
        stage_state: cute.Tensor,
        eps: Float32,
        m: Int32,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        token, _, _ = cute.arch.block_idx()
        cluster_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        base_fragment = cluster_rank * self.threads + tidx

        prenorm_fragments = cute.make_rmem_tensor(
            cute.make_layout((self.trips, VEC_BF16), stride=(VEC_BF16, 1)),
            BFloat16,
        )
        prenorm_fragments.fill(BFloat16(0.0))
        gamma_fragments = cute.make_rmem_tensor(
            cute.make_layout((self.trips, VEC_BF16), stride=(VEC_BF16, 1)),
            BFloat16,
        )
        gamma_fragments.fill(BFloat16(0.0))
        residual_fragments = cute.make_rmem_tensor(
            cute.make_layout((self.trips, VEC_BF16), stride=(VEC_BF16, 1)),
            BFloat16,
        )
        residual_fragments.fill(BFloat16(0.0))

        for trip in cutlass.range_constexpr(self.trips):
            fragment = base_fragment + trip * self.fragment_stride
            if fragment < self.fragments:
                gamma_element = Int64(fragment) * VEC_BF16
                gamma_pointer = cute.make_ptr(
                    BFloat16,
                    (gamma.iterator + gamma_element).llvm_ptr,
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                )
                gamma_fragments[trip, None].store(
                    packed_u32x4_to_bf16x8(load_global_u32x4(gamma_pointer))
                )
                residual_element = Int64(token) * self.hidden + gamma_element
                residual_pointer = cute.make_ptr(
                    BFloat16,
                    (residual_source.iterator + residual_element).llvm_ptr,
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                )
                residual_fragments[trip, None].store(
                    packed_u32x4_to_bf16x8(load_global_u32x4(residual_pointer))
                )

        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_wait()

        active_stage = load_volatile_u32(stage_state.iterator + ACTIVE_STAGE)
        for trip in cutlass.range_constexpr(self.trips):
            fragment = base_fragment + trip * self.fragment_stride
            lane_packed = cute.make_rmem_tensor(cute.make_layout((self.tp, 4)), Uint32)
            lane_packed.fill(Uint32(0))
            dirty = fragment < self.fragments
            while dirty:
                dirty = False
                for source_rank in cutlass.range_constexpr(self.tp):
                    if fragment < self.fragments:
                        source_element = (
                            (Int64(active_stage) * self.tp + Int64(source_rank))
                            * self.capacity_m
                            + Int64(token)
                        ) * self.hidden + Int64(fragment) * VEC_BF16
                        source_pointer = cute.make_ptr(
                            BFloat16,
                            (contribution_mailbox.iterator + source_element).llvm_ptr,
                            cute.AddressSpace.gmem,
                            assumed_align=16,
                        )
                        packed = load_global_u32x4(source_pointer, volatile=True)
                        dirty = dirty | fragment_has_negative_zero(packed)
                        for word in cutlass.range_constexpr(4):
                            lane_packed[source_rank, word] = packed[word]

            if fragment < self.fragments:
                prenorm = cute.make_rmem_tensor(cute.make_layout((VEC_BF16,)), Float32)
                prenorm.fill(Float32(0.0))
                for source_rank in cutlass.range_constexpr(self.tp):
                    packed = cute.make_rmem_tensor(cute.make_layout((4,)), Uint32)
                    for word in cutlass.range_constexpr(4):
                        packed[word] = lane_packed[source_rank, word]
                    prenorm.store(
                        prenorm.load()
                        + packed_u32x4_to_bf16x8(packed.load()).to(Float32)
                    )
                prenorm_bf16 = (
                    prenorm.load() + residual_fragments[trip, None].load().to(Float32)
                ).to(BFloat16)
                prenorm_fragments[trip, None].store(prenorm_bf16)
                output_element = Int64(token) * self.hidden + Int64(fragment) * VEC_BF16
                store_global_u32x4(
                    Int64((residual_output.iterator + output_element).toint()),
                    bf16x8_to_packed_u32x4(prenorm_bf16),
                )

        if token == 0 and cluster_rank == 0 and tidx == 0:
            store_global_u32(
                (stage_state.iterator + NEXT_STAGE).toint(),
                (active_stage + Uint32(1)) % Uint32(LAMPORT_GENERATIONS),
            )
        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_launch_dependents()

        for trip in cutlass.range_constexpr(self.trips):
            fragment = base_fragment + trip * self.fragment_stride
            for source_rank in cutlass.range_constexpr(self.tp):
                if fragment < self.fragments:
                    source_element = (
                        (Int64(active_stage) * self.tp + Int64(source_rank))
                        * self.capacity_m
                        + Int64(token)
                    ) * self.hidden + Int64(fragment) * VEC_BF16
                    store_lamport_sentinel_u32x4(
                        Int64((contribution_mailbox.iterator + source_element).toint())
                    )

        thread_sum = Float32(0.0)
        for trip in cutlass.range_constexpr(self.trips):
            fragment = base_fragment + trip * self.fragment_stride
            if fragment < self.fragments:
                values = prenorm_fragments[trip, None].load().to(Float32)
                thread_sum = thread_sum + (values * values).reduce(
                    cute.ReductionOp.ADD,
                    init_val=Float32(0.0),
                    reduction_profile=0,
                )

        smem = cutlass.utils.SmemAllocator()
        warp_sums = smem.allocate_array(Float32, self.warps)
        cluster_sums = smem.allocate_array(Float32, self.cluster_size)
        cta_sum = group_leader_block_sum(thread_sum, warp_sums, self.warps, 1)
        if tidx < self.cluster_size:
            local_slot = cluster_sums + cluster_rank
            remote_slot = map_shared_to_peer(local_slot, Int32(tidx))
            store_shared_cluster_f32(remote_slot, cta_sum)
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()

        full_sum = Float32(0.0)
        for peer in cutlass.range_constexpr(self.cluster_size):
            full_sum = full_sum + cute.arch.load(
                (cluster_sums + peer).llvm_ptr,
                Float32,
            )
        inv_rms = cute.math.rsqrt(
            full_sum / Float32(self.hidden) + eps,
            fastmath=True,
        )
        padded_cols = Int32(self.scale_cols)
        for trip in cutlass.range_constexpr(self.trips):
            fragment = base_fragment + trip * self.fragment_stride
            # Gemma RMSNorm scales by (1 + weight); the unfused norm returns BF16.
            normed = (
                (
                    prenorm_fragments[trip, None].load().to(Float32)
                    * inv_rms
                    * (gamma_fragments[trip, None].load().to(Float32) + Float32(1.0))
                )
                .to(BFloat16)
                .to(Float32)
            )
            values = cute.make_rmem_tensor(cute.make_layout((VEC_BF16,)), Float32)
            values.store(normed)
            amax = Float32(0.0)
            for element in cutlass.range_constexpr(VEC_BF16):
                amax = cute.arch.fmax(
                    amax, cute.arch.fmax(values[element], -values[element])
                )
            # Out-of-range fragments are zero; every lane joins the shuffles.
            for offset in cutlass.range_constexpr(1, _FRAGMENTS_PER_BLOCK):
                if cutlass.const_expr((offset & (offset - 1)) == 0):
                    amax = cute.arch.fmax(
                        amax,
                        cute.arch.shuffle_sync_bfly(
                            amax, offset=offset, mask=-1, mask_and_clamp=31
                        ),
                    )
            exponent = _ue8m0_ceil(amax * Float32(1.0 / 448.0))
            inverse = _ue8m0_inverse(exponent)
            if fragment < self.fragments:
                low = _f32x4_to_e4m3x4(
                    values[0] * inverse,
                    values[1] * inverse,
                    values[2] * inverse,
                    values[3] * inverse,
                )
                high = _f32x4_to_e4m3x4(
                    values[4] * inverse,
                    values[5] * inverse,
                    values[6] * inverse,
                    values[7] * inverse,
                )
                quant_element = Int64(token) * self.hidden + Int64(fragment) * VEC_BF16
                _store_global_u32x2(
                    Int64((quant_output.iterator + quant_element).toint()), low, high
                )
                if fragment % _FRAGMENTS_PER_BLOCK == 0:
                    col = fragment // _FRAGMENTS_PER_BLOCK
                    scale_output[_swizzled_scale_offset(token, col, padded_cols)] = (
                        exponent.to(Uint8)
                    )

        # Zero the scales of the last row tile's padding rows. A 4-byte word of
        # a 512-byte block holds the 4 blocks of one row, so every thread of
        # the grid clears whole words.
        tile_words = Int32(self.scale_cols * _SCALE_ROW_TILE // 4)
        last_tile = (m - Int32(1)) // Int32(_SCALE_ROW_TILE)
        word = (token * Int32(self.cluster_size) + cluster_rank) * Int32(
            self.threads
        ) + tidx
        while word < tile_words:
            block_word = word % Int32(128)
            row = (
                last_tile * Int32(_SCALE_ROW_TILE)
                + block_word // Int32(4)
                + block_word % Int32(4) * Int32(32)
            )
            if row >= m:
                byte = (Int64(last_tile) * Int64(tile_words) + Int64(word)) * 4
                store_global_u32(
                    cute.make_ptr(
                        Uint32,
                        (scale_output.iterator + byte).llvm_ptr,
                        cute.AddressSpace.gmem,
                        assumed_align=4,
                    ).toint(),
                    Uint32(0),
                )
            word = word + m * Int32(self.cluster_size * self.threads)


def mxfp8_scale_numel(num_tokens: int, hidden_size: int) -> int:
    """Bytes of the 128x4 swizzled UE8M0 scales of ``num_tokens`` rows."""
    rows = math.ceil(num_tokens / _SCALE_ROW_TILE) * _SCALE_ROW_TILE
    cols = math.ceil(hidden_size // MXFP8_BLOCK / _SCALE_COL_TILE) * _SCALE_COL_TILE
    return rows * cols


class FinalizeAllReduceGemmaRMSNormMxfp8:
    """MoE top-k finalize, TP all-reduce, residual add, Gemma RMSNorm and MXFP8
    quantization in two PDL-chained launches.

    ``__call__`` takes an unfinalized MoE output plus its TP-partial shared-expert
    output and returns the MXFP8 normalized rows with the new residual. Each
    instance owns a Lamport mailbox for ``max_num_tokens`` rows that all layers
    share; construction is collective over the TP group.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        max_num_tokens: int,
        top_k: int,
        fp32_weights: bool,
        device: torch.device,
    ) -> None:
        group = get_tp_group().device_group
        self.hidden_size = hidden = hidden_size
        self.capacity = capacity = max_num_tokens
        self.top_k = top_k
        self.fp32_weights = fp32_weights
        tp = dist.get_world_size(group)
        rank = dist.get_rank(group)
        device = torch.device(device)

        # A cluster of 128-thread CTAs covers a token's fragments in one trip
        # (six for hidden 6144); the finalize stages all top_k routes at once.
        cluster_size = math.ceil(hidden // VEC_BF16 / 128)
        with torch.accelerator.device_index(device.index):
            self._publish = cute.compile(
                _QuadFinalizePublishDeviceKernel(
                    hidden=hidden,
                    top_k=top_k,
                    tp=tp,
                    rank=rank,
                    capacity_m=capacity,
                    threads=128,
                    # The routing weights already carry routed_scaling_factor.
                    routed_scaling_factor=1.0,
                    include_shared_expert=True,
                    load_shared_expert_before_pdl=False,
                    enable_pdl=True,
                    prefetch_group=top_k,
                    fp32_weights=fp32_weights,
                ),
                make_fake_dynamic_compact_tensor(
                    BFloat16, alignment=16, divisibility=hidden
                ),
                make_fake_dynamic_compact_tensor(
                    Float32 if fp32_weights else BFloat16,
                    alignment=4 if fp32_weights else 2,
                    divisibility=top_k,
                ),
                make_fake_dynamic_compact_tensor(
                    Int32, alignment=4, divisibility=top_k
                ),
                make_fake_dynamic_compact_tensor(
                    BFloat16, alignment=16, divisibility=hidden
                ),
                make_fake_compact_tensor(Int32, (2,), assumed_align=4),
                Int64(0),
                Int32(capacity),
                current_cu_stream(),
            )
            self._collective = cute.compile(
                _LamportGemmaRMSNormMxfp8Kernel(
                    hidden=hidden,
                    tp=tp,
                    capacity_m=capacity,
                    cluster_size=cluster_size,
                    threads=128,
                    enable_pdl=True,
                ),
                make_fake_compact_tensor(
                    BFloat16,
                    (LAMPORT_GENERATIONS * tp * capacity * hidden,),
                    assumed_align=16,
                ),
                make_fake_dynamic_compact_tensor(
                    BFloat16, alignment=16, divisibility=hidden
                ),
                make_fake_compact_tensor(BFloat16, (hidden,), assumed_align=16),
                make_fake_dynamic_compact_tensor(
                    BFloat16, alignment=16, divisibility=hidden
                ),
                make_fake_dynamic_compact_tensor(
                    Uint8, alignment=16, divisibility=hidden
                ),
                make_fake_dynamic_compact_tensor(
                    Uint8, alignment=16, divisibility=_SCALE_ROW_TILE
                ),
                make_fake_compact_tensor(Int32, (2,), assumed_align=4),
                Float32(0.0),
                Int32(capacity),
                current_cu_stream(),
            )

            self._mailbox = symm_mem.empty(
                (LAMPORT_GENERATIONS, tp, capacity, hidden),
                dtype=torch.bfloat16,
                device=device,
            )
            self._mailbox_handle = symm_mem.rendezvous(self._mailbox, group)
            multicast = int(self._mailbox_handle.multicast_ptr or 0)
            if not multicast or multicast % 16:
                raise RuntimeError("NVLink multicast mapping is unavailable")
            self._multicast = multicast
            # Every slot starts as the Lamport sentinel, BF16 negative zero.
            self._mailbox.view(torch.int16).fill_(-32768)
            self._stage_state = torch.zeros(2, dtype=torch.int32, device=device)
            torch.accelerator.synchronize()
        dist.barrier(group=group)

    def __call__(
        self,
        moe_output: MoEOutput,
        residual: torch.Tensor,
        norm_weight: torch.Tensor,
        norm_eps: float,
    ) -> tuple[QuantizedActivation, torch.Tensor]:
        """Return (MXFP8 normalized rows, new residual).

        Args:
            moe_output: Unfinalized TP-partial routed output and the TP-partial
                shared-expert output, with routed_scaling_factor in the weights.
            residual: [M, hidden] BF16 residual stream.
            norm_weight: [hidden] BF16 Gemma RMSNorm weight (scale is 1 + w).
            norm_eps: RMSNorm epsilon.

        """
        routed = moe_output.routed
        shared_output = moe_output.shared_output
        assert isinstance(routed, UnfinalizedMoEOutput)
        assert shared_output is not None
        assert moe_output.routed_scaling_factor == 1.0
        hidden, top_k = self.hidden_size, self.top_k
        m = residual.shape[0]
        if not 1 <= m <= self.capacity:
            raise ValueError(f"M={m} is outside [1, {self.capacity}]")
        if residual.shape != (m, hidden) or shared_output.shape != (m, hidden):
            raise ValueError("residual and shared_output must be [M, hidden]")
        weights_dtype = torch.float32 if self.fp32_weights else torch.bfloat16
        if (
            routed.expert_weights.dtype != weights_dtype
            or routed.expert_weights.numel() != m * top_k
        ):
            raise ValueError(f"expert_weights must be {weights_dtype} [M, top_k]")
        self._publish(
            to_cute_dynamic(routed.gemm2_permuted.flatten(), 16, divisibility=hidden),
            to_cute_dynamic(
                routed.expert_weights.flatten(),
                4 if self.fp32_weights else 2,
                divisibility=top_k,
            ),
            to_cute_dynamic(
                routed.expanded_idx_to_permuted_idx.flatten(), 4, divisibility=top_k
            ),
            to_cute_dynamic(shared_output.flatten(), 16, divisibility=hidden),
            to_cute(self._stage_state, 4),
            Int64(self._multicast),
            Int32(m),
            current_cu_stream(),
        )
        residual_output = torch.empty_like(residual)
        quantized = torch.empty(
            (m, hidden), dtype=torch.float8_e4m3fn, device=residual.device
        )
        scales = torch.empty(
            mxfp8_scale_numel(m, hidden), dtype=torch.uint8, device=residual.device
        )
        self._collective(
            to_cute(self._mailbox.flatten(), 16),
            to_cute_dynamic(residual.flatten(), 16, divisibility=hidden),
            to_cute(norm_weight, 16),
            to_cute_dynamic(residual_output.flatten(), 16, divisibility=hidden),
            to_cute_dynamic(
                quantized.view(torch.uint8).flatten(), 16, divisibility=hidden
            ),
            to_cute_dynamic(scales, 16, divisibility=_SCALE_ROW_TILE),
            to_cute(self._stage_state, 4),
            Float32(norm_eps),
            Int32(m),
            current_cu_stream(),
        )
        return (
            QuantizedActivation(
                quantized, scales, residual.dtype, quantized.shape, kMxfp8Dynamic
            ),
            residual_output,
        )


_fused_moe_finalize: FinalizeAllReduceGemmaRMSNormMxfp8 | None = None


def supports_fused_moe_finalize(vllm_config: VllmConfig) -> bool:
    """Whether to build the fused MoE finalize kernel: tensor parallel without
    expert parallel, a BF16 model on SM100-family GPUs, and NVLink multicast on
    every TP rank. Every TP rank calls it; the multicast check is collective.
    """
    from vllm.distributed.device_communicators.custom_all_reduce import (
        _all_ranks_true,
        _has_local_multicast_support,
    )

    if (
        get_tensor_model_parallel_world_size() <= 1
        or vllm_config.parallel_config.enable_expert_parallel
        or not current_platform.is_cuda()
        or not current_platform.is_device_capability_family(100)
        or vllm_config.model_config.dtype != torch.bfloat16
    ):
        return False
    device = torch.device(
        current_platform.device_type, torch.accelerator.current_device_index()
    )
    return _all_ranks_true(
        get_tp_group().cpu_group, _has_local_multicast_support(device)
    )


def init_fused_moe_finalize(
    *, hidden_size: int, max_num_tokens: int, top_k: int, fp32_weights: bool
) -> None:
    """Build the kernel every deferred MoE output is finalized with.

    Collective over the TP group, so every rank calls it, and only when
    ``supports_fused_moe_finalize`` holds.
    """
    global _fused_moe_finalize
    _fused_moe_finalize = FinalizeAllReduceGemmaRMSNormMxfp8(
        hidden_size=hidden_size,
        max_num_tokens=max_num_tokens,
        top_k=top_k,
        fp32_weights=fp32_weights,
        device=torch.device(
            current_platform.device_type, torch.accelerator.current_device_index()
        ),
    )


def finalize_allreduce_gemma_rms_norm_mxfp8(
    moe_output: MoEOutput,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float,
) -> tuple[QuantizedActivation, torch.Tensor]:
    """Finalize a deferred MoE output into the next layer's MXFP8 input and
    new residual."""
    assert _fused_moe_finalize is not None, "init_fused_moe_finalize was not called"
    return _fused_moe_finalize(moe_output, residual, norm_weight, norm_eps)
