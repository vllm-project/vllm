# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from FlashInfer's low-latency MNNVL CuTe DSL all-reduce
# (flashinfer/comm/mnnvl_cutedsl/kernel_ll, flashinfer-ai/flashinfer@139af6f6).

"""Lamport all-reduce fused with DSV4.1's mHC post, collapse and RMSNorm.

FlashInfer's LL kernels, copied: a publish kernel (or its MoE finalize
variant) multicasts each rank's contribution into a three-generation Lamport
mailbox, and a PDL-chained consumer spins until every rank's fragment lands,
reduces it and normalizes. The consumer's residual add and copy are replaced by
the mHC post-mix of the hc residual streams and their pre-mix collapse, which
the RMSNorm then normalizes. The finalize variant also reads FP32 routing
weights.
"""

from __future__ import annotations

import math

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from cutlass import BFloat16, Float32, Int32, Int64, Uint32
from cutlass.cute.runtime import make_fake_compact_tensor

from vllm.distributed import get_tp_group

from .primitives import (
    NEGATIVE_ZERO_BF16_BITS,
    QUAD_BF16,
    VEC_BF16,
    WARP_SIZE,
    bf16x4_to_packed_u32x2,
    bf16x8_to_packed_u32x4,
    current_cu_stream,
    f32_to_bf16_bits,
    fragment_has_negative_zero,
    load_global_bf16_as_f32,
    load_global_u32x2,
    load_global_u32x4,
    load_volatile_u32,
    make_fake_dynamic_compact_tensor,
    map_shared_to_peer,
    packed_u32x2_to_bf16x4,
    packed_u32x4_to_bf16x8,
    sanitize_negative_zero_u32x2,
    sanitize_negative_zero_u32x4,
    shuffle_sync_idx_u32,
    stmc_bf16x2,
    stmc_bf16x4,
    stmc_bf16x8,
    store_global_u32,
    store_global_u32x4,
    store_lamport_sentinel_u32x4,
    store_shared_cluster_f32,
    to_cute,
    to_cute_dynamic,
)

LAMPORT_GENERATIONS = 3
NEXT_STAGE = 0
ACTIVE_STAGE = 1


@cute.jit
def _group_leader_block_sum(
    value: Float32,
    warp_sums: cute.Tensor,
    warps: cutlass.Constexpr[int],
    leader_stride: cutlass.Constexpr[int],
) -> Float32:
    lane = cute.arch.lane_idx()
    warp = cute.arch.warp_idx()
    for offset in cutlass.range_constexpr(1, WARP_SIZE):
        if cutlass.const_expr(offset >= leader_stride and (offset & (offset - 1)) == 0):
            value = value + cute.arch.shuffle_sync_bfly(
                value,
                offset=offset,
                mask=-1,
                mask_and_clamp=31,
            )
    if lane == 0:
        cute.arch.store((warp_sums + warp).llvm_ptr, value)
    cute.arch.barrier()

    result = Float32(0.0)
    if warp == 0:
        if lane < Int32(warps):
            result = cute.arch.load(
                (warp_sums + lane).llvm_ptr,
                Float32,
            )
        result = cute.arch.warp_reduction_sum(result)
        if lane == 0:
            cute.arch.store(warp_sums.llvm_ptr, result)
    cute.arch.barrier()
    return cute.arch.load(warp_sums.llvm_ptr, Float32)


class _SharedOnlyPublishDeviceKernel:
    def __init__(
        self,
        *,
        hidden: int,
        tp: int,
        rank: int,
        capacity_m: int,
        elements_per_thread: int,
        threads: int,
        release_before_store: bool,
        enable_pdl: bool,
    ) -> None:
        if elements_per_thread not in (1, QUAD_BF16, VEC_BF16):
            raise ValueError("elements_per_thread must be 1, 4, or 8")
        if hidden <= 0 or hidden % elements_per_thread:
            raise ValueError("hidden must divide evenly across thread fragments")
        self.hidden = hidden
        self.tp = tp
        self.rank = rank
        self.capacity_m = capacity_m
        self.elements_per_thread = elements_per_thread
        self.threads = threads
        self.fragments = hidden // elements_per_thread
        self.ctas_per_token = math.ceil(self.fragments / threads)
        self.release_before_store = release_before_store
        self.enable_pdl = enable_pdl

    @cute.jit
    def __call__(
        self,
        shared_output: cute.Tensor,
        stage_state: cute.Tensor,
        contribution_mailbox_multicast_address: Int64,
        m: Int32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(
            shared_output,
            stage_state,
            contribution_mailbox_multicast_address,
        ).launch(
            grid=(m * self.ctas_per_token, 1, 1),
            block=(self.threads, 1, 1),
            stream=stream,
            use_pdl=self.enable_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        shared_output: cute.Tensor,
        stage_state: cute.Tensor,
        contribution_mailbox_multicast_address: Int64,
    ) -> None:
        block, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        token = block // self.ctas_per_token
        cta_in_token = block % self.ctas_per_token
        fragment = cta_in_token * self.threads + tidx

        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_wait()

        if cutlass.const_expr(self.elements_per_thread == 1):
            bits = Uint32(0)
            if fragment < self.fragments:
                element = Int64(token) * self.hidden + Int64(fragment)
                bits = f32_to_bf16_bits(
                    load_global_bf16_as_f32(
                        Int64((shared_output.iterator + element).toint())
                    )
                )
                if bits == Uint32(NEGATIVE_ZERO_BF16_BITS):
                    bits = Uint32(0)
            partner_bits = shuffle_sync_idx_u32(
                bits,
                Int32(cute.arch.lane_idx() | Int32(1)),
            )
            packed_word = bits | (partner_bits << Uint32(16))
        elif cutlass.const_expr(self.elements_per_thread == QUAD_BF16):
            packed_words = cute.make_rmem_tensor(cute.make_layout((2,)), Uint32)
            packed_words.fill(Uint32(0))
            if fragment < self.fragments:
                element = Int64(token) * self.hidden + Int64(fragment) * QUAD_BF16
                pointer = cute.make_ptr(
                    BFloat16,
                    (shared_output.iterator + element).llvm_ptr,
                    cute.AddressSpace.gmem,
                    assumed_align=8,
                )
                packed_words.store(
                    sanitize_negative_zero_u32x2(load_global_u32x2(pointer))
                )
        else:
            packed_words = cute.make_rmem_tensor(cute.make_layout((4,)), Uint32)
            packed_words.fill(Uint32(0))
            if fragment < self.fragments:
                element = Int64(token) * self.hidden + Int64(fragment) * VEC_BF16
                pointer = cute.make_ptr(
                    BFloat16,
                    (shared_output.iterator + element).llvm_ptr,
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                )
                packed_words.store(
                    sanitize_negative_zero_u32x4(load_global_u32x4(pointer))
                )

        stage = load_volatile_u32(stage_state.iterator + NEXT_STAGE)
        if cutlass.const_expr(self.release_before_store):
            if block == 0 and tidx == 0:
                store_global_u32(
                    stage_state.iterator + ACTIVE_STAGE,
                    stage,
                )
            cute.arch.barrier()
            if cutlass.const_expr(self.enable_pdl):
                cute.arch.griddepcontrol_launch_dependents()

        if cutlass.const_expr(self.elements_per_thread == 1):
            if (cute.arch.lane_idx() & Int32(1)) == Int32(
                0
            ) and fragment < self.fragments:
                mailbox_element = (
                    (Int64(stage) * self.tp + self.rank) * self.capacity_m
                    + Int64(token)
                ) * self.hidden + Int64(fragment)
                stmc_bf16x2(
                    contribution_mailbox_multicast_address + mailbox_element * 2,
                    packed_word,
                )
        elif cutlass.const_expr(self.elements_per_thread == QUAD_BF16):
            if fragment < self.fragments:
                mailbox_element = (
                    (Int64(stage) * self.tp + self.rank) * self.capacity_m
                    + Int64(token)
                ) * self.hidden + Int64(fragment) * QUAD_BF16
                stmc_bf16x4(
                    contribution_mailbox_multicast_address + mailbox_element * 2,
                    packed_words,
                )
        else:
            if fragment < self.fragments:
                mailbox_element = (
                    (Int64(stage) * self.tp + self.rank) * self.capacity_m
                    + Int64(token)
                ) * self.hidden + Int64(fragment) * VEC_BF16
                stmc_bf16x8(
                    contribution_mailbox_multicast_address + mailbox_element * 2,
                    packed_words,
                )

        if cutlass.const_expr(not self.release_before_store):
            if block == 0 and tidx == 0:
                store_global_u32(
                    stage_state.iterator + ACTIVE_STAGE,
                    stage,
                )
            cute.arch.barrier()
            if cutlass.const_expr(self.enable_pdl):
                cute.arch.griddepcontrol_launch_dependents()


class _QuadFinalizePublishDeviceKernel:
    def __init__(
        self,
        *,
        hidden: int,
        top_k: int,
        tp: int,
        rank: int,
        capacity_m: int,
        threads: int,
        routed_scaling_factor: float,
        include_shared_expert: bool,
        load_shared_expert_before_pdl: bool,
        enable_pdl: bool,
        prefetch_group: int,
        fp32_weights: bool = False,
    ) -> None:
        if hidden <= 0 or hidden % QUAD_BF16:
            raise ValueError("hidden must be a positive multiple of 4")
        self.hidden = hidden
        self.top_k = top_k
        self.tp = tp
        self.rank = rank
        self.capacity_m = capacity_m
        self.threads = threads
        self.routed_scaling_factor = routed_scaling_factor
        self.fragments = hidden // QUAD_BF16
        self.ctas_per_token = math.ceil(self.fragments / threads)
        self.include_shared_expert = include_shared_expert
        self.load_shared_expert_before_pdl = load_shared_expert_before_pdl
        self.enable_pdl = enable_pdl
        self.prefetch_group = prefetch_group
        self.prefetch_groups = (top_k + prefetch_group - 1) // prefetch_group
        self.fp32_weights = fp32_weights

    def smem_size_in_bytes(self) -> int:
        return self.top_k * 8

    @cute.jit
    def __call__(
        self,
        routed_output: cute.Tensor,
        expert_weights: cute.Tensor,
        permuted_indices: cute.Tensor,
        shared_output: cute.Tensor,
        stage_state: cute.Tensor,
        contribution_mailbox_multicast_address: Int64,
        m: Int32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(
            routed_output,
            expert_weights,
            permuted_indices,
            shared_output,
            stage_state,
            contribution_mailbox_multicast_address,
        ).launch(
            grid=(m * self.ctas_per_token, 1, 1),
            block=(self.threads, 1, 1),
            smem=self.smem_size_in_bytes(),
            stream=stream,
            use_pdl=self.enable_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        routed_output: cute.Tensor,
        expert_weights: cute.Tensor,
        permuted_indices: cute.Tensor,
        shared_output: cute.Tensor,
        stage_state: cute.Tensor,
        contribution_mailbox_multicast_address: Int64,
    ) -> None:
        block, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        token = block // self.ctas_per_token
        cta_in_token = block % self.ctas_per_token
        fragment = cta_in_token * self.threads + tidx

        smem = cutlass.utils.SmemAllocator()
        staged_indices = smem.allocate_array(Int32, self.top_k)
        staged_weights = smem.allocate_array(Float32, self.top_k)
        metadata_index = Int32(tidx)
        while metadata_index < Int32(self.top_k):
            element = Int64(token) * self.top_k + Int64(metadata_index)
            row = cute.arch.load(
                (permuted_indices.iterator + element).llvm_ptr,
                Int32,
            )
            if cutlass.const_expr(self.fp32_weights):
                weight = cute.arch.load(
                    (expert_weights.iterator + element).llvm_ptr,
                    Float32,
                )
            else:
                weight = load_global_bf16_as_f32(
                    Int64((expert_weights.iterator + element).toint())
                )
            if cutlass.const_expr(self.routed_scaling_factor != 1.0):
                weight = weight * Float32(self.routed_scaling_factor)
            if row == Int32(-1):
                weight = Float32(0.0)
            cute.arch.store(
                (staged_indices + metadata_index).llvm_ptr,
                row,
            )
            cute.arch.store(
                (staged_weights + metadata_index).llvm_ptr,
                weight,
            )
            metadata_index = metadata_index + self.threads
        cute.arch.barrier()

        if cutlass.const_expr(self.include_shared_expert):
            shared_values = cute.make_rmem_tensor(
                cute.make_layout((QUAD_BF16,)), BFloat16
            )
            shared_values.fill(BFloat16(0.0))
            if cutlass.const_expr(self.load_shared_expert_before_pdl):  # noqa: SIM102
                if fragment < self.fragments:
                    shared_element = (
                        Int64(token) * self.hidden + Int64(fragment) * QUAD_BF16
                    )
                    shared_pointer = cute.make_ptr(
                        BFloat16,
                        (shared_output.iterator + shared_element).llvm_ptr,
                        cute.AddressSpace.gmem,
                        assumed_align=8,
                    )
                    shared_values.store(
                        packed_u32x2_to_bf16x4(load_global_u32x2(shared_pointer))
                    )

        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_wait()

        if cutlass.const_expr(  # noqa: SIM102
            self.include_shared_expert and not self.load_shared_expert_before_pdl
        ):
            if fragment < self.fragments:
                shared_element = (
                    Int64(token) * self.hidden + Int64(fragment) * QUAD_BF16
                )
                shared_pointer = cute.make_ptr(
                    BFloat16,
                    (shared_output.iterator + shared_element).llvm_ptr,
                    cute.AddressSpace.gmem,
                    assumed_align=8,
                )
                shared_values.store(
                    packed_u32x2_to_bf16x4(load_global_u32x2(shared_pointer))
                )

        stage = load_volatile_u32(stage_state.iterator + NEXT_STAGE)
        if fragment < self.fragments:
            accumulator = cute.make_rmem_tensor(cute.make_layout((QUAD_BF16,)), Float32)
            accumulator.fill(Float32(0.0))
            if cutlass.const_expr(self.prefetch_group == 1):
                for k in cutlass.range_constexpr(self.top_k):
                    row = cute.arch.load((staged_indices + k).llvm_ptr, Int32)
                    weight = cute.arch.load(
                        (staged_weights + k).llvm_ptr,
                        Float32,
                    )
                    if row != Int32(-1):
                        source_element = (
                            Int64(row) * self.hidden + Int64(fragment) * QUAD_BF16
                        )
                        source_pointer = cute.make_ptr(
                            BFloat16,
                            (routed_output.iterator + source_element).llvm_ptr,
                            cute.AddressSpace.gmem,
                            assumed_align=8,
                        )
                        accumulator.store(
                            accumulator.load()
                            + packed_u32x2_to_bf16x4(
                                load_global_u32x2(source_pointer)
                            ).to(Float32)
                            * weight
                        )
            else:
                inputs = cute.make_rmem_tensor(
                    cute.make_layout((self.prefetch_group, 2)),
                    Uint32,
                )
                inputs.fill(Uint32(0))
                for group in cutlass.range_constexpr(self.prefetch_groups):
                    for item in cutlass.range_constexpr(self.prefetch_group):
                        k = group * self.prefetch_group + item
                        if cutlass.const_expr(k < self.top_k):
                            row = cute.arch.load(
                                (staged_indices + k).llvm_ptr,
                                Int32,
                            )
                            for word in cutlass.range_constexpr(2):
                                inputs[item, word] = Uint32(0)
                            if row != Int32(-1):
                                source_element = (
                                    Int64(row) * self.hidden
                                    + Int64(fragment) * QUAD_BF16
                                )
                                source_pointer = cute.make_ptr(
                                    BFloat16,
                                    (routed_output.iterator + source_element).llvm_ptr,
                                    cute.AddressSpace.gmem,
                                    assumed_align=8,
                                )
                                source = load_global_u32x2(source_pointer)
                                for word in cutlass.range_constexpr(2):
                                    inputs[item, word] = source[word]
                    for item in cutlass.range_constexpr(self.prefetch_group):
                        k = group * self.prefetch_group + item
                        if cutlass.const_expr(k < self.top_k):
                            source = cute.make_rmem_tensor(
                                cute.make_layout((2,)),
                                Uint32,
                            )
                            for word in cutlass.range_constexpr(2):
                                source[word] = inputs[item, word]
                            accumulator.store(
                                accumulator.load()
                                + packed_u32x2_to_bf16x4(source.load()).to(Float32)
                                * cute.arch.load(
                                    (staged_weights + k).llvm_ptr,
                                    Float32,
                                )
                            )
            result = accumulator.load()
            if cutlass.const_expr(self.include_shared_expert):
                result = result + shared_values.load().to(Float32)
            result_packed = bf16x4_to_packed_u32x2(result.to(BFloat16))
            packed = sanitize_negative_zero_u32x2(result_packed)
            mailbox_element = (
                (Int64(stage) * self.tp + self.rank) * self.capacity_m + Int64(token)
            ) * self.hidden + Int64(fragment) * QUAD_BF16
            stmc_bf16x4(
                contribution_mailbox_multicast_address + mailbox_element * 2,
                packed,
            )

        if block == 0 and tidx == 0:
            store_global_u32(
                stage_state.iterator + ACTIVE_STAGE,
                stage,
            )
        cute.arch.barrier()
        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_launch_dependents()


class _LamportMHCDeviceKernel:
    """FlashInfer's ``_LamportResidualRMSNormDeviceKernel`` with mHC.

    The residual add and residual copy become the mHC post-mix of the hc
    streams, stored as the residual output, and their pre-mix collapse, which
    the unchanged RMSNorm normalizes.
    """

    def __init__(
        self,
        *,
        hidden: int,
        hc: int,
        tp: int,
        capacity_m: int,
        cluster_size: int,
        rank_lanes: int,
        threads: int,
        enable_pdl: bool,
    ) -> None:
        if rank_lanes not in (1, 2, 4, 8):
            raise ValueError("rank_lanes must be 1, 2, 4, or 8")
        if tp % rank_lanes:
            raise ValueError("tp must be divisible by rank_lanes")
        if threads <= 0 or threads % WARP_SIZE or threads % rank_lanes:
            raise ValueError("threads must be a positive warp and rank-lane multiple")
        if hidden <= 0 or hidden % VEC_BF16:
            raise ValueError("hidden must be a positive multiple of 8")
        self.hidden = hidden
        self.hc = hc
        self.tp = tp
        self.capacity_m = capacity_m
        self.cluster_size = cluster_size
        self.rank_lanes = rank_lanes
        self.threads = threads
        self.enable_pdl = enable_pdl
        self.fragments = hidden // VEC_BF16
        self.groups_per_cta = threads // rank_lanes
        self.fragment_stride = cluster_size * self.groups_per_cta
        self.trips = math.ceil(self.fragments / self.fragment_stride)
        self.warps = threads // WARP_SIZE
        self.rank_waves = tp // rank_lanes

    def smem_size_in_bytes(self) -> int:
        return (self.warps + self.cluster_size) * 4

    @cute.jit
    def __call__(
        self,
        contribution_mailbox: cute.Tensor,
        residual_source: cute.Tensor,
        post: cute.Tensor,
        comb: cute.Tensor,
        pre: cute.Tensor,
        gamma: cute.Tensor,
        residual_output: cute.Tensor,
        norm_output: cute.Tensor,
        stage_state: cute.Tensor,
        eps: Float32,
        m: Int32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(
            contribution_mailbox,
            residual_source,
            post,
            comb,
            pre,
            gamma,
            residual_output,
            norm_output,
            stage_state,
            eps,
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
        post: cute.Tensor,
        comb: cute.Tensor,
        pre: cute.Tensor,
        gamma: cute.Tensor,
        residual_output: cute.Tensor,
        norm_output: cute.Tensor,
        stage_state: cute.Tensor,
        eps: Float32,
    ) -> None:
        hc = self.hc
        tidx, _, _ = cute.arch.thread_idx()
        token, _, _ = cute.arch.block_idx()
        cluster_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        rank_lane = tidx % self.rank_lanes
        group = tidx // self.rank_lanes
        base_fragment = cluster_rank * self.groups_per_cta + group

        prenorm_fragments = cute.make_rmem_tensor(
            cute.make_layout(
                (self.trips, VEC_BF16),
                stride=(VEC_BF16, 1),
            ),
            BFloat16,
        )
        prenorm_fragments.fill(BFloat16(0.0))
        gamma_fragments = cute.make_rmem_tensor(
            cute.make_layout(
                (self.trips, VEC_BF16),
                stride=(VEC_BF16, 1),
            ),
            BFloat16,
        )
        gamma_fragments.fill(BFloat16(0.0))
        residual_fragments = cute.make_rmem_tensor(
            cute.make_layout(
                (self.trips, hc, VEC_BF16),
                stride=(hc * VEC_BF16, VEC_BF16, 1),
            ),
            BFloat16,
        )
        residual_fragments.fill(BFloat16(0.0))

        for trip in cutlass.range_constexpr(self.trips):
            fragment = base_fragment + trip * self.fragment_stride
            if fragment < self.fragments and rank_lane == 0:
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
                for source in cutlass.range_constexpr(hc):
                    residual_element = (
                        Int64(token) * hc + source
                    ) * self.hidden + gamma_element
                    residual_pointer = cute.make_ptr(
                        BFloat16,
                        (residual_source.iterator + residual_element).llvm_ptr,
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    )
                    residual_fragments[trip, source, None].store(
                        packed_u32x4_to_bf16x8(load_global_u32x4(residual_pointer))
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
                comb_index = (Int64(token) * hc + source) * hc + target
                comb_mix[source, target] = cute.arch.load(
                    (comb.iterator + comb_index).llvm_ptr, Float32
                )

        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_wait()

        active_stage = load_volatile_u32(stage_state.iterator + ACTIVE_STAGE)
        for trip in cutlass.range_constexpr(self.trips):
            fragment = base_fragment + trip * self.fragment_stride
            lane_packed = cute.make_rmem_tensor(
                cute.make_layout((self.rank_waves, 4)),
                Uint32,
            )
            lane_packed.fill(Uint32(0))
            dirty = fragment < self.fragments
            while dirty:
                dirty = False
                for wave in cutlass.range_constexpr(self.rank_waves):
                    source_rank = wave * self.rank_lanes + rank_lane
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
                        packed = load_global_u32x4(
                            source_pointer,
                            volatile=True,
                        )
                        dirty = dirty | fragment_has_negative_zero(packed)
                        for word in cutlass.range_constexpr(4):
                            lane_packed[wave, word] = packed[word]

            lane_sum = cute.make_rmem_tensor(cute.make_layout((VEC_BF16,)), Float32)
            lane_sum.fill(Float32(0.0))
            for wave in cutlass.range_constexpr(self.rank_waves):
                packed = cute.make_rmem_tensor(cute.make_layout((4,)), Uint32)
                for word in cutlass.range_constexpr(4):
                    packed[word] = lane_packed[wave, word]
                lane_sum.store(
                    lane_sum.load() + packed_u32x4_to_bf16x8(packed.load()).to(Float32)
                )
            for offset in cutlass.range_constexpr(1, 5):
                if cutlass.const_expr(offset < self.rank_lanes and offset in (1, 2, 4)):
                    for element in cutlass.range_constexpr(VEC_BF16):
                        lane_sum[element] = lane_sum[
                            element
                        ] + cute.arch.shuffle_sync_bfly(
                            lane_sum[element],
                            offset=offset,
                            mask=-1,
                            mask_and_clamp=31,
                        )

            if fragment < self.fragments and rank_lane == 0:
                # The all-reduced sublayer output is BF16, as the unfused
                # collective returns it, before mHC post mixes it in.
                reduced = lane_sum.load().to(BFloat16).to(Float32)
                collapse = cute.make_rmem_tensor(cute.make_layout((VEC_BF16,)), Float32)
                collapse.fill(Float32(0.0))
                for target in cutlass.range_constexpr(hc):
                    mixed = reduced * post_mix[target]
                    for source in cutlass.range_constexpr(hc):
                        mixed = (
                            mixed
                            + residual_fragments[trip, source, None].load().to(Float32)
                            * comb_mix[source, target]
                        )
                    mixed_bf16 = mixed.to(BFloat16)
                    output_element = (Int64(token) * hc + target) * self.hidden + Int64(
                        fragment
                    ) * VEC_BF16
                    store_global_u32x4(
                        Int64((residual_output.iterator + output_element).toint()),
                        bf16x8_to_packed_u32x4(mixed_bf16),
                    )
                    collapse.store(
                        collapse.load() + mixed_bf16.to(Float32) * pre_mix[target]
                    )
                prenorm_fragments[trip, None].store(collapse.load().to(BFloat16))

        if token == 0 and cluster_rank == 0 and tidx == 0:
            store_global_u32(
                stage_state.iterator + NEXT_STAGE,
                (active_stage + Uint32(1)) % Uint32(LAMPORT_GENERATIONS),
            )
        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_launch_dependents()

        for trip in cutlass.range_constexpr(self.trips):
            fragment = base_fragment + trip * self.fragment_stride
            for wave in cutlass.range_constexpr(self.rank_waves):
                source_rank = wave * self.rank_lanes + rank_lane
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
            if fragment < self.fragments and rank_lane == 0:
                values = prenorm_fragments[trip, None].load().to(Float32)
                thread_sum = thread_sum + (values * values).reduce(
                    cute.ReductionOp.ADD,
                    init_val=Float32(0.0),
                    reduction_profile=0,
                )

        smem = cutlass.utils.SmemAllocator()
        warp_sums = smem.allocate_array(Float32, self.warps)
        cluster_sums = smem.allocate_array(Float32, self.cluster_size)
        cta_sum = _group_leader_block_sum(
            thread_sum,
            warp_sums,
            self.warps,
            self.rank_lanes,
        )
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
        for trip in cutlass.range_constexpr(self.trips):
            fragment = base_fragment + trip * self.fragment_stride
            if fragment < self.fragments and rank_lane == 0:
                gamma_values = gamma_fragments[trip, None].load().to(Float32)
                result = (
                    prenorm_fragments[trip, None].load().to(Float32)
                    * inv_rms
                    * gamma_values
                ).to(BFloat16)
                output_element = Int64(token) * self.hidden + Int64(fragment) * VEC_BF16
                store_global_u32x4(
                    Int64((norm_output.iterator + output_element).toint()),
                    bf16x8_to_packed_u32x4(result),
                )


class AllReduceMHC:
    """Lamport TP all-reduce fused with mHC post, collapse and RMSNorm.

    ``__call__`` reduces ``x`` across the TP group, writes the post-mixed hc
    streams and the normalized collapse, and leaves the next mixes to the
    caller. ``finalize`` does the same for an unfinalized MoE output, folding
    the top-k reduction and the shared-expert add into the publish. Each
    instance owns a Lamport mailbox; construction is collective over the TP
    group.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        hc_mult: int,
        max_num_tokens: int,
        top_k: int,
        device: torch.device,
    ) -> None:
        group = get_tp_group().device_group
        self.hidden_size = hidden = hidden_size
        self.hc_mult = hc = hc_mult
        self.capacity = capacity = max_num_tokens
        self.top_k = top_k
        tp = dist.get_world_size(group)
        rank = dist.get_rank(group)
        device = torch.device(device)

        # FlashInfer's LL_ALL_REDUCE_GB300_TP{4,8}_H5120 and
        # LL_FINALIZE_..._H5120_K* presets: a cluster of 5 x 128 threads covers
        # a 640-fragment token in one fully used trip, and the finalize stages
        # all top_k rows at once.
        with torch.accelerator.device_index(device.index):
            self._publish = cute.compile(
                _SharedOnlyPublishDeviceKernel(
                    hidden=hidden,
                    tp=tp,
                    rank=rank,
                    capacity_m=capacity,
                    elements_per_thread=VEC_BF16,
                    threads=128,
                    release_before_store=False,
                    enable_pdl=True,
                ),
                make_fake_dynamic_compact_tensor(
                    BFloat16, alignment=16, divisibility=hidden
                ),
                make_fake_compact_tensor(Int32, (2,), assumed_align=4),
                Int64(0),
                Int32(capacity),
                current_cu_stream(),
            )
            self._finalize_publish = cute.compile(
                _QuadFinalizePublishDeviceKernel(
                    hidden=hidden,
                    top_k=top_k,
                    tp=tp,
                    rank=rank,
                    capacity_m=capacity,
                    threads=128,
                    # DSV4.1's router folds the scale into the weights.
                    routed_scaling_factor=1.0,
                    include_shared_expert=True,
                    load_shared_expert_before_pdl=False,
                    enable_pdl=True,
                    prefetch_group=top_k,
                    fp32_weights=True,
                ),
                make_fake_dynamic_compact_tensor(
                    BFloat16, alignment=16, divisibility=hidden
                ),
                make_fake_dynamic_compact_tensor(
                    Float32, alignment=4, divisibility=top_k
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
                _LamportMHCDeviceKernel(
                    hidden=hidden,
                    hc=hc,
                    tp=tp,
                    capacity_m=capacity,
                    cluster_size=5,
                    rank_lanes=1,
                    threads=128,
                    enable_pdl=True,
                ),
                make_fake_compact_tensor(
                    BFloat16,
                    (LAMPORT_GENERATIONS * tp * capacity * hidden,),
                    assumed_align=16,
                ),
                make_fake_dynamic_compact_tensor(
                    BFloat16, alignment=16, divisibility=hc * hidden
                ),
                make_fake_dynamic_compact_tensor(Float32, alignment=4, divisibility=hc),
                make_fake_dynamic_compact_tensor(
                    Float32, alignment=4, divisibility=hc * hc
                ),
                make_fake_dynamic_compact_tensor(Float32, alignment=4, divisibility=hc),
                make_fake_compact_tensor(BFloat16, (hidden,), assumed_align=16),
                make_fake_dynamic_compact_tensor(
                    BFloat16, alignment=16, divisibility=hc * hidden
                ),
                make_fake_dynamic_compact_tensor(
                    BFloat16, alignment=16, divisibility=hidden
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
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
        pre: torch.Tensor,
        norm_weight: torch.Tensor,
        norm_eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (post-mixed residual streams, normalized layer input).

        Args:
            x: [M, hidden] BF16 TP-partial sublayer output.
            residual: [M, hc, hidden] BF16 residual streams.
            post: [M, hc(, 1)] FP32 post-mix.
            comb: [M, hc, hc] FP32 residual mix, indexed [source, target].
            pre: [M, hc] FP32 carried pre-mix for the collapse.
            norm_weight: [hidden] BF16 RMSNorm weight.
            norm_eps: RMSNorm epsilon.

        """
        m = x.shape[0]
        self._check(m, residual, post, comb, pre)
        if x.shape != (m, self.hidden_size):
            raise ValueError("unexpected x shape")
        self._publish(
            to_cute_dynamic(x.flatten(), 16, divisibility=self.hidden_size),
            to_cute(self._stage_state, 4),
            Int64(self._multicast),
            Int32(m),
            current_cu_stream(),
        )
        return self._collect(residual, post, comb, pre, norm_weight, norm_eps, m)

    def finalize(
        self,
        gemm2_permuted: torch.Tensor,
        expert_weights: torch.Tensor,
        expanded_idx_to_permuted_idx: torch.Tensor,
        shared_output: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
        pre: torch.Tensor,
        norm_weight: torch.Tensor,
        norm_eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``__call__`` for a MoE output left unfinalized.

        Args:
            gemm2_permuted: [rows, hidden] BF16 unweighted GEMM2 output in the
                MoE's permuted order.
            expert_weights: [M, top_k] FP32 routing weights, already scaled.
            expanded_idx_to_permuted_idx: [M, top_k] int32 row of each route,
                -1 for an expert this rank does not hold.
            shared_output: [M, hidden] BF16 TP-partial shared-expert output.
            residual: [M, hc, hidden] BF16 residual streams.
            post: [M, hc(, 1)] FP32 post-mix.
            comb: [M, hc, hc] FP32 residual mix, indexed [source, target].
            pre: [M, hc] FP32 carried pre-mix for the collapse.
            norm_weight: [hidden] BF16 RMSNorm weight.
            norm_eps: RMSNorm epsilon.

        """
        m = shared_output.shape[0]
        hidden, top_k = self.hidden_size, self.top_k
        self._check(m, residual, post, comb, pre)
        if gemm2_permuted.dim() != 2 or gemm2_permuted.shape[1] != hidden:
            raise ValueError("gemm2_permuted must be [rows, hidden]")
        if shared_output.shape != (m, hidden):
            raise ValueError("unexpected shared_output shape")
        if expert_weights.dtype != torch.float32 or expert_weights.numel() != m * top_k:
            raise ValueError("expert_weights must be FP32 [M, top_k]")
        if (
            expanded_idx_to_permuted_idx.dtype != torch.int32
            or expanded_idx_to_permuted_idx.numel() != m * top_k
        ):
            raise ValueError("expanded_idx_to_permuted_idx must be int32 [M, top_k]")
        self._finalize_publish(
            to_cute_dynamic(gemm2_permuted.flatten(), 16, divisibility=hidden),
            to_cute_dynamic(expert_weights.flatten(), 4, divisibility=top_k),
            to_cute_dynamic(
                expanded_idx_to_permuted_idx.flatten(), 4, divisibility=top_k
            ),
            to_cute_dynamic(shared_output.flatten(), 16, divisibility=hidden),
            to_cute(self._stage_state, 4),
            Int64(self._multicast),
            Int32(m),
            current_cu_stream(),
        )
        return self._collect(residual, post, comb, pre, norm_weight, norm_eps, m)

    def _check(
        self,
        m: int,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
        pre: torch.Tensor,
    ) -> None:
        hc = self.hc_mult
        if not 1 <= m <= self.capacity:
            raise ValueError(f"M={m} is outside [1, {self.capacity}]")
        if residual.shape != (m, hc, self.hidden_size):
            raise ValueError("unexpected residual shape")
        if (
            post.numel() != m * hc
            or pre.numel() != m * hc
            or comb.numel() != m * hc * hc
        ):
            raise ValueError("unexpected mix shapes")

    def _collect(
        self,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
        pre: torch.Tensor,
        norm_weight: torch.Tensor,
        norm_eps: float,
        m: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden, hc = self.hidden_size, self.hc_mult
        residual_output = torch.empty_like(residual)
        layer_input = torch.empty(
            (m, hidden), dtype=torch.bfloat16, device=residual.device
        )
        self._collective(
            to_cute(self._mailbox.flatten(), 16),
            to_cute_dynamic(residual.flatten(), 16, divisibility=hc * hidden),
            to_cute_dynamic(post.flatten(), 4, divisibility=hc),
            to_cute_dynamic(comb.flatten(), 4, divisibility=hc * hc),
            to_cute_dynamic(pre.flatten(), 4, divisibility=hc),
            to_cute(norm_weight, 16),
            to_cute_dynamic(residual_output.flatten(), 16, divisibility=hc * hidden),
            to_cute_dynamic(layer_input.flatten(), 16, divisibility=hidden),
            to_cute(self._stage_state, 4),
            Float32(norm_eps),
            Int32(m),
            current_cu_stream(),
        )
        return residual_output, layer_input
