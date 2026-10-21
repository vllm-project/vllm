# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Adapted from FlashInfer's low-latency and balanced MNNVL CuTe DSL all-reduces
# (flashinfer/comm/mnnvl_cutedsl/kernel_ll and kernel_bt,
# flashinfer-ai/flashinfer@139af6f6), which carry the notice below. Modified
# by the vLLM contributors: the consumer applies DeepSeek-V4.1's mHC post-mix
# and collapse before its RMSNorm, and the kernels are fitted to its MoE
# finalize and decode shapes.
#
# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# FlashInfer's NOTICE: FlashInfer / Copyright 2025-2026 NVIDIA /
# Copyright 2023-2026 FlashInfer community (https://flashinfer.ai/)

"""Lamport all-reduce fused with DSV4.1's mHC post, collapse and RMSNorm.

FlashInfer's LL kernels, copied: a publish kernel (or its MoE finalize
variant) multicasts each rank's contribution into a three-generation Lamport
mailbox, and a PDL-chained consumer spins until every rank's fragment lands,
reduces it and normalizes. The consumer's residual add and copy are replaced by
the mHC post-mix of the hc residual streams and their pre-mix collapse, which
the RMSNorm then normalizes. The finalize variant also reads FP32 routing
weights.

Every rank receives every rank's whole contribution in LL, so its NVLink
traffic grows with ``tp`` times the rows. From ``BT_MIN_ROWS`` rows on
(``BT_MIN_FINALIZE_ROWS`` for the MoE finalize), the
two-shot protocol of FlashInfer's BT kernels takes over: the publish kernels
store each token's contribution only into its owner rank's mailbox (token
``t`` belongs to rank ``t % tp``), an owner kernel reduces its tokens and
multicasts the BF16 sums into a second mailbox, and the same mHC consumer reads
that one reduced value instead of ``tp`` contributions::

    LL:  publish ──multicast──► mailbox[gen][rank][token] ─► consumer (sum, mHC)
    BT:  publish ──unicast to owner──► mailbox[gen][rank][token // tp]
           ─► owner reduce ──multicast──► sums[gen][token] ─► consumer (mHC)

The owner sums the ranks in the same order and precision as the LL consumer
(FP32 from rank 0, rounded to BF16), so both protocols give the same bits.

Optionally the consumer also writes the normalized output as MXFP8 with UE8M0
scales in the swizzled 128x4 layout FlashInfer's MXFP8 GEMMs read. It quantizes
the BF16-rounded output with FlashInfer's own quantize helpers, so the result is
bit-identical to ``mxfp8_quantize`` of the BF16 output, without its launch.
"""

from __future__ import annotations

import math
from typing import Any, NamedTuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from cutlass import BFloat16, Float32, Int32, Int64, Uint8, Uint32
from cutlass.cute.runtime import make_fake_compact_tensor

from vllm.distributed import get_tp_group
from vllm.models.deepseek_v41.common.ops.query_quant import mxfp8_scale_bytes

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
    load_global_u64,
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
    store_global_u32x2,
    store_global_u32x4,
    store_lamport_sentinel_u32x4,
    store_shared_cluster_f32,
    to_cute,
    to_cute_dynamic,
)

LAMPORT_GENERATIONS = 3
NEXT_STAGE = 0
ACTIVE_STAGE = 1
# The fewest rows reduced with the two-shot (BT) protocol, for the plain
# all-reduce and for the MoE finalize (whose publish also folds the top-k
# rows and the shared expert); fewer use LL. BT reads each contribution once,
# at its owner, but waits twice (owner, then consumer), which costs more where
# ranks arrive skewed, as in a decode step. An instance whose capacity is
# below BT_MIN_ROWS builds no BT kernels or buffers.
BT_MIN_ROWS = 64
BT_MIN_FINALIZE_ROWS = 64
# One UE8M0 scale per 32 values: four adjacent threads' BF16 fragments.
MXFP8_BLOCK = 32
FRAGMENTS_PER_MXFP8_BLOCK = MXFP8_BLOCK // VEC_BF16
# The swizzled scale layout tiles 128 rows by 4 scale columns.
SCALE_TILE_ROWS = 128
SCALE_TILE_COLS = 4


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
    """Publishes a rank's contribution. LL (``owner_capacity`` None): the
    Int64 argument is the mailbox's multicast address. BT: it is the address
    of an int64 table of every rank's unicast mailbox address, and each token
    goes only to its owner's mailbox, ``owner_capacity`` tokens a rank."""

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
        owner_capacity: int | None = None,
    ) -> None:
        if elements_per_thread not in (1, QUAD_BF16, VEC_BF16):
            raise ValueError("elements_per_thread must be 1, 4, or 8")
        if owner_capacity is not None and elements_per_thread != VEC_BF16:
            raise ValueError("the owner (BT) publish stores 8-element fragments")
        self.owner_capacity = owner_capacity
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
        elif cutlass.const_expr(self.owner_capacity is None):
            if fragment < self.fragments:
                mailbox_element = (
                    (Int64(stage) * self.tp + self.rank) * self.capacity_m
                    + Int64(token)
                ) * self.hidden + Int64(fragment) * VEC_BF16
                stmc_bf16x8(
                    contribution_mailbox_multicast_address + mailbox_element * 2,
                    packed_words,
                )
        else:
            if fragment < self.fragments:
                mailbox_element = (
                    (Int64(stage) * self.tp + self.rank) * self.owner_capacity
                    + Int64(token // self.tp)
                ) * self.hidden + Int64(fragment) * VEC_BF16
                owner_mailbox = load_global_u64(
                    contribution_mailbox_multicast_address + Int64(token % self.tp) * 8
                )
                store_global_u32x4(owner_mailbox + mailbox_element * 2, packed_words)

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
    """The MoE finalize (top-k weighted sum plus the shared expert) fused with
    the publish, ``elements_per_thread`` (4 or 8) values a thread."""

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
        owner_capacity: int | None = None,
        elements_per_thread: int = QUAD_BF16,
    ) -> None:
        if elements_per_thread not in (QUAD_BF16, VEC_BF16):
            raise ValueError("elements_per_thread must be 4 or 8")
        if hidden <= 0 or hidden % elements_per_thread:
            raise ValueError("hidden must be a multiple of elements_per_thread")
        # Fragment width: 8- or 16-byte loads and stores.
        self.elements = elements_per_thread
        self.words = elements_per_thread // 2
        quad = elements_per_thread == QUAD_BF16
        self.load_words = load_global_u32x2 if quad else load_global_u32x4
        self.unpack = packed_u32x2_to_bf16x4 if quad else packed_u32x4_to_bf16x8
        self.pack = bf16x4_to_packed_u32x2 if quad else bf16x8_to_packed_u32x4
        self.sanitize = (
            sanitize_negative_zero_u32x2 if quad else sanitize_negative_zero_u32x4
        )
        self.store_multicast = stmc_bf16x4 if quad else stmc_bf16x8
        self.store_unicast = store_global_u32x2 if quad else store_global_u32x4
        # As in _SharedOnlyPublishDeviceKernel: None for LL, else BT.
        self.owner_capacity = owner_capacity
        self.hidden = hidden
        self.top_k = top_k
        self.tp = tp
        self.rank = rank
        self.capacity_m = capacity_m
        self.threads = threads
        self.routed_scaling_factor = routed_scaling_factor
        self.fragments = hidden // elements_per_thread
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
                cute.make_layout((self.elements,)), BFloat16
            )
            shared_values.fill(BFloat16(0.0))
            if cutlass.const_expr(self.load_shared_expert_before_pdl):  # noqa: SIM102
                if fragment < self.fragments:
                    shared_element = (
                        Int64(token) * self.hidden + Int64(fragment) * self.elements
                    )
                    shared_pointer = cute.make_ptr(
                        BFloat16,
                        (shared_output.iterator + shared_element).llvm_ptr,
                        cute.AddressSpace.gmem,
                        assumed_align=2 * self.elements,
                    )
                    shared_values.store(self.unpack(self.load_words(shared_pointer)))

        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_wait()

        if cutlass.const_expr(  # noqa: SIM102
            self.include_shared_expert and not self.load_shared_expert_before_pdl
        ):
            if fragment < self.fragments:
                shared_element = (
                    Int64(token) * self.hidden + Int64(fragment) * self.elements
                )
                shared_pointer = cute.make_ptr(
                    BFloat16,
                    (shared_output.iterator + shared_element).llvm_ptr,
                    cute.AddressSpace.gmem,
                    assumed_align=2 * self.elements,
                )
                shared_values.store(self.unpack(self.load_words(shared_pointer)))

        stage = load_volatile_u32(stage_state.iterator + NEXT_STAGE)
        if fragment < self.fragments:
            accumulator = cute.make_rmem_tensor(
                cute.make_layout((self.elements,)), Float32
            )
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
                            Int64(row) * self.hidden + Int64(fragment) * self.elements
                        )
                        source_pointer = cute.make_ptr(
                            BFloat16,
                            (routed_output.iterator + source_element).llvm_ptr,
                            cute.AddressSpace.gmem,
                            assumed_align=2 * self.elements,
                        )
                        accumulator.store(
                            accumulator.load()
                            + self.unpack(self.load_words(source_pointer)).to(Float32)
                            * weight
                        )
            else:
                inputs = cute.make_rmem_tensor(
                    cute.make_layout((self.prefetch_group, self.words)),
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
                            for word in cutlass.range_constexpr(self.words):
                                inputs[item, word] = Uint32(0)
                            if row != Int32(-1):
                                source_element = (
                                    Int64(row) * self.hidden
                                    + Int64(fragment) * self.elements
                                )
                                source_pointer = cute.make_ptr(
                                    BFloat16,
                                    (routed_output.iterator + source_element).llvm_ptr,
                                    cute.AddressSpace.gmem,
                                    assumed_align=2 * self.elements,
                                )
                                source = self.load_words(source_pointer)
                                for word in cutlass.range_constexpr(self.words):
                                    inputs[item, word] = source[word]
                    for item in cutlass.range_constexpr(self.prefetch_group):
                        k = group * self.prefetch_group + item
                        if cutlass.const_expr(k < self.top_k):
                            source = cute.make_rmem_tensor(
                                cute.make_layout((self.words,)),
                                Uint32,
                            )
                            for word in cutlass.range_constexpr(self.words):
                                source[word] = inputs[item, word]
                            accumulator.store(
                                accumulator.load()
                                + self.unpack(source.load()).to(Float32)
                                * cute.arch.load(
                                    (staged_weights + k).llvm_ptr,
                                    Float32,
                                )
                            )
            result = accumulator.load()
            if cutlass.const_expr(self.include_shared_expert):
                result = result + shared_values.load().to(Float32)
            result_packed = self.pack(result.to(BFloat16))
            packed = self.sanitize(result_packed)
            if cutlass.const_expr(self.owner_capacity is None):
                mailbox_element = (
                    (Int64(stage) * self.tp + self.rank) * self.capacity_m
                    + Int64(token)
                ) * self.hidden + Int64(fragment) * self.elements
                self.store_multicast(
                    contribution_mailbox_multicast_address + mailbox_element * 2,
                    packed,
                )
            else:
                mailbox_element = (
                    (Int64(stage) * self.tp + self.rank) * self.owner_capacity
                    + Int64(token // self.tp)
                ) * self.hidden + Int64(fragment) * self.elements
                owner_mailbox = load_global_u64(
                    contribution_mailbox_multicast_address + Int64(token % self.tp) * 8
                )
                self.store_unicast(owner_mailbox + mailbox_element * 2, packed)

        if block == 0 and tidx == 0:
            store_global_u32(
                stage_state.iterator + ACTIVE_STAGE,
                stage,
            )
        cute.arch.barrier()
        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_launch_dependents()


class _OwnerReduceMulticastDeviceKernel:
    """FlashInfer's BT ``_OwnerReduceMulticastDeviceKernel`` without its
    residual add: each rank waits for every rank's contribution to the tokens
    it owns, sums them in FP32 in rank order, and multicasts the BF16 sums into
    every rank's reduced mailbox ``[generation][token]``. It then advances the
    generation and resets its contribution slots."""

    def __init__(
        self,
        *,
        hidden: int,
        tp: int,
        rank: int,
        capacity_m: int,
        owner_capacity: int,
        threads: int,
        enable_pdl: bool,
    ) -> None:
        self.hidden = hidden
        self.tp = tp
        self.rank = rank
        self.capacity_m = capacity_m
        self.owner_capacity = owner_capacity
        self.threads = threads
        self.enable_pdl = enable_pdl
        self.fragments = hidden // VEC_BF16
        self.ctas_per_token = math.ceil(self.fragments / threads)

    @cute.jit
    def __call__(
        self,
        contribution_mailbox: cute.Tensor,
        stage_state: cute.Tensor,
        reduced_mailbox_multicast_address: Int64,
        m: Int32,
        stream: cuda.CUstream,
    ) -> None:
        owned_tokens = (m + Int32(self.tp - 1)) // Int32(self.tp)
        self.kernel(
            contribution_mailbox,
            stage_state,
            reduced_mailbox_multicast_address,
            m,
        ).launch(
            grid=(owned_tokens * self.ctas_per_token, 1, 1),
            block=(self.threads, 1, 1),
            stream=stream,
            use_pdl=self.enable_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        contribution_mailbox: cute.Tensor,
        stage_state: cute.Tensor,
        reduced_mailbox_multicast_address: Int64,
        m: Int32,
    ) -> None:
        block, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        local_token = block // self.ctas_per_token
        fragment = (block % self.ctas_per_token) * self.threads + tidx
        token = local_token * self.tp + self.rank
        active = token < m and fragment < self.fragments

        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_wait()

        stage = load_volatile_u32(stage_state.iterator + ACTIVE_STAGE)
        rank_values = cute.make_rmem_tensor(cute.make_layout((self.tp, 4)), Uint32)
        rank_values.fill(Uint32(0))
        dirty = active
        while dirty:
            dirty = False
            for source_rank in cutlass.range_constexpr(self.tp):
                source_element = (
                    (Int64(stage) * self.tp + source_rank) * self.owner_capacity
                    + Int64(local_token)
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
                    rank_values[source_rank, word] = packed[word]
        if active:
            reduced = cute.make_rmem_tensor(cute.make_layout((VEC_BF16,)), Float32)
            reduced.fill(Float32(0.0))
            for source_rank in cutlass.range_constexpr(self.tp):
                packed = cute.make_rmem_tensor(cute.make_layout((4,)), Uint32)
                for word in cutlass.range_constexpr(4):
                    packed[word] = rank_values[source_rank, word]
                reduced.store(
                    reduced.load() + packed_u32x4_to_bf16x8(packed.load()).to(Float32)
                )
            reduced_element = (
                Int64(stage) * self.capacity_m + Int64(token)
            ) * self.hidden + Int64(fragment) * VEC_BF16
            stmc_bf16x8(
                reduced_mailbox_multicast_address + reduced_element * 2,
                sanitize_negative_zero_u32x4(
                    bf16x8_to_packed_u32x4(reduced.load().to(BFloat16))
                ),
            )

        if block == 0 and tidx == 0:
            store_global_u32(
                stage_state.iterator + NEXT_STAGE,
                (stage + Uint32(1)) % Uint32(LAMPORT_GENERATIONS),
            )
        cute.arch.barrier()
        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_launch_dependents()

        if active:
            for source_rank in cutlass.range_constexpr(self.tp):
                source_element = (
                    (Int64(stage) * self.tp + source_rank) * self.owner_capacity
                    + Int64(local_token)
                ) * self.hidden + Int64(fragment) * VEC_BF16
                store_lamport_sentinel_u32x4(
                    Int64((contribution_mailbox.iterator + source_element).toint())
                )


class _LamportMHCDeviceKernel:
    """FlashInfer's ``_LamportResidualRMSNormDeviceKernel`` with mHC.

    The residual add and residual copy become the mHC post-mix of the hc
    streams, stored as the residual output, and their pre-mix collapse, which
    the unchanged RMSNorm normalizes. With ``quantize``, the normalized output
    is also written as MXFP8 with swizzled 128x4 UE8M0 scales, including the
    zero scales of the padding rows up to a multiple of 128. With
    ``residual_only``, only the residual output is written; the pre-mix, gamma
    and normalized output are unused. With ``reduced_input`` (BT), the mailbox
    holds each token's reduced sum, ``[generation][token]``, from
    ``_OwnerReduceMulticastDeviceKernel``, which also advanced the generation.
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
        quantize: bool = False,
        residual_only: bool = False,
        reduced_input: bool = False,
    ) -> None:
        if rank_lanes not in (1, 2, 4, 8):
            raise ValueError("rank_lanes must be 1, 2, 4, or 8")
        if reduced_input and rank_lanes != 1:
            raise ValueError("a reduced mailbox has one source: rank_lanes must be 1")
        self.reduced_input = reduced_input
        if tp % rank_lanes:
            raise ValueError("tp must be divisible by rank_lanes")
        if threads <= 0 or threads % WARP_SIZE or threads % rank_lanes:
            raise ValueError("threads must be a positive warp and rank-lane multiple")
        if hidden <= 0 or hidden % VEC_BF16:
            raise ValueError("hidden must be a positive multiple of 8")
        # A scale block's four fragments must sit in four adjacent lanes.
        if quantize and (rank_lanes != 1 or hidden % (MXFP8_BLOCK * SCALE_TILE_COLS)):
            raise ValueError("MXFP8 output needs rank_lanes == 1 and hidden % 128 == 0")
        if quantize and residual_only:
            raise ValueError("a residual-only consumer has no layer input to quantize")
        self.quantize = quantize
        # FlashInfer's MXFP8 quantize helpers are FlashInfer internals: only the
        # quantized variant, which the specialized decode builds, imports them.
        # Untyped modules.
        self.fp4_ops: Any = None
        self.mxfp8_ops: Any = None
        if quantize:
            from flashinfer.cute_dsl import fp4_common
            from flashinfer.quantization import quantization_cute_dsl_utils

            self.fp4_ops = fp4_common
            self.mxfp8_ops = quantization_cute_dsl_utils
        self.write_layer_input = not residual_only
        self.hidden = hidden
        self.hc = hc
        self.tp = tp
        self.capacity_m = capacity_m
        self.cluster_size = cluster_size
        self.rank_lanes = rank_lanes
        self.threads = threads
        self.enable_pdl = enable_pdl
        self.fragments = hidden // VEC_BF16
        self.scale_cols = hidden // MXFP8_BLOCK
        self.groups_per_cta = threads // rank_lanes
        self.fragment_stride = cluster_size * self.groups_per_cta
        self.trips = math.ceil(self.fragments / self.fragment_stride)
        self.warps = threads // WARP_SIZE
        self.rank_waves = 1 if reduced_input else tp // rank_lanes

    def source_element(self, stage, source_rank, token, fragment) -> Int64:
        """Mailbox element of a fragment: of ``source_rank``'s contribution,
        or of the reduced sum (``reduced_input``)."""
        if self.reduced_input:
            slot = Int64(stage) * self.capacity_m + Int64(token)
        else:
            slot = (
                Int64(stage) * self.tp + Int64(source_rank)
            ) * self.capacity_m + Int64(token)
        return slot * self.hidden + Int64(fragment) * VEC_BF16

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
        quant_output: cute.Tensor,
        quant_scales: cute.Tensor,
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
            quant_output,
            quant_scales,
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
        quant_output: cute.Tensor,
        quant_scales: cute.Tensor,
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
                if cutlass.const_expr(self.write_layer_input):
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
            if cutlass.const_expr(self.write_layer_input):
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
                        source_element = self.source_element(
                            active_stage, source_rank, token, fragment
                        )
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
                    if cutlass.const_expr(self.write_layer_input):
                        collapse.store(
                            collapse.load() + mixed_bf16.to(Float32) * pre_mix[target]
                        )
                if cutlass.const_expr(self.write_layer_input):
                    prenorm_fragments[trip, None].store(collapse.load().to(BFloat16))

        if cutlass.const_expr(not self.reduced_input):  # noqa: SIM102
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
                    source_element = self.source_element(
                        active_stage, source_rank, token, fragment
                    )
                    store_lamport_sentinel_u32x4(
                        Int64((contribution_mailbox.iterator + source_element).toint())
                    )

        if cutlass.const_expr(self.write_layer_input):
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
                if cutlass.const_expr(self.quantize):
                    normalized = cute.make_rmem_tensor(
                        cute.make_layout((VEC_BF16,)), BFloat16
                    )
                    normalized.fill(BFloat16(0.0))
                if fragment < self.fragments and rank_lane == 0:
                    gamma_values = gamma_fragments[trip, None].load().to(Float32)
                    result = (
                        prenorm_fragments[trip, None].load().to(Float32)
                        * inv_rms
                        * gamma_values
                    ).to(BFloat16)
                    output_element = (
                        Int64(token) * self.hidden + Int64(fragment) * VEC_BF16
                    )
                    store_global_u32x4(
                        Int64((norm_output.iterator + output_element).toint()),
                        bf16x8_to_packed_u32x4(result),
                    )
                    if cutlass.const_expr(self.quantize):
                        normalized.store(result)
                if cutlass.const_expr(self.quantize):
                    # FlashInfer's 4-threads-per-scale MXFP8 quantize of the BF16
                    # output. Every lane joins the shuffles; fragments are a
                    # multiple of 4, so a scale block is wholly in or out of range.
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
                                quant_output,
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
                # Zero the scales of the padding rows, as FlashInfer's quantize
                # does, one 4-column word per store, spread over the whole grid.
                m, _, _ = cute.arch.grid_dim()
                words_per_row = self.scale_cols // SCALE_TILE_COLS
                padded_m = (
                    (m + SCALE_TILE_ROWS - 1) // SCALE_TILE_ROWS * SCALE_TILE_ROWS
                )
                pad_words = (padded_m - m) * words_per_row
                word = (token * self.cluster_size + cluster_rank) * self.threads + tidx
                while word < pad_words:
                    row = m + word // words_per_row
                    offset = self.mxfp8_ops.compute_sf_index_swizzled_128x4_gpu(
                        row,
                        word % words_per_row * SCALE_TILE_COLS,
                        Int32(self.scale_cols),
                    )
                    store_global_u32(quant_scales.iterator + offset, Uint32(0))
                    word = word + m * self.cluster_size * self.threads


class _Consumer(NamedTuple):
    """A call's validated consumer: the compiled variant and its outputs."""

    # The JIT executor ``cute.compile`` returns; CuTe DSL exposes no type for it.
    kernel: Any
    # None for the residual-only variant, which reads no pre-mix: the launch
    # passes the post-mix (same shape) in its place.
    pre: torch.Tensor | None
    norm_weight: torch.Tensor
    residual_output: torch.Tensor
    norm_output: torch.Tensor
    layer_input: torch.Tensor | None
    quant_args: tuple[cute.Tensor, cute.Tensor]


class AllReduceMHC:
    """Lamport TP all-reduce fused with mHC post, collapse and RMSNorm.

    ``__call__`` reduces ``x`` across the TP group, writes the post-mixed hc
    streams and the normalized collapse, and leaves the next mixes to the
    caller. ``finalize`` does the same for an unfinalized MoE output, folding
    the top-k reduction and the shared-expert add into the publish. Each
    instance owns a Lamport mailbox; construction is collective over the TP
    group. With ``quantize_layer_input``, both also accept ``mxfp8_out``
    buffers that receive the normalized collapse as MXFP8. With
    ``residual_only``, both also accept ``pre`` and ``norm_weight`` of None and
    then write only the post-mixed streams, for a consumer that collapses them
    itself. Calls of ``bt_min_rows`` rows or more (finalize:
    ``bt_min_finalize_rows``) use the two-shot BT protocol, with its own
    mailboxes and generation state.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        hc_mult: int,
        max_num_tokens: int,
        top_k: int,
        device: torch.device,
        quantize_layer_input: bool = False,
        residual_only: bool = False,
    ) -> None:
        group = get_tp_group().device_group
        self.hidden_size = hidden = hidden_size
        self.hc_mult = hc = hc_mult
        self.capacity = capacity = max_num_tokens
        self.top_k = top_k
        self.quantize_layer_input = quantize_layer_input
        self.residual_only = residual_only
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
            # BT, for instances that can take BT_MIN_ROWS rows; a smaller one
            # (thresholds above its capacity) builds none of its kernels or
            # buffers. Tokens are owned round-robin, so a rank owns at most
            # owner_capacity of a call's rows.
            bt_enabled = capacity >= BT_MIN_ROWS
            self.bt_min_rows = BT_MIN_ROWS if bt_enabled else capacity + 1
            self.bt_min_finalize_rows = (
                BT_MIN_FINALIZE_ROWS if bt_enabled else capacity + 1
            )
            self._owner_capacity = owner_capacity = math.ceil(capacity / tp)
            if bt_enabled:
                self._compile_bt(hidden, top_k, tp, rank, capacity, owner_capacity)
            # Stand-ins for the MXFP8 outputs of the unquantized variants,
            # which never touch them.
            self._no_quant_output = torch.empty(16, dtype=torch.uint8, device=device)
            no_quant_cute = to_cute(self._no_quant_output, 16)
            self._no_quant_args = (no_quant_cute, no_quant_cute)

            def compile_collective(
                quantize: bool, residual_only: bool, reduced_input: bool
            ):
                if quantize:
                    quant_fakes = (
                        make_fake_dynamic_compact_tensor(
                            Uint8, alignment=16, divisibility=hidden
                        ),
                        make_fake_dynamic_compact_tensor(
                            Uint8, alignment=16, divisibility=16
                        ),
                    )
                else:
                    quant_fakes = (
                        make_fake_compact_tensor(Uint8, (16,), assumed_align=16),
                    ) * 2
                return cute.compile(
                    _LamportMHCDeviceKernel(
                        hidden=hidden,
                        hc=hc,
                        tp=tp,
                        capacity_m=capacity,
                        cluster_size=5,
                        rank_lanes=1,
                        threads=128,
                        enable_pdl=True,
                        quantize=quantize,
                        residual_only=residual_only,
                        reduced_input=reduced_input,
                    ),
                    make_fake_compact_tensor(
                        BFloat16,
                        (
                            LAMPORT_GENERATIONS
                            * (1 if reduced_input else tp)
                            * capacity
                            * hidden,
                        ),
                        assumed_align=16,
                    ),
                    make_fake_dynamic_compact_tensor(
                        BFloat16, alignment=16, divisibility=hc * hidden
                    ),
                    make_fake_dynamic_compact_tensor(
                        Float32, alignment=4, divisibility=hc
                    ),
                    make_fake_dynamic_compact_tensor(
                        Float32, alignment=4, divisibility=hc * hc
                    ),
                    make_fake_dynamic_compact_tensor(
                        Float32, alignment=4, divisibility=hc
                    ),
                    make_fake_compact_tensor(BFloat16, (hidden,), assumed_align=16),
                    make_fake_dynamic_compact_tensor(
                        BFloat16, alignment=16, divisibility=hc * hidden
                    ),
                    make_fake_dynamic_compact_tensor(
                        BFloat16, alignment=16, divisibility=hidden
                    ),
                    *quant_fakes,
                    make_fake_compact_tensor(Int32, (2,), assumed_align=4),
                    Float32(0.0),
                    Int32(capacity),
                    current_cu_stream(),
                )

            # Consumers by (BT, variant): "plain", "quant" (also the MXFP8
            # layer input) and "residual" (residual streams only).
            self._consumers: dict[tuple[bool, str], Any] = {}
            for bt in (False, True) if bt_enabled else (False,):
                self._consumers[bt, "plain"] = compile_collective(False, False, bt)
                if quantize_layer_input:
                    self._consumers[bt, "quant"] = compile_collective(True, False, bt)
                if residual_only:
                    self._consumers[bt, "residual"] = compile_collective(
                        False, True, bt
                    )
            if residual_only:
                # Stand-in for the norm weight the variant never reads.
                self._no_norm_weight = torch.empty(
                    hidden, dtype=torch.bfloat16, device=device
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
            if bt_enabled:
                self._allocate_bt(group, tp, capacity, owner_capacity, device)
            torch.accelerator.synchronize()
        dist.barrier(group=group)

    def _compile_bt(
        self,
        hidden: int,
        top_k: int,
        tp: int,
        rank: int,
        capacity: int,
        owner_capacity: int,
    ) -> None:
        """Compile the BT publish, finalize publish and owner-reduce kernels."""
        self._bt_publish = cute.compile(
            _SharedOnlyPublishDeviceKernel(
                hidden=hidden,
                tp=tp,
                rank=rank,
                capacity_m=capacity,
                elements_per_thread=VEC_BF16,
                threads=128,
                release_before_store=False,
                enable_pdl=True,
                owner_capacity=owner_capacity,
            ),
            make_fake_dynamic_compact_tensor(
                BFloat16, alignment=16, divisibility=hidden
            ),
            make_fake_compact_tensor(Int32, (2,), assumed_align=4),
            Int64(0),
            Int32(capacity),
            current_cu_stream(),
        )
        self._bt_finalize_publish = cute.compile(
            _QuadFinalizePublishDeviceKernel(
                hidden=hidden,
                top_k=top_k,
                tp=tp,
                rank=rank,
                capacity_m=capacity,
                threads=128,
                routed_scaling_factor=1.0,
                include_shared_expert=True,
                load_shared_expert_before_pdl=False,
                enable_pdl=True,
                prefetch_group=top_k,
                fp32_weights=True,
                owner_capacity=owner_capacity,
                elements_per_thread=VEC_BF16,
            ),
            make_fake_dynamic_compact_tensor(
                BFloat16, alignment=16, divisibility=hidden
            ),
            make_fake_dynamic_compact_tensor(Float32, alignment=4, divisibility=top_k),
            make_fake_dynamic_compact_tensor(Int32, alignment=4, divisibility=top_k),
            make_fake_dynamic_compact_tensor(
                BFloat16, alignment=16, divisibility=hidden
            ),
            make_fake_compact_tensor(Int32, (2,), assumed_align=4),
            Int64(0),
            Int32(capacity),
            current_cu_stream(),
        )
        # FlashInfer's BT presets for TP4/H5120 reduce with 32 threads a
        # CTA up to 192 rows and 320 above.
        self._owner_reduce = {
            threads: cute.compile(
                _OwnerReduceMulticastDeviceKernel(
                    hidden=hidden,
                    tp=tp,
                    rank=rank,
                    capacity_m=capacity,
                    owner_capacity=owner_capacity,
                    threads=threads,
                    enable_pdl=True,
                ),
                make_fake_compact_tensor(
                    BFloat16,
                    (LAMPORT_GENERATIONS * tp * owner_capacity * hidden,),
                    assumed_align=16,
                ),
                make_fake_compact_tensor(Int32, (2,), assumed_align=4),
                Int64(0),
                Int32(capacity),
                current_cu_stream(),
            )
            for threads in (32, 320)
        }

    def _allocate_bt(
        self,
        group: dist.ProcessGroup,
        tp: int,
        capacity: int,
        owner_capacity: int,
        device: torch.device,
    ) -> None:
        """Allocate the BT mailboxes (collective over ``group``)."""
        # BT: contributions [generation][source rank][owned token] on each
        # owner, stored by unicast through a table of the owners' addresses,
        # and the owners' sums [generation][token] on every rank.
        self._bt_mailbox = symm_mem.empty(
            (LAMPORT_GENERATIONS, tp, owner_capacity, self.hidden_size),
            dtype=torch.bfloat16,
            device=device,
        )
        self._bt_mailbox_handle = symm_mem.rendezvous(self._bt_mailbox, group)
        # get_remote_tensor keeps the rendezvous offset of a pooled buffer.
        owner_mailboxes = [
            self._bt_mailbox_handle.get_remote_tensor(
                peer, self._bt_mailbox.shape, self._bt_mailbox.dtype
            ).data_ptr()
            for peer in range(tp)
        ]
        if not all(owner_mailboxes):
            raise RuntimeError("NVLink peer mapping is unavailable")
        self._bt_owner_mailboxes = torch.tensor(
            owner_mailboxes, dtype=torch.int64, device=device
        )
        self._bt_reduced = symm_mem.empty(
            (LAMPORT_GENERATIONS, capacity, self.hidden_size),
            dtype=torch.bfloat16,
            device=device,
        )
        self._bt_reduced_handle = symm_mem.rendezvous(self._bt_reduced, group)
        bt_multicast = int(self._bt_reduced_handle.multicast_ptr or 0)
        if not bt_multicast or bt_multicast % 16:
            raise RuntimeError("NVLink multicast mapping is unavailable")
        self._bt_multicast = bt_multicast
        self._bt_mailbox.view(torch.int16).fill_(-32768)
        self._bt_reduced.view(torch.int16).fill_(-32768)
        self._bt_stage_state = torch.zeros(2, dtype=torch.int32, device=device)

    def __call__(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
        pre: torch.Tensor | None,
        norm_weight: torch.Tensor | None,
        norm_eps: float,
        mxfp8_out: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return (post-mixed residual streams, normalized layer input).

        Args:
            x: [M, hidden] BF16 TP-partial sublayer output.
            residual: [M, hc, hidden] BF16 residual streams.
            post: [M, hc(, 1)] FP32 post-mix.
            comb: [M, hc, hc] FP32 residual mix, indexed [source, target].
            pre: [M, hc] FP32 carried pre-mix for the collapse. None (with
                ``norm_weight``, built with ``residual_only``): write only the
                residual streams and return no layer input.
            norm_weight: [hidden] BF16 RMSNorm weight.
            norm_eps: RMSNorm epsilon.
            mxfp8_out: optional [M, hidden] FP8 E4M3 values and swizzled
                128x4 UE8M0 scales (``mxfp8_scale_bytes(M, hidden)`` bytes)
                that also receive the layer input as MXFP8.

        """
        m = x.shape[0]
        self._check(m, residual, post, comb, pre)
        if x.shape != (m, self.hidden_size):
            raise ValueError("unexpected x shape")
        bt = m >= self.bt_min_rows
        consumer = self._consumer(residual, pre, norm_weight, m, mxfp8_out, bt)
        (self._bt_publish if bt else self._publish)(
            to_cute_dynamic(x.flatten(), 16, divisibility=self.hidden_size),
            *self._publish_target(bt),
            Int32(m),
            current_cu_stream(),
        )
        return self._collect(consumer, residual, post, comb, norm_eps, m, bt)

    def finalize(
        self,
        gemm2_permuted: torch.Tensor,
        expert_weights: torch.Tensor,
        expanded_idx_to_permuted_idx: torch.Tensor,
        shared_output: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
        pre: torch.Tensor | None,
        norm_weight: torch.Tensor | None,
        norm_eps: float,
        mxfp8_out: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
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
            pre: [M, hc] FP32 carried pre-mix for the collapse (None: residual
                only, as in ``__call__``).
            norm_weight: [hidden] BF16 RMSNorm weight (None with ``pre``).
            norm_eps: RMSNorm epsilon.
            mxfp8_out: as in ``__call__``.

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
        bt = m >= self.bt_min_finalize_rows
        consumer = self._consumer(residual, pre, norm_weight, m, mxfp8_out, bt)
        (self._bt_finalize_publish if bt else self._finalize_publish)(
            to_cute_dynamic(gemm2_permuted.flatten(), 16, divisibility=hidden),
            to_cute_dynamic(expert_weights.flatten(), 4, divisibility=top_k),
            to_cute_dynamic(
                expanded_idx_to_permuted_idx.flatten(), 4, divisibility=top_k
            ),
            to_cute_dynamic(shared_output.flatten(), 16, divisibility=hidden),
            *self._publish_target(bt),
            Int32(m),
            current_cu_stream(),
        )
        return self._collect(consumer, residual, post, comb, norm_eps, m, bt)

    def _publish_target(self, bt: bool) -> tuple[cute.Tensor, Int64]:
        """The publish kernels' generation state and mailbox address: LL's
        multicast mailbox, or BT's table of the owners' mailboxes."""
        if bt:
            return to_cute(self._bt_stage_state, 4), Int64(
                self._bt_owner_mailboxes.data_ptr()
            )
        return to_cute(self._stage_state, 4), Int64(self._multicast)

    def _check(
        self,
        m: int,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
        pre: torch.Tensor | None,
    ) -> None:
        hc = self.hc_mult
        if not 1 <= m <= self.capacity:
            raise ValueError(f"M={m} is outside [1, {self.capacity}]")
        if residual.shape != (m, hc, self.hidden_size):
            raise ValueError("unexpected residual shape")
        if (
            post.numel() != m * hc
            or (pre is not None and pre.numel() != m * hc)
            or comb.numel() != m * hc * hc
        ):
            raise ValueError("unexpected mix shapes")

    def _consumer(
        self,
        residual: torch.Tensor,
        pre: torch.Tensor | None,
        norm_weight: torch.Tensor | None,
        m: int,
        mxfp8_out: tuple[torch.Tensor, torch.Tensor] | None,
        bt: bool,
    ) -> _Consumer:
        """Select and validate the consumer variant of a call and allocate its
        outputs. Callers do this before the call's first launch, so a bad
        argument raises with the mailboxes and generation state untouched."""
        hidden = self.hidden_size
        residual_output = torch.empty_like(residual)
        if (pre is None) != (norm_weight is None):
            raise ValueError("pre and norm_weight are both given or both None")
        if pre is None or norm_weight is None:
            if (bt, "residual") not in self._consumers:
                raise ValueError("built without residual_only")
            if mxfp8_out is not None:
                raise ValueError("a residual-only boundary has no layer input")
            # The variant reads neither the pre-mix nor the norm weight and
            # writes no layer input: stand-ins fill their arguments (the
            # post-mix has the pre-mix's shape).
            return _Consumer(
                kernel=self._consumers[bt, "residual"],
                pre=None,
                norm_weight=self._no_norm_weight,
                residual_output=residual_output,
                norm_output=residual_output,
                layer_input=None,
                quant_args=self._no_quant_args,
            )
        layer_input = torch.empty(
            (m, hidden), dtype=torch.bfloat16, device=residual.device
        )
        if mxfp8_out is None:
            kernel, quant_args = self._consumers[bt, "plain"], self._no_quant_args
        else:
            if (bt, "quant") not in self._consumers:
                raise ValueError("built without quantize_layer_input")
            values, scales = mxfp8_out
            if values.shape != (m, hidden) or scales.numel() != (
                mxfp8_scale_bytes(m, hidden)
            ):
                raise ValueError("mxfp8_out does not match the layer input")
            kernel = self._consumers[bt, "quant"]
            quant_args = (
                to_cute_dynamic(
                    values.view(torch.uint8).flatten(), 16, divisibility=hidden
                ),
                to_cute_dynamic(scales.flatten(), 16, divisibility=16),
            )
        return _Consumer(
            kernel=kernel,
            pre=pre,
            norm_weight=norm_weight,
            residual_output=residual_output,
            norm_output=layer_input,
            layer_input=layer_input,
            quant_args=quant_args,
        )

    def _collect(
        self,
        consumer: _Consumer,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
        norm_eps: float,
        m: int,
        bt: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Launch the call's reduction after its publish: for BT the owner
        reduce, then the consumer that ``_consumer`` selected."""
        hidden, hc = self.hidden_size, self.hc_mult
        if bt:
            # The owners reduce first; the consumer then reads their sums.
            self._owner_reduce[32 if m <= 192 else 320](
                to_cute(self._bt_mailbox.flatten(), 16),
                to_cute(self._bt_stage_state, 4),
                Int64(self._bt_multicast),
                Int32(m),
                current_cu_stream(),
            )
        pre = post if consumer.pre is None else consumer.pre
        consumer.kernel(
            to_cute((self._bt_reduced if bt else self._mailbox).flatten(), 16),
            to_cute_dynamic(residual.flatten(), 16, divisibility=hc * hidden),
            to_cute_dynamic(post.flatten(), 4, divisibility=hc),
            to_cute_dynamic(comb.flatten(), 4, divisibility=hc * hc),
            to_cute_dynamic(pre.flatten(), 4, divisibility=hc),
            to_cute(consumer.norm_weight, 16),
            to_cute_dynamic(
                consumer.residual_output.flatten(), 16, divisibility=hc * hidden
            ),
            to_cute_dynamic(consumer.norm_output.flatten(), 16, divisibility=hidden),
            *consumer.quant_args,
            to_cute(self._bt_stage_state if bt else self._stage_state, 4),
            Float32(norm_eps),
            Int32(m),
            current_cu_stream(),
        )
        return consumer.residual_output, consumer.layer_input
