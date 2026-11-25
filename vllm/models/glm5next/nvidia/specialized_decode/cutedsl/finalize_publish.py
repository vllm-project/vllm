# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Copy of `_QuadFinalizePublishDeviceKernel` from
# vllm/models/deepseek_v41/nvidia/ops/cute_dsl/all_reduce_mhc.py, unchanged, which
# adapts FlashInfer's low-latency MNNVL CuTe DSL all-reduce
# (flashinfer/comm/mnnvl_cutedsl/kernel_ll, flashinfer-ai/flashinfer@139af6f6)
# and carries the notice below.
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

"""DeepSeek-V4.1's MoE finalize (top-k weighted sum plus the shared expert)
fused with the Lamport mailbox publish.
"""

from __future__ import annotations

import math

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64, Uint32

from vllm.models.common.ops.moe_tail_fusion.primitives import (
    ACTIVE_STAGE,
    NEXT_STAGE,
    QUAD_BF16,
    VEC_BF16,
    bf16x4_to_packed_u32x2,
    bf16x8_to_packed_u32x4,
    load_global_bf16_as_f32,
    load_global_u32x2,
    load_global_u32x4,
    load_global_u64,
    load_volatile_u32,
    packed_u32x2_to_bf16x4,
    packed_u32x4_to_bf16x8,
    sanitize_negative_zero_u32x2,
    sanitize_negative_zero_u32x4,
    stmc_bf16x4,
    stmc_bf16x8,
    store_global_u32,
    store_global_u32x2,
    store_global_u32x4,
)


class _QuadFinalizePublishDeviceKernel:
    """The MoE finalize (top-k weighted sum plus the shared expert) fused with
    the publish, `elements_per_thread` (4 or 8) values a thread."""

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
                (stage_state.iterator + ACTIVE_STAGE).toint(),
                stage,
            )
        cute.arch.barrier()
        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_launch_dependents()
