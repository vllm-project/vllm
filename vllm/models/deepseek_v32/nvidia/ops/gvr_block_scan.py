# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Adapted from NVIDIA/TensorRT-LLM at 3c68ae6ac79c48c6bad5816adcfcefc7d9897d55.
# Keep this upstream kernel mechanically comparable to its source.
# ruff: noqa

import math

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import from_dlpack
from cutlass.utils.smem_allocator import SmemAllocator

"""
block prefix sum kernel (input is loading from shared memory) in CuTe DSL.
The parallel strategy is one thread process one element from shared memory.
"""


@cute.jit
def fence_acq_rel_cta(*, loc=None, ip=None):
    llvm.inline_asm(
        res=None,
        operands_=[],
        asm_string="membar.cta;",
        constraints="",
        has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@cute.jit
def warp_scan(
    val: cutlass.Int32, tidx, lane_id, num_threads_per_warp: cutlass.Constexpr
):
    """Warp scan kernel"""
    mask_val = cutlass.const_expr(((1 << num_threads_per_warp) - 1) & 0xFFFFFFFF)
    mask_and_clamp_val = 0
    iteration = cute.arch.log2_of_pow2_int(cutlass.Int32(num_threads_per_warp))
    for i in cutlass.range(iteration, unroll_full=True):
        offset = 1 << i
        other = cute.arch.shuffle_sync_up(
            val, offset, mask=mask_val, mask_and_clamp=mask_and_clamp_val
        )
        if lane_id >= offset:
            val = val + other
    return val


@cute.jit
def block_prefix_sum_kernel(
    val: cutlass.Int32,
    warp_sums: cute.Tensor,
    tidx,
    num_threads,
    num_warps,
    barrier_id=1,
    need_total_sum=False,
):
    """Block prefix sum kernel in CuTe DSL"""
    # Thread and warp id
    warp_id = tidx // 32
    lane_id = tidx % 32

    assert num_threads % 32 == 0, (
        "num_threads must be divisible by 32, but got {}".format(num_threads)
    )
    assert num_warps >= 1, "num_warps must be >= 1, but got {}".format(num_warps)
    assert num_warps == 2 ** int(math.log2(num_warps)), "num_warps must be a power of 2"

    # Step 1: Warp-level prefix sum using shuffle
    val = warp_scan(val, tidx, lane_id, num_threads_per_warp=32)

    # Step 2: Store warp prefix sums
    if lane_id == 31:  # Last thread in warp stores warp sum
        warp_sums[warp_id] = val
    cute.arch.barrier(barrier_id=barrier_id, number_of_threads=num_threads)

    # Step 3: Prefix sum across warps
    if warp_id == 0:
        if lane_id < num_warps:
            warp_val = warp_sums[lane_id]
            # call warp-level prefix sum
            warp_val = warp_scan(
                warp_val, tidx, lane_id, num_threads_per_warp=num_warps
            )
            warp_sums[lane_id] = warp_val
    cute.arch.barrier(barrier_id=barrier_id, number_of_threads=num_threads)

    # Step 4: Add warp-level prefix
    if warp_id > 0:
        val = val + warp_sums[warp_id - 1]

    # Step 5: Get total sum if need_total_sum is True
    total_sum = 0
    if need_total_sum:
        total_sum = warp_sums[num_warps - 1]

    return val, total_sum


@cute.kernel
def block_prefix_sum(
    num_bins: cutlass.Constexpr,
    num_threads_per_block: cutlass.Constexpr,
    input: cute.Tensor,
    output: cute.Tensor,
):
    tidx, _, _ = cute.arch.thread_idx()

    num_warps = cutlass.const_expr(min(num_bins, num_threads_per_block) // 32)
    # Shared memory allocation used for cross-warp communication.
    smem = SmemAllocator()
    s_warp_sums = smem.allocate_tensor(
        element_type=cute.Int32,
        layout=cute.make_ordered_layout((num_warps,), order=(0,)),
        byte_alignment=128,
    )

    if cutlass.const_expr(num_bins < num_threads_per_block):
        if tidx < num_bins:
            val = input[tidx]
            val, total_sum = block_prefix_sum_kernel(
                val, s_warp_sums, tidx, num_bins, num_warps, barrier_id=1
            )
            output[tidx] = val
    elif cutlass.const_expr(num_bins == num_threads_per_block):
        val = input[tidx]
        val, total_sum = block_prefix_sum_kernel(
            val, s_warp_sums, tidx, num_bins, num_warps, barrier_id=1
        )
        output[tidx] = val
    else:
        assert num_bins % num_threads_per_block == 0
        previous_sum = 0
        val = 0
        total_sum = 0
        """
        i = 0: total_sum = 1th_tile sum; previous_sum = 0; out = 1st_tile scan + 0;
        i = 1: total_sum = 2th_tile sum; previous_sum = 1th_tile sum; out = 2nd_tile scan + previous_sum
        i = 2: total_sum = 3th_tile sum; previous_sum = 1th_tile sum + 2th_tile sum; out = 3rd_tile scan + previous_sum
        ...
        """
        for i in range(tidx, num_bins, num_threads_per_block):
            val = input[i]
            val, total_sum = block_prefix_sum_kernel(
                val,
                s_warp_sums,
                tidx,
                num_threads_per_block,
                num_warps,
                barrier_id=0,
                need_total_sum=True,
            )
            output[i] = val + previous_sum
            previous_sum = previous_sum + total_sum


# host function for testing
@cute.jit
def host_block_prefix_sum(
    num_bins: cutlass.Constexpr,
    num_threads_per_block: cutlass.Constexpr,
    input: cute.Tensor,
    output: cute.Tensor,
):
    block_prefix_sum(num_bins, num_threads_per_block, input, output).launch(
        grid=(1, 1, 1), block=(num_threads_per_block, 1, 1)
    )
    return output


def test_block_prefix_sum(num_bins=1024, num_threads_per_block=1024):
    import torch

    input = torch.randint(0, 100, (num_bins,), device="cuda").to(torch.int32)
    # input = torch.arange(1, num_bins + 1, dtype=torch.int32, device='cuda')
    # input = torch.ones(num_bins, dtype=torch.int32, device='cuda')
    output = torch.empty_like(input)

    input_cute_tensor = from_dlpack(input)
    output_cute_tensor = from_dlpack(output)

    compiled_func = cute.compile(
        host_block_prefix_sum,
        num_bins,
        num_threads_per_block,
        input_cute_tensor,
        output_cute_tensor,
    )
    compiled_func(input_cute_tensor, output_cute_tensor)

    torch_output = torch.cumsum(input, dim=0)

    torch.testing.assert_close(output.to(torch_output.dtype), torch_output)
    print(
        "Test passed for num_bins: {}, num_threads_per_block: {}".format(
            num_bins, num_threads_per_block
        )
    )


if __name__ == "__main__":
    test_block_prefix_sum(num_bins=1024, num_threads_per_block=1024)
    test_block_prefix_sum(num_bins=256, num_threads_per_block=1024)
    test_block_prefix_sum(num_bins=512, num_threads_per_block=1024)
    test_block_prefix_sum(num_bins=2048, num_threads_per_block=512)
    test_block_prefix_sum(num_bins=1024, num_threads_per_block=512)
