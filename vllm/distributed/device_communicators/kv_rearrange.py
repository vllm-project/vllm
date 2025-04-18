# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import triton
import triton.language as tl

@triton.jit
def rearrange_kernel_read(
    t1_ptr,
    t2_ptr,
    N,
    B,
    H,
    C,
    d,
    tensor_subset_size,
    block_size,
    token_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    curr_n = offsets // block_size
    curr_b = offsets // token_size % B
    curr_h = offsets // C % H 
    curr_c = offsets % C

    src_pos = offsets

    tp_group = curr_h * d // H
    dst_h = curr_h % (H // d)
    tp_group_offset = curr_n * (block_size // d) + curr_b * (H // d) * C + dst_h * C + curr_c

    dst_pos = tensor_subset_size * tp_group + tp_group_offset
    
    tl.store(t1_ptr + src_pos, tl.load(t2_ptr + dst_pos))

@triton.jit
def rearrange_kernel_write(
    t1_ptr,
    t2_ptr,
    N,
    B,
    H,
    C,
    d,
    tensor_subset_size,
    block_size,
    token_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    curr_n = offsets // block_size
    curr_b = offsets // token_size % B
    curr_h = offsets // C % H 
    curr_c = offsets % C

    src_pos = offsets

    tp_group = curr_h * d // H
    dst_h = curr_h % (H // d)
    tp_group_offset = curr_n * (block_size // d) + curr_b * (H // d) * C + dst_h * C + curr_c

    dst_pos = tensor_subset_size * tp_group + tp_group_offset
    
    tl.store(t2_ptr + dst_pos, tl.load(t1_ptr + src_pos))
    


def rearrange_tensors(t1: torch.Tensor, t2: torch.Tensor, d: int, direction: str):
    N, B, H, C = t1.shape
    
    assert t2.shape == (N, B, H, C), "Destination tensor must have same shape as source"
    assert H % d == 0, "H must be divisible by d"

    block_size = B * H * C
    token_size = H * C
    tensor_size = N * block_size
    tensor_subset_size = tensor_size // d
    
    BLOCK_SIZE = 1024
    grid = ((N * B * H * C + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    if direction == "read":
        rearrange_kernel_read[grid](
            t1, t2,
            N, B, H, C,
            d,
            tensor_subset_size,
            block_size,
            token_size,
            BLOCK_SIZE=BLOCK_SIZE
        )
    elif direction == "write":
        rearrange_kernel_write[grid](
            t1, t2,
            N, B, H, C,
            d,
            tensor_subset_size,
            block_size,
            token_size,
            BLOCK_SIZE=BLOCK_SIZE
        )
    else:
        raise ValueError(f"Invalid direction: {direction}")