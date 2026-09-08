# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Triton grouped dynamic convolution for DFlash2.

Adapted from LightSeek TokenSpeed's DFlash2 grouped-convolution kernel.
"""

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op


@triton.jit
def _dflash2_grouped_conv_kernel(
    x_ptr,
    delta_ptr,
    base_ptr,
    output_ptr,
    x_stride_row,
    delta_stride_row,
    delta_stride_tap,
    base_stride_tap,
    output_stride_row,
    num_elements,
    NUM_CHANNELS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    TAPS: tl.constexpr,
    ELEMENT_BLOCK: tl.constexpr,
) -> None:
    offsets = tl.program_id(0) * ELEMENT_BLOCK + tl.arange(0, ELEMENT_BLOCK)
    mask = offsets < num_elements
    row = offsets // NUM_CHANNELS
    channels = offsets % NUM_CHANNELS
    groups = channels // GROUP_SIZE
    position = row % BLOCK_SIZE

    delta_row = delta_ptr + row * delta_stride_row
    x_row = x_ptr + row * x_stride_row
    accumulator = (
        tl.load(base_ptr + channels, mask=mask, other=0.0).to(tl.float32)
        + tl.load(delta_row + groups, mask=mask, other=0.0).to(tl.float32)
    ) * tl.load(x_row + channels, mask=mask, other=0.0).to(tl.float32)

    for tap in tl.static_range(1, TAPS):
        tap_mask = mask & (position >= tap)
        coefficient = tl.load(
            base_ptr + tap * base_stride_tap + channels,
            mask=tap_mask,
            other=0.0,
        ).to(tl.float32) + tl.load(
            delta_row + tap * delta_stride_tap + groups,
            mask=tap_mask,
            other=0.0,
        ).to(tl.float32)
        x = tl.load(
            x_ptr + (row - tap) * x_stride_row + channels,
            mask=tap_mask,
            other=0.0,
        ).to(tl.float32)
        accumulator += coefficient * x

    tl.store(
        output_ptr + row * output_stride_row + channels,
        accumulator,
        mask=mask,
    )


def dflash2_grouped_conv_impl(
    x: torch.Tensor,
    delta: torch.Tensor,
    base: torch.Tensor,
    block_size: int,
    group_size: int,
) -> torch.Tensor:
    num_rows, num_channels = x.shape
    output = torch.empty_like(x)
    if num_rows == 0:
        return output

    num_elements = num_rows * num_channels
    element_block = 512
    _dflash2_grouped_conv_kernel[(triton.cdiv(num_elements, element_block),)](
        x,
        delta,
        base,
        output,
        x.stride(0),
        delta.stride(0),
        delta.stride(1),
        base.stride(0),
        output.stride(0),
        num_elements,
        NUM_CHANNELS=num_channels,
        BLOCK_SIZE=block_size,
        GROUP_SIZE=group_size,
        TAPS=base.shape[0],
        ELEMENT_BLOCK=element_block,
        num_warps=4,
    )
    return output


def dflash2_grouped_conv_fake(
    x: torch.Tensor,
    delta: torch.Tensor,
    base: torch.Tensor,
    block_size: int,
    group_size: int,
) -> torch.Tensor:
    return torch.empty_like(x)


direct_register_custom_op(
    op_name="dflash2_grouped_conv",
    op_func=dflash2_grouped_conv_impl,
    fake_impl=dflash2_grouped_conv_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)


def dflash2_grouped_conv(
    x: torch.Tensor,
    delta: torch.Tensor,
    base: torch.Tensor,
    block_size: int,
    group_size: int,
) -> torch.Tensor:
    return torch.ops.vllm.dflash2_grouped_conv(x, delta, base, block_size, group_size)
