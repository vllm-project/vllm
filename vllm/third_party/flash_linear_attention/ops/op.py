# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import os

from vllm.triton_utils import tl, tldevice, triton

from .utils import is_gather_supported

if os.environ.get("FLA_USE_FAST_OPS", "0") == "1":
    exp = tldevice.fast_expf
    exp2 = tl.exp2
    log = tldevice.fast_logf
    log2 = tldevice.fast_log2f
else:
    exp = tl.exp
    exp2 = tl.exp2
    log = tl.log
    log2 = tl.log2


if not is_gather_supported:

    @triton.jit
    def gather(src, index, axis, _builder=None):
        """
        Gather operation that works when tl.gather is not supported.
        This is a fallback implementation that returns None.
        Just to make triton compiler happy.
        """
        return None
else:
    gather = tl.gather

if hasattr(triton.language, "_experimental_make_tensor_descriptor"):
    # For Triton 3.3.x
    make_tensor_descriptor = triton.language._experimental_make_tensor_descriptor
elif hasattr(triton.language, "make_tensor_descriptor"):
    # For Triton 3.4.x and later
    make_tensor_descriptor = triton.language.make_tensor_descriptor
else:
    """
    Fallback implementation when TMA is not supported.
    Returns None to indicate TMA descriptors are unavailable.
    Just make triton compiler happy.
    """

    @triton.jit
    def make_tensor_descriptor(
        base,
        shape,
        strides,
        block_shape,
        _builder=None,
    ):
        return None


@triton.jit
def store_2d_vec4_16bit(
    base_ptr,
    val_f32,
    row_offs,
    col_vec4_offs,
    row_stride,
    T,
    COLS: tl.constexpr,
    ROWS: tl.constexpr,
    BCOLS: tl.constexpr,
):
    """Pack 4 adjacent 16-bit columns into uint64 so RDNA WMMA layouts emit
    coalesced global_store_b128 instead of 8KB-strided scalar buffer_store_b16."""
    u16 = val_f32.to(base_ptr.dtype.element_ty).to(tl.uint16, bitcast=True)
    lo16, hi16 = tl.split(tl.reshape(u16, [ROWS, BCOLS // 2, 2]))
    u32 = lo16.to(tl.uint32) | (hi16.to(tl.uint32) << 16)
    lo32, hi32 = tl.split(tl.reshape(u32, [ROWS, BCOLS // 4, 2]))
    u64 = lo32.to(tl.uint64) | (hi32.to(tl.uint64) << 32)
    ptr_u64 = base_ptr.to(tl.pointer_type(tl.uint64))
    mask = (row_offs[:, None] < T) & (col_vec4_offs[None, :] < (COLS // 4))
    tl.store(
        ptr_u64
        + row_offs[:, None].to(tl.int64) * (row_stride // 4)
        + col_vec4_offs[None, :],
        u64,
        mask=mask,
    )
