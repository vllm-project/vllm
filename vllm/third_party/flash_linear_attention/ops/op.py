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


@triton.jit
def _round_to_bf16(x):
    return x.to(tl.bfloat16).to(tl.float32)


@triton.jit
def l2norm_bf16(x, eps: tl.constexpr, axis: tl.constexpr):
    x_f32 = x.to(tl.float32)
    square = _round_to_bf16(x_f32 * x_f32)
    square_sum = _round_to_bf16(tl.sum(square, axis=axis, keep_dims=True))
    norm_square = _round_to_bf16(square_sum + eps)
    inverse_norm = _round_to_bf16(tl.rsqrt(norm_square))
    return _round_to_bf16(x_f32 * inverse_norm)


@triton.jit
def sigmoid_bf16(x):
    return _round_to_bf16(tl.sigmoid(x.to(tl.float32)))


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
