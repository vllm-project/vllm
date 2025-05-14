# SPDX-License-Identifier: Apache-2.0
from math import prod
from typing import Optional

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8)
from vllm.model_executor.layers.quantization.utils.int8_utils import (
    per_token_group_quant_int8, per_token_quant_int8)
from vllm.utils import cdiv


def _resize_cache(x: torch.Tensor, v: tuple[int, ...]) -> torch.Tensor:
    """
    Shrink the given tensor and apply the given view to it.  This is
    used to resize the intermediate fused_moe caches.
    """
    assert prod(
        v) <= x.numel(), f"{prod(v)} <= {x.numel()}"  # CUDAGRAPH unfriendly?
    return x.flatten()[:prod(v)].view(*v)


def _fp8_quantize(
    A: torch.Tensor,
    A_scale: Optional[torch.Tensor],
    per_act_token: bool,
    block_shape: Optional[list[int]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Perform fp8 quantization on the inputs.  If a block_shape
    is provided, the output will be blocked.
    """
    if block_shape is None:
        A, A_scale = ops.scaled_fp8_quant(
            A, A_scale, use_per_token_if_dynamic=per_act_token)
    else:
        assert len(block_shape) == 2
        _, block_k = block_shape[0], block_shape[1]
        A, A_scale = per_token_group_quant_fp8(A, block_k)
        assert cdiv(A.size(-1), block_k) == A_scale.size(-1)

    return A, A_scale


def _int8_quantize(
    A: torch.Tensor,
    A_scale: Optional[torch.Tensor],
    per_act_token: bool,
    block_shape: Optional[list[int]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Perform int8 quantization on the inputs.  If a block_shape
    is provided, the output will be blocked.
    """

    # If weights are per-channel (per_channel_quant=True), then
    # activations apply per-token quantization. Otherwise, assume
    # activation tensor-wise fp8/int8 quantization, dynamic or static
    if block_shape is None:
        assert per_act_token, \
            "int8 quantization only supports block or channel-wise"
        A, A_scale = per_token_quant_int8(A)
    else:
        assert len(block_shape) == 2
        _, block_k = block_shape[0], block_shape[1]
        A, A_scale = per_token_group_quant_int8(A, block_k)
        assert cdiv(A.size(-1), block_k) == A_scale.size(-1)

    return A, A_scale


def moe_kernel_quantize_input(
    A: torch.Tensor,
    A_scale: Optional[torch.Tensor],
    qtype: Optional[torch.dtype],
    per_channel_quant: bool,
    block_shape: Optional[list[int]] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    if qtype == torch.float8_e4m3fn:
        return _fp8_quantize(A, A_scale, per_channel_quant, block_shape)
    elif qtype == torch.int8:
        return _int8_quantize(A, A_scale, per_channel_quant, block_shape)
    else:
        assert A_scale is None
        return A, A_scale


def _fp8_perm(m: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    A permutation routine that works on fp8 types.
    """
    if torch.is_floating_point(m) and m.dtype.itemsize == 1:
        return m.view(dtype=torch.uint8)[idx, ...].view(dtype=m.dtype)
    else:
        return m[idx, ...]
