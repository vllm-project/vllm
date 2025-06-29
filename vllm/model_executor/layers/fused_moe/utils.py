# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import socket
from contextlib import closing
from math import prod
from typing import Optional

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8)
from vllm.model_executor.layers.quantization.utils.int8_utils import (
    per_token_group_quant_int8, per_token_quant_int8)
from vllm.triton_utils import tl, triton
from vllm.utils import cdiv


@triton.jit
def _count_expert_num_tokens(topk_ids_ptr, expert_num_tokens_ptr, num_experts,
                             topk_numel, expert_map,
                             HAS_EXPERT_MAP: tl.constexpr,
                             BLOCK_SIZE: tl.constexpr):

    curr_expert = tl.program_id(0)

    offsets = tl.arange(0, BLOCK_SIZE)
    topk_ids_ptrs = topk_ids_ptr + offsets

    acc = tl.zeros((BLOCK_SIZE, ), dtype=tl.int32)
    for x in range(tl.cdiv(topk_numel, BLOCK_SIZE)):
        mask = offsets < (topk_numel - x * BLOCK_SIZE)
        expert_ids = tl.load(topk_ids_ptrs, mask=mask, other=-1)
        if HAS_EXPERT_MAP:
            expert_map_ptrs = expert_map + expert_ids
            expert_map_mask = expert_ids >= 0
            expert_ids = tl.load(expert_map_ptrs,
                                 mask=expert_map_mask,
                                 other=-1)

        has_curr_expert = tl.where(expert_ids == curr_expert, 1, 0)
        acc = acc + has_curr_expert
        topk_ids_ptrs += BLOCK_SIZE

    if curr_expert < num_experts:
        tl.store(expert_num_tokens_ptr + curr_expert, tl.sum(acc))


def _resize_cache(x: torch.Tensor, v: tuple[int, ...]) -> torch.Tensor:
    """
    Shrink the given tensor and apply the given view to it.  This is
    used to resize the intermediate fused_moe caches.
    """
    assert prod(v) <= x.numel(
    ), f"{v} ({prod(v)}) <= {x.shape} ({x.numel()})"  # CUDAGRAPH unfriendly?
    return x.flatten()[:prod(v)].view(*v)


def count_expert_num_tokens(
        topk_ids: torch.Tensor, num_local_experts: int,
        expert_map: Optional[torch.Tensor]) -> torch.Tensor:
    assert topk_ids.dtype.is_signed, (
        "The kernel uses -1 to represent invalid topk_ids")
    expert_num_tokens = torch.empty((num_local_experts),
                                    device=topk_ids.device,
                                    dtype=torch.int32)

    grid = num_local_experts
    BLOCK_SIZE = min(topk_ids.numel(), 1024)
    BLOCK_SIZE = triton.next_power_of_2(BLOCK_SIZE)

    _count_expert_num_tokens[(grid, )](
        topk_ids,
        expert_num_tokens,
        num_local_experts,
        topk_ids.numel(),
        expert_map,
        HAS_EXPERT_MAP=expert_map is not None,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return expert_num_tokens


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


def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
