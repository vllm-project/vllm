# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from math import prod
from typing import Any, Optional, Union

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8)
from vllm.model_executor.layers.quantization.utils.int8_utils import (
    per_token_group_quant_int8, per_token_quant_int8)
from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
    quant_dequant_mxfp4)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils import cdiv
from vllm.utils.flashinfer import fp4_quantize


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


def count_expert_num_tokens(
        topk_ids: torch.Tensor, num_local_experts: int,
        expert_map: Optional[torch.Tensor]) -> torch.Tensor:
    """
    Count the number to tokens assigned to each expert.

    Parameters:
    - topk_ids (torch.Tensor): Tensor mapping each token to its
    list of experts.
    - num_local_experts (int): Number of experts in this rank.
    - expert_map (Optional[torch.Tensor]):  A tensor mapping expert indices
    from the global expert space to the local expert space of the expert
    parallel shard.

    Returns:
    A tensor of size num_local_experts, where tensor[i] holds the number
    of tokens assigned to the ith expert.
    """
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


def _resize_cache(x: torch.Tensor, v: tuple[int, ...]) -> torch.Tensor:
    """
    Shrink the given tensor and apply the given view to it.  This is
    used to resize the intermediate fused_moe caches.
    """
    assert prod(v) <= x.numel(
    ), f"{v} ({prod(v)}) <= {x.shape} ({x.numel()})"  # CUDAGRAPH unfriendly?
    return x.flatten()[:prod(v)].view(*v)


def _fp4_quantize(
    A: torch.Tensor,
    A_scale: Optional[torch.Tensor],
    is_sf_swizzled_layout: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    return fp4_quantize(A,
                        A_scale,
                        is_sf_swizzled_layout=is_sf_swizzled_layout)


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
        # TODO(luka): use QuantFP8 custom op
        #  https://github.com/vllm-project/vllm/issues/20711
        A, A_scale = ops.scaled_fp8_quant(
            A, A_scale, use_per_token_if_dynamic=per_act_token)
    else:
        assert not per_act_token
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
        assert not per_act_token
        assert len(block_shape) == 2
        _, block_k = block_shape[0], block_shape[1]
        A, A_scale = per_token_group_quant_int8(A, block_k)
        assert cdiv(A.size(-1), block_k) == A_scale.size(-1)

    return A, A_scale


def _mxfp4_quantize(
    A: torch.Tensor,
    A_scale: Optional[torch.Tensor],
    per_act_token_quant: bool,
    block_shape: Optional[list[int]] = None,
) -> tuple[torch.Tensor, None]:
    assert block_shape is None
    if not current_platform.supports_mx():
        A = quant_dequant_mxfp4(A)
    else:
        raise NotImplementedError()

    return A, None


def moe_kernel_quantize_input(
    A: torch.Tensor,
    A_scale: Optional[torch.Tensor],
    quant_dtype: Union[None, torch.dtype, str],
    per_act_token_quant: bool,
    block_shape: Optional[list[int]] = None,
    is_fp4_scale_swizzled: bool = True,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    if quant_dtype == torch.float8_e4m3fn:
        return _fp8_quantize(A, A_scale, per_act_token_quant, block_shape)
    elif quant_dtype == torch.int8:
        return _int8_quantize(A, A_scale, per_act_token_quant, block_shape)
    elif quant_dtype == torch.uint8:  # nvfp4
        return _fp4_quantize(A,
                             A_scale,
                             is_sf_swizzled_layout=is_fp4_scale_swizzled)
    elif quant_dtype == "mxfp4":
        return _mxfp4_quantize(A, A_scale, per_act_token_quant, block_shape)
    else:
        return A, A_scale


def _fp8_perm(m: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    A permutation routine that works on fp8 types.
    """
    if torch.is_floating_point(m) and m.dtype.itemsize == 1:
        return m.view(dtype=torch.uint8)[idx, ...].view(dtype=m.dtype)
    else:
        return m[idx, ...]


def normalize_scales_shape(
        scales: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if scales is not None:
        if scales.numel() == 1:
            scales = scales.view(1, 1)
        else:
            scales = scales.view(-1, scales.size(-1))
    return scales


def normalize_batched_scales_shape(
    scales: Optional[torch.Tensor],
    num_experts: int,
) -> Optional[torch.Tensor]:
    if scales is not None and scales.ndim < 3:
        if scales.numel() == 1:
            scales = scales.view(1)
            scales = torch.repeat_interleave(scales, num_experts,
                                             dim=0).view(num_experts, 1, 1)
        else:
            scales = scales.view(num_experts, -1, scales.size(-1))

    return scales


def _validate_scale_shape(
    a: torch.Tensor,
    a_scale: Optional[torch.Tensor],
    per_act_token_quant: bool,
    block_shape: Optional[list[int]],
) -> None:
    if a_scale is None:
        return

    if not per_act_token_quant and block_shape is None:
        assert a_scale.numel() == 1, f"{a_scale.shape}"
    elif per_act_token_quant:
        assert a_scale.shape[0] == a.shape[0] and a_scale.shape[1] == 1, (
            f"{a_scale.shape[0]} == {a.shape[0]} and {a_scale.shape[1]} == 1")
    else:
        assert block_shape is not None
        expected = (a.shape[0], cdiv(a.shape[1], block_shape[1]))
        assert a_scale.shape == expected, f"{a_scale.shape} == {expected}"


def extract_required_args(
    extra_args: Optional[dict[str, Any]],
    required_keys: list[str],
) -> tuple[Any, ...]:
    if extra_args is None:
        raise ValueError("`extra_args` must be provided.")

    missing_keys = [k for k in required_keys if k not in extra_args]
    if missing_keys:
        raise ValueError(f"Missing keys in `extra_args`: {missing_keys}")

    return tuple(extra_args[k] for k in required_keys)
