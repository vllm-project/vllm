# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import functools
from math import prod

import torch
import torch.nn.functional as F

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
)
from vllm.model_executor.layers.quantization.utils.int8_utils import (
    per_token_group_quant_int8,
    per_token_quant_int8,
)
from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
    quant_dequant_mxfp4,
)
from vllm.model_executor.layers.quantization.utils.mxfp6_utils import (
    quant_dequant_mxfp6,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    mxfp8_e4m3_quantize,
)
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import is_torch_equal_or_newer


@triton.jit
def _count_expert_num_tokens(
    topk_ids_ptr,
    expert_num_tokens_ptr,
    num_experts,
    topk_numel,
    expert_map,
    HAS_EXPERT_MAP: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    curr_expert = tl.program_id(0)

    offsets = tl.arange(0, BLOCK_SIZE)
    topk_ids_ptrs = topk_ids_ptr + offsets

    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    for x in range(tl.cdiv(topk_numel, BLOCK_SIZE)):
        mask = offsets < (topk_numel - x * BLOCK_SIZE)
        expert_ids = tl.load(topk_ids_ptrs, mask=mask, other=-1)
        if HAS_EXPERT_MAP:
            expert_map_ptrs = expert_map + expert_ids
            expert_map_mask = expert_ids >= 0
            expert_ids = tl.load(expert_map_ptrs, mask=expert_map_mask, other=-1)

        has_curr_expert = tl.where(expert_ids == curr_expert, 1, 0)
        acc = acc + has_curr_expert
        topk_ids_ptrs += BLOCK_SIZE

    if curr_expert < num_experts:
        tl.store(expert_num_tokens_ptr + curr_expert, tl.sum(acc))


def count_expert_num_tokens(
    topk_ids: torch.Tensor, num_local_experts: int, expert_map: torch.Tensor | None
) -> torch.Tensor:
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
    assert topk_ids.dtype.is_signed, "The kernel uses -1 to represent invalid topk_ids"
    expert_num_tokens = torch.empty(
        (num_local_experts), device=topk_ids.device, dtype=torch.int32
    )

    grid = num_local_experts
    BLOCK_SIZE = min(topk_ids.numel(), 1024)
    BLOCK_SIZE = triton.next_power_of_2(BLOCK_SIZE)

    _count_expert_num_tokens[(grid,)](
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
    assert prod(v) <= x.numel(), (
        f"{v} ({prod(v)}) <= {x.shape} ({x.numel()})"
    )  # CUDAGRAPH unfriendly?
    return x.flatten()[: prod(v)].view(*v)


def _nvfp4_quantize(
    A: torch.Tensor,
    A_scale: torch.Tensor | None,
    is_sf_swizzled_layout: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    return ops.scaled_fp4_quant(A, A_scale, is_sf_swizzled_layout=is_sf_swizzled_layout)


def _fp8_quantize(
    A: torch.Tensor,
    A_scale: torch.Tensor | None,
    per_act_token: bool,
    block_shape: list[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Perform fp8 quantization on the inputs.  If a block_shape
    is provided, the output will be blocked.
    """
    if block_shape is None:
        # TODO(luka): use QuantFP8 custom op
        #  https://github.com/vllm-project/vllm/issues/20711
        A, A_scale = ops.scaled_fp8_quant(
            A, A_scale, use_per_token_if_dynamic=per_act_token
        )
    else:
        assert not per_act_token
        assert len(block_shape) == 2
        _, block_k = block_shape[0], block_shape[1]
        A, A_scale = per_token_group_quant_fp8(A, block_k)
        assert cdiv(A.size(-1), block_k) == A_scale.size(-1)

    return A, A_scale


def _int8_quantize(
    A: torch.Tensor,
    A_scale: torch.Tensor | None,
    per_act_token: bool,
    block_shape: list[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Perform int8 quantization on the inputs.  If a block_shape
    is provided, the output will be blocked.
    """

    # If weights are per-channel (per_channel_quant=True), then
    # activations apply per-token quantization. Otherwise, assume
    # activation tensor-wise fp8/int8 quantization, dynamic or static
    if block_shape is None:
        assert per_act_token, "int8 quantization only supports block or channel-wise"
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
    A_scale: torch.Tensor | None,
    per_act_token_quant: bool,
    block_shape: list[int] | None = None,
) -> tuple[torch.Tensor, None]:
    assert block_shape is None
    # TODO: native mxfp4 is currently not integrated in vllm,
    # so simulating even on devices supporting this data type natively.
    # Once integrated, `current_platform.supports_mx()` should be used to
    # control quantize+dequantize, or simply quantize here down to mxfp4.
    A = quant_dequant_mxfp4(A)

    return A, None


def _mxfp8_e4m3_quantize(
    A: torch.Tensor,
    A_scale: torch.Tensor | None,
    per_act_token_quant: bool,
    block_shape: list[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert A_scale is None
    assert not per_act_token_quant
    assert block_shape is None
    return mxfp8_e4m3_quantize(A)


def _mxfp6_e3m2_quantize(
    A: torch.Tensor,
    A_scale: torch.Tensor | None,
    per_act_token_quant: bool,
    block_shape: list[int] | None = None,
) -> tuple[torch.Tensor, None]:
    assert block_shape is None

    # TODO: native mxfp6 is currently not integrated in vllm,
    # so simulating even on devices supporting this data type natively.
    # Eventually, there should be a check based on
    # `current_platform.supports_mx()` here.
    A = quant_dequant_mxfp6(A, quant_dtype="fp6_e3m2")

    return A, None


def _mxfp6_e2m3_quantize(
    A: torch.Tensor,
    A_scale: torch.Tensor | None,
    per_act_token_quant: bool,
    block_shape: list[int] | None = None,
) -> tuple[torch.Tensor, None]:
    assert block_shape is None

    # TODO: native mxfp6 is currently not integrated in vllm,
    # so simulating even on devices supporting this data type natively.
    # Eventually, there should be a check based on
    # `current_platform.supports_mx()` here.
    A = quant_dequant_mxfp6(A, quant_dtype="fp6_e2m3")

    return A, None


def moe_kernel_quantize_input(
    A: torch.Tensor,
    A_scale: torch.Tensor | None,
    quant_dtype: None | torch.dtype | str,
    per_act_token_quant: bool,
    block_shape: list[int] | None = None,
    is_fp4_scale_swizzled: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if quant_dtype == torch.float8_e4m3fn:
        return _fp8_quantize(A, A_scale, per_act_token_quant, block_shape)
    elif quant_dtype == torch.int8:
        return _int8_quantize(A, A_scale, per_act_token_quant, block_shape)
    elif quant_dtype == "nvfp4":
        return _nvfp4_quantize(A, A_scale, is_sf_swizzled_layout=is_fp4_scale_swizzled)
    elif quant_dtype == "mxfp4":
        return _mxfp4_quantize(A, A_scale, per_act_token_quant, block_shape)
    elif quant_dtype == "mxfp8":
        # TODO: `quant_dtype == "mxfp8"` is ambiguous,
        # should be fp8_e4m3. OCP MX also defines `fp8_e5m2`.
        return _mxfp8_e4m3_quantize(A, A_scale, per_act_token_quant, block_shape)
    elif quant_dtype == "mxfp6_e3m2":
        return _mxfp6_e3m2_quantize(A, A_scale, per_act_token_quant, block_shape)
    elif quant_dtype == "mxfp6_e2m3":
        return _mxfp6_e2m3_quantize(A, A_scale, per_act_token_quant, block_shape)
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


def normalize_scales_shape(scales: torch.Tensor | None) -> torch.Tensor | None:
    if scales is not None:
        if scales.numel() == 1:
            scales = scales.view(1, 1)
        else:
            scales = scales.view(-1, scales.size(-1))
    return scales


def normalize_batched_scales_shape(
    scales: torch.Tensor | None,
    num_experts: int,
) -> torch.Tensor | None:
    if scales is not None and scales.ndim < 3:
        if scales.numel() == 1:
            scales = scales.view(1)
            scales = torch.repeat_interleave(scales, num_experts, dim=0).view(
                num_experts, 1, 1
            )
        else:
            scales = scales.view(num_experts, -1, scales.size(-1))

    return scales


def _validate_scale_shape(
    a: torch.Tensor,
    a_scale: torch.Tensor | None,
    per_act_token_quant: bool,
    block_shape: list[int] | None,
) -> None:
    if a_scale is None:
        return

    if not per_act_token_quant and block_shape is None:
        assert a_scale.numel() == 1, f"{a_scale.shape}"
    elif per_act_token_quant:
        assert a_scale.shape[0] == a.shape[0] and a_scale.shape[1] == 1, (
            f"{a_scale.shape[0]} == {a.shape[0]} and {a_scale.shape[1]} == 1"
        )
    else:
        assert block_shape is not None
        expected = (a.shape[0], cdiv(a.shape[1], block_shape[1]))
        assert a_scale.shape == expected, f"{a_scale.shape} == {expected}"


def activation_without_mul(activation: str) -> str:
    return activation + "_no_mul"


RELU2_NO_MUL: str = activation_without_mul("relu2")
SILU_NO_MUL: str = activation_without_mul("silu")
GELU_NO_MUL: str = activation_without_mul("gelu")


def apply_moe_activation(
    activation: str,
    output: torch.Tensor,
    input: torch.Tensor,
) -> torch.Tensor:
    """
    Apply MoE activation function.

    For *_and_mul activations (silu, gelu, swigluoai):
        - Expects output.size(-1) * 2 == input.size(-1)

    For *_no_mul activations (silu_no_mul, gelu_no_mul, relu2_no_mul):
        - Expects output.size(-1) == input.size(-1)
    """
    is_no_mul = activation.endswith("_no_mul")
    if is_no_mul:
        assert output.size(-1) == input.size(-1), (
            f"{activation} expects equal sizes: {output.size(-1)} vs {input.size(-1)}"
        )
    else:
        assert output.size(-1) * 2 == input.size(-1), (
            f"{activation} expects 2x ratio: {output.size(-1) * 2} vs {input.size(-1)}"
        )

    # Activations with gated multiplication (gate × activation(up))
    if activation == "silu":
        torch.ops._C.silu_and_mul(output, input)
    elif activation == "gelu":
        torch.ops._C.gelu_and_mul(output, input)
    elif activation == "swigluoai":
        torch.ops._C.swigluoai_and_mul(output, input)
    elif activation == "swiglustep":
        from vllm.model_executor.layers.activation import swiglustep_and_mul_out
        swiglustep_and_mul_out(output, input)
    # Activations without gated multiplication
    elif activation == SILU_NO_MUL:
        output.copy_(F.silu(input))
    elif activation == GELU_NO_MUL:
        output.copy_(F.gelu(input))
    elif activation == RELU2_NO_MUL:
        F.relu(input, inplace=True)
        torch.square(input, out=output)
    else:
        raise ValueError(f"Unsupported FusedMoe activation: {activation}")

    return output


# Torch custom ops can't deal with outputs aliasing inputs so we need to
# disable inplace for torch >= 2.9.
# See https://github.com/vllm-project/vllm/issues/26378
@functools.cache
def disable_inplace() -> bool:
    return is_torch_equal_or_newer("2.9")
