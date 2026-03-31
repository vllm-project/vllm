# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import functools
from math import prod

import torch

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
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    per_tensor_dequantize,
)
from vllm.platforms import current_platform
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
    is_sf_swizzled_layout: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert A_scale is None
    assert not per_act_token_quant
    assert block_shape is None or block_shape == [1, 32]
    return mxfp8_e4m3_quantize(A, is_sf_swizzled_layout)


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
    ocp_mx_scheme: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    # Handle OCP MX scheme that requires QDQ (quantize-dequantize) for emulation
    if ocp_mx_scheme is not None:
        if ocp_mx_scheme in {"w_mxfp4", "w_mxfp4_a_mxfp4"}:
            pass  # No QDQ needed for these schemes
        elif ocp_mx_scheme.endswith("a_fp8"):
            # Perform QDQ (quantize and dequantize) on activation for emulation
            # purpose, because there is no native kernel for weight in ocp_mx_scheme
            # and activation in FP8. The implementation is based on existing
            # non-emulation ops.
            qA, qA_scale = ops.scaled_fp8_quant(
                A, A_scale, use_per_token_if_dynamic=False
            )
            A = per_tensor_dequantize(qA, qA_scale).to(A.dtype)
            # After QDQ, we don't need further quantization
            return A, None
        # else: For other schemes (e.g., *_a_mxfp6_e3m2, *_a_mxfp6_e2m3),
        # weights are already dequantized, and we proceed with normal
        # activation quantization below.

    if quant_dtype == current_platform.fp8_dtype():
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
        return _mxfp8_e4m3_quantize(
            A,
            A_scale,
            per_act_token_quant,
            block_shape,
            is_sf_swizzled_layout=is_fp4_scale_swizzled,
        )
    elif quant_dtype == "mxfp6_e3m2":
        return _mxfp6_e3m2_quantize(A, A_scale, per_act_token_quant, block_shape)
    elif quant_dtype == "mxfp6_e2m3":
        return _mxfp6_e2m3_quantize(A, A_scale, per_act_token_quant, block_shape)
    else:
        return A, A_scale


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


# Torch custom ops can't deal with outputs aliasing inputs so we need to
# disable inplace for torch >= 2.9.
# See https://github.com/vllm-project/vllm/issues/26378
@functools.cache
def disable_inplace() -> bool:
    return is_torch_equal_or_newer("2.9")


@triton.jit
def _pack_topk_ids_weights_kernel(
    topk_ids_ptr,
    topk_weights_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    USE_GDC: tl.constexpr,
    launch_pdl: tl.constexpr,  # triton metadata
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    if USE_GDC:
        tl.extra.cuda.gdc_launch_dependents()
        tl.extra.cuda.gdc_wait()
    expert_id = tl.load(topk_ids_ptr + offsets, mask=mask, other=0).to(tl.int32)
    expert_id_shifted = expert_id << 16

    weight = tl.load(topk_weights_ptr + offsets, mask=mask, other=0.0)
    weight_bf16 = weight.to(tl.bfloat16)
    weight_int16 = weight_bf16.to(tl.int16, bitcast=True)

    weight_int32 = weight_int16.to(tl.int32) & 0xFFFF

    packed = expert_id_shifted | weight_int32
    tl.store(output_ptr + offsets, packed, mask=mask)


def trtllm_moe_pack_topk_ids_weights(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    block_size: int = 1024,
) -> torch.Tensor:
    assert topk_ids.shape == topk_weights.shape
    assert topk_ids.is_contiguous() and topk_weights.is_contiguous()

    original_shape = topk_ids.shape
    ids_flat = topk_ids.reshape(-1)
    weights_flat = topk_weights.reshape(-1)

    n_elements = ids_flat.numel()
    output = torch.empty(n_elements, dtype=torch.int32, device=topk_ids.device)

    use_gdc = current_platform.is_cuda() and current_platform.has_device_capability(90)
    grid = (triton.cdiv(n_elements, block_size),)
    _pack_topk_ids_weights_kernel[grid](
        ids_flat,
        weights_flat,
        output,
        n_elements,
        BLOCK_SIZE=block_size,
        USE_GDC=use_gdc,
        launch_pdl=use_gdc,
    )
    return output.reshape(original_shape)


# @torch.compile(dynamic=True, backend=current_platform.simple_compile_backend)
# def trtllm_moe_pack_topk_ids_weights(
#     topk_ids: torch.Tensor, topk_weights: torch.Tensor
# ) -> torch.Tensor:
#     """
#     Pack topk_ids and topk_weights into a single int32 tensor.
#     Format: (expert_id << 16) | weight_bf16.view(int16)
#     """
#     return (topk_ids.to(torch.int32) << 16) | topk_weights.to(torch.bfloat16).view(
#         torch.int16
#     )
