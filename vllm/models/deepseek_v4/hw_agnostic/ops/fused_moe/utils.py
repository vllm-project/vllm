# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from math import prod

import torch
import torch.nn.functional as F

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv

# Note: only the FP8 quant kernel is imported here; the upstream
# vllm.model_executor.layers.fused_moe.utils also imported int8 /
# mxfp4 / mxfp6 / mxfp8 / nvfp4 / w8a8 quant kernels, but DSv4 with FP8
# experts exercises only the FP8 branch. The other ``_*_quantize``
# helpers and their corresponding ``moe_kernel_quantize_input``
# branches were dropped from this vendored copy. The
# ``quantization.utils.fp8_utils`` import is allowed via a dedicated
# lint carve-out (study doc §54) since the DSv4 path requires it and
# the FP8 quant primitives are needed by every backend that supports
# block-scaled FP8 MoE.


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


# Removed from this vendored copy: _int8_quantize / _mxfp4_quantize /
# _mxfp8_e4m3_quantize / _mxfp6_e3m2_quantize / _mxfp6_e2m3_quantize /
# _nvfp4_quantize. DSv4 hw-agnostic only exercises the FP8 path; the
# other dtypes are handled by registered quant-method subclasses
# upstream and aren't reached from this file.


def moe_kernel_quantize_input(
    A: torch.Tensor,
    A_scale: torch.Tensor | None,
    quant_dtype: None | torch.dtype | str,
    per_act_token_quant: bool,
    block_shape: list[int] | None = None,
    is_scale_swizzled: bool = True,
    ocp_mx_scheme: str | None = None,
    quantization_emulation: bool = False,
    mx_alignment: int = 0,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    # DSv4 hw-agnostic only exercises the FP8 quantization path. The
    # upstream copy supported nvfp4 / mxfp4 / mxfp8 / mxfp6_e3m2 /
    # mxfp6_e2m3 / int8 / OCP-MX QDQ schemes — these branches were
    # dropped here. Other dtypes raise NotImplementedError.
    if ocp_mx_scheme is not None:
        raise NotImplementedError(
            f"OCP MX schemes ({ocp_mx_scheme!r}) are not supported on the "
            "DSv4 hw-agnostic FusedMoE path."
        )

    if quant_dtype == current_platform.fp8_dtype():
        if quantization_emulation:
            raise NotImplementedError(
                f"moe_kernel_quantize_input does not support quant_dtype={quant_dtype}"
                " MOE quantization emulation. Please open an issue."
            )
        return _fp8_quantize(A, A_scale, per_act_token_quant, block_shape)
    elif quant_dtype is None:
        return A, A_scale
    else:
        raise NotImplementedError(
            f"DSv4 hw-agnostic FusedMoE supports only FP8 / unquantized "
            f"experts; got quant_dtype={quant_dtype!r}."
        )


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


@torch.compile(dynamic=True, backend=current_platform.simple_compile_backend)
def swiglu_limit_func(
    output: torch.Tensor,
    input: torch.Tensor,  # first half is gate, second half is up
    swiglu_limit: float = 0.0,
) -> None:
    d = input.shape[1] // 2
    gate = input[:, :d]
    up = input[:, d:]

    if swiglu_limit > 0:
        gate = torch.clamp(gate, max=swiglu_limit)
        up = torch.clamp(up, min=-swiglu_limit, max=swiglu_limit)

    output.copy_(F.silu(gate) * up)
