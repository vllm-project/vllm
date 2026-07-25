# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import functools
from math import prod
from typing import TYPE_CHECKING

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
    xpu_mxfp4_quantize,
)
from vllm.model_executor.layers.quantization.utils.mxfp6_utils import (
    quant_dequant_mxfp6,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    mxfp8_e4m3_quantize,
    xpu_mxfp8_quantize,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    ref_nvfp4_quant_dequant,
)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    per_tensor_dequantize,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig


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
        if per_act_token:
            A, A_scale = per_token_quant_int8(A)
        elif A_scale is not None:
            # Static per-tensor: use the optimized CUDA kernel
            A, A_scale, _ = ops.scaled_int8_quant(A, scale=A_scale)
        elif A_scale is None:
            # Dynamic per-tensor: compute scale then quantize via kernel
            A_scale = torch.clamp(A.abs().max() / 127.0, min=1e-10)
            A, A_scale, _ = ops.scaled_int8_quant(A, scale=A_scale)
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
    if current_platform.is_xpu():
        return xpu_mxfp4_quantize(A)
    assert block_shape is None
    # TODO: native mxfp4 is currently not integrated in vllm,
    # so simulating even on devices supporting this data type natively.
    # Once integrated, `current_platform.supports_mx()` should be used to
    # control quantize+dequantize, or simply quantize here down to mxfp4.
    A = quant_dequant_mxfp4(A)

    return A, None


def _fp8_quantize_dequantize(
    A: torch.Tensor,
    A_scale: torch.Tensor,
):
    qA, qA_scale = ops.scaled_fp8_quant(A, A_scale, use_per_token_if_dynamic=False)
    A = per_tensor_dequantize(qA, qA_scale).to(A.dtype)

    return A, None


def _mxfp8_e4m3_quantize(
    A: torch.Tensor,
    A_scale: torch.Tensor | None,
    per_act_token_quant: bool,
    block_shape: list[int] | None = None,
    is_sf_swizzled_layout: bool = False,
    mx_alignment: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if current_platform.is_xpu():
        return xpu_mxfp8_quantize(A)
    assert A_scale is None
    assert not per_act_token_quant
    assert block_shape is None or block_shape == [1, 32]
    return mxfp8_e4m3_quantize(A, is_sf_swizzled_layout, mx_alignment)


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
    is_scale_swizzled: bool = True,
    ocp_mx_scheme: str | None = None,
    quantization_emulation: bool = False,
    mx_alignment: int = 0,
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
            # TODO: Remove this `ocp_mx_scheme is not None` block and rely solely
            # on `quantization_emulation`.
            return _fp8_quantize_dequantize(A, A_scale)
        # else: For other schemes (e.g., *_a_mxfp6_e3m2, *_a_mxfp6_e2m3),
        # weights are already dequantized, and we proceed with normal
        # activation quantization below.
    if quant_dtype == current_platform.fp8_dtype():
        if quantization_emulation:
            return _fp8_quantize_dequantize(A, A_scale)
        else:
            return _fp8_quantize(A, A_scale, per_act_token_quant, block_shape)
    elif quant_dtype == torch.int8:
        if quantization_emulation:
            raise NotImplementedError(
                "moe_kernel_quantize_input does not support quant_dtype=torch.int8"
                " MOE quantization emulation. Please open an issue."
            )
        return _int8_quantize(A, A_scale, per_act_token_quant, block_shape)
    elif quant_dtype == "nvfp4":
        if not quantization_emulation:
            return _nvfp4_quantize(A, A_scale, is_sf_swizzled_layout=is_scale_swizzled)
        else:
            assert A_scale is not None
            A = ref_nvfp4_quant_dequant(A, A_scale, block_size=16)
            return A, None
    elif quant_dtype == "mxfp4":
        if not current_platform.is_xpu() and not quantization_emulation:
            raise NotImplementedError(
                "moe_kernel_quantize_input should not be used for native"
                " quant_dtype='mxfp4' MOE. Please open an issue."
            )
        return _mxfp4_quantize(A, A_scale, per_act_token_quant, block_shape)
    elif quant_dtype == "mxfp8":
        # TODO: `quant_dtype == "mxfp8"` is ambiguous,
        # should be fp8_e4m3. OCP MX also defines `fp8_e5m2`.
        if not current_platform.is_xpu() and quantization_emulation:
            raise NotImplementedError(
                "moe_kernel_quantize_input does not support quant_dtype='mxfp8' MOE "
                "quantization emulation. Please open an issue."
            )
        # Non-swizzled (M, K/32) uint8 UE8M0 scales; deepgemm_moe_permute packs
        # them for DeepGEMM, TRTLLM takes them as-is.
        return _mxfp8_e4m3_quantize(
            A,
            A_scale,
            per_act_token_quant,
            block_shape,
            is_sf_swizzled_layout=is_scale_swizzled,
            mx_alignment=mx_alignment,
        )
    elif quant_dtype == "mxfp6_e3m2":
        if not quantization_emulation:
            raise NotImplementedError(
                "moe_kernel_quantize_input should not be used for native "
                " quant_dtype='mxfp6_e3m2'MOE. Please open an issue."
            )

        return _mxfp6_e3m2_quantize(A, A_scale, per_act_token_quant, block_shape)
    elif quant_dtype == "mxfp6_e2m3":
        if not quantization_emulation:
            raise NotImplementedError(
                "moe_kernel_quantize_input should not be used for native"
                " quant_dtype='mxfp6_e2m3' MOE. Please open an issue."
            )

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


def fi_moe_largest_bucket(moe_config: "FusedMoEConfig") -> int:
    """Estimate FlashInfer's MoE autotuning maximum token count.

    All DP ranks may contribute `max_num_tokens` to one invocation.
    Keep FlashInfer's default moe `tune_max_num_tokens=8192`
    floor to avoid over-underestimation.
    DeepEP, SP, or PCP may make this underestimate, however overestimation
    may be dangerous, increasing tuning- cost and memory use.

    NOTE: The DP factor applies even when EP is disabled:
    > Without `--enable-expert-parallel`, MoE layers would use tensor parallelism.

    For a detailed explanation, see: `docs/serving/data_parallel_deployment.md`
    """
    return max(moe_config.max_num_tokens * moe_config.dp_size, 8192)


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
def _swiglu_limit_torch(
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


@triton.jit
def _swiglu_limit_pad_aware_kernel(
    input_ptr,  # [num_tokens, 2 * hidden_size]
    output_ptr,  # [num_tokens, hidden_size]
    topk_ids_ptr,  # [num_tokens, num_topk]
    expert_map_ptr,  # global -> local expert id, or -1 if non-local
    hidden_size,
    input_row_stride,
    num_tokens,
    swiglu_limit,
    HAS_LIMIT: tl.constexpr,
    HAS_EXPERT_MAP: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Persistent over rows: each CTA owns one column tile and processes a
    # strided set of token assignments.
    pid = tl.program_id(0)
    row_stride = tl.num_programs(0)
    column_tile = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = column_tile < hidden_size

    for row in tl.range(pid, num_tokens, row_stride):
        expert_id = tl.load(topk_ids_ptr + row)
        should_compute = expert_id != -1
        if HAS_EXPERT_MAP:
            local_expert_id = tl.load(
                expert_map_ptr + expert_id,
                mask=expert_id >= 0,
                other=-1,
            )
            should_compute = should_compute & (local_expert_id != -1)

        if should_compute:
            gate_offsets = row.to(tl.int64) * input_row_stride + column_tile
            up_offsets = gate_offsets + hidden_size

            gate = tl.load(input_ptr + gate_offsets, mask=mask, other=0.0).to(
                tl.float32
            )

            up = tl.load(input_ptr + up_offsets, mask=mask, other=0.0).to(tl.float32)

            if HAS_LIMIT:
                gate = tl.minimum(gate, swiglu_limit)
                up = tl.maximum(up, -swiglu_limit)
                up = tl.minimum(up, swiglu_limit)

            silu_gate = gate / (1.0 + tl.exp(-gate))
            result = silu_gate * up
            tl.store(
                output_ptr + row.to(tl.int64) * hidden_size + column_tile,
                result.to(output_ptr.dtype.element_ty),
                mask=mask,
            )


def _swiglu_limit_pad_aware(
    output: torch.Tensor,
    input: torch.Tensor,
    topk_ids: torch.Tensor,
    swiglu_limit: float,
    expert_map: torch.Tensor | None = None,
) -> None:
    num_tokens, gate_up_size = input.shape
    hidden_size = gate_up_size // 2
    if num_tokens == 0:
        return

    BLOCK_SIZE = 1024
    grid = (min(num_tokens, 256), triton.cdiv(hidden_size, BLOCK_SIZE))
    _swiglu_limit_pad_aware_kernel[grid](
        input,
        output,
        topk_ids,
        expert_map,
        hidden_size,
        gate_up_size,
        num_tokens,
        swiglu_limit,
        HAS_LIMIT=swiglu_limit > 0,
        HAS_EXPERT_MAP=expert_map is not None,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )


def swiglu_limit_func(
    output: torch.Tensor,
    input: torch.Tensor,  # first half is gate, second half is up
    swiglu_limit: float = 0.0,
    topk_ids: torch.Tensor | None = None,
    expert_map: torch.Tensor | None = None,
) -> None:
    # The pad-aware Triton kernel skips unrouted token slots (topk_ids == -1)
    # and, when expert_map is given, slots routed to non-local experts, so it
    # requires topk_ids. Fall back to the torch implementation otherwise.
    if topk_ids is not None:
        _swiglu_limit_pad_aware(output, input, topk_ids, swiglu_limit, expert_map)
    else:
        _swiglu_limit_torch(output, input, swiglu_limit)


@functools.lru_cache
def enable_swap_ab(BLOCK_SIZE_M: int, BLOCK_SIZE_N: int) -> bool:
    return (
        current_platform.is_device_capability(90)
        and BLOCK_SIZE_M < 64
        and BLOCK_SIZE_N >= 64
    )
