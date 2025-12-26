# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import functools
from functools import lru_cache
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
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.flashinfer import flashinfer_fp4_quantize
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
    return flashinfer_fp4_quantize(
        A, A_scale, is_sf_swizzled_layout=is_sf_swizzled_layout
    )


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


# Torch custom ops can't deal with outputs aliasing inputs so we need to
# disable inplace for torch >= 2.9.
# See https://github.com/vllm-project/vllm/issues/26378
@functools.cache
def disable_inplace() -> bool:
    return is_torch_equal_or_newer("2.9")


@lru_cache
def supports_pdl(device: torch.device | None = None) -> bool:
    """
    Refer to: https://github.com/triton-lang/triton/blob/v3.5.0/python/tutorials/11-programmatic-dependent-launch.py
    """
    # PDL requires compute capability SM90 or above
    return current_platform.is_cuda() and current_platform.has_device_capability(90)


@triton.jit
def _update_accumulator(
    accumulator,
    tiled_a,
    tiled_b,
    token_mask,
    iter_k,
    a_scale_ptrs,
    b_scale_ptrs,
    stride_ask,
    stride_bsk,
    group_k,
    group_n,
    use_int8_w8a16: tl.constexpr,
    use_fp8_w8a8: tl.constexpr,
    use_int8_w8a8: tl.constexpr,
    compute_type: tl.constexpr = tl.float16,
):
    if use_int8_w8a16:
        accumulator = tl.dot(tiled_a, tiled_b.to(compute_type), acc=accumulator)
    elif use_fp8_w8a8 or use_int8_w8a8:
        if group_k > 0 and group_n > 0:
            offs_ks = iter_k // group_k
            a_scale = tl.load(
                a_scale_ptrs + offs_ks * stride_ask, mask=token_mask, other=0.0
            )
            b_scale = tl.load(b_scale_ptrs + offs_ks * stride_bsk)

            accumulator += (
                tl.dot(tiled_a, tiled_b) * a_scale[:, None] * b_scale[None, :]
            )
        else:
            if use_fp8_w8a8:
                # acc used to enable fp8_fast_accum
                accumulator = tl.dot(tiled_a, tiled_b, acc=accumulator)
            else:
                accumulator += tl.dot(tiled_a, tiled_b)
    else:
        accumulator += tl.dot(tiled_a, tiled_b)
    return accumulator


@triton.jit
def mm_k(
    a_ptrs,
    b_ptrs,
    ak_stride,
    bk_stride,
    token_mask,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    CAST_TYPE: tl.constexpr,
    b_dtype: tl.constexpr,
    USE_GDC: tl.constexpr,
    IS_PRIMARY: tl.constexpr,
    base_k,
    a_scale_ptrs=None,
    b_scale_ptrs=None,
    stride_ask=0,
    stride_bsk=0,
    group_k=0,
    group_n=0,
    use_int8_w8a16: tl.constexpr = False,
    use_fp8_w8a8: tl.constexpr = False,
    use_int8_w8a8: tl.constexpr = False,
    compute_type: tl.constexpr = tl.float16,
):
    """
    Given a_ptrs and b_ptrs, that identify the rows of A (m x k) and columns of
    B (k x n), iterate, through the K dimension to compute the partial/complete
    matrix block product.
    If SPLIT_K == 1, the output m x n product is complete.
    If SPLIT_K > 1, the thread block computes partial outputs. The partial
    outputs are then atomically summed in the caller code.
    Args:
        a_ptrs: Array of pointers, identifying rows of A
        b_ptrs: Array of pointers, identifying columns of B
        ak_stride: K dimension stride of the A matrix
        bk_stride: K dimension stride of the B matrix
        K: Length of the K dimension
        BLOCK_M: M dimension of the output block m x n
        BLOCK_N: N dimension of the output block m x n
        BLOCK_K: K dimension atom
        EVEN_K: True if the blocks of A and B can be loaded without any
          masking.
        SPLIT_K: Parameter signifying parallelism in the K dimension.
        CAST_TYPE: if True, cast the values from the A matrix to the B
          matrix dtype.
        b_dtype: datatype of the B matrix
        USE_GDC: Whether to use PDL. True indicates use.
        base_k: Base offset along K dimension for current SPLIT_K group
    """
    if USE_GDC and IS_PRIMARY:
        # GDC launch dependents hints the runtime system to launch dependent kernels.
        tl.extra.cuda.gdc_launch_dependents()

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    if USE_GDC and not IS_PRIMARY:
        tl.extra.cuda.gdc_wait()

    # Step size along K for each iteration
    STEP_K = BLOCK_K * SPLIT_K

    # Total number of iterations (compile-time constant)
    num_iters = tl.cdiv(K, STEP_K)

    for k in range(num_iters):
        # Current iteration's global K offset
        iter_k = k * STEP_K + base_k

        # Check if this iteration is completely valid (no masking needed)
        block_end = iter_k + BLOCK_K

        if EVEN_K:
            # K is divisible by BLOCK_K, no masking ever needed
            # pre-fetch lora weight
            tiled_b = tl.load(b_ptrs)
            if USE_GDC and not IS_PRIMARY:
                tl.extra.cuda.gdc_wait()
            tiled_a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
            if CAST_TYPE:
                tiled_a = tiled_a.to(b_dtype)
            accumulator = _update_accumulator(
                accumulator=accumulator,
                tiled_a=tiled_a,
                tiled_b=tiled_b,
                token_mask=token_mask,
                iter_k=iter_k,
                a_scale_ptrs=a_scale_ptrs,
                b_scale_ptrs=b_scale_ptrs,
                stride_ask=stride_ask,
                stride_bsk=stride_bsk,
                group_k=group_k,
                group_n=group_n,
                use_int8_w8a16=use_int8_w8a16,
                use_fp8_w8a8=use_fp8_w8a8,
                use_int8_w8a8=use_int8_w8a8,
                compute_type=compute_type,
            )
        else:
            # Check if we need element-wise masking
            if iter_k >= K:
                # Entire block out of range, skip
                pass
            elif block_end <= K:
                # Entire block in range, no masking needed (fast path)
                tiled_b = tl.load(b_ptrs)
                if USE_GDC and not IS_PRIMARY:
                    tl.extra.cuda.gdc_wait()
                tiled_a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
                if CAST_TYPE:
                    tiled_a = tiled_a.to(b_dtype)
                accumulator = _update_accumulator(
                    accumulator=accumulator,
                    tiled_a=tiled_a,
                    tiled_b=tiled_b,
                    token_mask=token_mask,
                    iter_k=iter_k,
                    a_scale_ptrs=a_scale_ptrs,
                    b_scale_ptrs=b_scale_ptrs,
                    stride_ask=stride_ask,
                    stride_bsk=stride_bsk,
                    group_k=group_k,
                    group_n=group_n,
                    use_int8_w8a16=use_int8_w8a16,
                    use_fp8_w8a8=use_fp8_w8a8,
                    use_int8_w8a8=use_int8_w8a8,
                    compute_type=compute_type,
                )
            else:
                # Partial block, need masking (only last iteration)
                k_offsets = tl.arange(0, BLOCK_K)
                mask = iter_k + k_offsets < K
                tiled_b = tl.load(b_ptrs, mask=mask[:, None], other=0.0)
                if USE_GDC and not IS_PRIMARY:
                    tl.extra.cuda.gdc_wait()
                tiled_a = tl.load(
                    a_ptrs, mask=token_mask[:, None] & mask[None, :], other=0.0
                )
                if CAST_TYPE:
                    tiled_a = tiled_a.to(b_dtype)
                accumulator = _update_accumulator(
                    accumulator=accumulator,
                    tiled_a=tiled_a,
                    tiled_b=tiled_b,
                    token_mask=token_mask,
                    iter_k=iter_k,
                    a_scale_ptrs=a_scale_ptrs,
                    b_scale_ptrs=b_scale_ptrs,
                    stride_ask=stride_ask,
                    stride_bsk=stride_bsk,
                    group_k=group_k,
                    group_n=group_n,
                    use_int8_w8a16=use_int8_w8a16,
                    use_fp8_w8a8=use_fp8_w8a8,
                    use_int8_w8a8=use_int8_w8a8,
                    compute_type=compute_type,
                )

        a_ptrs += STEP_K * ak_stride
        b_ptrs += STEP_K * bk_stride

    return accumulator
