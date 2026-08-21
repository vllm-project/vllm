# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM89/SM12x Triton FP8 einsum kernels for DeepSeek V4."""

import torch

from vllm.distributed import get_tensor_model_parallel_rank
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    _upcast_e8m0_to_fp32,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.deep_gemm import fp8_einsum


@triton.jit(
    do_not_specialize=[
        "num_tokens",
        "a_stride_group",
        "a_scale_stride_group",
        "a_scale_stride_hidden",
    ]
)
def _deepseek_v4_sm12x_fp8_einsum_kernel(
    a_ptr,
    a_scale_ptr,
    b_ptr,
    b_scale_ptr,
    out_ptr,
    num_tokens,
    num_groups: tl.constexpr,
    out_rank: tl.constexpr,
    hidden_size: tl.constexpr,
    a_stride_token: tl.constexpr,
    a_stride_group,
    a_stride_hidden: tl.constexpr,
    a_scale_stride_token: tl.constexpr,
    a_scale_stride_group,
    a_scale_stride_hidden,
    b_stride_group: tl.constexpr,
    b_stride_out: tl.constexpr,
    b_stride_hidden: tl.constexpr,
    b_scale_stride_group: tl.constexpr,
    b_scale_stride_out: tl.constexpr,
    b_scale_stride_hidden: tl.constexpr,
    out_stride_token: tl.constexpr,
    out_stride_group: tl.constexpr,
    out_stride_rank: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    BLOCK_OUT: tl.constexpr,
    BLOCK_HIDDEN: tl.constexpr,
    UPCAST_FP8: tl.constexpr,
) -> None:
    # Preserve alignment without specializing on request-dependent strides.
    a_stride_group = tl.multiple_of(a_stride_group, 16)
    a_scale_stride_group = tl.multiple_of(a_scale_stride_group, 16)
    token_block = tl.program_id(0)
    out_block = tl.program_id(1)
    group = tl.program_id(2)

    token_offsets = token_block * BLOCK_TOKENS + tl.arange(0, BLOCK_TOKENS)
    out_offsets = out_block * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
    hidden_offsets = tl.arange(0, BLOCK_HIDDEN)
    accum = tl.zeros((BLOCK_TOKENS, BLOCK_OUT), dtype=tl.float32)

    for hidden_start in range(0, hidden_size, BLOCK_HIDDEN):
        hidden = hidden_start + hidden_offsets
        a = tl.load(
            a_ptr
            + token_offsets[:, None] * a_stride_token
            + group * a_stride_group
            + hidden[None, :] * a_stride_hidden,
            mask=(token_offsets[:, None] < num_tokens)
            & (hidden[None, :] < hidden_size),
            other=0.0,
        )
        b = tl.load(
            b_ptr
            + group * b_stride_group
            + out_offsets[None, :] * b_stride_out
            + hidden[:, None] * b_stride_hidden,
            mask=(out_offsets[None, :] < out_rank) & (hidden[:, None] < hidden_size),
            other=0.0,
        )
        if UPCAST_FP8:
            # SM89/Ada: Triton may not lower an FP8-operand tl.dot on this arch,
            # so upcast to bf16 before the MMA. SM12x keeps the native FP8 dot.
            a = a.to(tl.bfloat16)
            b = b.to(tl.bfloat16)
        raw = tl.dot(a, b, out_dtype=tl.float32)
        hidden_scale_block = hidden_start // BLOCK_HIDDEN
        a_scale = tl.load(
            a_scale_ptr
            + token_offsets * a_scale_stride_token
            + group * a_scale_stride_group
            + hidden_scale_block * a_scale_stride_hidden,
            mask=token_offsets < num_tokens,
            other=0.0,
        )
        b_scale = tl.load(
            b_scale_ptr
            + group * b_scale_stride_group
            + (out_offsets // 128) * b_scale_stride_out
            + hidden_scale_block * b_scale_stride_hidden,
            mask=out_offsets < out_rank,
            other=0.0,
        )
        accum += raw * a_scale[:, None] * b_scale[None, :]

    tl.store(
        out_ptr
        + token_offsets[:, None] * out_stride_token
        + group * out_stride_group
        + out_offsets[None, :] * out_stride_rank,
        accum,
        mask=(token_offsets[:, None] < num_tokens) & (out_offsets[None, :] < out_rank),
    )


def deepseek_v4_sm12x_fp8_einsum(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """Compute ``bhr,hdr->bhd`` with FP32 block scales on SM89/SM12x.

    ``a`` is the transposed output of ``fused_inv_rope_fp8_quant`` with shape
    ``[tokens, groups, hidden]``. ``b`` is ``wo_a`` reshaped to
    ``[groups, out_rank, hidden]``.
    """
    num_tokens, num_groups, hidden_size = a.shape
    b_groups, out_rank, b_hidden_size = b.shape
    assert b_groups == num_groups
    assert b_hidden_size == hidden_size
    assert out.shape == (num_tokens, num_groups, out_rank)
    assert hidden_size % 128 == 0
    assert out_rank % 128 == 0
    assert a.dtype == torch.float8_e4m3fn
    assert b.dtype == torch.float8_e4m3fn
    e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
    if a_scale.dtype == e8m0_dtype:
        a_scale = _upcast_e8m0_to_fp32(a_scale)
    if b_scale.dtype == e8m0_dtype:
        b_scale = _upcast_e8m0_to_fp32(b_scale)
    assert a_scale.dtype == torch.float32
    assert b_scale.dtype == torch.float32

    if num_tokens == 0:
        return

    block_tokens = 16
    block_out = 128
    block_hidden = 128
    # SM12x has native FP8 tensor-core dot; SM89/Ada upcasts FP8->bf16 first.
    upcast_fp8 = not current_platform.is_device_capability_family(120)
    grid = (
        triton.cdiv(num_tokens, block_tokens),
        triton.cdiv(out_rank, block_out),
        num_groups,
    )
    _deepseek_v4_sm12x_fp8_einsum_kernel[grid](
        a,
        a_scale,
        b,
        b_scale,
        out,
        num_tokens,
        num_groups,
        out_rank,
        hidden_size,
        a.stride(0),
        a.stride(1),
        a.stride(2),
        a_scale.stride(0),
        a_scale.stride(1),
        a_scale.stride(2),
        b.stride(0),
        b.stride(1),
        b.stride(2),
        b_scale.stride(0),
        b_scale.stride(1),
        b_scale.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        BLOCK_TOKENS=block_tokens,
        BLOCK_OUT=block_out,
        BLOCK_HIDDEN=block_hidden,
        UPCAST_FP8=upcast_fp8,
        num_warps=4,
        num_stages=3,
    )


def deepseek_v4_fp8_einsum_config(
    capability_major: int,
) -> tuple[tuple[int, int, int], bool]:
    if capability_major == 10:
        return (1, 1, 128), True
    return (1, 128, 128), False


def _deepseek_v4_sm120_cutlass_compiled(capability: int) -> bool:
    try:
        return torch.ops._C.deepseek_v4_fp8_bmm_sm120_supported(capability)
    except AttributeError:
        return False


def use_deepseek_v4_sm120_cutlass_fp8_einsum(
    num_tokens: int,
    num_groups: int,
    out_rank: int,
    hidden_size: int,
) -> bool:
    capability = current_platform.get_device_capability()
    return (
        capability is not None
        and capability.major == 12
        and _deepseek_v4_sm120_cutlass_compiled(capability.to_int())
        and num_tokens >= 256
        and num_groups in (2, 8)
        and out_rank == 1024
        and hidden_size == 4096
    )


def _use_deepseek_v4_sm12x_triton_fp8_einsum(
    equation: str,
    recipe: list[int],
    b_scale: torch.Tensor,
) -> bool:
    capability = current_platform.get_device_capability()
    e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
    # SM12x (major 12) and SM89/Ada (8.9) both use the Triton FP8 einsum here:
    # they lack the SM90/SM100-only DeepGEMM fp8_einsum kernel.
    is_supported_arch = capability is not None and (
        capability.major == 12 or (capability.major, capability.minor) == (8, 9)
    )
    return (
        is_supported_arch
        and equation == "bhr,hdr->bhd"
        and tuple(recipe) == (1, 128, 128)
        and b_scale.dtype in (torch.float32, e8m0_dtype)
    )


def deepseek_v4_fp8_einsum(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    out: torch.Tensor,
    equation: str,
    recipe: list[int],
) -> None:
    if equation == "bhr,hdr->bhd" and b.dim() == 2:
        num_groups = out.shape[1]
        out_rank = out.shape[2]
        hidden_size = a.shape[2]
        if b.shape[0] % out_rank != 0:
            raise RuntimeError(
                "DeepSeek V4 fp8 einsum weight rows must be divisible by "
                f"out_rank={out_rank}, got {b.shape[0]}"
            )
        b_groups = b.shape[0] // out_rank
        group_start = 0
        if b_groups != num_groups:
            if b_groups % num_groups != 0:
                raise RuntimeError(
                    "DeepSeek V4 fp8 einsum weight groups must match the "
                    "TP-local output groups or be an integer multiple of "
                    f"them, got weight_groups={b_groups}, "
                    f"output_groups={num_groups}"
                )
            group_partitions = b_groups // num_groups
            group_start = (
                get_tensor_model_parallel_rank() % group_partitions
            ) * num_groups
        b = b.view(b_groups, out_rank, hidden_size)
        if group_start != 0 or b_groups != num_groups:
            b = b.narrow(0, group_start, num_groups)

        if b_scale.dim() == 2:
            scale_mn = recipe[1]
            scale_k_pack = 4 if b_scale.dtype == torch.int32 else 1
            scale_k = recipe[2] * scale_k_pack
            scale_out_blocks = (out_rank + scale_mn - 1) // scale_mn
            scale_hidden_blocks = (hidden_size + scale_k - 1) // scale_k
            if b_scale.shape[0] % scale_out_blocks != 0:
                raise RuntimeError(
                    "DeepSeek V4 fp8 einsum scale rows must be divisible by "
                    f"scale_out_blocks={scale_out_blocks}, "
                    f"got {b_scale.shape[0]}"
                )
            scale_groups = b_scale.shape[0] // scale_out_blocks
            if scale_groups not in (num_groups, b_groups):
                raise RuntimeError(
                    "DeepSeek V4 fp8 einsum scale groups must match the "
                    "TP-local output groups or weight groups, got "
                    f"scale_groups={scale_groups}, output_groups={num_groups}, "
                    f"weight_groups={b_groups}"
                )
            b_scale = b_scale.view(
                scale_groups,
                scale_out_blocks,
                scale_hidden_blocks,
            )
            if scale_groups == b_groups and scale_groups != num_groups:
                b_scale = b_scale.narrow(0, group_start, num_groups)
        elif b_scale.dim() == 3 and b_scale.shape[0] == b_groups:
            if b_groups != num_groups:
                b_scale = b_scale.narrow(0, group_start, num_groups)

        use_sm120_cutlass = (
            tuple(recipe) == (1, 128, 128)
            and a_scale.dtype == torch.float32
            and b_scale.dtype == torch.float32
            and use_deepseek_v4_sm120_cutlass_fp8_einsum(
                a.shape[0],
                num_groups,
                out_rank,
                hidden_size,
            )
        )
        if use_sm120_cutlass:
            a_group_major = a.transpose(0, 1)
            a_scale_cutlass = a_scale.transpose(0, 1).permute(0, 2, 1)
            b_cutlass = b.transpose(1, 2)
            if (
                a_group_major.is_contiguous()
                and a_scale_cutlass.is_contiguous()
                and b_scale.is_contiguous()
                and out.is_contiguous()
            ):
                torch.ops._C.deepseek_v4_fp8_bmm_sm120(
                    out,
                    a_group_major,
                    b_cutlass,
                    a_scale_cutlass,
                    b_scale,
                )
                return

        if _use_deepseek_v4_sm12x_triton_fp8_einsum(equation, recipe, b_scale):
            deepseek_v4_sm12x_fp8_einsum(a, a_scale, b, b_scale, out)
            return

    fp8_einsum(equation, (a, a_scale), (b, b_scale), out, recipe=tuple(recipe))
