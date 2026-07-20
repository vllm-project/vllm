# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4 XPU FP8 BMM op wrapper.

Registers a vLLM custom op so model code can call ``torch.ops.vllm`` instead
of directly invoking ``torch.ops._xpu_C``.
"""

import torch

from vllm.utils.torch_utils import direct_register_custom_op


def _xpu_fp8_bmm_impl(
    a: torch.Tensor,
    b: torch.Tensor,
    out_dtype: torch.dtype,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    return torch.ops._xpu_C.fp8_bmm(a, b, out_dtype, a_scale, b_scale, bias)


def _xpu_fp8_bmm_fake(
    a: torch.Tensor,
    b: torch.Tensor,
    out_dtype: torch.dtype,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    # [G, M, K] @ [G, K, N] => [G, M, N]
    return torch.empty(
        (a.shape[0], a.shape[1], b.shape[2]),
        dtype=out_dtype,
        device=a.device,
    )


direct_register_custom_op(
    op_name="xpu_fp8_bmm",
    op_func=_xpu_fp8_bmm_impl,
    fake_impl=_xpu_fp8_bmm_fake,
)


def xpu_fp8_bmm(
    a: torch.Tensor,
    b: torch.Tensor,
    out_dtype: torch.dtype,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """XPU FP8 batched GEMM wrapper registered as ``torch.ops.vllm.xpu_fp8_bmm``.

    Computes batched matrix multiplication over the leading group dimension:
    ``[G, M, K] @ [G, K, N] -> [G, M, N]``.

    Args:
        a: FP8 activation tensor with shape ``[G, M, K]``.
            Does not need to be contiguous;
        b: FP8 weight tensor with shape ``[G, K, N]``.
            Does not need to be contiguous.
        out_dtype: Output dtype accepted by the kernel (typically
            ``torch.bfloat16`` for DeepSeek-V4 O-proj path).
        a_scale: Activation scale tensor for ``a``.
            In current DeepSeek-V4 XPU usage it is block-scaled with shape
            ``[G, M, K/bs]`` (``bs`` is the quant block size, e.g. 128).
            Must be contiguous.
        b_scale: Weight scale tensor for ``b``.
            In current DeepSeek-V4 XPU usage it is block-scaled with shape
            ``[G, K/bs, N/bs]`` (``bs`` is the quant block size, e.g. 128).
            Must be contiguous.
        bias: Optional bias tensor. Pass ``None`` when no bias is required.

    Returns:
        Output tensor with shape ``[G, M, N]`` and dtype ``out_dtype``.

    Notes:
        - This API centralizes access to ``torch.ops._xpu_C.fp8_bmm``.
        - For DeepSeek-V4 O-proj callers, both ``a_scale`` and ``b_scale``
            should be contiguous. ``a`` and ``b`` may be non-contiguous views.
    """
    return torch.ops.vllm.xpu_fp8_bmm(a, b, out_dtype, a_scale, b_scale, bias)
