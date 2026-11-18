# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-pass Triton RMSNorm for an fp32 weight on a bf16/fp16 activation.

GemmaRMSNorm passes ``weight.float() + 1.0``, which the vllm_c kernels do not
accept, so under Inductor these calls lower to a looped reduction. One program
per row reads it once and writes the outputs in the same pass, launched with
programmatic dependent launch where the GPU supports it.
"""

import torch
from torch import Tensor

from vllm import ir
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON, tl, tldevice, triton
from vllm.utils.torch_utils import direct_register_custom_op

_MAX_HIDDEN_SIZE = 8192
_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)


@triton.jit
def _rms_norm_kernel(
    x_ptr,
    residual_ptr,
    weight_ptr,
    out_ptr,
    residual_out_ptr,
    eps,
    H: tl.constexpr,
    BLOCK: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    # Every operand may be written by the previous kernel, the weight included
    # (Inductor computes the Gemma offset weight), so all loads follow the wait.
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK)
    mask = offs < H
    base = row * H
    h = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    if HAS_RESIDUAL:
        h += tl.load(residual_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(
            residual_out_ptr + base + offs,
            h.to(residual_out_ptr.dtype.element_ty),
            mask=mask,
        )
    w = tl.load(weight_ptr + offs, mask=mask, other=0.0)
    rstd = tldevice.rsqrt(tl.sum(h * h, axis=0) / H + eps)
    y = h * rstd * w
    # Dependents read the outputs only after their own wait, which returns
    # once this grid has completed.
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()
    tl.store(out_ptr + base + offs, y.to(out_ptr.dtype.element_ty), mask=mask)


def _launch(
    x: Tensor, residual: Tensor | None, weight: Tensor, epsilon: float
) -> tuple[Tensor, Tensor | None]:
    hidden_size = x.shape[-1]
    num_rows = x.numel() // hidden_size
    out = torch.empty_like(x)
    residual_out = None if residual is None else torch.empty_like(x)
    if num_rows == 0:
        return out, residual_out
    block = triton.next_power_of_2(hidden_size)
    use_pdl = current_platform.is_arch_support_pdl()
    _rms_norm_kernel[(num_rows,)](
        x,
        x if residual is None else residual,
        weight,
        out,
        out if residual_out is None else residual_out,
        epsilon,
        H=hidden_size,
        BLOCK=block,
        HAS_RESIDUAL=residual is not None,
        USE_PDL=use_pdl,
        num_warps=min(max(block // 256, 1), 8),
        **({"launch_pdl": True} if use_pdl else {}),
    )
    return out, residual_out


def _rms_norm_triton(x: Tensor, weight: Tensor, epsilon: float) -> Tensor:
    return _launch(x, None, weight, epsilon)[0]


def _fused_add_rms_norm_triton(
    x: Tensor, residual: Tensor, weight: Tensor, epsilon: float
) -> tuple[Tensor, Tensor]:
    return _launch(x, residual, weight, epsilon)


def _rms_norm_triton_fake(x: Tensor, weight: Tensor, epsilon: float) -> Tensor:
    return torch.empty_like(x)


def _fused_add_rms_norm_triton_fake(
    x: Tensor, residual: Tensor, weight: Tensor, epsilon: float
) -> tuple[Tensor, Tensor]:
    return torch.empty_like(x), torch.empty_like(x)


SUPPORTED = HAS_TRITON and current_platform.is_cuda()

if SUPPORTED:
    # Opaque to Inductor so the PDL launch option reaches the kernel.
    direct_register_custom_op(
        "rms_norm_fp32_weight_triton",
        _rms_norm_triton,
        fake_impl=_rms_norm_triton_fake,
    )
    direct_register_custom_op(
        "fused_add_rms_norm_fp32_weight_triton",
        _fused_add_rms_norm_triton,
        fake_impl=_fused_add_rms_norm_triton_fake,
    )


def _supports(x: Tensor, weight: Tensor | None, variance_size: int | None) -> bool:
    return (
        variance_size is None
        and weight is not None
        and weight.dtype == torch.float32
        and x.dtype in _SUPPORTED_DTYPES
        and x.is_cuda
        and x.is_contiguous()
        and 0 < x.shape[-1] <= _MAX_HIDDEN_SIZE
        and weight.shape == (x.shape[-1],)
        and weight.is_contiguous()
    )


def _supports_rms_norm(
    x: Tensor, weight: Tensor | None, epsilon: float, variance_size: int | None = None
) -> bool:
    return _supports(x, weight, variance_size)


def _supports_fused_add_rms_norm(
    x: Tensor,
    x_residual: Tensor,
    weight: Tensor | None,
    epsilon: float,
    variance_size: int | None = None,
) -> bool:
    return (
        _supports(x, weight, variance_size)
        and x_residual.dtype == x.dtype
        and x_residual.shape == x.shape
        and x_residual.is_contiguous()
    )


@ir.ops.rms_norm.register_impl(
    "triton", supported=SUPPORTED, supports_args=_supports_rms_norm
)
def rms_norm(
    x: Tensor, weight: Tensor | None, epsilon: float, variance_size: int | None = None
) -> Tensor:
    assert weight is not None and variance_size is None
    return torch.ops.vllm.rms_norm_fp32_weight_triton(x, weight, epsilon)


@ir.ops.fused_add_rms_norm.register_impl(
    "triton", supported=SUPPORTED, supports_args=_supports_fused_add_rms_norm
)
def fused_add_rms_norm(
    x: Tensor,
    x_residual: Tensor,
    weight: Tensor | None,
    epsilon: float,
    variance_size: int | None = None,
) -> tuple[Tensor, Tensor]:
    assert weight is not None and variance_size is None
    return torch.ops.vllm.fused_add_rms_norm_fp32_weight_triton(
        x, x_residual, weight, epsilon
    )
