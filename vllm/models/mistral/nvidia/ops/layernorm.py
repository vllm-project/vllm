# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One-launch LayerNorm: fp32 statistics and affine, output in the input dtype."""

import torch

from vllm.models.mistral.nvidia.ops.ssmax import _round_bf16
from vllm.triton_utils import tl, triton


@triton.jit
def _layer_norm_kernel(
    x_ptr,
    x_stride,
    w_ptr,
    b_ptr,
    eps,
    out_ptr,
    out_stride,
    hidden_size: tl.constexpr,
    block: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, block)
    mask = offs < hidden_size
    x = tl.load(x_ptr + row * x_stride + offs, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / hidden_size
    diff = tl.where(mask, x - mean, 0.0)
    rstd = tl.rsqrt(tl.sum(diff * diff, axis=0) / hidden_size + eps)
    w = tl.load(w_ptr + offs, mask=mask).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask).to(tl.float32)
    out = _round_bf16(diff * rstd * w + b)
    tl.store(out_ptr + row * out_stride + offs, out.to(tl.bfloat16), mask=mask)


def layer_norm(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float
) -> torch.Tensor:
    """LayerNorm over the last dim of a bf16 ``[T, H]`` tensor."""
    assert x.ndim == 2 and x.stride(-1) == 1 and x.dtype == torch.bfloat16
    num_tokens, hidden_size = x.shape
    assert weight.shape == bias.shape == (hidden_size,)
    out = torch.empty_like(x)
    if num_tokens == 0:
        return out
    block = triton.next_power_of_2(hidden_size)
    _layer_norm_kernel[(num_tokens,)](
        x,
        x.stride(0),
        weight,
        bias,
        eps,
        out,
        out.stride(0),
        hidden_size=hidden_size,
        block=block,
        num_warps=min(max(block // 256, 1), 8),
    )
    return out
