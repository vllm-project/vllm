# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KDA f_b projection on the row-strided f_a slice of the fused in_proj output.

The generic ROCm unquantized GEMM materializes a row-strided activation with
.contiguous() before its skinny kernels, which costs one extra copy launch per
KDA layer at decode. f_a has a tiny K (head_dim), so a single-tile Triton GEMM
that takes the row stride reads it in place.
"""

import torch

from vllm.model_executor.layers.utils import rocm_unquantized_gemm_impl
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

# Covers decode and spec-decode verify (c48 x 7 tokens = 336 rows) and small
# mixed batches. The kernel stays ahead of the generic GEMM up to about 2k rows
# on MI355X; above the cap (prefill) the generic GEMM is kept.
KDA_FB_GEMM_MAX_TOKENS = 1024


@triton.jit
def _kda_fb_gemm_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    num_tokens,
    stride_x,
    stride_w,
    stride_out,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, K)
    mask_m = offs_m < num_tokens
    mask_n = offs_n < N

    # K fits one tile, so there is no K loop.
    x = tl.load(
        x_ptr + offs_m[:, None] * stride_x + offs_k[None, :],
        mask=mask_m[:, None],
        other=0.0,
    )
    w = tl.load(
        w_ptr + offs_n[:, None] * stride_w + offs_k[None, :],
        mask=mask_n[:, None],
        other=0.0,
    )
    acc = tl.dot(x, tl.trans(w))
    tl.store(
        out_ptr + offs_m[:, None] * stride_out + offs_n[None, :],
        acc.to(out_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_n[None, :],
    )


def kda_fb_gemm_supported(x: torch.Tensor, weight: torch.Tensor) -> bool:
    k = weight.shape[1]
    return (
        x.dim() == 2
        and x.dtype in (torch.bfloat16, torch.float16)
        and weight.dtype == x.dtype
        and x.stride(-1) == 1
        and weight.stride(-1) == 1
        and x.shape[1] == k
        and 16 <= k <= 256
        and k & (k - 1) == 0
    )


def kda_fb_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    block_m: int | None = None,
    block_n: int | None = None,
    num_warps: int | None = None,
) -> torch.Tensor:
    """Return x @ weight.T for a row-strided x without copying x."""
    num_tokens, k = x.shape
    n = weight.shape[0]
    out = torch.empty((num_tokens, n), dtype=x.dtype, device=x.device)
    if num_tokens == 0:
        return out
    # 16x32 tiles with 2 warps were the best or within 0.1 us of the best
    # config for 1-256 rows at K=128, N=1536 on MI355X, and stay within
    # 0.2 us of 32x64 / 4 warps up to the 1024-row cap.
    block_m = block_m or 16
    block_n = block_n or 32
    num_warps = num_warps or 2
    grid = (triton.cdiv(n, block_n), triton.cdiv(num_tokens, block_m))
    _kda_fb_gemm_kernel[grid](
        x,
        weight,
        out,
        num_tokens,
        x.stride(0),
        weight.stride(0),
        out.stride(0),
        N=n,
        K=k,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=num_warps,
    )
    return out


def _kda_fb_proj(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    # The row count is only known here at runtime (the compiled graph has a
    # symbolic batch), so the token-count gate lives inside the op.
    if x.shape[0] <= KDA_FB_GEMM_MAX_TOKENS and kda_fb_gemm_supported(x, weight):
        return kda_fb_gemm(x, weight)
    return rocm_unquantized_gemm_impl(x, weight)


def _kda_fb_proj_fake(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return x.new_empty((x.shape[0], weight.shape[0]))


direct_register_custom_op(
    op_name="kimi_k3_kda_fb_proj",
    op_func=_kda_fb_proj,
    fake_impl=_kda_fb_proj_fake,
)
