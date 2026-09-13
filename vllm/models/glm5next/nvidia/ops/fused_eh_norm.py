# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _rms_norm(x, w, eps, HIDDEN_SIZE: tl.constexpr):
    x = x.to(tl.float32)
    mean_sq = tl.sum(x * x, axis=0) / HIDDEN_SIZE
    rrms = tl.rsqrt(mean_sq + eps)
    w = w.to(tl.float32)
    return (x * rrms) * w


@triton.jit
def _fused_eh_norm_kernel(
    pos_ptr,
    embeds_ptr,
    embeds_stride,
    prev_ptr,
    prev_stride,
    enorm_w_ptr,
    hnorm_w_ptr,
    eps,
    out_ptr,
    out_stride,
    H: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """MTP input fusion: zero embeds at position 0, RMSNorm(embeds) with enorm
    and RMSNorm(prev_hidden) with hnorm, written side-by-side into ``out``
    ([N, 2H]) ready for the eh_proj GEMM. Replaces where + 2x RMSNorm + cat."""
    tok = tl.program_id(0)
    off = tl.arange(0, BLOCK)
    mask = off < H

    pos = tl.load(pos_ptr + tok)
    e = tl.load(embeds_ptr + tok * embeds_stride + off, mask=mask, other=0.0)
    e = tl.where(pos == 0, 0.0, e.to(tl.float32))
    ew = tl.load(enorm_w_ptr + off, mask=mask)
    e_normed = _rms_norm(e, ew, eps, H)
    tl.store(out_ptr + tok * out_stride + off, e_normed, mask=mask)

    p = tl.load(prev_ptr + tok * prev_stride + off, mask=mask, other=0.0)
    hw = tl.load(hnorm_w_ptr + off, mask=mask)
    p_normed = _rms_norm(p, hw, eps, H)
    tl.store(out_ptr + tok * out_stride + H + off, p_normed, mask=mask)


def fused_eh_norm(
    positions: torch.Tensor,
    inputs_embeds: torch.Tensor,
    previous_hidden: torch.Tensor,
    enorm_w: torch.Tensor,
    hnorm_w: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Returns cat([enorm(masked embeds), hnorm(prev_hidden)]) -> [N, 2H]."""
    n, h = inputs_embeds.shape
    out = torch.empty(n, 2 * h, dtype=inputs_embeds.dtype, device=inputs_embeds.device)
    _fused_eh_norm_kernel[(n,)](
        positions,
        inputs_embeds,
        inputs_embeds.stride(0),
        previous_hidden,
        previous_hidden.stride(0),
        enorm_w,
        hnorm_w,
        eps,
        out,
        out.stride(0),
        h,
        triton.next_power_of_2(h),
    )
    return out
