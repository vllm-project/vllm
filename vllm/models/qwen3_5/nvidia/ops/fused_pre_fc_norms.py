# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _gemma_rms_norm(x, w, eps, HIDDEN_SIZE: tl.constexpr):
    x = x.to(tl.float32)
    mean_sq = tl.sum(x * x, axis=0) / HIDDEN_SIZE
    rrms = tl.rsqrt(mean_sq + eps)
    w = w.to(tl.float32) + 1.0
    return (x * rrms) * w


@triton.jit
def _fused_pre_fc_norms_kernel(
    embeds_ptr,
    embeds_stride,
    hidden_ptr,
    hidden_stride,
    emb_w_ptr,
    hid_w_ptr,
    eps,
    out_ptr,
    out_stride,
    H: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """MTP input fusion: GemmaRMSNorm(embeds) and GemmaRMSNorm(hidden) written
    side-by-side into ``out`` ([N, 2H]) ready for the fc GEMM. Replaces
    2x GemmaRMSNorm + cat."""
    tok = tl.program_id(0)
    off = tl.arange(0, BLOCK)
    mask = off < H

    e = tl.load(embeds_ptr + tok * embeds_stride + off, mask=mask, other=0.0)
    ew = tl.load(emb_w_ptr + off, mask=mask)
    e_normed = _gemma_rms_norm(e, ew, eps, H)
    tl.store(out_ptr + tok * out_stride + off, e_normed, mask=mask)

    h = tl.load(hidden_ptr + tok * hidden_stride + off, mask=mask, other=0.0)
    hw = tl.load(hid_w_ptr + off, mask=mask)
    h_normed = _gemma_rms_norm(h, hw, eps, H)
    tl.store(out_ptr + tok * out_stride + H + off, h_normed, mask=mask)


def fused_pre_fc_norms(
    inputs_embeds: torch.Tensor,
    hidden_states: torch.Tensor,
    emb_weight: torch.Tensor,
    hid_weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Returns cat([norm_e(inputs_embeds), norm_h(hidden_states)]) -> [N, 2H].

    Both norms are Gemma-style RMSNorm (scale by ``1 + weight``, fp32
    accumulation, multiply in fp32 before the cast back to the input dtype).
    """
    n, h = inputs_embeds.shape
    out = torch.empty(n, 2 * h, dtype=inputs_embeds.dtype, device=inputs_embeds.device)
    _fused_pre_fc_norms_kernel[(n,)](
        inputs_embeds,
        inputs_embeds.stride(0),
        hidden_states,
        hidden_states.stride(0),
        emb_weight,
        hid_weight,
        eps,
        out,
        out.stride(0),
        h,
        triton.next_power_of_2(h),
    )
    return out
