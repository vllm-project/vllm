# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.platforms import current_platform
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

    Both norms are Gemma-style RMSNorm (scale by `1 + weight`, fp32
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


@triton.jit
def _fused_pre_fc_norms_split_kernel(
    embeds_ptr,
    embeds_stride,
    hidden_ptr,
    hidden_stride,
    emb_w_ptr,
    hid_w_ptr,
    emb_out_ptr,
    emb_out_stride,
    hid_out_ptr,
    hid_out_stride,
    eps,
    H_EMB: tl.constexpr,
    H_HID: tl.constexpr,
    launch_pdl: tl.constexpr,
):
    BLOCK_EMB: tl.constexpr = triton.next_power_of_2(H_EMB)
    BLOCK_HID: tl.constexpr = triton.next_power_of_2(H_HID)
    tok = tl.program_id(0)
    off_e = tl.arange(0, BLOCK_EMB)
    mask_e = off_e < H_EMB
    off_h = tl.arange(0, BLOCK_HID)
    mask_h = off_h < H_HID

    if launch_pdl:
        tl.extra.cuda.gdc_wait()

    e = tl.load(embeds_ptr + tok * embeds_stride + off_e, mask=mask_e, other=0.0)
    ew = tl.load(emb_w_ptr + off_e, mask=mask_e, other=0.0)
    h = tl.load(hidden_ptr + tok * hidden_stride + off_h, mask=mask_h, other=0.0)
    hw = tl.load(hid_w_ptr + off_h, mask=mask_h, other=0.0)

    e_normed = _gemma_rms_norm(e, ew, eps, H_EMB)
    h_normed = _gemma_rms_norm(h, hw, eps, H_HID)

    tl.store(emb_out_ptr + tok * emb_out_stride + off_e, e_normed, mask=mask_e)
    tl.store(hid_out_ptr + tok * hid_out_stride + off_h, h_normed, mask=mask_h)

    if launch_pdl:
        tl.extra.cuda.gdc_launch_dependents()


def fused_pre_fc_norms_split(
    inputs_embeds: torch.Tensor,
    hidden_states: torch.Tensor,
    emb_weight: torch.Tensor,
    hid_weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Two-width variant of `fused_pre_fc_norms` with separate outputs.

    For drafters whose pre-FC norms have different widths and feed separate
    FC GEMMs (e.g. Qwen4Exp MTP), so one launch writes two contiguous
    outputs instead of a concat. Returns `(normed_embeds [N, H_EMB],
    normed_hidden [N, H_HID])`.
    """
    assert inputs_embeds.dim() == 2 and inputs_embeds.stride(1) == 1
    assert hidden_states.dim() == 2 and hidden_states.stride(1) == 1
    assert emb_weight.is_contiguous() and hid_weight.is_contiguous()

    n, h_emb = inputs_embeds.shape
    h_hid = hidden_states.shape[1]
    emb_out = torch.empty(
        n, h_emb, dtype=inputs_embeds.dtype, device=inputs_embeds.device
    )
    hid_out = torch.empty(
        n, h_hid, dtype=hidden_states.dtype, device=hidden_states.device
    )
    _fused_pre_fc_norms_split_kernel[(n,)](
        inputs_embeds,
        inputs_embeds.stride(0),
        hidden_states,
        hidden_states.stride(0),
        emb_weight,
        hid_weight,
        emb_out,
        emb_out.stride(0),
        hid_out,
        hid_out.stride(0),
        eps,
        H_EMB=h_emb,
        H_HID=h_hid,
        launch_pdl=current_platform.is_arch_support_pdl(),
        num_warps=max(4, min(32, triton.next_power_of_2(h_hid) // 1024)),
    )
    return emb_out, hid_out
