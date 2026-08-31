# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused PLE gating and normalization kernel.

Each program processes one token and one HC stream:

  key_n   = grouped_rmsnorm(key, nk, group=H)
  query_n = grouped_rmsnorm(hidden,       nq,  group=H)
  d       = dot(key_n, query_n) / sqrt(H)
  g       = sigmoid(sign(d) * sqrt(clamp_min(|d|, 1e-6)))
  gated   = g * value
  normed  = grouped_rmsnorm(gated, ncw, group=H)

Reductions use fp32. Normalized key/query, the gate, and gated output are
rounded to bf16 before their consumers.
"""

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op


@triton.jit(do_not_specialize=["num_tokens"])
def _ple_gate_kernel(
    key_ptr,  # [T, HC*H] bf16 with row stride key_rs
    value_ptr,  # [T, H] bf16 with row stride value_rs
    hidden_ptr,  # [T, HC*H] bf16
    nk_ptr,  # [HC*H] bf16 norm_key weight
    nq_ptr,  # [HC*H] bf16 norm_query weight
    ncw_ptr,  # [HC*H] bf16 norm_conv weight
    gated_ptr,  # [T, HC*H] bf16 out
    normed_ptr,  # [T, HC*H] bf16 out
    num_tokens,
    key_rs,
    value_rs,
    eps,
    H: tl.constexpr,
    HC: tl.constexpr,
    BLOCK_H: tl.constexpr,
    launch_pdl: tl.constexpr,
):
    t = tl.program_id(0)
    s = tl.program_id(1)
    lanes = tl.arange(0, BLOCK_H)
    mask = lanes < H
    offs = s * H + lanes

    if launch_pdl:
        tl.extra.cuda.gdc_wait()

    k = tl.load(key_ptr + t * key_rs + offs, mask=mask, other=0.0).to(tl.float32)
    q = tl.load(hidden_ptr + t * HC * H + offs, mask=mask, other=0.0).to(tl.float32)
    nk = tl.load(nk_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    nq = tl.load(nq_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    k_n = (k * tl.rsqrt(tl.sum(k * k) / H + eps) * (1.0 + nk)).to(tl.bfloat16)
    q_n = (q * tl.rsqrt(tl.sum(q * q) / H + eps) * (1.0 + nq)).to(tl.bfloat16)

    products = (k_n.to(tl.float32) * q_n.to(tl.float32)).to(tl.bfloat16)
    dot = tl.sum(products.to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    d = dot / tl.sqrt(float(H))
    d = d.to(tl.bfloat16).to(tl.float32)
    sign = tl.where(d < 0, -1.0, 0.0)
    sign = tl.where(d > 0, 1.0, sign)
    g = tl.sigmoid(sign * tl.sqrt(tl.maximum(tl.abs(d), 1e-6)))
    g = g.to(tl.bfloat16).to(tl.float32)

    v = tl.load(value_ptr + t * value_rs + lanes, mask=mask, other=0.0).to(tl.float32)
    gated = (g * v).to(tl.bfloat16)
    gf = gated.to(tl.float32)
    ncw = tl.load(ncw_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    normed = gf * tl.rsqrt(tl.sum(gf * gf) / H + eps) * (1.0 + ncw)

    if launch_pdl:
        tl.extra.cuda.gdc_launch_dependents()
    tl.store(gated_ptr + t * HC * H + offs, gated, mask=mask)
    tl.store(normed_ptr + t * HC * H + offs, normed, mask=mask)


def _ple_gate(
    key: torch.Tensor,
    value: torch.Tensor,
    hidden: torch.Tensor,
    norm_key_w: torch.Tensor,
    norm_query_w: torch.Tensor,
    norm_conv_w: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens = hidden.shape[0]
    h = value.shape[-1]
    hc = hidden.shape[-1] // h
    assert key.stride(1) == 1 and value.stride(1) == 1
    assert hidden.is_contiguous()
    assert key.dtype == value.dtype == hidden.dtype == torch.bfloat16
    gated = torch.empty_like(hidden)
    normed = torch.empty_like(hidden)
    _ple_gate_kernel[(num_tokens, hc)](
        key,
        value,
        hidden,
        norm_key_w,
        norm_query_w,
        norm_conv_w,
        gated,
        normed,
        num_tokens,
        key.stride(0),
        value.stride(0),
        eps,
        H=h,
        HC=hc,
        BLOCK_H=triton.next_power_of_2(h),
        num_warps=8,
        launch_pdl=current_platform.is_arch_support_pdl(),
    )
    return gated, normed


def _ple_gate_fake(
    key: torch.Tensor,
    value: torch.Tensor,
    hidden: torch.Tensor,
    norm_key_w: torch.Tensor,
    norm_query_w: torch.Tensor,
    norm_conv_w: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(hidden), torch.empty_like(hidden)


direct_register_custom_op(
    op_name="qwen4_exp_ple_gate",
    op_func=_ple_gate,
    mutates_args=[],
    fake_impl=_ple_gate_fake,
)


def ple_gate(
    key: torch.Tensor,
    value: torch.Tensor,
    hidden: torch.Tensor,
    norm_key_w: torch.Tensor,
    norm_query_w: torch.Tensor,
    norm_conv_w: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.vllm.qwen4_exp_ple_gate(
        key, value, hidden, norm_key_w, norm_query_w, norm_conv_w, eps
    )
