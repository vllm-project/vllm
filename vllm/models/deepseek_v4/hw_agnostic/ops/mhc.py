# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4 mHC (multi-head compression) ops — hw-agnostic native-only.

Vendored from ``vllm/model_executor/layers/mhc.py``. All four CustomOps
keep only the pure-PyTorch reference implementation; the
TileLang/aiter/Triton fast paths used by upstream
``forward_cuda``/``forward_hip``/``forward_xpu`` are dropped, and the
math (previously imported from ``vllm.model_executor.kernels.mhc``)
is inlined here so the module has no dependency on the upstream
kernel package.
"""

import torch
import torch.nn as nn


def _mhc_pre_torch(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure-PyTorch reference for the mHC pre block.

    Inputs:
        residual: (..., hc_mult, hidden_size), bfloat16
        fn: (hc_mult3, hc_mult * hidden_size), float32
        hc_scale: (3,), float32
        hc_base: (hc_mult3,), float32
    Returns:
        post_mix: (..., hc_mult, 1), float32
        comb_mix: (..., hc_mult, hc_mult), float32
        layer_input: (..., hidden_size), bfloat16
    """
    assert residual.dtype == torch.bfloat16
    assert fn.dtype == torch.float32
    assert hc_scale.dtype == torch.float32
    assert hc_base.dtype == torch.float32

    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    hc_mult2 = hc_mult * hc_mult
    hc_mult3 = hc_mult * 2 + hc_mult2
    hc_hidden_size = hc_mult * hidden_size
    assert fn.shape[0] == hc_mult3
    assert fn.shape[1] == hc_hidden_size
    assert hc_scale.shape == (3,)
    assert hc_base.shape == (hc_mult3,)

    outer_shape = residual.shape[:-2]
    residual_flat = residual.view(-1, hc_mult, hidden_size)
    num_tokens = residual_flat.shape[0]

    x = residual_flat.view(num_tokens, hc_hidden_size).to(torch.float32)
    mixes = torch.matmul(x, fn.t())
    sqrsum = x.square().sum(dim=-1, keepdim=True)
    mixes = mixes * torch.rsqrt(sqrsum / hc_hidden_size + rms_eps)

    pre_logits = mixes[:, :hc_mult] * hc_scale[0] + hc_base[:hc_mult]
    pre_mix = torch.sigmoid(pre_logits) + hc_pre_eps

    post_logits = (
        mixes[:, hc_mult : 2 * hc_mult] * hc_scale[1] + hc_base[hc_mult : 2 * hc_mult]
    )
    post_mix = torch.sigmoid(post_logits) * hc_post_mult_value

    comb_logits = mixes[:, 2 * hc_mult :].view(num_tokens, hc_mult, hc_mult) * hc_scale[
        2
    ] + hc_base[2 * hc_mult :].view(1, hc_mult, hc_mult)
    comb_mix = torch.softmax(comb_logits, dim=-1) + hc_sinkhorn_eps
    comb_mix = comb_mix / (comb_mix.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)
    for _ in range(sinkhorn_repeat - 1):
        comb_mix = comb_mix / (comb_mix.sum(dim=-1, keepdim=True) + hc_sinkhorn_eps)
        comb_mix = comb_mix / (comb_mix.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)

    layer_input = torch.sum(
        pre_mix.unsqueeze(-1) * residual_flat.to(torch.float32), dim=1
    ).to(torch.bfloat16)
    return (
        post_mix.view(*outer_shape, hc_mult, 1),
        comb_mix.view(*outer_shape, hc_mult, hc_mult),
        layer_input.view(*outer_shape, hidden_size),
    )


def _mhc_post_torch(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    mixed_residual = torch.einsum(
        "...ij,...ih->...jh",
        comb_res_mix.to(torch.float32),
        residual.to(torch.float32),
    )
    post_term = post_layer_mix.to(torch.float32) * x.unsqueeze(-2).to(torch.float32)
    return (mixed_residual + post_term).to(residual.dtype)


def _hc_head_fused_torch(
    hidden_states: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_norm_eps: float,
    hc_eps: float,
) -> torch.Tensor:
    """Pure-PyTorch reference for the HC head reduction.

    1. Weight-free RMSNorm over the flattened ``(hc_mult * hidden_size)`` axis.
    2. Linear projection by ``hc_fn``: per-stream gates.
    3. ``sigmoid(mixes * hc_scale + hc_base) + hc_eps`` → gate weights.
    4. ``out[t, h] = sum_m gate[t, m] * hidden_states[t, m, h]``.
    """
    assert hc_fn.dtype == torch.float32
    assert hc_scale.dtype == torch.float32
    assert hc_base.dtype == torch.float32

    outer_shape = hidden_states.shape[:-2]
    hc_mult, hidden_size = hidden_states.shape[-2:]
    out_dtype = hidden_states.dtype

    x = hidden_states.reshape(-1, hc_mult, hidden_size)
    x_flat = x.reshape(-1, hc_mult * hidden_size).to(torch.float32)
    rms = torch.rsqrt(x_flat.square().mean(dim=-1, keepdim=True) + rms_norm_eps)
    x_normed = x_flat * rms

    mixes = torch.matmul(x_normed, hc_fn.t())
    pre = torch.sigmoid(mixes * hc_scale + hc_base) + hc_eps

    out = torch.einsum("tm,tmh->th", pre, x.to(torch.float32))
    return out.to(out_dtype).view(*outer_shape, hidden_size)


class MHCPreOp(nn.Module):
    """mHC pre block.

    Computes mix logits from RMS-normalized HC residual streams, then
    returns ``post_mix``, ``comb_mix``, and
    ``layer_input = sum_i pre_mix_i * residual_i``.
    """

    def forward(
        self,
        residual: torch.Tensor,
        fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        rms_eps: float,
        hc_pre_eps: float,
        hc_sinkhorn_eps: float,
        hc_post_mult_value: float,
        sinkhorn_repeat: int,
        n_splits: int = 1,
        norm_weight: torch.Tensor | None = None,
        norm_eps: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _mhc_pre_torch(
            residual,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
        )


class MHCPostOp(nn.Module):
    """mHC post block.

    Combines the layer output with the HC residual streams:
    ``out_j = post_layer_mix_j * x + sum_i comb_res_mix_ij * residual_i``.
    """

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post_layer_mix: torch.Tensor,
        comb_res_mix: torch.Tensor,
    ) -> torch.Tensor:
        return _mhc_post_torch(x, residual, post_layer_mix, comb_res_mix)


class HCHeadOp(nn.Module):
    """HC head reduction for DeepSeek V4.

    Computes gates from the RMS-normalized flattened HC residual and
    returns ``out = sum_i gate_i * residual_i``, collapsing ``hc_mult``
    streams to one.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        rms_norm_eps: float,
        hc_eps: float,
    ) -> torch.Tensor:
        return _hc_head_fused_torch(
            hidden_states, hc_fn, hc_scale, hc_base, rms_norm_eps, hc_eps
        )


class MHCFusedPostPreOp(nn.Module):
    """Fused mHC post block followed by the next mHC pre block.

    Equivalent to applying ``MHCPostOp`` and then ``MHCPreOp`` to the
    updated residual streams, returning ``residual_cur``,
    ``post_mix_cur``, ``comb_mix_cur``, and ``layer_input_cur``.
    """

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post_layer_mix: torch.Tensor,
        comb_res_mix: torch.Tensor,
        fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        rms_eps: float,
        hc_pre_eps: float,
        hc_sinkhorn_eps: float,
        hc_post_mult_value: float,
        sinkhorn_repeat: int,
        n_splits: int = 1,
        tile_n: int = 1,
        norm_weight: torch.Tensor | None = None,
        norm_eps: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        residual_cur = _mhc_post_torch(x, residual, post_layer_mix, comb_res_mix)
        post_mix_cur, comb_mix_cur, layer_input_cur = _mhc_pre_torch(
            residual_cur,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
        )
        return residual_cur, post_mix_cur, comb_mix_cur, layer_input_cur
