# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm import _custom_ops as ops


def mhc_pre_cpu(
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
    """CPU-ported mHC pre block (see `mhc_pre_torch` for the eager reference).

    The ported kernel (`hc_pre_fused_cpu`) only exposes one merged `hc_eps`
    (used for both `hc_pre_eps`/`hc_sinkhorn_eps`) and hardcodes the post-mix
    multiplier to 2.0 -- true of every real call site in
    `models/deepseek_v4/cpu/model.py`, so this is not a capability loss here.
    """
    assert n_splits == 1, "mhc_pre_cpu does not support n_splits != 1"
    assert hc_pre_eps == hc_sinkhorn_eps, (
        "mhc_pre_cpu requires hc_pre_eps == hc_sinkhorn_eps (single merged hc_eps)"
    )
    assert hc_post_mult_value == 2.0, "mhc_pre_cpu hardcodes the post-mix multiplier"

    hc_mult, hidden_size = residual.shape[-2:]
    outer_shape = residual.shape[:-2]
    x_flat = residual.reshape(-1, hc_mult, hidden_size)

    layer_input, post, comb = ops.hc_pre_fused_cpu(
        x_flat,
        fn,
        hc_scale,
        hc_base,
        hc_mult,
        sinkhorn_repeat,
        rms_eps,
        hc_pre_eps,
    )

    post_mix = post.view(*outer_shape, hc_mult, 1)
    comb_mix = comb.view(*outer_shape, hc_mult, hc_mult)
    layer_input = layer_input.view(*outer_shape, hidden_size)
    return post_mix, comb_mix, layer_input


def mhc_post_cpu(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    """CPU-ported mHC post block (see `mhc_post_torch` for the eager reference)."""
    hc_mult, hidden_size = residual.shape[-2:]
    outer_shape = residual.shape[:-2]

    x_flat = x.reshape(-1, hidden_size)
    residual_flat = residual.reshape(-1, hc_mult, hidden_size)
    post_flat = post_layer_mix.reshape(-1, hc_mult).float()
    comb_flat = comb_res_mix.reshape(-1, hc_mult, hc_mult).float()

    out = ops.hc_post_fused_cpu(x_flat, residual_flat, post_flat, comb_flat)
    return out.view(*outer_shape, hc_mult, hidden_size)


def hc_head_fused_cpu(
    hidden_states: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_norm_eps: float,
    hc_eps: float,
) -> torch.Tensor:
    """CPU-ported HC head reduction (see `test_hc_head_cpu` in
    tests/kernels/test_mhc_kernels.py for the eager reference this is tested
    against).

    The ported kernel's C++ signature takes `(hc_eps, norm_eps)` -- the
    opposite order from this wrapper's `(rms_norm_eps, hc_eps)`, which matches
    the real call site in `models/deepseek_v4/cpu/model.py`.
    """
    hc_mult, hidden_size = hidden_states.shape[-2:]
    outer_shape = hidden_states.shape[:-2]
    hs_flat = hidden_states.reshape(-1, hc_mult, hidden_size)

    out = ops.hc_head_fused_cpu(hs_flat, hc_fn, hc_scale, hc_base, hc_eps, rms_norm_eps)
    return out.view(*outer_shape, hidden_size)
