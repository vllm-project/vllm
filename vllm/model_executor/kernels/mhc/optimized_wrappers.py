# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Python wrappers for optimized fused MHC kernels.

These wrappers handle tensor shape validation and provide a clean
interface matching the original separate kernel calls.
"""

from typing import Optional

import torch

from vllm.utils.torch_utils import direct_register_custom_op

from .optimized_fusions import (
    mhc_post_hc_head_fused_tilelang,
    mhc_post_hc_head_norm_fused_tilelang,
    mhc_post_hc_head_norm_fused_tilelang_mtp,
    mhc_post_mean_fused_tilelang,
)


def mhc_post_hc_head_fused(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    hc_head_fn: torch.Tensor,
    hc_head_scale: torch.Tensor,
    hc_head_base: torch.Tensor,
    rms_norm_eps: float,
    hc_eps: float,
) -> torch.Tensor:
    """Fused MHC post + HC head operation.

    Replaces the sequence:
        residual_out = mhc_post_tilelang(x, residual, post_layer_mix, comb_res_mix)
        output = hc_head_fused_kernel_tilelang(residual_out, hc_head_fn, ...)

    Args:
        x: [num_tokens, hidden_size] layer output
        residual: [num_tokens, hc_mult, hidden_size] input residual streams
        post_layer_mix: [num_tokens, hc_mult] or [num_tokens, hc_mult, 1]
        comb_res_mix: [num_tokens, hc_mult, hc_mult] combination mixing weights
        hc_head_fn: [hc_mult, hc_mult * hidden_size] HC head projection matrix
        hc_head_scale: [1] HC head scale factor
        hc_head_base: [hc_mult] HC head bias
        rms_norm_eps: RMS normalization epsilon
        hc_eps: HC head gate epsilon

    Returns:
        output: [num_tokens, hidden_size] compressed output
    """
    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    outer_shape = residual.shape[:-2]

    assert x.shape == (*outer_shape, hidden_size)
    assert post_layer_mix.shape in (
        (*outer_shape, hc_mult, 1),
        (*outer_shape, hc_mult),
    )
    assert comb_res_mix.shape == (*outer_shape, hc_mult, hc_mult)
    assert hc_head_fn.shape == (hc_mult, hc_mult * hidden_size)
    assert hc_head_scale.shape == (1,)
    assert hc_head_base.shape == (hc_mult,)

    # Flatten batch dimensions
    residual_flat = residual.view(-1, hc_mult, hidden_size)
    num_tokens = residual_flat.shape[0]
    x_flat = x.view(num_tokens, hidden_size)
    post_layer_mix_flat = post_layer_mix.view(num_tokens, hc_mult)
    comb_res_mix_flat = comb_res_mix.view(num_tokens, hc_mult, hc_mult)

    # Allocate output
    output = torch.empty(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=x.device
    )

    # Call fused kernel
    mhc_post_hc_head_fused_tilelang(
        comb_res_mix_flat,
        residual_flat,
        post_layer_mix_flat,
        x_flat,
        hc_head_fn,
        hc_head_scale,
        hc_head_base,
        output,
        hc_mult,
        hidden_size,
        rms_norm_eps,
        hc_eps,
    )

    return output.view(*outer_shape, hidden_size)


def mhc_post_hc_head_norm_fused(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    hc_head_fn: torch.Tensor,
    hc_head_scale: torch.Tensor,
    hc_head_base: torch.Tensor,
    norm_weight: torch.Tensor,
    rms_norm_eps: float,
    hc_eps: float,
    norm_eps: float,
    mtp_buffer: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fused MHC post + HC head + RMSNorm operation.

    Replaces the sequence:
        residual_out = mhc_post_tilelang(x, residual, post_layer_mix, comb_res_mix)
        hc_out = hc_head_fused_kernel_tilelang(residual_out, hc_head_fn, ...)
        output = rms_norm(hc_out, norm_weight, norm_eps)

    When mtp_buffer is provided, the pre-hc_head residual (MHC post output)
    is also written to it for the MTP draft model.

    Args:
        Same as mhc_post_hc_head_fused, plus:
        norm_weight: [hidden_size] RMSNorm weight
        norm_eps: RMSNorm epsilon
        mtp_buffer: optional [num_tokens, hc_mult * hidden_size] buffer for
            the MTP draft model's pre-hc_head residual stash

    Returns:
        output: [num_tokens, hidden_size] normalized compressed output
    """
    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    outer_shape = residual.shape[:-2]

    assert x.shape == (*outer_shape, hidden_size)
    assert norm_weight.shape == (hidden_size,)

    # Ensure norm_weight is contiguous and correct dtype
    if norm_weight.dtype != torch.bfloat16:
        norm_weight = norm_weight.to(torch.bfloat16)
    if not norm_weight.is_contiguous():
        norm_weight = norm_weight.contiguous()

    # Flatten batch dimensions
    residual_flat = residual.view(-1, hc_mult, hidden_size)
    num_tokens = residual_flat.shape[0]
    x_flat = x.view(num_tokens, hidden_size)
    post_layer_mix_flat = post_layer_mix.view(num_tokens, hc_mult)
    comb_res_mix_flat = comb_res_mix.view(num_tokens, hc_mult, hc_mult)

    # Allocate output
    output = torch.empty(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=x.device
    )

    # Choose kernel variant based on MTP buffer availability
    if mtp_buffer is not None:
        # Use MTP variant that also writes the pre-hc_head residual
        mhc_post_hc_head_norm_fused_tilelang_mtp(
            comb_res_mix_flat,
            residual_flat,
            post_layer_mix_flat,
            x_flat,
            hc_head_fn,
            hc_head_scale,
            hc_head_base,
            norm_weight,
            norm_eps,
            output,
            mtp_buffer,
            hc_mult,
            hidden_size,
            rms_norm_eps,
            hc_eps,
        )
    else:
        # Standard variant (no MTP output)
        mhc_post_hc_head_norm_fused_tilelang(
            comb_res_mix_flat,
            residual_flat,
            post_layer_mix_flat,
            x_flat,
            hc_head_fn,
            hc_head_scale,
            hc_head_base,
            norm_weight,
            norm_eps,
            output,
            hc_mult,
            hidden_size,
            rms_norm_eps,
            hc_eps,
        )

    return output.view(*outer_shape, hidden_size)


def mhc_post_mean_fused(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused MHC post + mean(dim=1) operation for aux layers.

    Replaces the sequence:
        residual_out = mhc_post_tilelang(x, residual, post_layer_mix, comb_res_mix)
        mean_out = residual_out.mean(dim=1)

    Args:
        x: [num_tokens, hidden_size] layer output
        residual: [num_tokens, hc_mult, hidden_size] input residual streams
        post_layer_mix: [num_tokens, hc_mult] or [num_tokens, hc_mult, 1]
        comb_res_mix: [num_tokens, hc_mult, hc_mult] combination mixing weights

    Returns:
        residual_out: [num_tokens, hc_mult, hidden_size] full MHC post output
        mean_out: [num_tokens, hidden_size] mean across hc_mult dimension
    """
    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    outer_shape = residual.shape[:-2]

    assert x.shape == (*outer_shape, hidden_size)

    # Flatten batch dimensions
    residual_flat = residual.view(-1, hc_mult, hidden_size)
    num_tokens = residual_flat.shape[0]
    x_flat = x.view(num_tokens, hidden_size)
    post_layer_mix_flat = post_layer_mix.view(num_tokens, hc_mult)
    comb_res_mix_flat = comb_res_mix.view(num_tokens, hc_mult, hc_mult)

    # Allocate outputs
    residual_out = torch.empty_like(residual_flat)
    mean_out = torch.empty(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=x.device
    )

    # Call fused kernel
    mhc_post_mean_fused_tilelang(
        comb_res_mix_flat,
        residual_flat,
        post_layer_mix_flat,
        x_flat,
        residual_out,
        mean_out,
        hc_mult,
        hidden_size,
    )

    return (
        residual_out.view(*outer_shape, hc_mult, hidden_size),
        mean_out.view(*outer_shape, hidden_size),
    )


def _mhc_post_hc_head_fused_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    hc_head_fn: torch.Tensor,
    hc_head_scale: torch.Tensor,
    hc_head_base: torch.Tensor,
    rms_norm_eps: float,
    hc_eps: float,
) -> torch.Tensor:
    """Fake implementation for shape inference."""
    hidden_size = residual.shape[-1]
    outer_shape = residual.shape[:-2]
    return torch.empty(
        *outer_shape, hidden_size, dtype=torch.bfloat16, device=x.device
    )


def _mhc_post_hc_head_norm_fused_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    hc_head_fn: torch.Tensor,
    hc_head_scale: torch.Tensor,
    hc_head_base: torch.Tensor,
    norm_weight: torch.Tensor,
    rms_norm_eps: float,
    hc_eps: float,
    norm_eps: float,
    mtp_buffer: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fake implementation for shape inference."""
    hidden_size = residual.shape[-1]
    outer_shape = residual.shape[:-2]
    return torch.empty(
        *outer_shape, hidden_size, dtype=torch.bfloat16, device=x.device
    )


def _mhc_post_mean_fused_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fake implementation for shape inference."""
    return torch.empty_like(residual), torch.empty_like(x)


# Register custom ops for torch.compile compatibility
direct_register_custom_op(
    op_name="mhc_post_hc_head_fused",
    op_func=mhc_post_hc_head_fused,
    mutates_args=[],
    fake_impl=_mhc_post_hc_head_fused_fake,
)

direct_register_custom_op(
    op_name="mhc_post_hc_head_norm_fused",
    op_func=mhc_post_hc_head_norm_fused,
    mutates_args=[],
    fake_impl=_mhc_post_hc_head_norm_fused_fake,
)

direct_register_custom_op(
    op_name="mhc_post_mean_fused",
    op_func=mhc_post_mean_fused,
    mutates_args=[],
    fake_impl=_mhc_post_mean_fused_fake,
)
