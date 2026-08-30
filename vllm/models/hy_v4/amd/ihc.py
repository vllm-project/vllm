# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm Triton kernels for HY V4 independent Hyper-Connections."""

import torch
from torch import nn
from transformers import PretrainedConfig

from vllm.models.hy_v4.nvidia.hc import (
    HYV4HCPostLayer as BaseHCPostLayer,
)
from vllm.models.hy_v4.nvidia.hc import (
    HYV4HCPreLayer as BaseHCPreLayer,
)
from vllm.triton_utils import tl, triton


@triton.jit
def _hy4_ihc_pre_stage1(
    x_ptr,
    fn_ptr,
    partial_ptr,
    K_TOTAL: tl.constexpr,
    HC_MULT: tl.constexpr,
    HC_POW2: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_K: tl.constexpr,
    PARTIAL_STRIDE: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    split_idx = tl.program_id(1)
    hc_idx = tl.arange(0, HC_POW2)
    hc_mask = hc_idx < HC_MULT

    k_offsets = split_idx * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = k_offsets < K_TOTAL
    x_tile = tl.load(
        x_ptr + token_idx * K_TOTAL + k_offsets, mask=k_mask, other=0.0
    ).to(tl.float32)

    fn_offsets = hc_idx[:, None] * K_TOTAL + k_offsets[None, :]
    fn_mask = hc_mask[:, None] & k_mask[None, :]
    mix_pre = tl.sum(
        tl.load(fn_ptr + fn_offsets, mask=fn_mask, other=0.0) * x_tile[None, :],
        axis=1,
    )
    mix_post = tl.sum(
        tl.load(
            fn_ptr + HC_MULT * K_TOTAL + fn_offsets,
            mask=fn_mask,
            other=0.0,
        )
        * x_tile[None, :],
        axis=1,
    )

    partial = partial_ptr + (token_idx * NUM_SPLITS + split_idx) * PARTIAL_STRIDE
    tl.store(partial, tl.sum(x_tile * x_tile, axis=0))
    tl.store(partial + 1 + hc_idx, mix_pre, mask=hc_mask)
    tl.store(partial + 1 + HC_POW2 + hc_idx, mix_post, mask=hc_mask)


@triton.jit
def _hy4_ihc_pre_stage2(
    x_ptr,
    partial_ptr,
    scale_ptr,
    base_ptr,
    output_ptr,
    post_ptr,
    HIDDEN_SIZE: tl.constexpr,
    K_TOTAL: tl.constexpr,
    HC_MULT: tl.constexpr,
    HC_POW2: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    PARTIAL_STRIDE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    MAGNITUDE: tl.constexpr,
    NORM_EPS: tl.constexpr,
    HC_EPS: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    block_idx = tl.program_id(1)
    hc_idx = tl.arange(0, HC_POW2)
    hc_mask = hc_idx < HC_MULT

    partial = partial_ptr + token_idx * NUM_SPLITS * PARTIAL_STRIDE
    sumsq = tl.zeros((), dtype=tl.float32)
    mix_pre = tl.zeros((HC_POW2,), dtype=tl.float32)
    mix_post = tl.zeros((HC_POW2,), dtype=tl.float32)
    for split_idx in tl.static_range(NUM_SPLITS):
        split = partial + split_idx * PARTIAL_STRIDE
        sumsq += tl.load(split)
        mix_pre += tl.load(split + 1 + hc_idx, mask=hc_mask, other=0.0)
        mix_post += tl.load(split + 1 + HC_POW2 + hc_idx, mask=hc_mask, other=0.0)

    inv_rms = tl.rsqrt(sumsq / K_TOTAL + NORM_EPS)
    pre = (
        tl.sigmoid(
            mix_pre * inv_rms * tl.load(scale_ptr)
            + tl.load(base_ptr + hc_idx, mask=hc_mask, other=0.0)
        )
        + HC_EPS
    )
    if block_idx == 0:
        post = (
            MAGNITUDE
            * tl.sigmoid(
                mix_post * inv_rms * tl.load(scale_ptr + 1)
                + tl.load(base_ptr + HC_MULT + hc_idx, mask=hc_mask, other=0.0)
            )
            + HC_EPS
        )
        tl.store(post_ptr + token_idx * HC_MULT + hc_idx, post, mask=hc_mask)

    d_offsets = block_idx * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_offsets < HIDDEN_SIZE
    x_row = x_ptr + token_idx * K_TOTAL
    output = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for channel_idx in tl.static_range(HC_MULT):
        channel = tl.load(
            x_row + channel_idx * HIDDEN_SIZE + d_offsets,
            mask=d_mask,
            other=0.0,
        )
        gate = tl.sum(tl.where(hc_idx == channel_idx, pre, 0.0), axis=0)
        output += gate * channel.to(tl.float32)
    tl.store(
        output_ptr + token_idx * HIDDEN_SIZE + d_offsets,
        output.to(output_ptr.dtype.element_ty),
        mask=d_mask,
    )


@triton.jit
def _hy4_ihc_post_kernel(
    output_ptr,
    residual_ptr,
    post_ptr,
    result_ptr,
    HIDDEN_SIZE: tl.constexpr,
    HC_MULT: tl.constexpr,
    HC_POW2: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    block_idx = tl.program_id(1)
    hc_idx = tl.arange(0, HC_POW2)
    hc_mask = hc_idx < HC_MULT
    post = tl.load(post_ptr + token_idx * HC_MULT + hc_idx, mask=hc_mask, other=0.0)

    d_offsets = block_idx * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_offsets < HIDDEN_SIZE
    output = tl.load(
        output_ptr + token_idx * HIDDEN_SIZE + d_offsets,
        mask=d_mask,
        other=0.0,
    ).to(tl.float32)
    residual_row = residual_ptr + token_idx * HC_MULT * HIDDEN_SIZE
    result_row = result_ptr + token_idx * HC_MULT * HIDDEN_SIZE
    for channel_idx in tl.static_range(HC_MULT):
        residual = tl.load(
            residual_row + channel_idx * HIDDEN_SIZE + d_offsets,
            mask=d_mask,
            other=0.0,
        )
        gate = tl.sum(tl.where(hc_idx == channel_idx, post, 0.0), axis=0)
        result = gate * output + residual.to(tl.float32)
        tl.store(
            result_row + channel_idx * HIDDEN_SIZE + d_offsets,
            result.to(result_ptr.dtype.element_ty),
            mask=d_mask,
        )


def fused_hy4_ihc_pre(
    x: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    magnitude: float,
    norm_eps: float,
    hc_eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Execute HY V4 iHC pre-processing in two fused Triton stages."""
    assert x.dim() == 3
    assert hc_fn.dtype == torch.float32
    assert hc_scale.dtype == torch.float32
    assert hc_base.dtype == torch.float32

    x = x.contiguous()
    hc_fn = hc_fn.contiguous()
    num_tokens, hc_mult, hidden_size = x.shape
    k_total = hc_mult * hidden_size
    assert hc_fn.shape == (2 * hc_mult, k_total)
    assert hc_scale.shape == (2,)
    assert hc_base.shape == (2 * hc_mult,)

    output = torch.empty((num_tokens, hidden_size), dtype=x.dtype, device=x.device)
    post = torch.empty((num_tokens, hc_mult), dtype=torch.float32, device=x.device)
    if num_tokens == 0:
        return output, post

    block_k = 1024
    block_d = 1024
    hc_pow2 = triton.next_power_of_2(hc_mult)
    num_splits = triton.cdiv(k_total, block_k)
    partial_stride = 1 + 2 * hc_pow2
    partial = torch.empty(
        (num_tokens, num_splits, partial_stride),
        dtype=torch.float32,
        device=x.device,
    )

    _hy4_ihc_pre_stage1[(num_tokens, num_splits)](
        x,
        hc_fn,
        partial,
        K_TOTAL=k_total,
        HC_MULT=hc_mult,
        HC_POW2=hc_pow2,
        NUM_SPLITS=num_splits,
        BLOCK_K=block_k,
        PARTIAL_STRIDE=partial_stride,
        num_warps=8,
        enable_fp_fusion=False,
    )
    _hy4_ihc_pre_stage2[(num_tokens, triton.cdiv(hidden_size, block_d))](
        x,
        partial,
        hc_scale,
        hc_base,
        output,
        post,
        HIDDEN_SIZE=hidden_size,
        K_TOTAL=k_total,
        HC_MULT=hc_mult,
        HC_POW2=hc_pow2,
        NUM_SPLITS=num_splits,
        PARTIAL_STRIDE=partial_stride,
        BLOCK_D=block_d,
        MAGNITUDE=magnitude,
        NORM_EPS=norm_eps,
        HC_EPS=hc_eps,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output, post


def fused_hy4_ihc_post(
    output: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
) -> torch.Tensor:
    """Execute HY V4 iHC post-processing in one fused Triton kernel."""
    assert output.dim() == 2
    assert post.dtype == torch.float32

    output = output.contiguous()
    residual = residual.contiguous()
    post = post.contiguous()
    num_tokens, hidden_size = output.shape
    hc_mult = post.shape[-1]
    assert residual.shape == (num_tokens, hc_mult, hidden_size)
    assert post.shape == (num_tokens, hc_mult)

    result = torch.empty(
        (num_tokens, hc_mult, hidden_size), dtype=output.dtype, device=output.device
    )
    if num_tokens == 0:
        return result

    block_d = 1024
    _hy4_ihc_post_kernel[(num_tokens, triton.cdiv(hidden_size, block_d))](
        output,
        residual,
        post,
        result,
        HIDDEN_SIZE=hidden_size,
        HC_MULT=hc_mult,
        HC_POW2=triton.next_power_of_2(hc_mult),
        BLOCK_D=block_d,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return result


class HYV4HCPreLayer(BaseHCPreLayer):
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return fused_hy4_ihc_pre(
            x,
            self.hc_fn.weight,
            self.hc_scale,
            self.hc_base,
            self.magnitude,
            self.layernorm_epsilon,
            self.hc_eps,
        )


class HYV4HCPostLayer(BaseHCPostLayer):
    def forward(
        self, x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor
    ) -> torch.Tensor:
        return fused_hy4_ihc_post(x, residual, post)


class HYV4HCLayer(nn.Module):
    """HY V4 iHC boundary backed by ROCm Triton kernels."""

    def __init__(
        self,
        config: PretrainedConfig,
        layer_idx: int,
        init_std: float = 6e-3,
        base_noise_std: float = 0.0,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.enable_ihc = getattr(config, "enable_ihc", False)
        if self.enable_ihc:
            self.hc_pre = HYV4HCPreLayer(
                config,
                config.hidden_size,
                config.hc_mult,
                config.hc_magnitude,
                init_std,
                base_noise_std,
                config.hc_eps,
                config.rms_norm_eps,
                prefix=f"{prefix}.hc_pre",
            )
            self.hc_post = HYV4HCPostLayer(config)

    def prepare_input(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self.enable_ihc:
            return hidden_states
        if hidden_states.dim() == 3:
            return hidden_states
        if hidden_states.dim() != 2:
            raise RuntimeError(
                f"HC expects a 2D/3D tensor, got shape={tuple(hidden_states.shape)}"
            )

        num_tokens, width = hidden_states.shape
        hidden_size = self.config.hidden_size
        hc_mult = self.config.hc_mult
        if width == hidden_size:
            return hidden_states.unsqueeze(1).repeat(1, hc_mult, 1)
        if width == hc_mult * hidden_size:
            return hidden_states.reshape(num_tokens, hc_mult, hidden_size)
        raise RuntimeError(
            f"HC expects last dim to be hidden_size ({hidden_size}) or "
            f"hc_mult*hidden_size ({hc_mult * hidden_size}), got {width}."
        )

    def pre(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        if not self.enable_ihc:
            return hidden_states, None, hidden_states
        reduced, post_gates = self.hc_pre(hidden_states)
        return reduced, post_gates, hidden_states

    def post(
        self,
        output_with_bias: torch.Tensor,
        residual: torch.Tensor,
        post_gates: torch.Tensor | None,
    ) -> torch.Tensor:
        if not self.enable_ihc:
            return output_with_bias + residual
        assert post_gates is not None
        return self.hc_post(output_with_bias, residual, post_gates)
