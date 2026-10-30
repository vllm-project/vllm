# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shape contract for ``QwenGatedDeltaNetAttention._output_projection``.

Every ``forward_*`` already allocates ``core_attn_out`` and ``z`` as
``(N, H, D)``. The helper now norms those tensors as-is and flattens once
for ``out_proj``; it used to be rank-agnostic via ``reshape(z_shape_og)``.
"""

from __future__ import annotations

import types

import pytest
import torch

from vllm.model_executor.layers.layernorm import RMSNormGated
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)
from vllm.platforms import current_platform


@pytest.mark.parametrize("dtype", [torch.float32])
@torch.inference_mode()
def test_output_projection_norms_per_head_and_flattens(
    default_vllm_config,
    dtype: torch.dtype,
) -> None:
    num_tokens, num_heads, head_dim = 3, 4, 8
    hidden = num_heads * head_dim

    layer = types.SimpleNamespace()
    layer.norm = RMSNormGated(
        head_dim,
        eps=1e-5,
        group_size=None,
        norm_before_gate=True,
        device="cpu",
        dtype=dtype,
    )
    layer.out_proj = lambda x: (x, None)
    layer._output_projection = types.MethodType(
        QwenGatedDeltaNetAttention._output_projection, layer
    )

    core_attn_out = torch.randn(num_tokens, num_heads, head_dim, dtype=dtype)
    z = torch.randn(num_tokens, num_heads, head_dim, dtype=dtype)
    out = layer._output_projection(core_attn_out, z)

    assert core_attn_out.shape == (num_tokens, num_heads, head_dim)
    assert z.shape == (num_tokens, num_heads, head_dim)
    assert out.shape == (num_tokens, hidden)
    assert out.dtype == dtype


@pytest.mark.parametrize("num_tokens", [1, 37])
@pytest.mark.parametrize("norm_before_gate", [True, False])
@pytest.mark.parametrize("activation", ["silu", "sigmoid"])
@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="the fused gated RMS norm Triton kernel requires CUDA",
)
@torch.inference_mode()
def test_fused_gated_rms_norm_strided_z_bitwise(
    num_tokens: int, norm_before_gate: bool, activation: str
) -> None:
    """The Qwen3.5 fused gated RMS norm reads z from its packed strided view
    of the qkvz projection and matches ``rmsnorm_fn`` on the copied/reshaped
    z bitwise."""
    from vllm.models.qwen3_5.nvidia.ops.fused_gdn_gated_norm import (
        fused_gdn_gated_rms_norm,
    )
    from vllm.third_party.flash_linear_attention.ops.layernorm_guard import (
        rmsnorm_fn,
    )

    torch.manual_seed(0)
    dtype = torch.bfloat16
    num_heads, head_dim = 4, 128
    qkvz_row = 2 * 2 * 128 + 2 * num_heads * head_dim
    qkvz = torch.randn(num_tokens, qkvz_row, dtype=dtype, device="cuda")
    z_view = qkvz[:, qkvz_row - num_heads * head_dim :]
    x = torch.randn(num_tokens, num_heads, head_dim, dtype=dtype, device="cuda")
    weight = torch.randn(head_dim, dtype=torch.float32, device="cuda")
    eps = 1e-6

    ref = rmsnorm_fn(
        x,
        weight,
        None,
        z=z_view.reshape(num_tokens, num_heads, head_dim),
        eps=eps,
        group_size=None,
        norm_before_gate=norm_before_gate,
        activation=activation,
    )

    out = torch.empty(num_tokens, num_heads * head_dim, dtype=dtype, device="cuda")
    fused_gdn_gated_rms_norm(
        x=x.view(num_tokens, -1),
        z=z_view,
        weight=weight,
        out=out,
        eps=eps,
        num_v_heads=num_heads,
        head_v_dim=head_dim,
        norm_before_gate=norm_before_gate,
        activation=activation,
    )

    torch.testing.assert_close(
        out.view(num_tokens, num_heads, head_dim), ref, rtol=0, atol=0
    )


# 48/24/12 value heads per rank: TP1/2/4 of a 48-head model.
@pytest.mark.parametrize("num_heads", [48, 24, 12])
@pytest.mark.parametrize("num_tokens", [1, 37, 300])
@pytest.mark.skipif(
    not current_platform.is_cuda() or not current_platform.has_device_capability(89),
    reason="the FP8 (e4m3fn) output of the fused gated RMS norm requires NVIDIA SM89+",
)
@torch.inference_mode()
def test_fused_gated_rms_norm_fp8_matches_chain(
    num_tokens: int, num_heads: int
) -> None:
    """The FP8 output (fused out_proj input quant) equals the bf16 kernel
    followed by static scaled_fp8_quant."""
    from vllm import _custom_ops as ops
    from vllm.models.qwen3_5.nvidia.ops.fused_gdn_gated_norm import (
        fused_gdn_gated_rms_norm,
    )

    g = torch.Generator(device="cuda").manual_seed(num_tokens)
    dtype = torch.bfloat16
    head_dim = 128
    qkvz_row = 2 * (num_heads // 3) * 128 + 2 * num_heads * head_dim
    qkvz = torch.randn(num_tokens, qkvz_row, generator=g, dtype=dtype, device="cuda")
    z_view = qkvz[:, qkvz_row - num_heads * head_dim :]
    x = torch.randn(
        num_tokens, num_heads * head_dim, generator=g, dtype=dtype, device="cuda"
    )
    weight = (torch.rand(head_dim, generator=g, device="cuda") + 0.5).to(dtype)
    # Small enough that the largest outputs saturate at +-448.
    scale = torch.tensor([0.004], device="cuda")
    kwargs = dict(
        z=z_view,
        weight=weight,
        eps=1e-6,
        num_v_heads=num_heads,
        head_v_dim=head_dim,
        norm_before_gate=True,
        activation="silu",
    )

    ref = torch.empty_like(x)
    fused_gdn_gated_rms_norm(x=x, out=ref, **kwargs)
    ref_q, _ = ops.scaled_fp8_quant(ref, scale, group_shape=(-1, -1))
    q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    fused_gdn_gated_rms_norm(x=x, out=q, scale=scale, **kwargs)

    assert ref_q.float().abs().max().item() == 448.0

    # Arch-dependent compilation flips occasional bf16 rounding boundaries;
    # codes must be within one FP8 step (+0 == -0).
    def codes(t: torch.Tensor) -> torch.Tensor:
        b = t.view(torch.uint8).int()
        return torch.where(b >= 128, -(b & 0x7F), b)

    dist = (codes(q) - codes(ref_q)).abs()
    assert dist.max().item() <= 1
    assert (dist > 0).float().mean().item() <= 1e-4
