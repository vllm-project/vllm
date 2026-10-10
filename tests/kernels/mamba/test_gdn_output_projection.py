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
    GDN_AITER_NORM_OUT_PROJ_AVAILABLE,
    QwenGatedDeltaNetAttention,
)


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


@pytest.mark.skipif(
    not GDN_AITER_NORM_OUT_PROJ_AVAILABLE,
    reason="needs ROCm with the aiter FlyDSL GDN norm + out_proj kernel",
)
# At this shape the aiter kernel runs 1 to 5 tokens. 6 and 64 tokens run the Triton
# gated norm and then the GEMM of out_proj, which is also the path of every prefill.
@pytest.mark.parametrize("num_tokens", [1, 3, 5, 6, 64])
@torch.inference_mode()
def test_gated_rmsnorm_out_proj_op_matches_unfused(
    default_vllm_config,
    num_tokens: int,
) -> None:
    # One rank of a GDN layer with 16 value heads of 128 and hidden size 8192.
    num_heads, head_dim, hidden = 16, 128, 8192
    eps = 1e-6
    dtype = torch.bfloat16
    torch.manual_seed(0)

    norm_weight = torch.empty(head_dim, device="cuda", dtype=dtype).normal_(1.0, 0.1)
    weight = (0.02 * torch.randn(hidden, num_heads * head_dim, device="cuda")).to(dtype)
    core_attn_out = torch.randn(
        num_tokens, num_heads, head_dim, device="cuda", dtype=dtype
    )
    z = torch.randn_like(core_attn_out)
    out = torch.ops.vllm.qwen_gdn_gated_rmsnorm_out_proj(
        core_attn_out, z, norm_weight, weight, eps
    )

    normed = RMSNormGated.forward_static(
        core_attn_out,
        z,
        norm_weight,
        eps,
        dtype,
        group_size=None,
        norm_before_gate=True,
        activation="silu",
    )
    ref = torch.nn.functional.linear(normed.flatten(-2).float(), weight.float())
    assert out.shape == (num_tokens, hidden)
    assert out.dtype == dtype
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)
