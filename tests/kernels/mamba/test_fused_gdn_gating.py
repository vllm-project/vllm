# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Accuracy tests for fused_gdn_gating_kernel."""

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    fused_gdn_gating,
)
from vllm.platforms import current_platform

requires_gpu = pytest.mark.skipif(
    not (current_platform.is_cuda() or current_platform.is_xpu()),
    reason="requires CUDA or XPU",
)

DEVICE = torch.device("xpu") if current_platform.is_xpu() else torch.device("cuda")


def fused_gdn_gating_ref(A_log, a, b, dt_bias, beta=1.0, threshold=20.0):
    x = a.float() + dt_bias.float()
    softplus_x = F.softplus(x, beta=beta, threshold=threshold)
    g = -torch.exp(A_log.float()) * softplus_x
    g = g.unsqueeze(0)
    beta_output = torch.sigmoid(b.float()).to(b.dtype)
    beta_output = beta_output.unsqueeze(0)
    return g, beta_output


CONFIGS = [
    pytest.param(1, 8, torch.float32, id="B1_H8_f32"),
    pytest.param(1, 16, torch.float32, id="B1_H16_f32"),
    pytest.param(2, 32, torch.float32, id="B2_H32_f32"),
    pytest.param(4, 64, torch.float32, id="B4_H64_f32"),
    pytest.param(8, 128, torch.float32, id="B8_H128_f32"),
    pytest.param(16, 256, torch.float32, id="B16_H256_f32"),
    pytest.param(1, 8, torch.bfloat16, id="B1_H8_bf16"),
    pytest.param(2, 16, torch.bfloat16, id="B2_H16_bf16"),
    pytest.param(4, 32, torch.bfloat16, id="B4_H32_bf16"),
    pytest.param(8, 64, torch.bfloat16, id="B8_H64_bf16"),
    pytest.param(16, 128, torch.bfloat16, id="B16_H128_bf16"),
    pytest.param(1, 8, torch.float16, id="B1_H8_f16"),
    pytest.param(2, 16, torch.float16, id="B2_H16_f16"),
    pytest.param(4, 32, torch.float16, id="B4_H32_f16"),
    pytest.param(8, 64, torch.float16, id="B8_H64_f16"),
    pytest.param(16, 128, torch.float16, id="B16_H128_f16"),
    # Edge cases: num_heads < BLK_HEADS=8, odd, non-power-of-two
    pytest.param(1, 1, torch.float32, id="B1_H1_f32"),
    pytest.param(1, 7, torch.float32, id="B1_H7_f32"),
    pytest.param(32, 9, torch.float32, id="B32_H9_f32"),
    # Large
    pytest.param(32, 512, torch.bfloat16, id="B32_H512_bf16"),
    pytest.param(64, 256, torch.float16, id="B64_H256_f16"),
]


@requires_gpu
@pytest.mark.parametrize("batch,num_heads,dtype", CONFIGS)
@torch.inference_mode()
def test_fused_gdn_gating(batch, num_heads, dtype):
    A_log = torch.randn(num_heads, device=DEVICE, dtype=torch.float32)
    a = torch.randn(batch, num_heads, device=DEVICE, dtype=dtype)
    b = torch.randn(batch, num_heads, device=DEVICE, dtype=dtype)
    dt_bias = torch.randn(num_heads, device=DEVICE, dtype=torch.float32)

    g_out, beta_out = fused_gdn_gating(A_log, a, b, dt_bias)
    g_ref, beta_ref = fused_gdn_gating_ref(A_log, a, b, dt_bias)

    assert beta_out.dtype == b.dtype, (
        f"beta_output dtype mismatch: expected {b.dtype}, got {beta_out.dtype}"
    )
    torch.testing.assert_close(g_out, g_ref, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(
        beta_out.float(), beta_ref.float(), atol=1e-3, rtol=1e-3
    )


@requires_gpu
@pytest.mark.parametrize(
    "beta_val,threshold_val",
    [
        pytest.param(1.0, 20.0, id="default"),
        pytest.param(0.5, 10.0, id="beta0.5_thr10"),
        pytest.param(2.0, 40.0, id="beta2_thr40"),
    ],
)
@torch.inference_mode()
def test_fused_gdn_gating_params(beta_val, threshold_val):
    batch, num_heads = 4, 32
    A_log = torch.randn(num_heads, device=DEVICE, dtype=torch.float32)
    a = torch.randn(batch, num_heads, device=DEVICE, dtype=torch.float32)
    b = torch.randn(batch, num_heads, device=DEVICE, dtype=torch.float32)
    dt_bias = torch.randn(num_heads, device=DEVICE, dtype=torch.float32)

    g_out, beta_out = fused_gdn_gating(
        A_log, a, b, dt_bias, beta=beta_val, threshold=threshold_val
    )
    g_ref, beta_ref = fused_gdn_gating_ref(
        A_log, a, b, dt_bias, beta=beta_val, threshold=threshold_val
    )

    assert beta_out.dtype == b.dtype, (
        f"beta_output dtype mismatch: expected {b.dtype}, got {beta_out.dtype}"
    )
    torch.testing.assert_close(g_out, g_ref, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(
        beta_out.float(), beta_ref.float(), atol=1e-3, rtol=1e-3
    )
