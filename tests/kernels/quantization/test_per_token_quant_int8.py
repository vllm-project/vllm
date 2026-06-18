# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Precision tests for vllm's per_token_quant_int8 Triton operator.

For each row the kernel computes absmax, derives a symmetric scale
(absmax / 127), and rounds x * 127 / absmax to int8. Compared against a
float32 PyTorch reference.

Source: vllm/model_executor/layers/quantization/utils/int8_utils.py
"""

import pytest
import torch

from vllm.model_executor.layers.quantization.utils.int8_utils import (
    per_token_quant_int8,
)
from vllm.platforms import current_platform

DEVICE = current_platform.device_type


def per_token_quant_int8_ref(x):
    """Naive per-token symmetric INT8 quantization reference (float32)."""
    original_shape = x.shape
    x = x.reshape(-1, original_shape[-1]).float()
    absmax = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10)
    scale = absmax / 127.0
    x_q = (x * (127.0 / absmax)).round().clamp(-128, 127).to(torch.int8)
    return (
        x_q.view(original_shape),
        scale.view(*original_shape[:-1], 1),
    )


@pytest.mark.parametrize(
    "M,N", [(1, 128), (16, 256), (64, 512), (4, 1024)], ids=lambda v: str(v)
)
@torch.inference_mode()
def test_per_token_quant_int8(M, N):
    """per_token_quant_int8 must match the PyTorch reference (fp32)."""
    torch.manual_seed(0)
    x = torch.randn(M, N, device=DEVICE, dtype=torch.float32)

    x_q, scale = per_token_quant_int8(x)
    x_q_ref, scale_ref = per_token_quant_int8_ref(x)

    # int8 outputs may differ by one ULP at rounding ties.
    torch.testing.assert_close(x_q, x_q_ref, atol=1, rtol=0)
    torch.testing.assert_close(scale, scale_ref, atol=1e-7, rtol=1e-5)


@pytest.mark.parametrize("M,N", [(8, 128), (32, 256)], ids=lambda v: str(v))
@torch.inference_mode()
def test_per_token_quant_int8_bfloat16(M, N):
    """bfloat16 input (cast to fp32 inside the kernel) matches the reference."""
    torch.manual_seed(0)
    x = torch.randn(M, N, device=DEVICE, dtype=torch.bfloat16)

    x_q, scale = per_token_quant_int8(x)
    x_q_ref, scale_ref = per_token_quant_int8_ref(x)

    torch.testing.assert_close(x_q, x_q_ref, atol=1, rtol=0)
    torch.testing.assert_close(scale, scale_ref, atol=1e-5, rtol=1e-3)


@torch.inference_mode()
def test_per_token_quant_int8_3d():
    """3D input is flattened to rows; output keeps the original shape."""
    torch.manual_seed(0)
    x = torch.randn(2, 16, 256, device=DEVICE, dtype=torch.float32)

    x_q, scale = per_token_quant_int8(x)
    x_q_ref, scale_ref = per_token_quant_int8_ref(x)

    assert x_q.shape == x.shape
    assert scale.shape == (2, 16, 1)
    torch.testing.assert_close(x_q, x_q_ref, atol=1, rtol=0)
    torch.testing.assert_close(scale, scale_ref, atol=1e-7, rtol=1e-5)


@torch.inference_mode()
def test_per_token_quant_int8_roundtrip():
    """Dequantized output (x_q * scale) must stay close to the input."""
    torch.manual_seed(0)
    x = torch.randn(16, 512, device=DEVICE, dtype=torch.float32)

    x_q, scale = per_token_quant_int8(x)
    x_deq = x_q.float() * scale

    rel_err = (x_deq - x).abs() / (x.abs() + 1e-10)
    assert rel_err.mean().item() < 0.05
