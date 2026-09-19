# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPTQ dequantization helper for the DFlash context-KV precompute path.

The DFlash context-KV precompute rebuilds float K/V weights from GPTQ-packed
parameters when the draft model is quantized (see _dequantize_gptq_proj_weight
in qwen3_dflash.py). These tests pin the packed-layout assumptions (bit order,
grouping, zero points) so that a regression in the format handling fails
loudly without needing a quantized checkpoint.
"""

import torch

from vllm.model_executor.models.qwen3_dflash import (
    _dequantize_gptq_proj_weight,
)


class _FakeGptqLinear:

    def __init__(self, qweight: torch.Tensor, scales: torch.Tensor,
                 qzeros: torch.Tensor):
        self.qweight = qweight
        self.scales = scales
        self.qzeros = qzeros


def _pack_last_axis(codes: torch.Tensor) -> torch.Tensor:
    """Pack 8 4-bit codes (0..15) per int32, LSB first, along the last dim.

    codes: [..., n] with n % 8 == 0 -> packed [..., n // 8].
    """
    n = codes.shape[-1]
    c = codes.reshape(*codes.shape[:-1], n // 8, 8)
    packed = torch.zeros(*codes.shape[:-1], n // 8, dtype=torch.int32)
    for b in range(8):
        packed += (c[..., b] << (4 * b)).int()
    return packed


def test_dequantize_gptq_proj_weight_roundtrip():
    """Random asymmetric codes/zero points round-trip exactly.

    Exercises the bit order of both the input-axis code packing and the
    output-axis zero-point packing, plus the per-group scale broadcast.
    """
    torch.manual_seed(0)
    n_in, n_out, group_size = 512, 384, 128
    num_groups = n_in // group_size
    codes = torch.randint(0, 16, (n_in, n_out), dtype=torch.int32)
    zeros = torch.randint(0, 16, (num_groups, n_out), dtype=torch.int32)
    scales = torch.rand(num_groups, n_out, dtype=torch.float16) * 0.05 + 1e-3

    # qweight: codes packed LSB-first along in -> [in/8, out]
    qweight = _pack_last_axis(codes.t()).t()
    # qzeros: zero points packed LSB-first along out -> [in/gs, out/8]
    qzeros = _pack_last_axis(zeros)

    rebuilt = _dequantize_gptq_proj_weight(
        _FakeGptqLinear(qweight, scales, qzeros))  # [out, in]

    group_ids = torch.arange(n_in) // group_size
    expected = ((codes.float() - zeros[group_ids].float()) *
                scales[group_ids].float()).t()
    torch.testing.assert_close(rebuilt, expected, atol=1e-2, rtol=1e-2)


def test_dequantize_gptq_proj_weight_symmetric():
    """Symmetric quantizers store zero point 8 in every nibble."""
    torch.manual_seed(1)
    n_in, n_out, group_size = 256, 192, 64
    num_groups = n_in // group_size
    weight = torch.randn(n_out, n_in)
    # Reference grouped symmetric int4 quantization.
    grouped = weight.t().reshape(num_groups, group_size, n_out)
    scale = grouped.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 8.0
    # Codes carry the unsigned +8 offset.
    codes = torch.clamp(torch.round(grouped / scale), -8, 7).to(
        torch.int32) + 8
    codes = codes.reshape(n_in, n_out)

    qweight = _pack_last_axis(codes.t()).t()
    # Zero point 8 (0x8 in every nibble) along the output axis; the int32
    # signed form of 0x88888888 is -2004318072.
    qzeros = torch.full((num_groups, n_out // 8),
                        -2004318072,
                        dtype=torch.int32)
    rebuilt = _dequantize_gptq_proj_weight(
        _FakeGptqLinear(qweight, scale.squeeze(1).to(torch.float16), qzeros))

    group_ids = torch.arange(n_in) // group_size
    expected = ((codes.float() - 8.0) *
                scale.squeeze(1)[group_ids].float()).t()
    torch.testing.assert_close(rebuilt, expected, atol=1e-2, rtol=1e-2)
