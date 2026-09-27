# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fused Kimi-K3 MLA output gate must match attn_out * sigmoid(gate)."""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="The fused output gate is a ROCm path"
)

# Kimi-K3 MLA at TP8: 12 heads x 128 value dims per rank.
WIDTH = 12 * 128


@pytest.mark.parametrize("num_tokens", [0, 1, 8, 16, 257])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@torch.inference_mode()
def test_sigmoid_gate_mul_matches_reference(num_tokens: int, dtype: torch.dtype):
    from vllm.models.kimi_k3.amd.ops.sigmoid_gate import sigmoid_gate_mul

    torch.manual_seed(num_tokens)
    x = torch.randn(num_tokens, WIDTH, dtype=dtype, device="cuda")
    gate = torch.randn(num_tokens, WIDTH, dtype=dtype, device="cuda") * 4

    out = sigmoid_gate_mul(x, gate)

    expected = (x.float() * gate.float().sigmoid()).to(dtype)
    assert out.shape == x.shape and out.dtype == dtype
    torch.testing.assert_close(out, expected)


@torch.inference_mode()
def test_sigmoid_gate_mul_falls_back_for_noncontiguous_input():
    from vllm.models.kimi_k3.amd.ops.sigmoid_gate import sigmoid_gate_mul

    x = torch.randn(8, 2 * WIDTH, dtype=torch.bfloat16, device="cuda")[:, ::2]
    gate = torch.randn(8, WIDTH, dtype=torch.bfloat16, device="cuda")

    torch.testing.assert_close(sigmoid_gate_mul(x, gate), x * gate.sigmoid())
