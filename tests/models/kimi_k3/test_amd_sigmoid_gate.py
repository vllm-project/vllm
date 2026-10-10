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


@pytest.mark.parametrize(
    ("num_tokens", "dtype", "contiguous"),
    [
        *[
            (n, dt, True)
            for n in (0, 1, 8, 16, 257)
            for dt in (
                torch.bfloat16,
                torch.float16,
                torch.float32,
            )
        ],
        # Non-contiguous input keeps the torch expression, which sigmoids in
        # the activation dtype rather than fp32.
        (8, torch.bfloat16, False),
    ],
)
@torch.inference_mode()
def test_sigmoid_gate_mul_matches_reference(
    num_tokens: int, dtype: torch.dtype, contiguous: bool
):
    from vllm.models.kimi_k3.amd.ops.sigmoid_gate import sigmoid_gate_mul

    torch.manual_seed(num_tokens)
    gate = torch.randn(num_tokens, WIDTH, dtype=dtype, device="cuda") * 4
    if contiguous:
        x = torch.randn(num_tokens, WIDTH, dtype=dtype, device="cuda")
        expected = (x.float() * gate.float().sigmoid()).to(dtype)
    else:
        x = torch.randn(num_tokens, 2 * WIDTH, dtype=dtype, device="cuda")[:, ::2]
        expected = x * gate.sigmoid()

    out = sigmoid_gate_mul(x, gate)

    assert out.shape == x.shape and out.dtype == dtype
    torch.testing.assert_close(out, expected)
