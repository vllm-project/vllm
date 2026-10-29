# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused shared-expert gate on ROCm matches the unfused sigmoid(x @ w) * out."""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm()
    or not hasattr(torch.ops._rocm_C, "shared_expert_gate_rdna3"),
    reason="ROCm shared expert gate kernel",
)


@pytest.mark.parametrize("num_tokens", [1, 4, 33, 2048])
@pytest.mark.parametrize("hidden_size", [2048, 2560])
def test_shared_expert_gate_matches_unfused(num_tokens: int, hidden_size: int):
    g = torch.Generator(device="cuda").manual_seed(num_tokens)
    x = torch.randn(
        num_tokens, hidden_size, generator=g, device="cuda", dtype=torch.float16
    )
    weight = 0.05 * torch.randn(
        1, hidden_size, generator=g, device="cuda", dtype=torch.float16
    )
    out = torch.randn(
        num_tokens, hidden_size, generator=g, device="cuda", dtype=torch.float16
    )
    gate = (x.float() @ weight.float().T).half()
    expected = torch.sigmoid(gate) * out

    torch.ops._rocm_C.shared_expert_gate_rdna3(out, x, weight)

    torch.testing.assert_close(out, expected, atol=2e-3, rtol=2e-3)
