# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F

from vllm.models.qwen4_exp.amd.ops.hc import (
    MAX_FUSED_UP_M,
    hc_gate_mix,
    hc_up_gate_mix,
)
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="AMD HC kernels require ROCm"
)

HC = 4
HIDDEN_SIZE = 2560
HYPER_HIDDEN_SIZE = HC * HIDDEN_SIZE
LORA_RANK = 320


@pytest.mark.parametrize("num_tokens", [1, 3, 16, 17, 48, 49, 100, MAX_FUSED_UP_M])
def test_hc_up_gate_mix_matches_unfused(num_tokens: int) -> None:
    torch.manual_seed(0)
    xn = torch.randn(num_tokens, HYPER_HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
    lowrank = torch.randn(num_tokens, LORA_RANK, dtype=torch.bfloat16, device="cuda")
    weight = (
        torch.randn(HYPER_HIDDEN_SIZE, LORA_RANK, dtype=torch.bfloat16, device="cuda")
        / LORA_RANK**0.5
    )

    actual = hc_up_gate_mix(xn, lowrank, weight, HC)
    expected = hc_gate_mix(xn, F.linear(lowrank, weight), HC)

    # Only the GEMM accumulation order differs; the bf16 gate can round one ulp
    # apart before the sigmoid.
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
