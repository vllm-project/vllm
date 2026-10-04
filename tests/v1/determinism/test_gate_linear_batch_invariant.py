# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test that GateLinear router logits do not depend on the batch size."""

import pytest
import torch
from utils import skip_unsupported

from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.platforms import current_platform

DEVICE_TYPE = current_platform.device_type


@skip_unsupported
@pytest.mark.parametrize("params_dtype", [torch.bfloat16, torch.float32])
# (3072, 256) has a model-specific fp32 router kernel, (4096, 128) does not.
@pytest.mark.parametrize("hidden_size,num_experts", [(3072, 256), (4096, 128)])
@pytest.mark.parametrize("batch_size", [8, 64, 1024])
def test_gate_linear_logits_are_batch_invariant(
    dist_init,
    default_vllm_config,
    params_dtype: torch.dtype,
    hidden_size: int,
    num_experts: int,
    batch_size: int,
):
    torch.manual_seed(0)
    gate = GateLinear(
        hidden_size,
        num_experts,
        out_dtype=torch.float32,
        params_dtype=params_dtype,
    ).to(DEVICE_TYPE)
    torch.nn.init.normal_(gate.weight, std=0.02)
    x = torch.randn(batch_size, hidden_size, dtype=torch.bfloat16, device=DEVICE_TYPE)

    alone, _ = gate(x[:1])
    batched, _ = gate(x)

    assert batched.dtype == torch.float32
    assert torch.equal(alone[0], batched[0])
