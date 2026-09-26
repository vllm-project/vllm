# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch


def _situ(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    gate = 4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)
    up = 25.0 * torch.tanh(up / 25.0)
    return gate * up


def _silu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return gate.float() * torch.sigmoid(gate.float()) * up.float()


def _reference(x: torch.Tensor, weight: torch.Tensor, activation: str) -> torch.Tensor:
    gate_weight, up_weight = weight.chunk(2, dim=0)
    gate = x.float() @ gate_weight.float().T
    up = x.float() @ up_weight.float().T
    output = _silu(gate, up) if activation == "silu" else _situ(gate, up)
    return output.to(torch.bfloat16)


@pytest.mark.parametrize("activation", ("silu", "situ"))
@pytest.mark.parametrize(
    "tokens,intermediate,k,tactic",
    ((1, 64, 256, 2), (8, 192, 1536, 2), (16, 128, 1536, 8)),
)
def test_fused_gated_matches_fp32_reference(
    activation, tokens, intermediate, k, tactic
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SM100 required")

    from vllm.model_executor.kernels.linear.cute_dsl.cutedsl_bf16_gated_gemm import (
        cutedsl_bf16_gated_gemm,
    )

    torch.manual_seed(20260924 + tokens + intermediate + tactic)
    x = torch.randn(tokens, k, device="cuda", dtype=torch.bfloat16) * 0.2
    weight = torch.randn(2 * intermediate, k, device="cuda", dtype=torch.bfloat16) * 0.2

    actual = cutedsl_bf16_gated_gemm(x, weight, activation=activation, tactic=tactic)
    torch.testing.assert_close(
        actual, _reference(x, weight, activation), rtol=2e-2, atol=0.25
    )


@pytest.mark.parametrize("activation", ("silu", "situ"))
@pytest.mark.parametrize("tactic", (0, 2, 8, 11, 19, 24, 25))
def test_fused_gated_tactic_shape_contract(activation, tactic):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SM100 required")

    from vllm.model_executor.kernels.linear.cute_dsl.cutedsl_bf16_gated_gemm import (
        CUTEDSL_GATED_TACTICS,
        cutedsl_bf16_gated_gemm,
    )

    cta_m, cta_n, _, use_2cta = CUTEDSL_GATED_TACTICS[tactic]
    tokens = cta_n + 1
    intermediate = cta_m * (2 if use_2cta else 1)
    k = 256
    torch.manual_seed(1000 + tactic)
    x = torch.randn(tokens, k, device="cuda", dtype=torch.bfloat16) * 0.2
    weight = torch.randn(2 * intermediate, k, device="cuda", dtype=torch.bfloat16) * 0.2

    actual = cutedsl_bf16_gated_gemm(x, weight, activation=activation, tactic=tactic)
    torch.testing.assert_close(
        actual, _reference(x, weight, activation), rtol=2e-2, atol=0.25
    )
