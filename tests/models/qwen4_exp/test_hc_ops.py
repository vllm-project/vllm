# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.models.qwen4_exp.nvidia.ops.hc import (
    grouped_gemma_rmsnorm,
    hc_combine,
    hc_combine_norm,
    hc_gate_mix,
)
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda() or not HAS_TRITON,
    reason="HC kernels require CUDA and Triton",
)

HC = 4
HIDDEN_SIZE = 2560
HYPER_HIDDEN_SIZE = HC * HIDDEN_SIZE
EPS = 1e-6


def test_grouped_gemma_rmsnorm() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, HYPER_HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(HYPER_HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")

    actual = grouped_gemma_rmsnorm(x, weight, EPS, HC)

    grouped = x.float().unflatten(-1, (HC, HIDDEN_SIZE))
    variance = grouped.square().mean(-1, keepdim=True)
    expected = grouped * torch.rsqrt(variance + EPS)
    expected = expected.flatten(-2) * (1.0 + weight.float())
    torch.testing.assert_close(actual, expected.to(torch.bfloat16))


def test_hc_gate_mix() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, HYPER_HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
    gate = torch.randn(2, HYPER_HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")

    actual = hc_gate_mix(x, gate, HC)
    expected = (
        torch.sigmoid(gate.float().unflatten(-1, (HC, HIDDEN_SIZE)))
        * x.float().unflatten(-1, (HC, HIDDEN_SIZE))
    ).mean(-2)

    torch.testing.assert_close(actual, expected.to(torch.bfloat16))


def test_hc_combine() -> None:
    torch.manual_seed(0)
    block_output = torch.randn(2, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
    residual = torch.randn(2, HYPER_HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
    injection = torch.randn(2, HC, dtype=torch.bfloat16, device="cuda")

    actual = hc_combine(residual, block_output, injection, HC)
    injection_weight = 2.0 * torch.sigmoid(injection.float() / HC)
    expected = residual.float().unflatten(-1, (HC, HIDDEN_SIZE))
    expected = expected + block_output.float().unsqueeze(
        -2
    ) * injection_weight.unsqueeze(-1)

    torch.testing.assert_close(actual, expected.flatten(-2).to(torch.bfloat16))


def test_hc_combine_unit_injection() -> None:
    torch.manual_seed(0)
    block_output = torch.randn(2, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
    residual = torch.randn(2, HYPER_HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")

    actual = hc_combine(residual, block_output, None, HC)
    expected = residual.unflatten(-1, (HC, HIDDEN_SIZE))
    expected = expected + block_output.unsqueeze(-2)

    assert torch.equal(actual, expected.flatten(-2))


def test_hc_combine_norm() -> None:
    torch.manual_seed(0)
    block_output = torch.randn(2, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
    residual = torch.randn(2, HYPER_HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
    injection = torch.randn(2, HC, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(HYPER_HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")

    actual, actual_norm = hc_combine_norm(
        residual, block_output, injection, weight, EPS, HC
    )

    injection_weight = 2.0 * torch.sigmoid(injection.float() / HC)
    expected = residual.float().unflatten(-1, (HC, HIDDEN_SIZE))
    expected = expected + block_output.float().unsqueeze(
        -2
    ) * injection_weight.unsqueeze(-1)
    expected = expected.flatten(-2).to(residual.dtype)
    grouped = expected.float().unflatten(-1, (HC, HIDDEN_SIZE))
    variance = grouped.square().mean(-1, keepdim=True)
    expected_norm = grouped * torch.rsqrt(variance + EPS)
    expected_norm = expected_norm.flatten(-2) * (1.0 + weight.float())

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_norm, expected_norm.to(torch.bfloat16))


@pytest.mark.parametrize("num_tokens", [1, 17, 2048])
def test_hc_combine_norm_unit_injection(num_tokens: int) -> None:
    torch.manual_seed(0)
    embedding = torch.randn(
        num_tokens, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda"
    )
    hidden = torch.randn(
        num_tokens, HC, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda"
    )
    weight = torch.randn(HYPER_HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")

    actual, actual_norm = hc_combine_norm(
        hidden.flatten(1), embedding, None, weight, EPS, HC
    )

    expected = (hidden + embedding.unsqueeze(1)).flatten(1)
    expected_norm = grouped_gemma_rmsnorm(expected, weight, EPS, HC)
    assert torch.equal(actual, expected)
    torch.testing.assert_close(actual_norm, expected_norm)
