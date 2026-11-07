# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for the AMD Qwen4Exp hyper-connection kernels."""

import pytest
import torch
import torch.nn.functional as F

from vllm.models.qwen4_exp.amd.ops.hc import (
    grouped_gemma_rmsnorm,
    hc_combine,
    hc_combine_norm,
    hc_gate_mix,
    hc_silu,
)
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm() or not HAS_TRITON,
    reason="AMD HC kernels require ROCm and Triton",
)

HC = 4
HIDDEN_SIZE = 2560
HYPER_HIDDEN_SIZE = HC * HIDDEN_SIZE
EPS = 1e-6
LORA_RANK = 320
DOWN_N = LORA_RANK + HC + 12  # merged down+inject weight, 16-row padded
NUM_TOKENS = [1, 17, 2048]
# A [HIDDEN_SIZE] norm weight is shared by all streams; a [HYPER_HIDDEN_SIZE]
# one is per branch (hc_per_branch_norm).
NORM_SIZES = [HIDDEN_SIZE, HYPER_HIDDEN_SIZE]


def _randn(*shape: int) -> torch.Tensor:
    return torch.randn(*shape, dtype=torch.bfloat16, device="cuda")


def _ref_grouped_gemma_rmsnorm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    grouped = x.float().unflatten(-1, (HC, HIDDEN_SIZE))
    variance = grouped.square().mean(-1, keepdim=True)
    normed = grouped * torch.rsqrt(variance + EPS)
    weight = weight.float().reshape(-1, HIDDEN_SIZE)
    return (normed * (1.0 + weight)).flatten(-2).to(torch.bfloat16)


def _ref_hc_combine(
    residual: torch.Tensor, block_output: torch.Tensor, injection: torch.Tensor
) -> torch.Tensor:
    injection_weight = 2.0 * torch.sigmoid(injection.float() / HC)
    expected = residual.float().unflatten(-1, (HC, HIDDEN_SIZE))
    expected = expected + block_output.float().unsqueeze(
        -2
    ) * injection_weight.unsqueeze(-1)
    return expected.flatten(-2).to(torch.bfloat16)


@pytest.mark.parametrize("norm_size", NORM_SIZES)
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
def test_grouped_gemma_rmsnorm(num_tokens: int, norm_size: int) -> None:
    torch.manual_seed(0)
    x = _randn(num_tokens, HYPER_HIDDEN_SIZE)
    weight = _randn(norm_size)

    actual = grouped_gemma_rmsnorm(x, weight, EPS, HC)

    torch.testing.assert_close(actual, _ref_grouped_gemma_rmsnorm(x, weight))


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
def test_hc_silu(num_tokens: int) -> None:
    # lowrank is a column slice of the merged down+inject output, as in the model.
    torch.manual_seed(0)
    lowrank = _randn(num_tokens, DOWN_N)[:, :LORA_RANK]

    actual = hc_silu(lowrank, HC)

    expected = F.silu(lowrank.float() / HC).to(torch.bfloat16)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
def test_hc_gate_mix(num_tokens: int) -> None:
    torch.manual_seed(0)
    x = _randn(num_tokens, HYPER_HIDDEN_SIZE)
    gate = _randn(num_tokens, HYPER_HIDDEN_SIZE)

    actual = hc_gate_mix(x, gate, HC)
    expected = (
        torch.sigmoid(gate.float().unflatten(-1, (HC, HIDDEN_SIZE)))
        * x.float().unflatten(-1, (HC, HIDDEN_SIZE))
    ).mean(-2)

    torch.testing.assert_close(actual, expected.to(torch.bfloat16))


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
def test_hc_combine(num_tokens: int) -> None:
    # injection is a column slice of the merged down+inject output, as in the model.
    torch.manual_seed(0)
    block_output = _randn(num_tokens, HIDDEN_SIZE)
    residual = _randn(num_tokens, HYPER_HIDDEN_SIZE)
    injection = _randn(num_tokens, DOWN_N)[:, LORA_RANK : LORA_RANK + HC]

    actual = hc_combine(residual, block_output, injection, HC)

    torch.testing.assert_close(
        actual, _ref_hc_combine(residual, block_output, injection)
    )


@pytest.mark.parametrize("norm_size", NORM_SIZES)
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
def test_hc_combine_norm(num_tokens: int, norm_size: int) -> None:
    torch.manual_seed(0)
    block_output = _randn(num_tokens, HIDDEN_SIZE)
    residual = _randn(num_tokens, HYPER_HIDDEN_SIZE)
    injection = _randn(num_tokens, DOWN_N)[:, LORA_RANK : LORA_RANK + HC]
    weight = _randn(norm_size)

    actual, actual_norm = hc_combine_norm(
        residual, block_output, injection, weight, EPS, HC
    )

    expected = _ref_hc_combine(residual, block_output, injection)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        actual_norm, _ref_grouped_gemma_rmsnorm(expected, weight)
    )
