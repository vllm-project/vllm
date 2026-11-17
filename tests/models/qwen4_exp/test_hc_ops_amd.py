# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for the AMD Qwen4Exp hyper-connection kernels."""

import pytest
import torch
import torch.nn.functional as F

from tests.kernels.utils import opcheck
from vllm.model_executor.layers.utils import (
    rocm_unquantized_gemm,
    rocm_unquantized_gemm_impl,
)
from vllm.models.qwen4_exp.amd.ops import hc as hc_ops
from vllm.models.qwen4_exp.amd.ops.hc import (
    MIN_FUSED_UP_M,
    grouped_gemma_rmsnorm,
    hc_combine,
    hc_combine_norm,
    hc_gate_mix,
    hc_silu,
    hc_up_gate_mix,
    hc_up_gate_mix_supported,
)
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON
from vllm.utils.torch_utils import set_random_seed

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
    set_random_seed(0)
    x = _randn(num_tokens, HYPER_HIDDEN_SIZE)
    weight = _randn(norm_size)

    actual = grouped_gemma_rmsnorm(x, weight, EPS, HC)

    torch.testing.assert_close(actual, _ref_grouped_gemma_rmsnorm(x, weight))


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
def test_hc_silu(num_tokens: int) -> None:
    # lowrank is a column slice of the merged down+inject output, as in the model.
    set_random_seed(0)
    lowrank = _randn(num_tokens, DOWN_N)[:, :LORA_RANK]

    actual = hc_silu(lowrank, HC)

    expected = F.silu(lowrank.float() / HC).to(torch.bfloat16)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
def test_hc_gate_mix(num_tokens: int) -> None:
    set_random_seed(0)
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
    set_random_seed(0)
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
    set_random_seed(0)
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


def _up_gate_mix_inputs(
    num_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    w_up = _randn(HYPER_HIDDEN_SIZE, LORA_RANK) / LORA_RANK**0.5
    if not hc_up_gate_mix_supported(w_up, HC):
        pytest.skip("the fused HC up projection + gate mix requires gfx950")
    # lowrank is hc_silu's output, as in the model; the atol below assumes its range.
    lowrank = hc_silu(_randn(num_tokens, LORA_RANK), HC)
    xn = _randn(num_tokens, HYPER_HIDDEN_SIZE)
    return lowrank, xn, w_up


def _hc_up_gate_mix_counting_unfused(
    monkeypatch: pytest.MonkeyPatch, *args: torch.Tensor
) -> tuple[torch.Tensor, int]:
    # A spy on the unfused GEMM rather than torch.profiler, which records
    # nothing while the CI kernel recorder is registered.
    calls = 0

    def spy(*gemm_args):
        nonlocal calls
        calls += 1
        return rocm_unquantized_gemm_impl(*gemm_args)

    with monkeypatch.context() as m:
        m.setattr(hc_ops, "rocm_unquantized_gemm_impl", spy)
        out = hc_up_gate_mix(*args)
    return out, calls


# 4097 leaves a one-row last row tile.
@pytest.mark.parametrize("num_tokens", [MIN_FUSED_UP_M, 4097])
def test_hc_up_gate_mix(monkeypatch: pytest.MonkeyPatch, num_tokens: int) -> None:
    set_random_seed(0)
    lowrank, xn, w_up = _up_gate_mix_inputs(num_tokens)

    actual, unfused_gemms = _hc_up_gate_mix_counting_unfused(
        monkeypatch, lowrank, xn, w_up
    )

    assert unfused_gemms == 0
    # The kernel rounds the gate to bf16 like the unfused GEMM output; its
    # accumulation order can still move a gate by one bf16 ulp.
    gate = (lowrank.float() @ w_up.float().T).to(torch.bfloat16).float()
    expected = (
        torch.sigmoid(gate.unflatten(-1, (HC, HIDDEN_SIZE)))
        * xn.float().unflatten(-1, (HC, HIDDEN_SIZE))
    ).mean(-2)
    torch.testing.assert_close(
        actual, expected.to(torch.bfloat16), atol=1e-3, rtol=1.6e-2
    )
    opcheck(torch.ops.vllm.qwen4_exp_hc_up_gate_mix, (lowrank, xn, w_up))


def test_hc_up_gate_mix_below_threshold_is_unfused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_random_seed(0)
    lowrank, xn, w_up = _up_gate_mix_inputs(MIN_FUSED_UP_M - 1)

    actual, unfused_gemms = _hc_up_gate_mix_counting_unfused(
        monkeypatch, lowrank, xn, w_up
    )

    # The kernel can match the unfused chain bit for bit, so the call count is
    # what tells the two paths apart.
    assert unfused_gemms == 1
    expected = hc_gate_mix(xn, rocm_unquantized_gemm(None, lowrank, w_up), HC)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
