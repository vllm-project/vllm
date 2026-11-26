# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for the AMD Qwen4Exp hyper-connection kernels."""

import pytest
import torch
import torch.nn.functional as F

from tests.kernels.utils import opcheck
from vllm._aiter_ops import rocm_aiter_ops
from vllm.models.qwen4_exp.amd import hyperconnection  # noqa: F401
from vllm.models.qwen4_exp.amd.ops.hc import (
    grouped_gemma_rmsnorm,
    hc_combine,
    hc_combine_norm,
    hc_gate_mix,
    hc_silu,
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


@pytest.fixture
def aiter_enabled(monkeypatch: pytest.MonkeyPatch):
    # CI does not set VLLM_ROCM_USE_AITER, which gates the two-stage kernels.
    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    rocm_aiter_ops.refresh_env_variables()
    yield
    monkeypatch.undo()
    rocm_aiter_ops.refresh_env_variables()


def _assert_combine_and_mix_close(
    actual: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection: torch.Tensor,
    norm_weight: torch.Tensor,
    w_down_inject: torch.Tensor,
    w_up: torch.Tensor,
) -> None:
    actual_out, actual_input, actual_injection = actual
    expected = _ref_hc_combine(residual, block_output, injection)
    xn = _ref_grouped_gemma_rmsnorm(expected, norm_weight).float()
    down = xn @ w_down_inject.float().T
    gate = F.silu(down[:, :LORA_RANK] / HC) @ w_up.float().T
    expected_input = (
        torch.sigmoid(gate.unflatten(-1, (HC, HIDDEN_SIZE)))
        * xn.unflatten(-1, (HC, HIDDEN_SIZE))
    ).mean(-2)
    torch.testing.assert_close(actual_out, expected)
    torch.testing.assert_close(
        actual_input, expected_input.to(torch.bfloat16), atol=3e-2, rtol=2e-2
    )
    torch.testing.assert_close(
        actual_injection,
        down[:, LORA_RANK : LORA_RANK + HC].to(torch.bfloat16),
        atol=3e-2,
        rtol=2e-2,
    )


# Up to the cutoff (4) the op runs the two-stage skinny decode kernels; 5, 8 and
# 64 run the Triton chain.
@pytest.mark.parametrize("norm_size", NORM_SIZES)
@pytest.mark.parametrize("num_tokens", [1, 2, 3, 4, 5, 8, 64])
def test_hc_two_stage_combine_and_mix(
    aiter_enabled: None, num_tokens: int, norm_size: int
) -> None:
    if not rocm_aiter_ops.is_hc_two_stage_enabled():
        pytest.skip("AITER FlyDSL two-stage HC kernels are not supported here")
    set_random_seed(0)
    residual = _randn(num_tokens, HYPER_HIDDEN_SIZE)
    block_output = _randn(num_tokens, HIDDEN_SIZE)
    injection = _randn(num_tokens, DOWN_N)[:, LORA_RANK : LORA_RANK + HC]
    norm_weight = _randn(norm_size)
    w_down_inject = _randn(DOWN_N, HYPER_HIDDEN_SIZE) / HYPER_HIDDEN_SIZE**0.5
    w_up = _randn(HYPER_HIDDEN_SIZE, LORA_RANK) / LORA_RANK**0.5
    w_down_inject_folded = rocm_aiter_ops.hc_fold_norm_weight(
        w_down_inject, norm_weight, HC
    )
    max_tokens = rocm_aiter_ops.hc_two_stage_max_tokens(
        HYPER_HIDDEN_SIZE, DOWN_N, HC, LORA_RANK
    )
    assert max_tokens == rocm_aiter_ops._HC_TWO_STAGE_MAX_TOKENS
    args = (
        residual,
        block_output,
        injection,
        norm_weight,
        w_down_inject,
        w_down_inject_folded,
        w_up,
        EPS,
        HC,
        LORA_RANK,
        max_tokens,
    )
    op = torch.ops.vllm.qwen4_exp_hc_two_stage_combine_and_mix

    _assert_combine_and_mix_close(
        op(*args), residual, block_output, injection, norm_weight, w_down_inject, w_up
    )
    if num_tokens in (2, 5):
        opcheck(op, args)


# Covers the module wiring: the norm fold in process_weights_after_loading and
# the dispatch to the two-stage kernels at 1-4 tokens only.
@pytest.mark.parametrize("hc_per_branch_norm", [False, True])
@pytest.mark.parametrize("num_tokens", [1, 4, 8])
def test_gated_residual_combine_and_mix(
    dist_init,
    aiter_enabled: None,
    monkeypatch: pytest.MonkeyPatch,
    num_tokens: int,
    hc_per_branch_norm: bool,
) -> None:
    if not rocm_aiter_ops.is_hc_two_stage_enabled():
        pytest.skip("AITER FlyDSL two-stage HC kernels are not supported here")
    from vllm.models.qwen4_exp.common.hyperconnection import HyperConnectionConfig

    config = HyperConnectionConfig(
        hc_count=HC,
        hidden_size=HIDDEN_SIZE,
        hc_lowrank=LORA_RANK,
        rms_norm_eps=EPS,
        hc_per_branch_norm=hc_per_branch_norm,
    )
    with torch.device("cuda"):
        module = hyperconnection.GatedResidual(config)
    max_tokens = rocm_aiter_ops._HC_TWO_STAGE_MAX_TOKENS
    assert module.hc_two_stage_max_tokens == max_tokens

    set_random_seed(0)
    norm_weight = module.hc_norm.weight
    w_down_inject = module.input_mix_weight_down_block_inject.weight
    w_up = module.input_mix_weight_up.weight
    with torch.no_grad():
        norm_weight.copy_(_randn(norm_weight.numel()))
        w_down_inject.copy_(_randn(DOWN_N, HYPER_HIDDEN_SIZE) / HYPER_HIDDEN_SIZE**0.5)
        # Checkpoints leave the 16-row alignment padding zero.
        w_down_inject[LORA_RANK + HC :] = 0
        w_up.copy_(_randn(HYPER_HIDDEN_SIZE, LORA_RANK) / LORA_RANK**0.5)
    module.process_weights_after_loading()

    calls: list[int] = []
    two_stage = rocm_aiter_ops.hc_two_stage_combine_and_mix

    def spy(**kwargs):
        calls.append(kwargs["residual"].shape[0])
        return two_stage(**kwargs)

    monkeypatch.setattr(rocm_aiter_ops, "hc_two_stage_combine_and_mix", spy)

    residual = _randn(num_tokens, HYPER_HIDDEN_SIZE)
    block_output = _randn(num_tokens, HIDDEN_SIZE)
    injection = _randn(num_tokens, DOWN_N)[:, LORA_RANK : LORA_RANK + HC]
    with torch.no_grad():
        actual = module.combine_and_mix(residual, block_output, injection)

    assert calls == ([num_tokens] if num_tokens <= max_tokens else [])
    _assert_combine_and_mix_close(
        actual, residual, block_output, injection, norm_weight, w_down_inject, w_up
    )

    # A LoRA wrapper around a projection must keep the module chain, so its
    # delta is applied.
    class Wrapped(torch.nn.Module):
        def __init__(self, base: torch.nn.Module) -> None:
            super().__init__()
            self.base = base

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.base(x)

    module.input_mix_weight_up = Wrapped(module.input_mix_weight_up)
    calls.clear()
    with torch.no_grad():
        actual = module.combine_and_mix(residual, block_output, injection)
    assert calls == []
    _assert_combine_and_mix_close(
        actual, residual, block_output, injection, norm_weight, w_down_inject, w_up
    )


def test_gated_residual_two_stage_needs_bf16(
    dist_init, aiter_enabled: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    if not rocm_aiter_ops.is_hc_two_stage_enabled():
        pytest.skip("AITER FlyDSL two-stage HC kernels are not supported here")
    from types import SimpleNamespace

    from vllm.models.qwen4_exp.common.hyperconnection import HyperConnectionConfig

    vllm_config = SimpleNamespace(model_config=SimpleNamespace(dtype=torch.float16))
    monkeypatch.setattr(hyperconnection, "get_current_vllm_config", lambda: vllm_config)
    config = HyperConnectionConfig(
        hc_count=HC, hidden_size=HIDDEN_SIZE, hc_lowrank=LORA_RANK, rms_norm_eps=EPS
    )
    with torch.device("cuda"):
        module = hyperconnection.GatedResidual(config)
    assert module.hc_two_stage_max_tokens == 0
