# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the NVFP4 W4A16 choice between Marlin and CuTe-DSL.

The first group needs no GPU. The rest needs an sm_12x GPU, the only place where the
dispatch is selected.
"""

import pytest
import torch
import torch.nn as nn

from vllm.config import KernelConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.kernels.linear import (
    FlashInferCuteDslNvFp4W4A16LinearKernel,
    MarlinNvFp4LinearKernel,
    init_nvfp4_linear_kernel,
)
from vllm.model_executor.kernels.linear.nvfp4.base import NvFp4LinearLayerConfig
from vllm.model_executor.kernels.linear.nvfp4.dispatch import (
    _CALIBRATION_TOKENS,
    MarlinCuteDslNvFp4W4A16DispatchKernel,
    _longest_win_run,
    _Plan,
)
from vllm.platforms import current_platform

T, F = True, False


@pytest.mark.parametrize(
    ("wins", "expected"),
    [
        pytest.param(
            [F, F, F, F, T, T, T, T, T, T],
            (F, F, F, F, T, T, T, T, T, T),
            id="run-to-the-end",
        ),
        pytest.param([F] * 10, (F,) * 10, id="never"),
        pytest.param([T] * 10, (T,) * 10, id="always"),
        pytest.param([F, F, F, F, T, F, F, F, F, F], (F,) * 10, id="isolated-is-noise"),
        pytest.param(
            [T, T, T, F, T, T, T, T, F, F],
            (F, F, F, F, T, T, T, T, F, F),
            id="longest-of-two",
        ),
        pytest.param(
            [T, T, T, F, T, T, F, F, F, F],
            (T, T, T, F, F, F, F, F, F, F),
            id="first-of-two-equal-or-longer",
        ),
    ],
)
def test_longest_win_run_keeps_one_run_of_at_least_two(wins, expected):
    assert _longest_win_run(wins) == expected


def test_plan_follows_the_calibrated_count_at_or_above_the_token_count():
    first_cutedsl = len(_CALIBRATION_TOKENS) // 2
    plan = _Plan(tuple(i >= first_cutedsl for i in range(len(_CALIBRATION_TOKENS))))
    for i, tokens in enumerate(_CALIBRATION_TOKENS):
        assert plan.pick_cute(tokens) == (i >= first_cutedsl)
        # One token more rounds up to the next calibrated count.
        if i + 1 < len(_CALIBRATION_TOKENS):
            assert plan.pick_cute(tokens + 1) == (i + 1 >= first_cutedsl)
    # Above the last calibrated count the last decision holds.
    assert plan.pick_cute(_CALIBRATION_TOKENS[-1] * 4)
    assert plan.any_cute


def test_plan_without_a_win_never_picks_cutedsl():
    plan = _Plan((False,) * len(_CALIBRATION_TOKENS))
    assert not plan.any_cute
    assert not any(plan.pick_cute(tokens) for tokens in (1, 100, 100_000))


IS_SM12X = current_platform.is_cuda() and (
    current_platform.is_device_capability_family(120)
)
requires_sm12x = pytest.mark.skipif(
    not IS_SM12X, reason="the dispatch is only selected on sm_12x GPUs"
)


def _make_layer(n: int, k: int, seed: int) -> nn.Module:
    """A layer with random NVFP4 weights, as the kernels expect it after loading."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    layer = nn.Module()
    weight = torch.randint(
        0, 256, (n, k // 2), dtype=torch.uint8, device="cuda", generator=gen
    )
    scale = torch.rand((n, k // 16), device="cuda", generator=gen) * 1.5 + 0.5
    layer.weight = nn.Parameter(weight, requires_grad=False)
    layer.weight_scale = nn.Parameter(
        scale.to(torch.float8_e4m3fn), requires_grad=False
    )
    layer.weight_global_scale = nn.Parameter(
        torch.tensor(0.01, dtype=torch.float32, device="cuda"), requires_grad=False
    )
    layer.output_size_per_partition = n
    layer.input_size_per_partition = k
    layer.params_dtype = torch.bfloat16
    return layer


@requires_sm12x
@pytest.mark.parametrize(("n", "k"), [(2048, 1024), (18560, 4096)])
def test_output_is_that_of_the_kernel_the_dispatch_picks(n, k):
    config = NvFp4LinearLayerConfig()
    marlin = MarlinNvFp4LinearKernel(config)
    cutedsl = FlashInferCuteDslNvFp4W4A16LinearKernel(config)
    dispatch = MarlinCuteDslNvFp4W4A16DispatchKernel(config)
    marlin_layer, cutedsl_layer, dispatch_layer = (_make_layer(n, k, 0) for _ in "abc")
    marlin.process_weights_after_loading(marlin_layer)
    cutedsl.process_weights_after_loading(cutedsl_layer)
    dispatch.process_weights_after_loading(dispatch_layer)

    for tokens in (1, 8, 64, 65, 96, 128, 300, 2048):
        x = torch.randn(tokens, k, dtype=torch.bfloat16, device="cuda")
        from_marlin = marlin.apply_weights(marlin_layer, x)
        from_cutedsl = cutedsl.apply_weights(cutedsl_layer, x)
        if dispatch.uses_cutedsl(tokens):
            # The test can only tell the kernels apart if their results differ.
            assert not torch.equal(from_marlin, from_cutedsl)
            expected = from_cutedsl
        else:
            expected = from_marlin
        torch.testing.assert_close(
            dispatch.apply_weights(dispatch_layer, x), expected, rtol=0, atol=0
        )


@requires_sm12x
def test_layers_of_one_shape_make_the_same_choice():
    n, k = 4096, 2048
    config = NvFp4LinearLayerConfig()
    first = MarlinCuteDslNvFp4W4A16DispatchKernel(config)
    second = MarlinCuteDslNvFp4W4A16DispatchKernel(config)
    first_layer, second_layer = _make_layer(n, k, 1), _make_layer(n, k, 2)
    first.process_weights_after_loading(first_layer)
    second.process_weights_after_loading(second_layer)
    assert [first.uses_cutedsl(t) for t in _CALIBRATION_TOKENS] == [
        second.uses_cutedsl(t) for t in _CALIBRATION_TOKENS
    ]


@requires_sm12x
def test_selection_picks_the_dispatch_by_default():
    with set_current_vllm_config(VllmConfig()):
        kernel = init_nvfp4_linear_kernel(use_a16=True)
    assert type(kernel) is MarlinCuteDslNvFp4W4A16DispatchKernel


@requires_sm12x
def test_selection_keeps_marlin_when_the_dispatch_is_disabled(monkeypatch):
    monkeypatch.setenv("VLLM_NVFP4_W4A16_DISPATCH", "0")
    with set_current_vllm_config(VllmConfig()):
        kernel = init_nvfp4_linear_kernel(use_a16=True)
    assert type(kernel) is MarlinNvFp4LinearKernel


@requires_sm12x
@pytest.mark.parametrize(
    ("backend", "expected"),
    [
        ("marlin", MarlinNvFp4LinearKernel),
        ("flashinfer_cutedsl", FlashInferCuteDslNvFp4W4A16LinearKernel),
    ],
)
def test_selection_honors_an_explicit_backend(backend, expected):
    config = VllmConfig(kernel_config=KernelConfig(linear_backend=backend))
    with set_current_vllm_config(config):
        kernel = init_nvfp4_linear_kernel(use_a16=True)
    assert type(kernel) is expected
