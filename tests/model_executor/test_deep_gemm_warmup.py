# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

import vllm.model_executor.warmup.deep_gemm_warmup as deep_gemm_warmup


@pytest.fixture(autouse=True)
def clear_fp8_einsum_warmup_cache() -> None:
    deep_gemm_warmup.FP8_EINSUM_WARMUP_CACHE.clear()


def test_fp8_einsum_relax_warmup_covers_exact_small_shapes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_DEEP_GEMM_WARMUP", "relax")
    monkeypatch.setattr(
        deep_gemm_warmup,
        "_generate_optimal_warmup_m_values",
        lambda max_tokens, n, device: [1, 16, 64],
    )
    weight = torch.empty((2, 128, 256), dtype=torch.float8_e4m3fn)

    assert deep_gemm_warmup._get_fp8_einsum_m_values(weight, 64) == [
        *range(1, 33),
        64,
    ]


def test_fp8_einsum_full_warmup_covers_every_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_DEEP_GEMM_WARMUP", "full")
    weight = torch.empty((2, 128, 256), dtype=torch.float8_e4m3fn)

    assert deep_gemm_warmup._get_fp8_einsum_m_values(weight, 5) == [1, 2, 3, 4, 5]


@pytest.mark.parametrize(
    ("tma_aligned_scales", "expected_scale_dtype", "expected_scale_inner"),
    [
        (False, torch.float32, 2),
        (True, torch.int32, 1),
    ],
)
def test_fp8_einsum_warmup_matches_runtime_layout(
    monkeypatch: pytest.MonkeyPatch,
    tma_aligned_scales: bool,
    expected_scale_dtype: torch.dtype,
    expected_scale_inner: int,
) -> None:
    monkeypatch.setattr(
        deep_gemm_warmup,
        "_get_fp8_einsum_m_values",
        lambda weight, max_tokens: [1, 3],
    )
    monkeypatch.setattr(
        deep_gemm_warmup,
        "get_tma_aligned_size",
        lambda size, element_size: (size + 3) // 4 * 4,
    )

    calls = []

    def record_fp8_einsum(equation, a, b, out, *, recipe):
        calls.append((equation, a, b, out, recipe))

    monkeypatch.setattr(deep_gemm_warmup, "fp8_einsum", record_fp8_einsum)

    weight = torch.empty((2, 128, 256), dtype=torch.float8_e4m3fn)
    weight_scale = torch.empty((2, 1, 2), dtype=expected_scale_dtype)
    recipe = (1, 1, 128)

    deep_gemm_warmup._deepgemm_fp8_einsum_warmup(
        weight,
        weight_scale,
        recipe,
        tma_aligned_scales,
        max_tokens=3,
    )

    assert len(calls) == 2
    for num_tokens, (equation, a, b, out, actual_recipe) in zip([1, 3], calls):
        aq, aq_scale = a
        assert equation == "bhr,hdr->bhd"
        assert aq.shape == (num_tokens, 2, 256)
        assert aq.stride() == (256, num_tokens * 256, 1)
        assert aq_scale.shape == (num_tokens, 2, expected_scale_inner)
        assert aq_scale.dtype == expected_scale_dtype
        assert b[0] is weight
        assert b[1] is weight_scale
        assert out.shape == (num_tokens, 2, 128)
        assert actual_recipe == recipe

    assert {
        (weight.size(), recipe, tma_aligned_scales)
    } == deep_gemm_warmup.FP8_EINSUM_WARMUP_CACHE


def test_count_warmup_iterations_includes_unique_fp8_einsum_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = torch.nn.Module()
    target = torch.nn.Module()
    target.wo_a = SimpleNamespace(
        weight=torch.empty((2, 128, 256), dtype=torch.float8_e4m3fn)
    )
    target._einsum_recipe = (1, 1, 128)
    target._tma_aligned_scales = True
    model.add_module("target", target)

    monkeypatch.setattr(
        deep_gemm_warmup,
        "_fp8_linear_may_use_deep_gemm",
        lambda module: False,
    )
    monkeypatch.setattr(
        deep_gemm_warmup,
        "_fp8_einsum_may_use_deep_gemm",
        lambda module: module is target,
    )
    monkeypatch.setattr(
        deep_gemm_warmup,
        "_fused_moe_grouped_gemm_may_use_deep_gemm",
        lambda module: False,
    )
    monkeypatch.setattr(
        deep_gemm_warmup,
        "_get_fp8_einsum_m_values",
        lambda weight, max_tokens: [1, 2, 3],
    )

    assert deep_gemm_warmup._count_warmup_iterations(model, 3) == 3
