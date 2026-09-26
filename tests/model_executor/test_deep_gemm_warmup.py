# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.kernels.linear.scaled_mm.deep_gemm import (
    DeepGemmFp8BlockScaledMMKernel,
)
from vllm.model_executor.warmup import deep_gemm_warmup


def _block_fp8_layer(n: int, k: int, scale_name: str = "weight_scale"):
    layer = SimpleNamespace(
        weight=torch.empty((n, k), dtype=torch.float8_e4m3fn),
        weight_block_size=[128, 128],
        deep_gemm_warmup_provider=object.__new__(DeepGemmFp8BlockScaledMMKernel),
    )
    setattr(layer, scale_name, torch.empty((n // 128, k // 128), dtype=torch.float32))
    return layer


def _run_warmup(monkeypatch, layers) -> list[dict]:
    calls: list[dict] = []
    monkeypatch.setattr(
        deep_gemm_warmup,
        "get_mk_alignment_for_contiguous_layout",
        lambda: [128, 128],
    )
    monkeypatch.setattr(
        deep_gemm_warmup,
        "_deepgemm_fp8_gemm_nt_warmup",
        lambda **kwargs: calls.append(kwargs),
    )
    model = SimpleNamespace(modules=lambda: iter(layers))
    deep_gemm_warmup.deepgemm_fp8_gemm_nt_warmup(model, max_tokens=16)
    return calls


@pytest.mark.parametrize("scale_name", ["weight_scale", "weight_scale_inv"])
def test_registered_deep_gemm_layer_is_warmed_up(monkeypatch, scale_name) -> None:
    """Any layer stamped by the DeepGEMM kernel is warmed, regardless of the
    quantization method that owns it or the name of its scale parameter."""
    layer = _block_fp8_layer(256, 128, scale_name)

    calls = _run_warmup(monkeypatch, [layer])

    assert len(calls) == 1
    assert calls[0]["w"] is layer.weight
    assert calls[0]["ws"] is getattr(layer, scale_name)


def test_n_multiple_of_64_matches_kernel_selection(monkeypatch) -> None:
    """DeepGEMM accepts N % 64 == 0 (e.g. DeepSeek kv_a_proj_with_mqa, N=576),
    so warmup must not require N % 128 == 0."""
    assert len(_run_warmup(monkeypatch, [_block_fp8_layer(576, 256)])) == 1


def test_unstamped_and_mismatched_block_layers_are_skipped(monkeypatch) -> None:
    mismatched = _block_fp8_layer(256, 128)
    mismatched.weight_block_size = [1, 32]
    layers = [SimpleNamespace(), mismatched, _block_fp8_layer(256, 128)]

    calls = _run_warmup(monkeypatch, layers)

    assert len(calls) == 1
    assert calls[0]["w"] is layers[-1].weight


@pytest.mark.parametrize("is_bmm", [False, True])
def test_kernel_registers_itself_as_warmup_provider(is_bmm) -> None:
    """Bmm layers bypass the kernel at runtime, so they must not be warmed."""
    kernel = object.__new__(DeepGemmFp8BlockScaledMMKernel)
    kernel.is_deep_gemm_supported = False
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(
        torch.empty((256, 128), dtype=torch.float8_e4m3fn), requires_grad=False
    )
    layer.weight_scale = torch.nn.Parameter(
        torch.empty((2, 1), dtype=torch.float32), requires_grad=False
    )
    layer.weight_block_size = [128, 128]
    layer.is_bmm = is_bmm

    kernel.process_weights_after_loading(layer)

    provider = getattr(layer, "deep_gemm_warmup_provider", None)
    assert provider is (None if is_bmm else kernel)


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
        "_deep_gemm_linear_data",
        lambda module: None,
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
