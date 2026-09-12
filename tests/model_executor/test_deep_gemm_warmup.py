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
    """bmm layers bypass the kernel at runtime, so they must not be warmed."""
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
