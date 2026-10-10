# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest
import torch

from vllm.model_executor.kernels.linear.scaled_mm.deep_gemm import (
    DeepGemmFp8BlockScaledMMKernel,
)
from vllm.model_executor.warmup import deep_gemm_warmup, kernel_warmup


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


@pytest.mark.cpu_test
@pytest.mark.parametrize("has_draft", [False, True])
@pytest.mark.parametrize(
    "supported, mode", [(True, "relax"), (True, "skip"), (False, "relax")]
)
def test_kernel_warmup_covers_draft_model(monkeypatch, has_draft, supported, mode):
    """Draft-model kernels must be warmed before serving, just like the target's."""
    target = torch.nn.Module()
    draft = torch.nn.Module() if has_draft else None
    worker = Mock(use_v2_model_runner=True)
    worker.get_model.return_value = target
    worker.get_draft_model.return_value = draft
    worker.scheduler_config.max_num_batched_tokens = 16
    worker.vllm_config.kernel_config.enable_jit_warmup = False
    worker.vllm_config.kernel_config.enable_cutedsl_warmup = False
    worker.vllm_config.kernel_config.enable_flashinfer_autotune = False
    worker.vllm_config.compilation_config.cudagraph_capture_sizes = []
    worker.model_runner.attn_groups = []

    for name in (
        "qwen_triton_warmup",
        "qwen_vl_triton_warmup",
        "mamba_triton_warmup",
        "_warmup_gemm_rs_ar",
        "flashinfer_sparse_mla_decode_autotune_warmup",
        "deepseek_v4_sparse_mla_attention_warmup",
        "b12x_warmup",
    ):
        monkeypatch.setattr(kernel_warmup, name, Mock())
    platform = Mock()
    platform.has_device_capability.return_value = False
    platform.is_rocm.return_value = False
    monkeypatch.setattr(kernel_warmup, "current_platform", platform)
    monkeypatch.setattr(kernel_warmup, "is_deep_gemm_supported", lambda: supported)
    monkeypatch.setenv("VLLM_DEEP_GEMM_WARMUP", mode)
    monkeypatch.setenv("VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC", "0")
    warm_models = Mock()
    monkeypatch.setattr(kernel_warmup, "deep_gemm_warmup", warm_models)

    kernel_warmup.kernel_warmup(worker)

    expected = []
    if supported and mode != "skip":
        expected = [call(target, 16)]
        if has_draft:
            expected.append(call(draft, 16))
    assert warm_models.call_args_list == expected
