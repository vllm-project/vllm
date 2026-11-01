# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

import vllm.model_executor.kernels.linear as linear
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.kernels.linear.nvfp4 import NvFp4LinearLayerConfig
from vllm.platforms import PlatformEnum


@pytest.fixture
def unavailable_backends(monkeypatch):
    monkeypatch.setattr(linear.current_platform, "_enum", PlatformEnum.CUDA)
    monkeypatch.setattr(linear.current_platform, "get_device_capability", lambda: None)
    monkeypatch.setattr(linear.envs, "VLLM_BATCH_INVARIANT", False)
    monkeypatch.setattr(linear.envs, "VLLM_DISABLED_KERNELS", [])
    monkeypatch.setattr(linear, "prefers_humming", lambda cc: False)
    monkeypatch.setattr(linear, "prioritize_humming", lambda kernels, cc: kernels)
    for kernel in (
        linear.MarlinNvFp4LinearKernel,
        linear.HummingNvFp4LinearKernel,
        linear.FlashInferCuteDslNvFp4W4A16LinearKernel,
    ):
        monkeypatch.setattr(
            kernel, "is_supported", lambda *args: (False, "unavailable in test")
        )


def test_use_a16_falls_back_when_optimized_backends_unavailable(unavailable_backends):
    with set_current_vllm_config(VllmConfig()):
        kernel = linear.init_nvfp4_linear_kernel(use_a16=True)
    assert isinstance(kernel, linear.EmulationA16NvFp4LinearKernel)


def test_use_a16_prefers_marlin_when_available(unavailable_backends, monkeypatch):
    monkeypatch.setattr(
        linear.MarlinNvFp4LinearKernel, "is_supported", lambda *args: (True, None)
    )
    with set_current_vllm_config(VllmConfig()):
        kernel = linear.init_nvfp4_linear_kernel(use_a16=True)
    assert isinstance(kernel, linear.MarlinNvFp4LinearKernel)


def test_missing_marlin_does_not_skip_humming(unavailable_backends, monkeypatch):
    monkeypatch.setattr(
        linear.HummingNvFp4LinearKernel, "is_supported", lambda *args: (True, None)
    )
    monkeypatch.setattr(
        linear.HummingNvFp4LinearKernel, "can_implement", lambda *args: (True, None)
    )
    with set_current_vllm_config(VllmConfig()):
        kernel = linear.init_nvfp4_linear_kernel(use_a16=True)
    assert isinstance(kernel, linear.HummingNvFp4LinearKernel)


def test_disabled_marlin_falls_back(unavailable_backends, monkeypatch):
    monkeypatch.setattr(
        linear.MarlinNvFp4LinearKernel, "is_supported", lambda *args: (True, None)
    )
    monkeypatch.setattr(
        linear.envs, "VLLM_DISABLED_KERNELS", ["MarlinNvFp4LinearKernel"]
    )
    with set_current_vllm_config(VllmConfig()):
        kernel = linear.init_nvfp4_linear_kernel(use_a16=True)
    assert isinstance(kernel, linear.EmulationA16NvFp4LinearKernel)


def test_explicit_unavailable_marlin_errors(unavailable_backends):
    config = VllmConfig()
    config.kernel_config.linear_backend = "marlin"
    with (
        set_current_vllm_config(config),
        pytest.raises(ValueError, match="unavailable"),
    ):
        linear.init_nvfp4_linear_kernel(use_a16=True)


def test_explicit_emulation_uses_weight_only_kernel(unavailable_backends):
    config = VllmConfig()
    config.kernel_config.linear_backend = "emulation"
    with set_current_vllm_config(config):
        kernel = linear.init_nvfp4_linear_kernel(use_a16=True)
    assert isinstance(kernel, linear.EmulationA16NvFp4LinearKernel)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_auto_selection_with_optimized_kernels_disabled(monkeypatch):
    # Real platform and registration probes, rather than synthetic availability.
    monkeypatch.setattr(
        linear.envs,
        "VLLM_DISABLED_KERNELS",
        [
            "MarlinNvFp4LinearKernel",
            "HummingNvFp4LinearKernel",
            "FlashInferCuteDslNvFp4W4A16LinearKernel",
        ],
    )
    with set_current_vllm_config(VllmConfig()):
        kernel = linear.init_nvfp4_linear_kernel(use_a16=True)
    assert isinstance(kernel, linear.EmulationA16NvFp4LinearKernel)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("with_bias", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_weight_only_emulation_preserves_activations(dtype, with_bias, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    # Every E2M1 code, both signs, unequal block scales and a nonunit global
    # scale. There is deliberately no activation-scale attribute on the layer.
    values = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6]
    )
    codes = torch.arange(16, dtype=torch.uint8).repeat(2)
    packed = (codes[::2] | (codes[1::2] << 4)).to(device)
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(packed.repeat(2, 1), requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(
        torch.tensor([[0.5, 2.0], [1.0, 0.25]], device=device).to(torch.float8_e4m3fn),
        requires_grad=False,
    )
    layer.weight_global_scale = torch.tensor(0.5, device=device)
    layer.output_size_per_partition = 2
    x = torch.linspace(-1.13, 0.97, 192, device=device).reshape(2, 3, 32).to(dtype)
    reference_weight = (
        values.to(device).repeat(2).repeat(2, 1)
        * layer.weight_scale.float().repeat_interleave(16, dim=1)
        * 0.5
    ).to(dtype)
    bias = torch.tensor([0.25, -0.5], dtype=dtype, device=device) if with_bias else None
    expected = x @ reference_weight.t()
    if bias is not None:
        expected = expected + bias
    kernel = linear.EmulationA16NvFp4LinearKernel(NvFp4LinearLayerConfig())
    kernel.process_weights_after_loading(layer)
    result = kernel.apply_weights(layer, x, bias)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    if device == "cuda":
        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            for _ in range(3):
                kernel.apply_weights(layer, x, bias)
        torch.cuda.current_stream().wait_stream(warmup_stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured_result = kernel.apply_weights(layer, x, bias)
        graph.replay()
        torch.testing.assert_close(captured_result, expected, rtol=0, atol=0)
