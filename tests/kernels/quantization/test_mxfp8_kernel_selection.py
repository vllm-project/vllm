# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for MXFP8 linear kernel selection logic (CPU-only)

Run `pytest tests/kernels/quantization/test_mxfp8_kernel_selection.py`.
"""

from contextlib import contextmanager
from unittest.mock import patch

import pytest
import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.config.kernel import KernelConfig
from vllm.model_executor.kernels.linear import (
    EmulationMxfp8LinearKernel,
    FlashInferCutedslMxfp8LinearKernel,
    FlashInferCutlassMxfp8LinearKernel,
    FlashInferTrtllmMxfp8LinearKernel,
    HummingMxfp8LinearKernel,
    MarlinMxfp8LinearKernel,
    Mxfp8LinearLayerConfig,
    TorchMxFp8LinearKernel,
    init_mxfp8_linear_kernel,
)
from vllm.model_executor.layers.quantization.quark.schemes.quark_ocp_mx import (
    QuarkOCP_MX,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kMxfp8Dynamic,
    kMxfp8Static,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.platforms import PlatformEnum
from vllm.platforms.interface import DeviceCapability

pytestmark = pytest.mark.cpu_test

# Kernels backed by FlashInfer mm_mxfp8, which requires N, K >= 128 and
# K % 32 == 0.
_MM_MXFP8_KERNELS = [
    FlashInferCutedslMxfp8LinearKernel,
    FlashInferCutlassMxfp8LinearKernel,
]

_SUPPORTED_SHAPE = (4096, 4096)
# (N, K): N < 128, K < 128, K % 32 != 0.
_UNSUPPORTED_SHAPES = [(64, 4096), (4096, 64), (4096, 4112)]

# (N, K) with N < 128 or K < 128, which only Marlin runs.
_MARLIN_ONLY_SHAPES = [(32, 256), (64, 256), (96, 256), (256, 64)]
# (N, K): N <= 0, K <= 0, K % 32 != 0.
_MARLIN_UNSUPPORTED_SHAPES = [(0, 256), (256, 0), (256, 100)]

_SELECTION_CASES = [
    (100, (128, 128), FlashInferCutedslMxfp8LinearKernel),
    (100, (130, 256), FlashInferCutedslMxfp8LinearKernel),
    (103, (128, 128), FlashInferCutedslMxfp8LinearKernel),
    (103, (130, 256), FlashInferCutedslMxfp8LinearKernel),
    (120, (128, 128), FlashInferCutlassMxfp8LinearKernel),
    (120, (130, 256), MarlinMxfp8LinearKernel),
    (121, (128, 128), FlashInferCutlassMxfp8LinearKernel),
    (121, (130, 256), MarlinMxfp8LinearKernel),
    *[
        (capability, shape, MarlinMxfp8LinearKernel)
        for capability in (100, 103, 120, 121)
        for shape in _MARLIN_ONLY_SHAPES
    ],
]


@pytest.mark.parametrize("kernel_cls", _MM_MXFP8_KERNELS)
def test_mm_mxfp8_kernels_accept_supported_shape(kernel_cls):
    config = Mxfp8LinearLayerConfig(weight_shape=_SUPPORTED_SHAPE)
    can_implement, reason = kernel_cls.can_implement(config)
    assert can_implement, reason


@pytest.mark.parametrize("kernel_cls", _MM_MXFP8_KERNELS)
@pytest.mark.parametrize("weight_shape", _UNSUPPORTED_SHAPES)
def test_mm_mxfp8_kernels_reject_unsupported_shape(kernel_cls, weight_shape):
    config = Mxfp8LinearLayerConfig(weight_shape=weight_shape)
    can_implement, reason = kernel_cls.can_implement(config)
    assert not can_implement
    assert reason


@pytest.mark.parametrize("weight_shape", _MARLIN_UNSUPPORTED_SHAPES)
def test_marlin_mxfp8_rejects_unsupported_shape(weight_shape):
    config = Mxfp8LinearLayerConfig(weight_shape=weight_shape)
    can_implement, reason = MarlinMxfp8LinearKernel.can_implement(config)
    assert not can_implement
    assert reason


@contextmanager
def _patch_cuda_platform(capability: int):
    """Patch the platform so kernel selection runs as on a CUDA GPU with the
    given compute capability. The FlashInfer and Marlin kernels report as
    supported, CuTe-DSL only on SM100/103."""
    with (
        patch("vllm.model_executor.kernels.linear.current_platform") as platform,
        patch(
            "vllm.platforms.current_platform.get_device_capability",
            return_value=DeviceCapability(capability // 10, capability % 10),
        ),
        patch(
            "vllm.model_executor.kernels.linear.mxfp8.flashinfer.current_platform."
            "is_device_capability_family",
            side_effect=lambda cap, device_id=0: cap // 10 == capability // 10,
        ),
        patch.object(
            FlashInferCutedslMxfp8LinearKernel,
            "is_supported",
            return_value=(capability in (100, 103), None),
        ),
        patch.object(
            FlashInferCutlassMxfp8LinearKernel,
            "is_supported",
            return_value=(True, None),
        ),
        patch.object(
            MarlinMxfp8LinearKernel, "is_supported", return_value=(True, None)
        ),
    ):
        platform._enum = PlatformEnum.CUDA
        yield


@pytest.mark.parametrize(
    ("capability", "expected"),
    [(100, True), (103, True), (120, False), (121, False)],
)
def test_cutlass_mxfp8_requires_aligned_n_only_on_sm12x(capability, expected):
    """(130, 256) passes the generic mm_mxfp8 limits. Only the SM12x CUTLASS
    backend needs N % 32 == 0, so SM100/103 must keep accepting it."""
    config = Mxfp8LinearLayerConfig(weight_shape=(130, 256))
    with _patch_cuda_platform(capability):
        can_implement, _ = FlashInferCutlassMxfp8LinearKernel.can_implement(config)
    assert can_implement == expected


@pytest.mark.parametrize(
    ("capability", "weight_shape", "expected_kernel_cls"), _SELECTION_CASES
)
def test_init_mxfp8_linear_kernel(capability, weight_shape, expected_kernel_cls):
    """A layer that mm_mxfp8 cannot handle must fall through to the next
    kernel in the CUDA priority list instead of being selected and failing
    in apply_weights."""
    with _patch_cuda_platform(capability):
        kernel = init_mxfp8_linear_kernel(weight_shape=weight_shape)

    assert isinstance(kernel, expected_kernel_cls)
    assert kernel.config == Mxfp8LinearLayerConfig(weight_shape=weight_shape)


@pytest.mark.parametrize("activation_key", [None, kMxfp8Dynamic])
def test_quark_mxfp8_preserves_activation_precision(activation_key, dist_init):
    """A weight-only Quark layer must not acquire FP8 activations on SM100."""
    scheme = QuarkOCP_MX(kMxfp8Static, activation_key)
    layer = torch.nn.Module()
    with _patch_cuda_platform(100):
        scheme.create_weights(
            layer,
            output_partition_sizes=[128],
            input_size_per_partition=128,
            params_dtype=torch.bfloat16,
            weight_loader=default_weight_loader,
        )
    expected = (
        MarlinMxfp8LinearKernel
        if activation_key is None
        else FlashInferCutedslMxfp8LinearKernel
    )
    assert isinstance(scheme.ocp_mx_linear, expected)


@pytest.mark.parametrize("capability", [100, 103, 120, 121])
def test_mxfp8_weight_only_skips_activation_quantization(capability):
    with _patch_cuda_platform(capability):
        kernel = init_mxfp8_linear_kernel(
            weight_shape=(128, 128), activation_quant_key=None
        )
    assert isinstance(kernel, MarlinMxfp8LinearKernel)


def test_mxfp8_weight_only_emulation_fallback(monkeypatch):
    monkeypatch.setenv(
        "VLLM_DISABLED_KERNELS",
        "MarlinMxfp8LinearKernel,FlashInferCutedslMxfp8LinearKernel,"
        "FlashInferCutlassMxfp8LinearKernel,B12xMxfp8LinearKernel",
    )
    with _patch_cuda_platform(100):
        kernel = init_mxfp8_linear_kernel(
            weight_shape=(128, 128), activation_quant_key=None
        )
    assert isinstance(kernel, EmulationMxfp8LinearKernel)


@pytest.mark.parametrize(
    ("backend", "kernel_cls"),
    [
        ("flashinfer_cutlass", FlashInferCutlassMxfp8LinearKernel),
        ("flashinfer_cutedsl", FlashInferCutedslMxfp8LinearKernel),
        ("flashinfer_trtllm", FlashInferTrtllmMxfp8LinearKernel),
        ("humming", HummingMxfp8LinearKernel),
        ("torch", TorchMxFp8LinearKernel),
    ],
)
def test_mxfp8_weight_only_rejects_forced_w8a8_backend(backend, kernel_cls):
    config = VllmConfig(kernel_config=KernelConfig(linear_backend=backend))
    with (
        set_current_vllm_config(config),
        _patch_cuda_platform(100),
        patch.object(kernel_cls, "is_supported", return_value=(True, None)),
        pytest.raises(ValueError, match="quantization scheme"),
    ):
        init_mxfp8_linear_kernel(weight_shape=(128, 128), activation_quant_key=None)
