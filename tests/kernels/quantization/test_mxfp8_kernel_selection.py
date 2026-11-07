# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for MXFP8 linear kernel selection logic (CPU-only)

Run `pytest tests/kernels/quantization/test_mxfp8_kernel_selection.py`.
"""

from contextlib import contextmanager
from unittest.mock import patch

import pytest

from vllm.model_executor.kernels.linear import (
    _POSSIBLE_MXFP8_KERNELS,
    EmulationMxfp8LinearKernel,
    FlashInferCutedslMxfp8LinearKernel,
    FlashInferCutlassMxfp8LinearKernel,
    HummingMxfp8LinearKernel,
    MarlinMxfp8LinearKernel,
    Mxfp8LinearLayerConfig,
    init_mxfp8_linear_kernel,
)
from vllm.platforms import PlatformEnum

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
        # The Humming-vs-Marlin priority reads the real device's capability;
        # pin it to the mocked one so the tests do not depend on the host.
        patch(
            "vllm.model_executor.layers.quantization.utils.humming.priority."
            "prefers_humming",
            side_effect=lambda compute_capability=None: capability == 90,
        ),
    ):
        platform._enum = PlatformEnum.CUDA
        platform.is_cuda.return_value = True
        yield


@contextmanager
def _humming_supported(supported: bool):
    with patch.object(
        HummingMxfp8LinearKernel,
        "is_supported",
        return_value=(supported, None if supported else "patched out"),
    ):
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


def test_only_row_independent_kernels_claim_batch_invariance():
    """Marlin and the FlashInfer kernels pick their tiling from the token
    count, so a row's output changes with the batch; they must not be
    selectable under VLLM_BATCH_INVARIANT."""
    claiming = {
        kernel
        for kernel in _POSSIBLE_MXFP8_KERNELS[PlatformEnum.CUDA]
        if kernel.supports_batch_invariance()
    }
    assert claiming == {HummingMxfp8LinearKernel, EmulationMxfp8LinearKernel}


@pytest.mark.parametrize("capability", [80, 89, 90, 100, 120])
@pytest.mark.parametrize("weight_shape", [_SUPPORTED_SHAPE, (130, 256), (32, 256)])
def test_batch_invariance_selects_humming(monkeypatch, capability, weight_shape):
    """Every shape that would otherwise land on FlashInfer or Marlin goes to
    Humming under VLLM_BATCH_INVARIANT, on every architecture."""
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    with _patch_cuda_platform(capability), _humming_supported(True):
        kernel = init_mxfp8_linear_kernel(weight_shape=weight_shape)

    assert isinstance(kernel, HummingMxfp8LinearKernel)


@pytest.mark.parametrize(
    ("linear_backend", "expected_kernel_cls"),
    [
        ("marlin", HummingMxfp8LinearKernel),
        ("flashinfer_cutedsl", HummingMxfp8LinearKernel),
        ("humming", HummingMxfp8LinearKernel),
        ("emulation", EmulationMxfp8LinearKernel),
    ],
)
def test_batch_invariance_overrides_non_invariant_linear_backend(
    monkeypatch, linear_backend, expected_kernel_cls
):
    """--linear-backend is honored only when it names a batch-invariant
    kernel; a request for Marlin or FlashInfer is overridden."""
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    with (
        _patch_cuda_platform(100),
        _humming_supported(True),
        patch(
            "vllm.model_executor.kernels.linear._get_linear_backend",
            return_value=linear_backend,
        ),
    ):
        kernel = init_mxfp8_linear_kernel(weight_shape=_SUPPORTED_SHAPE)

    assert isinstance(kernel, expected_kernel_cls)


def test_batch_invariance_falls_back_to_emulation_without_humming(monkeypatch):
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    with _patch_cuda_platform(90), _humming_supported(False):
        kernel = init_mxfp8_linear_kernel(weight_shape=_SUPPORTED_SHAPE)

    assert isinstance(kernel, EmulationMxfp8LinearKernel)


def test_batch_invariance_error_names_excluded_kernels(monkeypatch):
    """When no batch-invariant kernel can run, the error says which kernels
    were excluded for batch invariance rather than hiding them."""
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    with (
        _patch_cuda_platform(90),
        _humming_supported(False),
        patch.object(
            EmulationMxfp8LinearKernel,
            "is_supported",
            return_value=(False, "patched out"),
        ),
        pytest.raises(ValueError, match="MarlinMxfp8LinearKernel: not batch invariant"),
    ):
        init_mxfp8_linear_kernel(weight_shape=_SUPPORTED_SHAPE)
