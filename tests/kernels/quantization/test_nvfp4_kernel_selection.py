# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for NVFP4 linear kernel selection order (CPU-only)."""

from collections.abc import Callable

import pytest

from vllm.model_executor.kernels.linear import (
    _POSSIBLE_NVFP4_KERNELS,
    CutlassNvFp4LinearKernel,
    FlashInferB12xNvFp4LinearKernel,
    FlashInferCuteDslNvFp4W4A16LinearKernel,
    FlashInferCutlassNvFp4LinearKernel,
)
from vllm.platforms.interface import PlatformEnum

# W4A4 kernels that run on SM120/121, where the head of the list is gated to
# sm_10x and selection falls through to whatever follows.
W4A4_KERNELS_ON_SM12X = (
    FlashInferCutlassNvFp4LinearKernel,
    FlashInferB12xNvFp4LinearKernel,
    CutlassNvFp4LinearKernel,
)


def _constant_support(available: bool) -> Callable[[int], bool]:
    return lambda cc: available


@pytest.mark.parametrize("w4a4_kernel", W4A4_KERNELS_ON_SM12X)
def test_w4a16_kernel_does_not_precede_w4a4_kernels(w4a4_kernel):
    candidates = _POSSIBLE_NVFP4_KERNELS[PlatformEnum.CUDA]
    w4a16_index = candidates.index(FlashInferCuteDslNvFp4W4A16LinearKernel)
    assert candidates.index(w4a4_kernel) < w4a16_index, (
        f"{w4a4_kernel.__name__} must be preferred over "
        f"{FlashInferCuteDslNvFp4W4A16LinearKernel.__name__}"
    )


def test_dynamic_rejects_uncalibrated_weight_only_checkpoint(monkeypatch):
    """An A16 checkpoint cannot supply the required calibrated A4 input scales."""
    from vllm.model_executor.kernels import linear

    monkeypatch.setattr(linear, "_get_linear_backend", lambda **kwargs: "nvfp4_dynamic")
    with pytest.raises(ValueError, match="calibrated W4A4 checkpoint"):
        linear.init_nvfp4_linear_kernel(use_a16=True)


@pytest.mark.parametrize("batch_invariant_supported", [False, True])
def test_dynamic_rejects_batch_invariant_mode(monkeypatch, batch_invariant_supported):
    """Explicit dynamic selection must not silently override determinism."""
    from vllm.model_executor.kernels import linear

    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    monkeypatch.setattr(linear, "_get_linear_backend", lambda **kwargs: "nvfp4_dynamic")
    monkeypatch.setattr(
        CutlassNvFp4LinearKernel,
        "is_supported",
        lambda: (batch_invariant_supported, "test platform"),
    )
    with pytest.raises(ValueError, match="does not support VLLM_BATCH_INVARIANT"):
        linear.init_nvfp4_linear_kernel()


@pytest.mark.parametrize(
    "name,bits,match",
    [
        ("marlin", 16, "Unknown canonical"),
        ("flashinfer_trtllm", 4, "Unknown canonical"),
        ("flashinfer_cutlass", 16, "does not implement W4A16"),
        ("flashinfer_cutedsl_native", 4, "does not implement W4A4"),
    ],
)
def test_dynamic_rejects_incompatible_candidate(name, bits, match):
    from vllm.model_executor.kernels.linear.nvfp4.dynamic_backends import (
        get_dynamic_backend,
    )

    with pytest.raises(ValueError, match=match):
        get_dynamic_backend(name, bits)


def test_dynamic_rejects_candidate_with_different_weight_layout(monkeypatch):
    from dataclasses import replace

    from vllm.model_executor.kernels.linear.nvfp4 import dynamic_backends

    candidate = dynamic_backends.get_dynamic_backend("flashinfer_cutlass", 4)
    monkeypatch.setitem(
        dynamic_backends._BACKENDS,
        "other_layout",
        replace(candidate, weight_layout="shuffled"),
    )
    with pytest.raises(ValueError, match="different weight layout"):
        dynamic_backends.get_dynamic_backend("other_layout", 4)


@pytest.mark.parametrize("a4_available", [False, True])
def test_dynamic_default_does_not_require_cutedsl_w4a4(monkeypatch, a4_available):
    from dataclasses import replace
    from types import SimpleNamespace

    from vllm.config.kernel import KernelConfig
    from vllm.model_executor.kernels.linear.nvfp4 import dynamic, dynamic_backends

    config = KernelConfig(linear_backend="nvfp4_dynamic")
    monkeypatch.setattr(
        dynamic,
        "get_current_vllm_config",
        lambda: SimpleNamespace(kernel_config=config),
    )
    for name, bits, available in (
        ("flashinfer_cutlass", 4, a4_available),
        ("flashinfer_cutedsl_native", 16, True),
    ):
        candidate = dynamic_backends.get_dynamic_backend(name, bits)
        monkeypatch.setitem(
            dynamic_backends._BACKENDS,
            name,
            replace(candidate, is_supported=_constant_support(available)),
        )
    monkeypatch.delitem(dynamic_backends._BACKENDS, "flashinfer_cutedsl")
    supported, reason = dynamic.DynamicNvFp4LinearKernel.is_supported(121)
    assert supported == a4_available
    if not supported:
        assert "flashinfer_cutlass" in reason
