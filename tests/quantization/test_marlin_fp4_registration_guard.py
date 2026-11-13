# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Marlin FP4 availability must reflect CUDA custom-op registration.

`is_fp4_marlin_supported()` previously checked only platform + compute
capability, so source builds without the Marlin CUDA extensions would
select Marlin and then fail late (during weight processing) with an opaque
missing-op error. These tests pin the registration-aware behavior.

Run `pytest tests/quantization/test_marlin_fp4_registration_guard.py`.
"""

import pytest

from vllm.model_executor.kernels.linear.nvfp4.marlin import MarlinNvFp4LinearKernel
from vllm.model_executor.layers.quantization.utils import marlin_utils_fp4


def _patch_platform_ok(monkeypatch):
    monkeypatch.setattr(marlin_utils_fp4.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        marlin_utils_fp4.current_platform,
        "has_device_capability",
        lambda capability: True,
    )


def test_marlin_fp4_unavailable_when_cuda_ops_missing(monkeypatch):
    _patch_platform_ok(monkeypatch)
    monkeypatch.setattr(
        marlin_utils_fp4,
        "_has_cuda_kernel",
        lambda qualname: qualname != "_C::gptq_marlin_repack",
    )

    assert not marlin_utils_fp4.is_fp4_marlin_supported()
    reason = marlin_utils_fp4.get_fp4_marlin_unavailable_reason()
    assert reason is not None
    assert "_C::gptq_marlin_repack" in reason

    supported, kernel_reason = MarlinNvFp4LinearKernel.is_supported()
    assert not supported
    assert kernel_reason is not None
    assert "_C::gptq_marlin_repack" in kernel_reason


def test_marlin_fp4_available_when_cuda_ops_registered(monkeypatch):
    _patch_platform_ok(monkeypatch)
    monkeypatch.setattr(marlin_utils_fp4, "_has_cuda_kernel", lambda qualname: True)

    assert marlin_utils_fp4.get_fp4_marlin_unavailable_reason() is None
    assert marlin_utils_fp4.is_fp4_marlin_supported()

    supported, reason = MarlinNvFp4LinearKernel.is_supported()
    assert supported
    assert reason is None


def test_marlin_fp4_reason_reports_all_missing_ops(monkeypatch):
    _patch_platform_ok(monkeypatch)
    monkeypatch.setattr(marlin_utils_fp4, "_has_cuda_kernel", lambda qualname: False)

    reason = marlin_utils_fp4.get_fp4_marlin_unavailable_reason()
    assert reason is not None
    for qualname in marlin_utils_fp4._FP4_MARLIN_REQUIRED_CUDA_OPS:
        assert qualname in reason


def test_unregistered_schema_is_unavailable(monkeypatch):
    def missing_schema(*args):
        raise RuntimeError("operator does not exist")

    monkeypatch.setattr(
        marlin_utils_fp4.torch._C,
        "_dispatch_has_kernel_for_dispatch_key",
        missing_schema,
    )
    assert not marlin_utils_fp4._has_cuda_kernel("_C::missing_test_op")


def test_unexpected_dispatch_error_propagates(monkeypatch):
    def unexpected_error(*args):
        raise TypeError("bad dispatcher invocation")

    monkeypatch.setattr(
        marlin_utils_fp4.torch._C,
        "_dispatch_has_kernel_for_dispatch_key",
        unexpected_error,
    )
    with pytest.raises(TypeError, match="bad dispatcher invocation"):
        marlin_utils_fp4._has_cuda_kernel("_C::missing_test_op")
