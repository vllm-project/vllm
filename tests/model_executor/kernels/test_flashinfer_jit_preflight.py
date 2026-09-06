# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer JIT preflight for kernel selection.

Kernels that FlashInfer JIT-compiles with nvcc must decline at selection time
when FlashInfer cannot target the current GPU (e.g. SM 12.x with a CUDA
toolkit older than 12.9). Otherwise they are selected and engine startup dies
in the first JIT build with a misleading "FlashInfer requires GPUs with sm75
or higher" error (vllm-project/vllm#50705).
"""

import sys
import types

import pytest

import vllm.utils.flashinfer as fi_utils
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability

SM120 = DeviceCapability(12, 0)
TOOLKIT_REASON = "SM 12.x requires CUDA >= 12.9"


@pytest.fixture(autouse=True)
def _reset_preflight_cache():
    fi_utils.flashinfer_jit_unsupported_reason.cache_clear()
    fi_utils.flashinfer_b12x_unsupported_reason.cache_clear()
    yield
    fi_utils.flashinfer_jit_unsupported_reason.cache_clear()
    fi_utils.flashinfer_b12x_unsupported_reason.cache_clear()


def _install_fake_flashinfer(monkeypatch, targets, normalize_error=None):
    """Stand in for the FlashInfer internals the preflight reads."""
    jit_core = types.SimpleNamespace(
        current_compilation_context=types.SimpleNamespace(TARGET_CUDA_ARCHS=targets)
    )

    class CompilationContext:
        @staticmethod
        def _normalize_cuda_arch(major, minor):
            if normalize_error is not None:
                raise RuntimeError(normalize_error)
            return (major, str(minor))

    monkeypatch.setitem(sys.modules, "flashinfer.jit.core", jit_core)
    monkeypatch.setitem(
        sys.modules,
        "flashinfer.compilation_context",
        types.SimpleNamespace(CompilationContext=CompilationContext),
    )


def _fake_cuda_device(monkeypatch, capability=SM120):
    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        current_platform, "get_device_capability", lambda device_id=0: capability
    )


def test_preflight_passes_when_flashinfer_targets_current_gpu(monkeypatch):
    _fake_cuda_device(monkeypatch)
    _install_fake_flashinfer(monkeypatch, targets={(12, "0f")})

    assert fi_utils.flashinfer_jit_unsupported_reason() is None


def test_preflight_reports_toolkit_error_swallowed_by_flashinfer(monkeypatch):
    """An empty target set means FlashInfer logged and dropped the arch
    error; the preflight must surface that real reason."""
    _fake_cuda_device(monkeypatch)
    _install_fake_flashinfer(monkeypatch, targets=set(), normalize_error=TOOLKIT_REASON)

    assert fi_utils.flashinfer_jit_unsupported_reason() == TOOLKIT_REASON


def test_preflight_rejects_gpu_missing_from_target_set(monkeypatch):
    """Targets for another GPU in the box do not make this one compilable."""
    _fake_cuda_device(monkeypatch)
    _install_fake_flashinfer(monkeypatch, targets={(9, "0a")})

    reason = fi_utils.flashinfer_jit_unsupported_reason()

    assert reason is not None
    assert "sm_120" in reason


def test_preflight_is_neutral_without_flashinfer(monkeypatch):
    _fake_cuda_device(monkeypatch)
    monkeypatch.setitem(sys.modules, "flashinfer.jit.core", None)

    assert fi_utils.flashinfer_jit_unsupported_reason() is None


def test_preflight_is_neutral_off_cuda(monkeypatch):
    monkeypatch.setattr(current_platform, "is_cuda", lambda: False)

    assert fi_utils.flashinfer_jit_unsupported_reason() is None


def _fp8_scaled_mm_env(monkeypatch):
    import vllm.model_executor.kernels.linear.scaled_mm.flashinfer as mod

    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(mod, "has_flashinfer", lambda: True)
    return mod, mod.FlashInferFP8ScaledMMLinearKernel


def _nvfp4_cutlass_env(monkeypatch):
    import vllm.model_executor.kernels.linear.nvfp4.flashinfer as mod
    import vllm.model_executor.layers.quantization.utils.nvfp4_utils as nvfp4_utils

    monkeypatch.setattr(
        current_platform, "has_device_capability", lambda cap, device_id=0: True
    )
    monkeypatch.setattr(nvfp4_utils, "cutlass_fp4_supported", lambda: True)
    monkeypatch.setattr(mod, "has_flashinfer", lambda: True)
    return mod, mod.FlashInferCutlassNvFp4LinearKernel


def _mxfp8_cutlass_env(monkeypatch):
    import vllm.model_executor.kernels.linear.mxfp8.flashinfer as mod

    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        current_platform, "has_device_capability", lambda cap, device_id=0: True
    )
    monkeypatch.setattr(mod, "has_flashinfer", lambda: True)
    return mod, mod.FlashInferCutlassMxfp8LinearKernel


LINEAR_KERNEL_ENVS = pytest.mark.parametrize(
    "setup_env",
    [_fp8_scaled_mm_env, _nvfp4_cutlass_env, _mxfp8_cutlass_env],
    ids=["fp8_scaled_mm", "nvfp4_cutlass", "mxfp8_cutlass"],
)


@LINEAR_KERNEL_ENVS
def test_flashinfer_linear_kernel_declines_when_jit_unavailable(monkeypatch, setup_env):
    mod, kernel_cls = setup_env(monkeypatch)
    monkeypatch.setattr(
        mod, "flashinfer_jit_unsupported_reason", lambda: TOOLKIT_REASON
    )

    supported, reason = kernel_cls.is_supported(120)

    assert not supported
    assert TOOLKIT_REASON in reason


@LINEAR_KERNEL_ENVS
def test_flashinfer_linear_kernel_accepts_when_jit_available(monkeypatch, setup_env):
    mod, kernel_cls = setup_env(monkeypatch)
    monkeypatch.setattr(mod, "flashinfer_jit_unsupported_reason", lambda: None)

    assert kernel_cls.is_supported(120) == (True, None)


def test_preflight_reports_import_time_failure(monkeypatch):
    """FlashInfer builds its compilation context at import time and a
    malformed FLASHINFER_CUDA_ARCH_LIST raises ValueError there; the
    preflight must turn that into a reason, not crash kernel selection."""
    _fake_cuda_device(monkeypatch)
    broken = types.ModuleType("flashinfer.jit.core")

    def _raise(name):
        raise ValueError("not enough values to unpack (expected 2, got 1)")

    vars(broken)["__getattr__"] = _raise
    monkeypatch.setitem(sys.modules, "flashinfer.jit.core", broken)

    reason = fi_utils.flashinfer_jit_unsupported_reason()

    assert reason is not None
    assert "not enough values to unpack" in reason


def _install_fake_cuda_version(monkeypatch, version):
    from packaging.version import Version

    monkeypatch.setitem(
        sys.modules,
        "flashinfer.jit.cpp_ext",
        types.SimpleNamespace(get_cuda_version=lambda: Version(version)),
    )


@pytest.mark.parametrize(
    ("cuda_version", "expected"),
    [("13.0", None), ("13.1", None), ("12.8", "12.8"), ("12.9", "12.9")],
)
def test_b12x_preflight_mirrors_flashinfer_cuda13_requirement(
    monkeypatch, cuda_version, expected
):
    """FlashInfer's b12x kernels raise at call time below CUDA 13; the
    preflight must decline them at selection with the toolkit version."""
    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
    _install_fake_cuda_version(monkeypatch, cuda_version)

    reason = fi_utils.flashinfer_b12x_unsupported_reason()

    if expected is None:
        assert reason is None
    else:
        assert reason is not None
        assert "CUDA >= 13" in reason
        assert expected in reason


def test_b12x_preflight_is_neutral_without_flashinfer(monkeypatch):
    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
    monkeypatch.setitem(sys.modules, "flashinfer.jit.cpp_ext", None)

    assert fi_utils.flashinfer_b12x_unsupported_reason() is None


B12X_REASON = "FlashInfer b12x kernels require CUDA >= 13 (local toolkit 12.8)"


def _b12x_nvfp4_env(monkeypatch):
    import vllm.model_executor.kernels.linear.nvfp4.flashinfer as mod

    monkeypatch.setattr(
        current_platform, "has_device_capability", lambda cap, device_id=0: True
    )
    monkeypatch.setattr(mod, "has_flashinfer_b12x_gemm", lambda: True)
    return mod, mod.FlashInferB12xNvFp4LinearKernel


def test_b12x_nvfp4_kernel_declines_below_cuda13(monkeypatch):
    mod, kernel_cls = _b12x_nvfp4_env(monkeypatch)
    monkeypatch.setattr(mod, "flashinfer_b12x_unsupported_reason", lambda: B12X_REASON)

    supported, reason = kernel_cls.is_supported(120)

    assert not supported
    assert reason == B12X_REASON


def test_b12x_nvfp4_kernel_accepts_on_cuda13(monkeypatch):
    mod, kernel_cls = _b12x_nvfp4_env(monkeypatch)
    monkeypatch.setattr(mod, "flashinfer_b12x_unsupported_reason", lambda: None)

    assert kernel_cls.is_supported(120) == (True, None)


@pytest.mark.parametrize("b12x_reason", [None, B12X_REASON])
def test_b12x_moe_experts_follow_cuda13_preflight(monkeypatch, b12x_reason):
    import vllm.model_executor.layers.fused_moe.experts.flashinfer_b12x_moe as mod

    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        current_platform,
        "is_device_capability_family",
        lambda family, device_id=0: family == 120,
    )
    monkeypatch.setattr(mod, "has_flashinfer_b12x_moe", lambda: True)
    monkeypatch.setattr(mod, "flashinfer_b12x_unsupported_reason", lambda: b12x_reason)

    assert mod.FlashInferB12xExperts._supports_current_device() is (b12x_reason is None)
