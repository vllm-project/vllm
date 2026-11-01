# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kernels that FlashInfer JIT-compiles must decline at selection time when
FlashInfer cannot target the current GPU (vllm-project/vllm#50705)."""

import pytest

from vllm.platforms import current_platform


def _fp8_scaled_mm(monkeypatch):
    import vllm.model_executor.kernels.linear.scaled_mm.flashinfer as mod

    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(mod, "has_flashinfer", lambda: True)
    return mod, lambda: mod.FlashInferFP8ScaledMMLinearKernel.is_supported(120)[0]


def _nvfp4_cutlass(monkeypatch):
    import vllm.model_executor.kernels.linear.nvfp4.flashinfer as mod
    import vllm.model_executor.layers.quantization.utils.nvfp4_utils as nvfp4_utils

    monkeypatch.setattr(
        current_platform, "has_device_capability", lambda cap, device_id=0: True
    )
    monkeypatch.setattr(nvfp4_utils, "cutlass_fp4_supported", lambda: True)
    monkeypatch.setattr(mod, "has_flashinfer", lambda: True)
    return mod, lambda: mod.FlashInferCutlassNvFp4LinearKernel.is_supported(120)[0]


def _mxfp8_cutlass(monkeypatch):
    import vllm.model_executor.kernels.linear.mxfp8.flashinfer as mod

    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        current_platform, "has_device_capability", lambda cap, device_id=0: True
    )
    monkeypatch.setattr(mod, "has_flashinfer", lambda: True)
    return mod, lambda: mod.FlashInferCutlassMxfp8LinearKernel.is_supported(120)[0]


def _cutlass_moe(monkeypatch):
    import vllm.model_executor.layers.fused_moe.experts.flashinfer_cutlass_moe as mod

    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        current_platform,
        "is_device_capability_family",
        lambda family, device_id=0: family == 120,
    )
    monkeypatch.setattr(mod, "has_flashinfer_cutlass_fused_moe", lambda: True)
    return mod, mod.FlashInferExperts._supports_current_device


@pytest.mark.parametrize(
    "setup", [_fp8_scaled_mm, _nvfp4_cutlass, _mxfp8_cutlass, _cutlass_moe]
)
@pytest.mark.parametrize("reason", [None, "SM 12.x requires CUDA >= 12.9"])
def test_flashinfer_jit_kernels_follow_preflight(monkeypatch, setup, reason):
    mod, is_supported = setup(monkeypatch)
    monkeypatch.setattr(mod, "flashinfer_jit_unsupported_reason", lambda: reason)

    assert is_supported() is (reason is None)
