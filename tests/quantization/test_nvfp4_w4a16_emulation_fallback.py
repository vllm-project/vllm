# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""W4A16 NVFP4 must stay loadable when Marlin FP4 kernels are unavailable.

With --linear-backend=auto and use_a16=True, kernel selection forces Marlin.
On builds without the Marlin CUDA extensions that previously raised (or,
before the registration guard, crashed during weight processing). Selection
should instead fall back to the weight-only emulation kernel with a warning.

Run `pytest tests/quantization/test_nvfp4_w4a16_emulation_fallback.py`.
"""

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.kernels.linear import (
    EmulationA16NvFp4LinearKernel,
    MarlinNvFp4LinearKernel,
    init_nvfp4_linear_kernel,
)
from vllm.model_executor.layers.quantization.utils import marlin_utils_fp4


def _patch_platform_ok(monkeypatch):
    monkeypatch.setattr(marlin_utils_fp4.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        marlin_utils_fp4.current_platform,
        "has_device_capability",
        lambda capability: True,
    )


def test_use_a16_falls_back_to_emulation_when_marlin_ops_missing(monkeypatch):
    _patch_platform_ok(monkeypatch)
    monkeypatch.setattr(marlin_utils_fp4, "_has_cuda_kernel", lambda qualname: False)

    with set_current_vllm_config(VllmConfig()):
        kernel = init_nvfp4_linear_kernel(use_a16=True)

    assert isinstance(kernel, EmulationA16NvFp4LinearKernel)


def test_use_a16_prefers_marlin_when_available(monkeypatch):
    _patch_platform_ok(monkeypatch)
    monkeypatch.setattr(marlin_utils_fp4, "_has_cuda_kernel", lambda qualname: True)

    with set_current_vllm_config(VllmConfig()):
        kernel = init_nvfp4_linear_kernel(use_a16=True)

    assert isinstance(kernel, MarlinNvFp4LinearKernel)
