# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""On WSL2, pinned host memory follows the model runner unless
VLLM_WSL2_ENABLE_PIN_MEMORY decides it directly.

The gate is CudaPlatformBase.is_pin_memory_available; the cached wrapper in
vllm.utils.platform_utils is not under test."""

import pytest

from vllm.platforms import current_platform

# ROCm keeps its own WSL2 gate in vllm/platforms/rocm.py.
pytestmark = pytest.mark.skipif(
    current_platform.is_rocm(), reason="the gate under test is CudaPlatformBase's"
)
cuda_platform = pytest.importorskip("vllm.platforms.cuda")


@pytest.fixture
def wsl2(monkeypatch):
    monkeypatch.setattr(cuda_platform, "in_wsl", lambda: True)
    monkeypatch.setattr(cuda_platform, "_get_wsl_kernel_version", lambda: (6, 6, 87))
    monkeypatch.delenv("VLLM_WSL2_ENABLE_PIN_MEMORY", raising=False)
    monkeypatch.delenv("VLLM_USE_V2_MODEL_RUNNER", raising=False)
    return cuda_platform.CudaPlatformBase


@pytest.mark.parametrize(
    ("pin_memory_env", "runner_env", "expected"),
    [
        (None, None, True),
        (None, "1", True),
        (None, "0", False),
        ("0", None, False),
        ("0", "1", False),
        ("1", "0", True),
    ],
)
def test_wsl2_pinned_memory_follows_the_model_runner(
    wsl2, monkeypatch, pin_memory_env, runner_env, expected
):
    if pin_memory_env is not None:
        monkeypatch.setenv("VLLM_WSL2_ENABLE_PIN_MEMORY", pin_memory_env)
    if runner_env is not None:
        monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", runner_env)

    assert wsl2.is_pin_memory_available() is expected


def test_wsl2_kernel_without_pinned_memory_support(wsl2, monkeypatch):
    monkeypatch.setattr(cuda_platform, "_get_wsl_kernel_version", lambda: (4, 19, 0))
    monkeypatch.setenv("VLLM_WSL2_ENABLE_PIN_MEMORY", "1")

    assert wsl2.is_pin_memory_available() is False
