# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ``large_gpu_mark`` memory-detection fallback.

On MIG runners the platform NVML query (``nvmlDeviceGetMemoryInfo``) raises
"Insufficient Permissions" because it reads the parent device. Previously this
forced the detected memory to 0, so the test skipped at *any* ``min_gb``. The
mark must fall back to the CUDA runtime (torch), which reports the MIG
instance's own memory without NVML privileges.
"""

from tests.utils import large_gpu_mark
from vllm.platforms import current_platform
from vllm.utils.mem_constants import GB_bytes


def _skips(mark) -> bool:
    # pytest.mark.skipif stores the skip condition as the first positional arg.
    return bool(mark.args[0])


def _deny(*args, **kwargs):
    raise RuntimeError("Insufficient Permissions")


def test_reproduces_mig_skip_without_fallback(monkeypatch):
    """Regression: NVML denial with no working fallback -> skip at any min_gb."""
    import torch

    monkeypatch.setattr(current_platform, "is_cpu", lambda: False)
    monkeypatch.setattr(current_platform, "get_device_total_memory", _deny)
    monkeypatch.setattr(torch.cuda, "get_device_properties", _deny)

    # Even a 1 GB requirement skips, because memory could not be read.
    assert _skips(large_gpu_mark(min_gb=1)) is True


def test_falls_back_to_torch_when_nvml_denied(monkeypatch):
    """Fix: NVML denial -> torch reports the MIG memory -> the test runs."""
    import torch

    class _Props:
        total_memory = 35 * GB_bytes

    monkeypatch.setattr(current_platform, "is_cpu", lambda: False)
    monkeypatch.setattr(current_platform, "get_device_total_memory", _deny)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda *a, **k: _Props)

    # 35 GB (via torch) >= 32 -> not skipped.
    assert _skips(large_gpu_mark(min_gb=32)) is False
    # ...but a genuinely-too-large requirement still skips correctly.
    assert _skips(large_gpu_mark(min_gb=40)) is True
