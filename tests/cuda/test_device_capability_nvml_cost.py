# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`has_device_capability` must not pay an NVML init/shutdown per call.

Regression test for issue #50381. `NvmlCudaPlatform.has_device_capability` only
reads `get_device_capability()`, which is already cached and already wraps
itself in an NVML context. Decorating it with `with_nvml_context` as well cost
an `nvmlInit()`/`nvmlShutdown()` pair on every call, and it is called per
attention layer per step from `triton_reshape_and_cache_flash` when the KV
cache dtype is fp8 or bfloat16.
"""

from collections.abc import Generator

import pytest

from vllm.platforms.cuda import DeviceCapability, NvmlCudaPlatform


@pytest.fixture
def nvml_counter(monkeypatch: pytest.MonkeyPatch) -> Generator[dict[str, int]]:
    """Count NVML init/shutdown calls and stub out the device query.

    Targets `NvmlCudaPlatform` rather than the `CudaPlatform` alias: the alias
    resolves to `NonNvmlCudaPlatform` where NVML is unavailable, and that class
    reads `torch.cuda` instead, which needs a real device. Pinning the class
    keeps this test hermetic on CPU-only machines.
    """
    calls = {"init": 0, "shutdown": 0}

    import vllm.platforms.cuda as cuda_mod

    monkeypatch.setattr(
        cuda_mod.pynvml,
        "nvmlInit",
        lambda: calls.__setitem__("init", calls["init"] + 1),
    )
    monkeypatch.setattr(
        cuda_mod.pynvml,
        "nvmlShutdown",
        lambda: calls.__setitem__("shutdown", calls["shutdown"] + 1),
    )
    monkeypatch.setattr(
        NvmlCudaPlatform,
        "visible_device_id_to_physical_device_id",
        classmethod(lambda cls, device_id: device_id),
    )
    monkeypatch.setattr(
        cuda_mod.pynvml, "nvmlDeviceGetHandleByIndex", lambda index: object()
    )
    monkeypatch.setattr(
        cuda_mod.pynvml, "nvmlDeviceGetCudaComputeCapability", lambda handle: (9, 0)
    )

    # get_device_capability is process-cached; start from a known state.
    NvmlCudaPlatform.get_device_capability.cache_clear()
    yield calls
    NvmlCudaPlatform.get_device_capability.cache_clear()


@pytest.mark.cpu_test
def test_repeated_capability_checks_do_not_reinit_nvml(nvml_counter: dict[str, int]):
    """Only the first capability query may touch NVML.

    The Triton reshape-and-cache path calls this per layer per step, so the
    cost must not scale with the number of calls.
    """
    assert NvmlCudaPlatform.has_device_capability(80)
    for _ in range(20):
        NvmlCudaPlatform.has_device_capability(80)
        NvmlCudaPlatform.has_device_capability(89)
        NvmlCudaPlatform.has_device_capability((9, 0))

    assert nvml_counter["init"] == 1
    assert nvml_counter["shutdown"] == 1


@pytest.mark.cpu_test
def test_capability_comparison_still_correct(nvml_counter: dict[str, int]):
    """Removing the NVML wrapper must not change the answers."""
    assert NvmlCudaPlatform.get_device_capability() == DeviceCapability(
        major=9, minor=0
    )
    assert NvmlCudaPlatform.has_device_capability(80)
    assert NvmlCudaPlatform.has_device_capability(90)
    assert NvmlCudaPlatform.has_device_capability((9, 0))
    assert not NvmlCudaPlatform.has_device_capability(100)
    assert not NvmlCudaPlatform.has_device_capability((10, 0))
