# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ctypes
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from vllm.platforms import current_platform


def check_cuda_context():
    """Check CUDA driver context status."""
    try:
        cuda = ctypes.CDLL("libcuda.so")
        device = ctypes.c_int()
        result = cuda.cuCtxGetDevice(ctypes.byref(device))
        return (True, device.value) if result == 0 else (False, None)
    except Exception:
        return False, None


def run_cuda_test_in_thread(device_input, expected_device_id):
    """Run CUDA context test in separate thread for isolation."""
    try:
        # New thread should have no CUDA context initially
        valid_before, device_before = check_cuda_context()
        if valid_before:
            return (
                False,
                "CUDA context should not exist in new thread, "
                f"got device {device_before}",
            )

        # Test setting CUDA context
        current_platform.set_device(device_input)

        # Verify context is created correctly
        valid_after, device_id = check_cuda_context()
        if not valid_after:
            return False, "CUDA context should be valid after set_cuda_context"
        if device_id != expected_device_id:
            return False, f"Expected device {expected_device_id}, got {device_id}"

        return True, "Success"
    except Exception as e:
        return False, f"Exception in thread: {str(e)}"


class TestSetCudaContext:
    """Test suite for the set_cuda_context function."""

    @pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA not available")
    @pytest.mark.parametrize(
        argnames="device_input,expected_device_id",
        argvalues=[
            (0, 0),
            (torch.device("cuda:0"), 0),
            ("cuda:0", 0),
        ],
        ids=["int", "torch_device", "string"],
    )
    def test_set_cuda_context_parametrized(self, device_input, expected_device_id):
        """Test setting CUDA context in isolated threads."""
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                run_cuda_test_in_thread, device_input, expected_device_id
            )
            success, message = future.result(timeout=30)
        assert success, message

    @pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA not available")
    def test_set_cuda_context_invalid_device_type(self):
        """Test error handling for invalid device type."""
        with pytest.raises(ValueError, match="Expected a cuda device"):
            current_platform.set_device(torch.device("cpu"))


def test_get_device_capability_uses_visible_device_ordinal(monkeypatch):
    import vllm.platforms.interface as platform_interface
    from vllm.platforms.cuda import NvmlCudaPlatform, pynvml

    seen_indices: list[int] = []

    def record_handle(index: int) -> str:
        seen_indices.append(index)
        return f"handle-{index}"

    monkeypatch.setattr(platform_interface, "_assigned_physical_gpu_ids", [1])
    monkeypatch.setenv(NvmlCudaPlatform.device_control_env_var, "0,1")
    monkeypatch.setattr(
        NvmlCudaPlatform,
        "device_control_id_to_physical_device_id",
        classmethod(lambda _cls, device_id: int(device_id)),
    )
    monkeypatch.setattr(pynvml, "nvmlInit", lambda: None)
    monkeypatch.setattr(pynvml, "nvmlShutdown", lambda: None)
    monkeypatch.setattr(
        pynvml,
        "nvmlDeviceGetHandleByIndex",
        record_handle,
    )
    monkeypatch.setattr(
        pynvml,
        "nvmlDeviceGetCudaComputeCapability",
        lambda _handle: (9, 0),
    )
    NvmlCudaPlatform.get_device_capability.cache_clear()

    capability = NvmlCudaPlatform.get_device_capability(device_id=1)

    assert capability is not None
    assert capability.to_int() == 90
    assert seen_indices == [1]


def _stub_nvml(monkeypatch) -> dict[str, int]:
    """Stub NVML to report SM 9.0 and count init/shutdown pairs.

    Pins `NvmlCudaPlatform` rather than the `CudaPlatform` alias: that alias is
    `NonNvmlCudaPlatform` where NVML is unavailable, and that class reads
    `torch.cuda` instead, which needs a real device.
    """
    from vllm.platforms.cuda import NvmlCudaPlatform, pynvml

    calls = {"init": 0, "shutdown": 0}

    monkeypatch.setattr(
        pynvml, "nvmlInit", lambda: calls.__setitem__("init", calls["init"] + 1)
    )
    monkeypatch.setattr(
        pynvml,
        "nvmlShutdown",
        lambda: calls.__setitem__("shutdown", calls["shutdown"] + 1),
    )
    # Pin the visible-device mapping so the test does not depend on whatever
    # CUDA_VISIBLE_DEVICES happens to be set to in the environment.
    monkeypatch.setenv(NvmlCudaPlatform.device_control_env_var, "0")
    monkeypatch.setattr(
        NvmlCudaPlatform,
        "device_control_id_to_physical_device_id",
        classmethod(lambda _cls, device_id: int(device_id)),
    )
    monkeypatch.setattr(
        pynvml, "nvmlDeviceGetHandleByIndex", lambda index: f"handle-{index}"
    )
    monkeypatch.setattr(
        pynvml, "nvmlDeviceGetCudaComputeCapability", lambda _handle: (9, 0)
    )
    NvmlCudaPlatform.get_device_capability.cache_clear()
    return calls


def test_has_device_capability_does_not_reinit_nvml(monkeypatch):
    """Repeated capability checks must not re-enter an NVML context.

    `has_device_capability` only reads the cached `get_device_capability`, which
    carries its own NVML context. Wrapping it in `with_nvml_context` as well
    cost an nvmlInit()/nvmlShutdown() pair per call, and
    `triton_reshape_and_cache_flash` calls it per attention layer per step for
    fp8 and bfloat16 KV caches (issue #50381).
    """
    from vllm.platforms.cuda import NvmlCudaPlatform

    calls = _stub_nvml(monkeypatch)
    try:
        assert NvmlCudaPlatform.has_device_capability(80)
        for _ in range(20):
            NvmlCudaPlatform.has_device_capability(80)
            NvmlCudaPlatform.has_device_capability(89)
            NvmlCudaPlatform.has_device_capability((9, 0))

        assert calls["init"] == 1
        assert calls["shutdown"] == 1
    finally:
        NvmlCudaPlatform.get_device_capability.cache_clear()


def test_has_device_capability_comparisons(monkeypatch):
    """Dropping the redundant NVML context must not change the answers."""
    from vllm.platforms.cuda import NvmlCudaPlatform

    _stub_nvml(monkeypatch)
    try:
        assert NvmlCudaPlatform.has_device_capability(80)
        assert NvmlCudaPlatform.has_device_capability(90)
        assert NvmlCudaPlatform.has_device_capability((9, 0))
        assert not NvmlCudaPlatform.has_device_capability(100)
        assert not NvmlCudaPlatform.has_device_capability((10, 0))
    finally:
        NvmlCudaPlatform.get_device_capability.cache_clear()


@pytest.fixture
def nvml_warnings(monkeypatch):
    from vllm.logger import _print_warning_once
    from vllm.platforms.cuda import pynvml

    names = ["Model A", "Model A", "Model B"]
    failures: dict[str, Exception] = {}
    seen = []

    def query(index, operation):
        if operation == "handle":
            seen.append(index)
        if index == 1 and operation in failures:
            raise failures[operation]
        return index if operation == "handle" else names[index]

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.delenv("CUDA_DEVICE_ORDER", raising=False)
    monkeypatch.setattr(pynvml, "nvmlInit", lambda: None)
    monkeypatch.setattr(pynvml, "nvmlShutdown", lambda: None)
    monkeypatch.setattr(pynvml, "nvmlDeviceGetCount", lambda: len(names))
    monkeypatch.setattr(
        pynvml, "nvmlDeviceGetHandleByIndex", lambda i: query(i, "handle")
    )
    monkeypatch.setattr(pynvml, "nvmlDeviceGetName", lambda i: query(i, "name"))
    # Keep deduplication enabled so unhashable NVML errors cannot go unnoticed.
    _print_warning_once.cache_clear()
    try:
        yield names, failures, seen
    finally:
        _print_warning_once.cache_clear()


@pytest.mark.parametrize("operation", [None, "handle", "name"])
@pytest.mark.parametrize(
    "last_name,order,expected_warning",
    [
        ("Model A", None, False),
        ("Model B", None, True),
        ("Model B", "PCI_BUS_ID", False),
    ],
)
def test_log_warnings_preserves_readable_device_warning(
    monkeypatch,
    nvml_warnings,
    caplog_vllm,
    operation,
    last_name,
    order,
    expected_warning,
):
    """An excluded bad GPU must not hide or fabricate GPU ordering warnings."""
    from vllm.platforms.cuda import NvmlCudaPlatform, pynvml

    names, failures, seen = nvml_warnings
    names[-1] = last_name
    expected = []
    if operation is not None:
        error = pynvml.NVMLError(pynvml.NVML_ERROR_UNKNOWN)
        failures[operation] = error
        expected.append(f"Failed to query NVML device 1: {error}")
    if order is not None:
        monkeypatch.setenv("CUDA_DEVICE_ORDER", order)
    if expected_warning:
        readable = names if operation is None else [names[0], names[2]]
        expected.append(
            "Detected different devices in the system: "
            f"{', '.join(readable)}. Please make sure to set "
            "`CUDA_DEVICE_ORDER=PCI_BUS_ID` to avoid unexpected behavior."
        )

    NvmlCudaPlatform.log_warnings()
    NvmlCudaPlatform.log_warnings()

    assert caplog_vllm.messages == expected
    assert seen == [0, 1, 2] * 2


@pytest.mark.parametrize("operation", ["handle", "name"])
def test_log_warnings_propagates_non_nvml_failure(nvml_warnings, operation):
    from vllm.platforms.cuda import NvmlCudaPlatform

    _, failures, _ = nvml_warnings
    error = RuntimeError(f"{operation} programming error")
    failures[operation] = error

    with pytest.raises(RuntimeError) as exc_info:
        NvmlCudaPlatform.log_warnings()

    assert exc_info.value is error


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
