# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ctypes
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform


def check_cuda_context():
    """Check CUDA driver context status"""
    try:
        cuda = ctypes.CDLL("libcuda.so")
        device = ctypes.c_int()
        result = cuda.cuCtxGetDevice(ctypes.byref(device))
        return (True, device.value) if result == 0 else (False, None)
    except Exception:
        return False, None


def run_cuda_test_in_thread(device_input, expected_device_id):
    """Run CUDA context test in separate thread for isolation"""
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


@pytest.mark.parametrize(
    "device_env_var",
    ["CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES"],
)
def test_get_device_total_memory_uses_mig_uuid_handle(monkeypatch, device_env_var):
    import vllm.platforms.interface as platform_interface
    from vllm.platforms.cuda import NvmlCudaPlatform, pynvml

    mig_uuid = "MIG-be6b2880-0e33-5698-a7c1-3972ccc1f981"
    mig_handle = object()
    total_memory = 33280 * 2**20

    monkeypatch.setattr(platform_interface, "_assigned_physical_gpu_ids", None)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("NVIDIA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv(device_env_var, mig_uuid)
    monkeypatch.setattr(pynvml, "nvmlInit", lambda: None)
    monkeypatch.setattr(pynvml, "nvmlShutdown", lambda: None)
    monkeypatch.setattr(
        pynvml,
        "nvmlDeviceGetHandleByUUID",
        lambda device_uuid: mig_handle if device_uuid == mig_uuid else None,
    )
    monkeypatch.setattr(
        pynvml,
        "nvmlDeviceGetHandleByIndex",
        lambda _index: pytest.fail("queried the parent GPU handle"),
    )
    monkeypatch.setattr(
        pynvml,
        "nvmlDeviceGetMemoryInfo",
        lambda handle: SimpleNamespace(total=total_memory)
        if handle is mig_handle
        else pytest.fail("queried memory for the wrong device"),
    )

    assert NvmlCudaPlatform.get_device_total_memory() == total_memory


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
