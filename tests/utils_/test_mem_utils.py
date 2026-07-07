# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from unittest.mock import MagicMock, patch

import torch
from vllm_test_utils.monitor import monitor

from vllm.platforms import current_platform
from vllm.utils.mem_utils import MemorySnapshot, memory_profiling

from ..utils import create_new_process_for_each_test


def _runtime_library():
    """Return a CUDA-flavored runtime wrapper together with helpers to allocate
    and free device memory outside of PyTorch's caching allocator.

    On CUDA/ROCm ``cudaMalloc`` immediately reserves physical memory. On XPU,
    Level Zero commits device pages lazily (only on first access), so the
    allocation is touched with a ``memset`` to make it observable through the
    runtime's usable-memory query."""
    if current_platform.is_xpu():
        from vllm.distributed.device_communicators.xpu_wrapper import XpuRTLibrary

        lib = XpuRTLibrary()

        def malloc_fn(size: int):
            ptr = lib.xpuMalloc(size)
            # touch the allocation so Level Zero commits the physical pages
            lib.xpuMemset(ptr, 0, size)
            lib.xpuDeviceSynchronize()
            return ptr

        return lib.xpuFree, malloc_fn

    from vllm.distributed.device_communicators.cuda_wrapper import CudaRTLibrary

    lib = CudaRTLibrary()
    return lib.cudaFree, lib.cudaMalloc


@create_new_process_for_each_test()
def test_memory_profiling():
    device = current_platform.device_type
    is_xpu = current_platform.is_xpu()

    free_fn, malloc_fn = _runtime_library()
    # 512 MiB allocation outside of this instance
    handle1 = malloc_fn(512 * 1024 * 1024)

    # Warm up PyTorch's accelerator context so that its internal initialization
    # overhead (streams, cuBLAS/oneDNN handles, etc.) is included in the
    # baseline and does not inflate non-torch increase which is larger on ROCm
    # than on CUDA
    _warmup = torch.zeros(1, device=device)
    del _warmup
    torch.accelerator.empty_cache()

    baseline_snapshot = MemorySnapshot()

    # load weights
    weights = torch.randn(128, 1024, 1024, device=device, dtype=torch.float32)

    weights_memory = 128 * 1024 * 1024 * 4  # 512 MiB

    def measure_current_non_torch():
        free, total = torch.accelerator.get_memory_info()
        current_used = total - free
        current_torch = torch.accelerator.memory_reserved()
        current_non_torch = current_used - current_torch
        return current_non_torch

    with (
        memory_profiling(
            baseline_snapshot=baseline_snapshot, weights_memory=weights_memory
        ) as result,
        monitor(measure_current_non_torch) as monitored_values,
    ):
        # make a memory spike, 1 GiB
        spike = torch.randn(256, 1024, 1024, device=device, dtype=torch.float32)
        del spike

        # Add some extra non-torch memory 256 MiB (simulate NCCL)
        handle2 = malloc_fn(256 * 1024 * 1024)

    # this is an analytic value, it is exact on CUDA/ROCm because cudaMalloc
    # reserves memory in exact byte counts. On XPU, Level Zero commits and
    # reports device memory at page granularity, so the measured delta is
    # within a few pages of 256 MiB rather than byte-exact.
    measured_diff = monitored_values.values[-1] - monitored_values.values[0]
    if is_xpu:
        # Level Zero commits and reports device memory at page granularity, so
        # the measured delta is within a few pages of 256 MiB rather than
        # byte-exact as on CUDA/ROCm.
        assert abs(measured_diff - 256 * 1024 * 1024) <= 4096
    else:
        assert measured_diff == 256 * 1024 * 1024

    non_torch_increase = result.non_torch_increase
    if is_xpu and non_torch_increase > result.torch_peak_increase:
        # On XPU, torch memory freed via empty_cache() at the end of the
        # profiled region is not always reclaimed by the Level Zero driver
        # before the final measurement. When that happens the transient
        # activation spike (torch_peak_increase) is still counted as non-torch
        # usage because it left torch's reserved pool but not the device's used
        # pool, so discount it here.
        non_torch_increase -= result.torch_peak_increase

    # Check that the memory usage is within 5% of the expected values
    # 5% tolerance is caused by cuda runtime.
    # we cannot control cuda runtime in the granularity of bytes,
    # which causes a small error (<10 MiB in practice)
    non_torch_ratio = non_torch_increase / (256 * 1024 * 1024)  # noqa
    assert abs(non_torch_ratio - 1) <= 0.05
    assert result.torch_peak_increase == 1024 * 1024 * 1024
    del weights
    free_fn(handle1)
    free_fn(handle2)


def test_memory_snapshot_uses_psutil_on_integrated_gpu():
    """On integrated (UMA) GPUs, free_memory should come from psutil."""
    mock_cuda_free = 40 * 1024**3
    mock_cuda_total = 120 * 1024**3
    mock_psutil_available = 100 * 1024**3

    with (
        patch("vllm.utils.mem_utils.current_platform") as mock_platform,
        patch("vllm.utils.mem_utils.psutil") as mock_psutil,
        patch("torch.accelerator") as mock_accelerator,
    ):
        mock_accelerator.get_memory_info.return_value = (
            mock_cuda_free,
            mock_cuda_total,
        )
        mock_platform.is_integrated_gpu.return_value = True
        mock_platform.memory_stats.return_value = {
            "allocated_bytes.all.peak": 0,
        }
        mock_accelerator.memory_reserved.return_value = 0
        mock_accelerator.current_device = lambda: "cuda:0"

        mock_vmem = MagicMock()
        mock_vmem.available = mock_psutil_available
        mock_psutil.virtual_memory.return_value = mock_vmem

        snapshot = MemorySnapshot(device="cuda:0")

        assert snapshot.free_memory == mock_psutil_available
        assert snapshot.total_memory == mock_cuda_total
        mock_psutil.virtual_memory.assert_called_once()


def test_memory_snapshot_uses_cuda_on_discrete_gpu():
    """On discrete GPUs, free_memory should come from accelerator  get_memory_info."""
    mock_cuda_free = 70 * 1024**3
    mock_cuda_total = 80 * 1024**3

    with (
        patch("vllm.utils.mem_utils.current_platform") as mock_platform,
        patch("vllm.utils.mem_utils.psutil") as mock_psutil,
        patch("torch.accelerator") as mock_accelerator,
    ):
        mock_accelerator.get_memory_info.return_value = (
            mock_cuda_free,
            mock_cuda_total,
        )
        mock_platform.is_integrated_gpu.return_value = False
        mock_accelerator.memory_stats.return_value = {
            "allocated_bytes.all.peak": 0,
        }
        mock_accelerator.memory_reserved.return_value = 0
        mock_accelerator.current_device = lambda: "cuda:0"

        snapshot = MemorySnapshot(device="cuda:0")

        assert snapshot.free_memory == mock_cuda_free
        assert snapshot.total_memory == mock_cuda_total
        mock_psutil.virtual_memory.assert_not_called()
