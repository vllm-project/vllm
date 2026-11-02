# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import multiprocessing as mp
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm_test_utils.monitor import monitor

from vllm.platforms import current_platform
from vllm.utils.mem_utils import MemorySnapshot, memory_profiling

from ..utils import create_new_process_for_each_test


@create_new_process_for_each_test()
def test_memory_profiling():
    # Fake out some model loading + inference memory usage to test profiling
    # Memory used by other processes will show up as cuda usage outside of torch
    from vllm.distributed.device_communicators.cuda_wrapper import CudaRTLibrary

    lib = CudaRTLibrary()
    # 512 MiB allocation outside of this instance
    handle1 = lib.cudaMalloc(512 * 1024 * 1024)

    # Warm up PyTorch's CUDA/ROCm context so that its internal initialization
    # overhead (streams, cuBLAS handles, etc.) is included in the baseline and
    # does not inflate non-torch increase which is larger on ROCm than on CUDA
    _warmup = torch.zeros(1, device="cuda")
    del _warmup
    torch.accelerator.empty_cache()

    baseline_snapshot = MemorySnapshot()

    # load weights

    weights = torch.randn(128, 1024, 1024, device="cuda", dtype=torch.float32)

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
        spike = torch.randn(256, 1024, 1024, device="cuda", dtype=torch.float32)
        del spike

        # Add some extra non-torch memory 256 MiB (simulate NCCL)
        handle2 = lib.cudaMalloc(256 * 1024 * 1024)

    # this is an analytic value, it is exact,
    # we only have 256 MiB non-torch memory increase
    measured_diff = monitored_values.values[-1] - monitored_values.values[0]
    assert measured_diff == 256 * 1024 * 1024

    # Check that the memory usage is within 5% of the expected values
    # 5% tolerance is caused by cuda runtime.
    # we cannot control cuda runtime in the granularity of bytes,
    # which causes a small error (<10 MiB in practice)
    non_torch_ratio = result.non_torch_increase / (256 * 1024 * 1024)  # noqa
    assert abs(non_torch_ratio - 1) <= 0.05
    assert result.torch_peak_increase == 1024 * 1024 * 1024

    expected_total_consumed = (256 + 512) * 1024 * 1024
    total_consumed_ratio = result.total_consumed / expected_total_consumed
    assert abs(total_consumed_ratio - 1) <= 0.05, (
        f"total_consumed={result.total_consumed}, "
        f"expected={expected_total_consumed}, "
        f"ratio={total_consumed_ratio}"
    )

    expected_non_kv = expected_total_consumed + 1024 * 1024 * 1024
    non_kv_ratio = result.non_kv_cache_memory / expected_non_kv
    assert abs(non_kv_ratio - 1) <= 0.05, (
        f"non_kv_cache_memory={result.non_kv_cache_memory}, "
        f"expected={expected_non_kv}, "
        f"ratio={non_kv_ratio}"
    )

    del weights
    lib.cudaFree(handle1)
    lib.cudaFree(handle2)


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


def _mock_measurements(mock_platform, mock_accelerator, free_values, process_values):
    """Feed one (free memory, process memory) pair per MemorySnapshot.measure()."""
    total = 80 * 1024**3
    mock_accelerator.get_memory_info.side_effect = [(f, total) for f in free_values]
    mock_accelerator.memory_stats.return_value = {
        "allocated_bytes.all.peak": 0,
        "allocated_bytes.all.current": 0,
    }
    mock_accelerator.memory_reserved.return_value = 0
    mock_accelerator.current_device = lambda: "cuda:0"
    mock_platform.is_integrated_gpu.return_value = False
    mock_platform.get_process_memory_usage.side_effect = list(process_values)


def test_memory_profiling_uses_process_scoped_consumption():
    """Another process allocating during load/profile must not be charged
    to this instance when per-process usage is available."""
    gib = 1024**3
    with (
        patch("vllm.utils.mem_utils.current_platform") as mock_platform,
        patch("torch.accelerator") as mock_accelerator,
    ):
        # baseline -> before_profile -> after_profile:
        # this process grows 1 GiB then 1 GiB more; another process takes
        # 2 GiB in the same window, so the device-wide delta is 4 GiB.
        _mock_measurements(
            mock_platform,
            mock_accelerator,
            free_values=[70 * gib, 69 * gib, 66 * gib],
            process_values=[1 * gib, 2 * gib, 3 * gib],
        )
        baseline = MemorySnapshot(device="cuda:0")
        with memory_profiling(baseline_snapshot=baseline) as result:
            pass

    assert baseline.process_memory == 1 * gib
    assert result.process_scoped
    assert result.total_consumed == 2 * gib
    assert result.non_kv_cache_memory == 2 * gib


def test_memory_profiling_falls_back_to_device_delta_without_process_usage():
    gib = 1024**3
    with (
        patch("vllm.utils.mem_utils.current_platform") as mock_platform,
        patch("torch.accelerator") as mock_accelerator,
    ):
        _mock_measurements(
            mock_platform,
            mock_accelerator,
            free_values=[70 * gib, 69 * gib, 66 * gib],
            process_values=[None, None, None],
        )
        baseline = MemorySnapshot(device="cuda:0")
        with memory_profiling(baseline_snapshot=baseline) as result:
            pass

    assert baseline.process_memory is None
    assert not result.process_scoped
    assert result.total_consumed == 4 * gib


def test_memory_snapshot_subtraction_keeps_process_memory():
    a = MemorySnapshot(device="cuda:0", auto_measure=False)
    b = MemorySnapshot(device="cuda:0", auto_measure=False)
    a.process_memory, b.process_memory = 5, 2
    assert (a - b).process_memory == 3
    b.process_memory = None
    assert (a - b).process_memory is None


def _hold_device_memory(num_bytes: int, ready, release):
    import torch

    buf = torch.empty(num_bytes, dtype=torch.uint8, device="cuda")
    torch.cuda.synchronize()
    ready.set()
    release.wait()
    del buf


@create_new_process_for_each_test()
def test_memory_profiling_ignores_other_process_allocations():
    """End-to-end on a real device: a second process allocates 512 MiB while
    this one is profiling; total_consumed must only reflect our own 256 MiB."""
    _warmup = torch.zeros(1, device="cuda")
    del _warmup
    torch.accelerator.empty_cache()
    if current_platform.get_process_memory_usage(torch.cuda.current_device()) is None:
        pytest.skip("platform cannot report per-process device memory usage")

    baseline_snapshot = MemorySnapshot()
    weights = torch.randn(64, 1024, 1024, device="cuda", dtype=torch.float32)
    weights_memory = 64 * 1024 * 1024 * 4  # 256 MiB

    ctx = mp.get_context("spawn")
    ready, release = ctx.Event(), ctx.Event()
    other = ctx.Process(
        target=_hold_device_memory, args=(512 * 1024 * 1024, ready, release)
    )
    other.start()
    try:
        with memory_profiling(
            baseline_snapshot=baseline_snapshot, weights_memory=weights_memory
        ) as result:
            assert ready.wait(timeout=120), "helper process did not allocate"
    finally:
        release.set()
        other.join(timeout=60)

    assert result.process_scoped
    ratio = result.total_consumed / weights_memory
    assert abs(ratio - 1) <= 0.05, (
        f"total_consumed={result.total_consumed}, expected~{weights_memory}"
    )
    del weights
