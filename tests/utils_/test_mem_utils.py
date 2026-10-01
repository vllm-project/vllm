# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm_test_utils.monitor import monitor

from vllm.platforms import current_platform
from vllm.utils.mem_constants import GiB_bytes as GiB
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


def test_memory_snapshot_on_integrated_gpu():
    """Integrated GPUs should use the platform's corrected memory info."""
    mock_cuda_free = 40 * GiB
    mock_cuda_total = 120 * GiB

    with (
        patch("vllm.utils.mem_utils.current_platform") as mock_platform,
        patch("torch.accelerator") as mock_accelerator,
    ):
        mock_accelerator.get_memory_info.return_value = (
            mock_cuda_free,
            mock_cuda_total,
        )
        mock_platform.is_integrated_gpu.return_value = True
        mock_platform.get_integrated_gpu_memory_info.return_value = (
            30 * GiB,
            100 * GiB,
        )
        mock_accelerator.memory_stats.return_value = {
            "allocated_bytes.all.peak": 0,
        }
        mock_accelerator.memory_reserved.return_value = 0

        snapshot = MemorySnapshot(device="cuda:0")

        mock_platform.get_integrated_gpu_memory_info.assert_called_once_with(
            0, mock_cuda_free, mock_cuda_total
        )
        assert snapshot.free_memory == 30 * GiB
        assert snapshot.total_memory == 100 * GiB
        assert snapshot.cuda_memory == 70 * GiB


# Cases measured on a 128 GiB Strix Halo (gfx1151). HIP reports the GTT limit
# while it serves allocations from GTT, and the carve-out otherwise.
@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm only")
@pytest.mark.parametrize(
    ("from_gtt", "hip_info", "host_available", "host_total", "expected"),
    [
        # GTT-backed, idle: HIP's GTT bound is the tighter one.
        (True, (100, 100), 121, 125, (100, 100)),
        # GTT-backed, host memory in use elsewhere: host bound is tighter.
        (True, (100, 100), 59, 125, (59, 100)),
        # GTT-backed with a GTT limit above the RAM the OS sees: clamp total.
        (True, (60.1, 100), 58.9, 62.6, (58.9, 62.6)),
        # Carve-out-backed: host usage does not consume the carve-out.
        (False, (63.8, 64), 38.3, 62.6, (63.8, 64)),
    ],
)
def test_rocm_integrated_gpu_memory_info(
    from_gtt: bool,
    hip_info: tuple[float, float],
    host_available: float,
    host_total: float,
    expected: tuple[float, float],
):
    from vllm.platforms.rocm import RocmPlatform

    def gib(x: float) -> int:
        return int(x * GiB)

    with (
        patch("vllm.platforms.rocm._apu_allocates_from_gtt", return_value=from_gtt),
        patch("vllm.platforms.rocm.psutil") as mock_psutil,
    ):
        mock_psutil.virtual_memory.return_value = MagicMock(
            available=gib(host_available), total=gib(host_total)
        )

        free, total = RocmPlatform.get_integrated_gpu_memory_info(
            0, gib(hip_info[0]), gib(hip_info[1])
        )

    assert (free, total) == (gib(expected[0]), gib(expected[1]))
    assert 0 <= free <= total


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
