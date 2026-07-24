# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math
from unittest.mock import MagicMock, patch

import torch
from vllm_test_utils.monitor import monitor

from vllm.utils.mem_constants import GiB_bytes
from vllm.utils.mem_utils import (
    MemorySnapshot,
    cap_unified_memory_budget,
    limit_torch_allocator_to_budget,
    memory_profiling,
    unified_memory_allocator_ceiling_bytes,
    unified_memory_host_reserve_bytes,
)

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


def test_unified_memory_host_reserve_uses_env_floor():
    """The reserve is the larger of the env floor and 5% of the pool."""
    total = 120 * GiB_bytes
    with patch("vllm.utils.mem_utils.envs") as mock_envs:
        mock_envs.VLLM_UNIFIED_MEMORY_HOST_RESERVE_GB = 8.0
        # 5% of 120 GiB = 6 GiB < 8 GiB floor -> floor wins.
        assert unified_memory_host_reserve_bytes(total) == 8 * GiB_bytes


def test_unified_memory_host_reserve_scales_with_pool():
    """For a large pool the proportional (5%) reserve dominates the floor."""
    total = 400 * GiB_bytes
    with patch("vllm.utils.mem_utils.envs") as mock_envs:
        mock_envs.VLLM_UNIFIED_MEMORY_HOST_RESERVE_GB = 8.0
        # 5% of 400 GiB = 20 GiB > 8 GiB floor -> proportional wins.
        assert unified_memory_host_reserve_bytes(total) == 20 * GiB_bytes


def test_allocator_ceiling_leaves_util_slice_at_low_util():
    """At moderate util, (1-util)*total dominates the floor -> ceiling=util*total."""
    total = 120 * GiB_bytes
    with patch("vllm.utils.mem_utils.envs") as mock_envs:
        mock_envs.VLLM_UNIFIED_MEMORY_HOST_RESERVE_GB = 8.0
        # (1-0.7)*120 = 36 GiB > 8 GiB floor -> ceiling = 120 - 36 = 84 GiB.
        ceiling = unified_memory_allocator_ceiling_bytes(total, 0.7)
    assert abs(ceiling - 84 * GiB_bytes) <= 2  # tolerate fp rounding on (1-util)


def test_allocator_ceiling_uses_reserve_floor_at_high_util():
    """At high util, the absolute reserve floor dominates (1-util)*total."""
    total = 120 * GiB_bytes
    with patch("vllm.utils.mem_utils.envs") as mock_envs:
        mock_envs.VLLM_UNIFIED_MEMORY_HOST_RESERVE_GB = 8.0
        # (1-0.97)*120 = 3.6 GiB < 8 GiB floor -> ceiling = 120 - 8 = 112 GiB.
        ceiling = unified_memory_allocator_ceiling_bytes(total, 0.97)
    assert abs(ceiling - 112 * GiB_bytes) <= 2  # tolerate fp rounding


def test_allocator_ceiling_is_at_or_above_budget():
    """The host-safety ceiling must not clip the available-bounded KV budget."""
    total = 120 * GiB_bytes
    available = 100 * GiB_bytes
    with (
        patch("vllm.utils.mem_utils.current_platform") as mock_platform,
        patch("vllm.utils.mem_utils.envs") as mock_envs,
    ):
        mock_platform.is_integrated_gpu.return_value = True
        mock_envs.VLLM_UNIFIED_MEMORY_HOST_RESERVE_GB = 8.0
        budget = cap_unified_memory_budget(
            "cuda:0", math.ceil(total * 0.7), available, total
        )
        ceiling = unified_memory_allocator_ceiling_bytes(total, 0.7)
    assert ceiling >= budget


def test_cap_unified_memory_budget_noop_on_discrete_gpu():
    """Discrete GPUs keep the full util*total budget (independent pools)."""
    requested = 60 * GiB_bytes
    with patch("vllm.utils.mem_utils.current_platform") as mock_platform:
        mock_platform.is_integrated_gpu.return_value = False
        capped = cap_unified_memory_budget(
            "cuda:0",
            requested_memory=requested,
            available_memory=10 * GiB_bytes,
            total_memory=80 * GiB_bytes,
        )
    assert capped == requested


def test_cap_unified_memory_budget_leaves_os_reserve():
    """On integrated GPUs the budget is capped to available - reserve."""
    total = 120 * GiB_bytes
    available = 100 * GiB_bytes
    # util=0.9 -> requested 108 GiB, more than available.
    requested = 108 * GiB_bytes
    with (
        patch("vllm.utils.mem_utils.current_platform") as mock_platform,
        patch("vllm.utils.mem_utils.envs") as mock_envs,
    ):
        mock_platform.is_integrated_gpu.return_value = True
        mock_envs.VLLM_UNIFIED_MEMORY_HOST_RESERVE_GB = 8.0
        capped = cap_unified_memory_budget(
            "cuda:0",
            requested_memory=requested,
            available_memory=available,
            total_memory=total,
        )
    # reserve = max(8, 6) GiB = 8 GiB -> cap = 100 - 8 = 92 GiB.
    assert capped == available - 8 * GiB_bytes


def test_cap_unified_memory_budget_keeps_budget_that_fits():
    """A budget already leaving the reserve free is untouched."""
    total = 120 * GiB_bytes
    available = 100 * GiB_bytes
    requested = 70 * GiB_bytes  # leaves 30 GiB free, above 8 GiB reserve.
    with (
        patch("vllm.utils.mem_utils.current_platform") as mock_platform,
        patch("vllm.utils.mem_utils.envs") as mock_envs,
    ):
        mock_platform.is_integrated_gpu.return_value = True
        mock_envs.VLLM_UNIFIED_MEMORY_HOST_RESERVE_GB = 8.0
        capped = cap_unified_memory_budget(
            "cuda:0",
            requested_memory=requested,
            available_memory=available,
            total_memory=total,
        )
    assert capped == requested


def test_limit_torch_allocator_noop_on_discrete_gpu():
    """No allocator cap is applied on discrete GPUs."""
    with (
        patch("vllm.utils.mem_utils.current_platform") as mock_platform,
        patch("torch.cuda.set_per_process_memory_fraction") as mock_set,
    ):
        mock_platform.is_integrated_gpu.return_value = False
        applied = limit_torch_allocator_to_budget(
            "cuda:0", budget_memory=60 * GiB_bytes, total_memory=80 * GiB_bytes
        )
    assert applied is False
    mock_set.assert_not_called()


def test_limit_torch_allocator_caps_fraction_on_integrated_gpu():
    """On integrated GPUs the allocator is capped to budget/total."""
    with (
        patch("vllm.utils.mem_utils.current_platform") as mock_platform,
        patch("torch.cuda.set_per_process_memory_fraction") as mock_set,
    ):
        mock_platform.is_integrated_gpu.return_value = True
        applied = limit_torch_allocator_to_budget(
            "cuda:0", budget_memory=84 * GiB_bytes, total_memory=120 * GiB_bytes
        )
    assert applied is True
    mock_set.assert_called_once()
    fraction, index = mock_set.call_args.args
    assert index == 0
    assert abs(fraction - 0.7) < 1e-6
