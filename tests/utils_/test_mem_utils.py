# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
from vllm_test_utils.monitor import monitor

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

    baseline_snapshot = MemorySnapshot()

    # load weights

    weights = torch.randn(128, 1024, 1024, device="cuda", dtype=torch.float32)

    weights_memory = 128 * 1024 * 1024 * 4  # 512 MiB

    def measure_current_non_torch():
        free, total = torch.cuda.mem_get_info()
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

    # total_consumed should reflect all GPU memory used between baseline and
    # after profiling, measured via mem_get_info(). This includes weights
    # (512 MiB) + non-torch (256 MiB) = 768 MiB.
    expected_total_consumed = (256 + 512) * 1024 * 1024  # 768 MiB
    total_consumed_ratio = result.total_consumed / expected_total_consumed
    assert abs(total_consumed_ratio - 1) <= 0.05, (
        f"total_consumed={result.total_consumed}, "
        f"expected={expected_total_consumed}, "
        f"ratio={total_consumed_ratio}"
    )

    # non_kv_cache_memory = total_consumed + transient_peak_headroom
    # transient_peak_headroom = torch_peak - torch_allocated (after profiling)
    # In this test no persistent torch allocations are created during
    # profiling (spike is deleted), so transient_headroom == torch_peak_increase
    # = 768 MiB + 1 GiB = 1.75 GiB
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


@create_new_process_for_each_test()
def test_memory_profiling_persistent_torch():
    """Test that persistent torch allocations during profiling are not
    double-counted. transient_peak_headroom should only capture the
    transient spike, not persistent allocations already in total_consumed."""
    from vllm.distributed.device_communicators.cuda_wrapper import CudaRTLibrary

    lib = CudaRTLibrary()
    handle1 = lib.cudaMalloc(512 * 1024 * 1024)  # 512 MiB external

    baseline_snapshot = MemorySnapshot()

    # load weights: 512 MiB
    weights = torch.randn(128, 1024, 1024, device="cuda", dtype=torch.float32)
    weights_memory = 128 * 1024 * 1024 * 4  # 512 MiB

    with memory_profiling(
        baseline_snapshot=baseline_snapshot, weights_memory=weights_memory
    ) as result:
        # Transient spike: 1 GiB (freed before persistent is created)
        spike = torch.randn(256, 1024, 1024, device="cuda", dtype=torch.float32)
        del spike

        # Persistent torch allocation: 256 MiB (NOT freed during profiling)
        persistent = torch.randn(  # noqa: F841
            64, 1024, 1024, device="cuda", dtype=torch.float32
        )

    persistent_size = 64 * 1024 * 1024 * 4  # 256 MiB

    # transient_peak_headroom = torch_peak - torch_allocated (after profiling)
    # torch_peak = weights(512) + spike(1024) = 1536 MiB (peak during profiling)
    # torch_allocated = weights(512) + persistent(256) = 768 MiB (after profiling)
    # transient_peak_headroom = 1536 - 768 = 768 MiB
    transient_peak_headroom = (
        result.after_profile.torch_peak - result.after_profile.torch_allocated
    )

    # Key assertion: transient_peak_headroom < torch_peak_increase by exactly
    # the persistent allocation size. This proves persistent torch allocations
    # (already in total_consumed) are correctly excluded from the headroom.
    assert result.torch_peak_increase - transient_peak_headroom == persistent_size, (
        f"torch_peak_increase={result.torch_peak_increase}, "
        f"transient_peak_headroom={transient_peak_headroom}, "
        f"diff={result.torch_peak_increase - transient_peak_headroom}, "
        f"expected_persistent={persistent_size}"
    )

    # Verify non_kv_cache_memory uses transient_peak_headroom, not
    # torch_peak_increase (which would double-count persistent by 256 MiB).
    assert result.non_kv_cache_memory == (
        result.total_consumed + transient_peak_headroom
    ), (
        f"non_kv_cache_memory={result.non_kv_cache_memory}, "
        f"expected={result.total_consumed + transient_peak_headroom}"
    )

    del persistent
    del weights
    lib.cudaFree(handle1)
