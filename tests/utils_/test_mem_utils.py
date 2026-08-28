# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from unittest.mock import MagicMock, patch

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


@create_new_process_for_each_test()
def test_memory_profiling_excludes_compilation_peak():
    """A compile-time spike must not be charged as activation memory.

    torch.compile and Inductor autotuning allocate large temporaries inside
    profile_run(), which runs inside the memory_profiling window. They are
    freed before the first real forward, so charging them as activation
    headroom shrinks the KV cache until the compile cache is warm (#54122).
    A peak recorded before compilation must still survive, because the
    multimodal encoder pass runs before the backbone compiles.
    """
    from vllm.compilation.monitor import monitor_torch_compile
    from vllm.config import CompilationMode, VllmConfig

    vllm_config = VllmConfig()
    vllm_config.compilation_config.mode = CompilationMode.VLLM_COMPILE

    _warmup = torch.zeros(1, device="cuda")
    del _warmup
    torch.accelerator.empty_cache()

    baseline_snapshot = MemorySnapshot()

    weights = torch.randn(64, 1024, 1024, device="cuda", dtype=torch.float32)
    weights_memory = 64 * 1024 * 1024 * 4  # 256 MiB

    with memory_profiling(
        baseline_snapshot=baseline_snapshot, weights_memory=weights_memory
    ) as result:
        # Encoder pass, before anything compiles: 512 MiB.
        encoder_spike = torch.randn(128, 1024, 1024, device="cuda", dtype=torch.float32)
        del encoder_spike

        # Compilation allocates far more than any forward, then frees it: 2 GiB.
        with monitor_torch_compile(vllm_config):
            compile_spike = torch.randn(
                512, 1024, 1024, device="cuda", dtype=torch.float32
            )
            del compile_spike

        # The profiling forward itself: 256 MiB.
        activation_spike = torch.randn(
            64, 1024, 1024, device="cuda", dtype=torch.float32
        )
        del activation_spike

    # The 2 GiB compile spike is excluded and the 512 MiB encoder peak, the
    # largest real one, is kept. Without the fix this is the 2 GiB spike;
    # discarding the peak outright instead would report the 256 MiB forward.
    assert result.torch_peak_increase == 512 * 1024 * 1024

    del weights


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
