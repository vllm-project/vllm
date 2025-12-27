# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
import gc
import time
from collections.abc import Generator
from dataclasses import dataclass, field
from functools import cache

import psutil
import torch
import torch.types

from vllm.logger import init_logger

from .mem_constants import GiB_bytes

logger = init_logger(__name__)


@cache
def get_max_shared_memory_bytes(gpu: int = 0) -> int:
    """Returns the maximum shared memory per thread block in bytes."""
    from vllm import _custom_ops as ops

    max_shared_mem = ops.get_max_shared_memory_per_block_device_attribute(gpu)
    # value 0 will cause MAX_SEQ_LEN become negative and test_attention.py
    # will fail
    assert max_shared_mem > 0, "max_shared_mem can not be zero"
    return int(max_shared_mem)


def get_cpu_memory() -> int:
    """Returns the total CPU memory of the node in bytes."""
    return psutil.virtual_memory().total


class DeviceMemoryProfiler:
    def __init__(self, device: torch.types.Device | None = None):
        self.device = device

    def current_memory_usage(self) -> float:
        # Return the memory usage in bytes.
        from vllm.platforms import current_platform

        gc.collect()
        return current_platform.get_current_memory_usage(self.device)

    def __enter__(self):
        self.initial_memory = self.current_memory_usage()
        # This allows us to call methods of the context manager if needed
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.final_memory = self.current_memory_usage()
        self.consumed_memory = self.final_memory - self.initial_memory

        # Force garbage collection
        gc.collect()


@dataclass
class MemorySnapshot:
    """Memory snapshot."""

    torch_peak: int = 0
    free_memory: int = 0
    total_memory: int = 0
    cuda_memory: int = 0
    torch_memory: int = 0
    non_torch_memory: int = 0
    timestamp: float = 0.0

    device: torch.types.Device = None
    auto_measure: bool = True

    def __post_init__(self) -> None:
        if self.device is None:
            from vllm.platforms import current_platform

            device_fn = current_platform.current_device
            assert device_fn is not None
            self.device_ = torch.device(device_fn())
        else:
            self.device_ = torch.device(self.device)

        if self.auto_measure:
            self.measure()

    def measure(self) -> None:
        from vllm.platforms import current_platform

        device = self.device_

        # we measure the torch peak memory usage via allocated_bytes,
        # rather than `torch.cuda.memory_reserved()` .
        # After `torch.cuda.reset_peak_memory_stats()`,
        # `torch.cuda.memory_reserved()` will keep growing, and only shrink
        # when we call `torch.cuda.empty_cache()` or OOM happens.
        self.torch_peak = torch.cuda.memory_stats(device).get(
            "allocated_bytes.all.peak", 0
        )

        self.free_memory, self.total_memory = torch.cuda.mem_get_info(device)
        shared_sysmem_device_mem_sms = ((8, 7), (11, 0), (12, 1))  # Orin, Thor, Spark
        if (
            current_platform.is_cuda()
            and current_platform.get_device_capability(device.index)
            in shared_sysmem_device_mem_sms
        ):
            # On UMA (Orin, Thor and Spark) platform,
            # where both CPU and GPU rely on system memory,
            # the cudaMemGetInfo function shows the amount of free system memory
            # rather than what’s actually available.
            # In the case,
            # torch.cuda.mem_get_info() only reports "free" memory,
            # which can be lower than what is actually
            # available due to not including cache memory.
            # There’s also a comprehensive reference page
            # that explains how you can compute the proper value yourself.
            # https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/#estimating-total-allocatable-device-memory-on-an-integrated-gpu-device
            self.free_memory = psutil.virtual_memory().available

        self.cuda_memory = self.total_memory - self.free_memory

        # torch.cuda.memory_reserved() is how many bytes
        # PyTorch gets from cuda (by calling cudaMalloc, etc.)
        # this is used to measure the non-torch memory usage
        self.torch_memory = torch.cuda.memory_reserved(device)

        self.non_torch_memory = self.cuda_memory - self.torch_memory
        self.timestamp = time.time()

    def __sub__(self, other: "MemorySnapshot") -> "MemorySnapshot":
        if self.device_ != other.device_:
            raise ValueError(
                "The two snapshots should be from the same device! "
                f"Found: {self.device_} vs. {other.device_}"
            )

        return MemorySnapshot(
            torch_peak=self.torch_peak - other.torch_peak,
            free_memory=self.free_memory - other.free_memory,
            total_memory=self.total_memory - other.total_memory,
            cuda_memory=self.cuda_memory - other.cuda_memory,
            torch_memory=self.torch_memory - other.torch_memory,
            non_torch_memory=self.non_torch_memory - other.non_torch_memory,
            timestamp=self.timestamp - other.timestamp,
            device=self.device_,
            auto_measure=False,
        )


@dataclass
class MemoryProfilingResult:
    """Memory profiling result. All numbers are in bytes."""

    non_kv_cache_memory: int = 0
    torch_peak_increase: int = 0
    non_torch_increase: int = 0
    weights_memory: float = 0
    before_create: MemorySnapshot = field(default_factory=MemorySnapshot)
    before_profile: MemorySnapshot = field(default_factory=MemorySnapshot)
    after_profile: MemorySnapshot = field(default_factory=MemorySnapshot)
    profile_time: float = 0.0

    def __repr__(self) -> str:
        return (
            f"Memory profiling takes {self.profile_time:.2f} seconds. "
            f"Total non KV cache memory: "
            f"{(self.non_kv_cache_memory / GiB_bytes):.2f}GiB; "
            f"torch peak memory increase: "
            f"{(self.torch_peak_increase / GiB_bytes):.2f}GiB; "
            f"non-torch forward increase memory: "
            f"{(self.non_torch_increase / GiB_bytes):.2f}GiB; "
            f"weights memory: {(self.weights_memory / GiB_bytes):.2f}GiB."
        )


@contextlib.contextmanager
def memory_profiling(
    baseline_snapshot: MemorySnapshot, weights_memory: int
) -> Generator[MemoryProfilingResult, None, None]:
    """Memory profiling context manager.
    baseline_snapshot: the memory snapshot before the current vLLM instance.
    weights_memory: memory used by PyTorch when loading the model weights.
        Note that, before loading the model weights, we also initialize the device
        and distributed environment, which may consume some memory. This part is not
        included in the weights_memory because PyTorch does not control it.

    The memory in one GPU can be classified into 3 categories:
    1. memory used by anything other than the current vLLM instance.
    2. memory used by torch in the current vLLM instance.
    3. memory used in the current vLLM instance, but not by torch.

    A quantitive example:

    Before creating the current vLLM instance:
        category 1: 1 GiB
        category 2: 0 GiB
        category 3: 0 GiB

    After creating the current vLLM instance and loading the model,
    (i.e. before profiling):
        category 1: 1 GiB
        category 2: 2 GiB (model weights take 2 GiB)
        category 3: 0.5 GiB (memory used by NCCL)

    During profiling (peak):
        category 1: 1 GiB
        category 2: 4 GiB (peak activation tensors take 2 GiB)
        category 3: 1 GiB (memory used by NCCL + buffers for some attention backends)

    After profiling:
        category 1: 1 GiB
        category 2: 3 GiB (after garbage-collecting activation tensors)
        category 3: 1 GiB (memory used by NCCL + buffers for some attention backends)

    In this case, non-kv cache takes 5 GiB in total, including:
    a. 2 GiB used by the model weights (category 2)
    b. 2 GiB reserved for the peak activation tensors (category 2)
    c. 1 GiB used by non-torch components (category 3)

    The memory used for loading weights (a.) is directly given from the argument `weights_memory`.

    The increase of `torch.cuda.memory_stats()["allocated_bytes.all.peak"]` during profiling gives (b.).

    The increase of `non_torch_memory` from creating the current vLLM instance until after profiling to get (c.).
    """  # noqa
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    result = MemoryProfilingResult()

    result.before_create = baseline_snapshot
    # the part of memory used for holding the model weights
    result.weights_memory = weights_memory

    result.before_profile.measure()

    yield result

    gc.collect()
    torch.cuda.empty_cache()

    result.after_profile.measure()

    diff_profile = result.after_profile - result.before_profile
    diff_from_create = result.after_profile - result.before_create
    result.torch_peak_increase = diff_profile.torch_peak
    result.non_torch_increase = diff_from_create.non_torch_memory
    result.profile_time = diff_profile.timestamp

    non_torch_memory = result.non_torch_increase
    peak_activation_memory = result.torch_peak_increase
    result.non_kv_cache_memory = (
        non_torch_memory + peak_activation_memory + result.weights_memory
    )  # noqa


class MemorySnapshotProfiler:
    """Memory snapshot profiler with start/stop API.

    This class provides a similar interface to torch.profiler.profile()
    for memory profiling, supporting:
    - start()/stop() methods for controlling profiling
    - Exception handling to dump snapshot on error
    - Context manager support

    Usage:
        profiler = MemorySnapshotProfiler(
            output_dir="/path/to/output",
            dump_on_exception=True,
        )

        # Method 1: start/stop
        profiler.start()
        ... # code to profile
        profiler.stop()

        # Method 2: context manager
        with profiler:
            ... # code to profile

    The output .pickle file can be visualized at https://pytorch.org/memory_viz
    """

    def __init__(
        self,
        output_dir: str,
        filename_prefix: str = "memory_snapshot",
        max_entries: int = 100000,
        dump_on_exception: bool = True,
    ):
        """
        Args:
            output_dir: Directory to save the memory snapshot file.
            filename_prefix: Prefix for the snapshot filename.
            max_entries: Maximum number of allocation entries to record.
            dump_on_exception: If True, dump snapshot when exception occurs.
        """
        self.output_dir = output_dir
        self.filename_prefix = filename_prefix
        self.max_entries = max_entries
        self.dump_on_exception = dump_on_exception
        self._recording = False
        self._rank: int | None = None

    def set_rank(self, rank: int) -> None:
        """Set the worker rank for filename disambiguation."""
        self._rank = rank

    def start(self) -> "MemorySnapshotProfiler":
        """Start recording memory allocation history."""
        import os

        if self._recording:
            logger.warning("Memory snapshot profiler is already recording.")
            return self

        os.makedirs(self.output_dir, exist_ok=True)

        # Clear existing memory state for a cleaner baseline
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Start recording memory history with full stack traces
        torch.cuda.memory._record_memory_history(
            enabled="all",  # Record all allocations
            stacks="all",  # Capture both Python and C++ stacks
            max_entries=self.max_entries,
        )
        self._recording = True
        logger.info("Memory snapshot profiling started.")
        return self

    def stop(self, suffix: str | None = None) -> str | None:
        """Stop recording and save the memory snapshot.

        Args:
            suffix: Optional suffix to add to the filename before timestamp.

        Returns:
            Path to the saved snapshot file, or None if not recording.
        """
        import os

        if not self._recording:
            logger.warning("Memory snapshot profiler is not recording.")
            return None

        torch.cuda.synchronize()

        # Build filename
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        parts = [self.filename_prefix]
        if self._rank is not None:
            parts.append(f"rank{self._rank}")
        if suffix:
            parts.append(suffix)
        parts.append(timestamp)
        filename = "_".join(parts) + ".pickle"

        snapshot_file = os.path.join(self.output_dir, filename)
        torch.cuda.memory._dump_snapshot(snapshot_file)
        torch.cuda.memory._record_memory_history(enabled=None)
        self._recording = False

        logger.info(
            "Memory snapshot saved to %s. Visualize at https://pytorch.org/memory_viz",
            snapshot_file,
        )
        return snapshot_file

    def dump_on_error(self) -> str | None:
        """Dump snapshot on error without stopping the profiler.

        Returns:
            Path to the saved snapshot file, or None if not recording.
        """
        import os

        if not self._recording:
            return None

        torch.cuda.synchronize()

        # Build filename with error suffix
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        parts = [self.filename_prefix]
        if self._rank is not None:
            parts.append(f"rank{self._rank}")
        parts.append("error")
        parts.append(timestamp)
        filename = "_".join(parts) + ".pickle"

        snapshot_file = os.path.join(self.output_dir, filename)
        try:
            torch.cuda.memory._dump_snapshot(snapshot_file)
            logger.info(
                "Memory snapshot (error) saved to %s. "
                "Visualize at https://pytorch.org/memory_viz",
                snapshot_file,
            )
            return snapshot_file
        except Exception as e:
            logger.warning("Failed to dump memory snapshot on error: %s", e)
            return None

    def __enter__(self) -> "MemorySnapshotProfiler":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is not None and self.dump_on_exception:
            # Dump snapshot on exception before stopping
            self.dump_on_error()
        self.stop()

    @property
    def is_recording(self) -> bool:
        """Check if profiler is currently recording."""
        return self._recording
