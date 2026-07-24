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

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.platforms import current_platform

from .mem_constants import GiB_bytes, KiB_bytes, MiB_bytes

logger = init_logger(__name__)


def format_kib(b: int) -> str:
    return f"{round(b / KiB_bytes, 2)}"


def format_mib(b: int) -> str:
    return f"{round(b / MiB_bytes, 2)}"


def format_gib(b: int) -> str:
    return f"{round(b / GiB_bytes, 2)}"


@cache
def get_max_shared_memory_bytes(gpu: int = 0) -> int:
    """Returns the maximum shared memory per thread block in bytes."""
    from vllm import _custom_ops as ops

    max_shared_mem = ops.get_max_shared_memory_per_block_device_attribute(gpu)
    # value 0 will cause MAX_SEQ_LEN become negative and test_attention.py
    # will fail
    assert max_shared_mem > 0, "max_shared_mem cannot be zero"
    return int(max_shared_mem)


def get_cpu_memory() -> int:
    """Returns the total CPU memory of the node in bytes."""
    return psutil.virtual_memory().total


_UMA_PRESSURE_THRESHOLD = 0.8
_UMA_MIN_RELEASE_BYTES = 512 * MiB_bytes


def release_device_memory_under_pressure(device: torch.device) -> bool:
    """On integrated (UMA) GPUs, release caching-allocator memory back to the
    OS when system memory pressure is high. The OS may start thrashing before
    an allocation failure would trigger PyTorch's own cache release.

    Returns:
        True if memory was released.
    """
    if device.type != "cuda" or not current_platform.is_integrated_gpu(device.index):
        return False

    releasable = torch.accelerator.memory_reserved(
        device
    ) - torch.accelerator.memory_allocated(device)
    if releasable < _UMA_MIN_RELEASE_BYTES:
        return False

    # cudaMemGetInfo underreports free memory on UMA, see MemorySnapshot.measure
    mem = psutil.virtual_memory()
    if mem.available > (1 - _UMA_PRESSURE_THRESHOLD) * mem.total:
        return False

    torch.accelerator.synchronize(device)
    torch.accelerator.empty_cache()
    logger.debug(
        "Released %sGiB of cached device memory under memory pressure",
        format_gib(releasable),
    )
    return True


def unified_memory_host_reserve_bytes(total_memory: int) -> int:
    """Host-memory headroom to keep free on integrated (unified-memory) GPUs.

    On integrated GPUs the CPU/OS and the GPU share one physical pool, so
    ``gpu_memory_utilization`` does not by itself reserve memory for the host.
    This returns how many bytes to keep free for the OS: the larger of an
    absolute floor (``VLLM_UNIFIED_MEMORY_HOST_RESERVE_GB``) and a small
    fraction of the pool, so the reserve scales with pool size.

    Args:
        total_memory: Total size of the unified memory pool, in bytes.

    Returns:
        The number of bytes to leave available for the OS.
    """
    absolute_floor = int(envs.VLLM_UNIFIED_MEMORY_HOST_RESERVE_GB * GiB_bytes)
    proportional_floor = total_memory // 20  # 5% of the shared pool
    return max(absolute_floor, proportional_floor)


def unified_memory_allocator_ceiling_bytes(
    total_memory: int, gpu_memory_utilization: float
) -> int:
    """Hard host-safety ceiling for torch allocations on integrated GPUs.

    Leaves the OS the larger of the host reserve and the ``(1 - util)`` slice of
    the pool, so a runaway profiling transient is stopped before it can wedge
    the host. This sits at or above the KV-cache budget from
    ``cap_unified_memory_budget`` (which is bounded by *available* memory), so
    normal allocations are not clipped -- only catastrophic over-allocation.

    Args:
        total_memory: Total size of the unified memory pool, in bytes.
        gpu_memory_utilization: The configured utilization fraction.

    Returns:
        The maximum bytes the process may allocate on the device.
    """
    reserve = max(
        unified_memory_host_reserve_bytes(total_memory),
        int((1.0 - gpu_memory_utilization) * total_memory),
    )
    return max(total_memory - reserve, 0)


def cap_unified_memory_budget(
    device: torch.types.Device,
    requested_memory: int,
    available_memory: int,
    total_memory: int,
) -> int:
    """Cap a memory budget so an integrated GPU leaves headroom for the OS.

    ``gpu_memory_utilization`` is expressed against total device memory. On an
    integrated (unified-memory) GPU that total is the same pool the OS uses, so
    the raw ``util * total`` budget can starve the host. This caps the budget by
    ``available_memory - reserve`` so the host keeps a floor of free memory.
    A no-op on discrete GPUs (the two pools are independent there).

    Args:
        device: The device the budget is for.
        requested_memory: The uncapped budget (``util * total``), in bytes.
        available_memory: Currently available host/device memory, in bytes.
        total_memory: Total unified pool size, in bytes.

    Returns:
        The capped budget in bytes (unchanged on discrete GPUs).
    """
    if device is None:
        return requested_memory
    device_ = torch.device(device)
    device_index = device_.index if device_.index is not None else 0
    if device_.type != "cuda" or not current_platform.is_integrated_gpu(device_index):
        return requested_memory

    reserve = unified_memory_host_reserve_bytes(total_memory)
    honest_cap = available_memory - reserve
    if requested_memory <= honest_cap:
        return requested_memory

    logger.warning(
        "Integrated (unified-memory) GPU detected: capping the memory budget "
        "from %sGiB to %sGiB to keep %sGiB free for the OS. On these devices "
        "gpu_memory_utilization does not reserve host memory; tune it (or "
        "VLLM_UNIFIED_MEMORY_HOST_RESERVE_GB) if you need a different split.",
        format_gib(requested_memory),
        format_gib(max(honest_cap, 0)),
        format_gib(reserve),
    )
    return max(honest_cap, 0)


def limit_torch_allocator_to_budget(
    device: torch.types.Device,
    budget_memory: int,
    total_memory: int,
) -> bool:
    """Hard-cap the torch caching allocator on integrated GPUs.

    Bounds torch allocations to ``budget_memory / total_memory`` of the pool so
    a startup profiling transient that would exceed the budget raises a clean
    ``torch.OutOfMemoryError`` instead of physically exhausting the shared pool
    and wedging the host. A no-op on discrete GPUs, where an over-allocation
    already fails cleanly against a separate VRAM pool.

    Args:
        device: The device to cap.
        budget_memory: The allocation ceiling, in bytes.
        total_memory: Total device memory, in bytes.

    Returns:
        True if a cap was applied.
    """
    if device is None or total_memory <= 0:
        return False
    device_ = torch.device(device)
    device_index = device_.index if device_.index is not None else 0
    if device_.type != "cuda" or not current_platform.is_integrated_gpu(device_index):
        return False

    fraction = max(0.0, min(1.0, budget_memory / total_memory))
    torch.cuda.set_per_process_memory_fraction(fraction, device_index)
    logger.info(
        "Integrated GPU: limiting torch allocations to %sGiB (%.1f%% of the "
        "unified pool) so profiling cannot wedge the host.",
        format_gib(budget_memory),
        fraction * 100,
    )
    return True


class DeviceMemoryProfiler:
    def __init__(self, device: torch.types.Device | None = None):
        self.device = device

    def current_memory_usage(self) -> float:
        # Return the memory usage in bytes.
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
    torch_allocated: int = 0
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
            device_fn = current_platform.current_device
            assert device_fn is not None
            self.device_ = torch.device(device_fn())
        else:
            self.device_ = torch.device(self.device)

        if self.auto_measure:
            self.measure()

    def measure(self) -> None:
        device = self.device_

        # we measure the torch peak memory usage via allocated_bytes,
        # rather than `torch.accelerator.memory_reserved()` .
        # After `torch.accelerator.reset_peak_memory_stats()`,
        # `torch.accelerator.memory_reserved()` will keep growing, and only shrink
        # when we call `torch.accelerator.empty_cache()` or OOM happens.
        stats = torch.accelerator.memory_stats(device)
        self.torch_peak = stats.get("allocated_bytes.all.peak", 0)
        self.torch_allocated = stats.get("allocated_bytes.all.current", 0)

        self.free_memory, self.total_memory = torch.accelerator.get_memory_info(device)
        if current_platform.is_integrated_gpu(device.index):
            # On UMA (Unified Memory Architecture) platforms where CPU and
            # GPU share physical memory (e.g. GH200, DGX Spark, Jetson Orin),
            # cudaMemGetInfo underreports free memory because it does not
            # account for reclaimable OS memory (page cache, buffers).
            # Use psutil to get the true available memory.
            # https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/#estimating-total-allocatable-device-memory-on-an-integrated-gpu-device
            self.free_memory = psutil.virtual_memory().available

        self.cuda_memory = self.total_memory - self.free_memory

        # torch.accelerator.memory_reserved() is how many bytes
        # PyTorch gets from cuda (by calling cudaMalloc, etc.)
        # this is used to measure the non-torch memory usage
        self.torch_memory = torch.accelerator.memory_reserved(device)

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
            torch_allocated=self.torch_allocated - other.torch_allocated,
            free_memory=self.free_memory - other.free_memory,
            total_memory=self.total_memory - other.total_memory,
            cuda_memory=self.cuda_memory - other.cuda_memory,
            torch_memory=self.torch_memory - other.torch_memory,
            non_torch_memory=self.non_torch_memory - other.non_torch_memory,
            timestamp=self.timestamp - other.timestamp,
            device=self.device_,
            auto_measure=False,
        )

    def __repr__(self) -> str:
        return (
            f"torch_peak={format_gib(self.torch_peak)}GiB, "
            f"torch_allocated={format_gib(self.torch_allocated)}GiB, "
            f"free_memory={format_gib(self.free_memory)}GiB, "
            f"total_memory={format_gib(self.total_memory)}GiB, "
            f"{current_platform.device_name}_memory={format_gib(self.cuda_memory)}GiB, "
            f"torch_memory={format_gib(self.torch_memory)}GiB, "
            f"non_torch_memory={format_gib(self.non_torch_memory)}GiB, "
            f"timestamp={self.timestamp}, "
            f"auto_measure={self.auto_measure}"
        )


@dataclass
class MemoryProfilingResult:
    """Memory profiling result. All numbers are in bytes."""

    non_kv_cache_memory: int = 0
    torch_peak_increase: int = 0
    non_torch_increase: int = 0
    total_consumed: int = 0
    transient_peak_headroom: int = 0
    weights_memory: int = 0
    before_create: MemorySnapshot = field(default_factory=MemorySnapshot)
    profile_time: float = 0.0

    def __post_init__(self) -> None:
        device = self.before_create.device_

        self.before_profile = MemorySnapshot(device=device, auto_measure=False)
        self.after_profile = MemorySnapshot(device=device, auto_measure=False)

    def __repr__(self) -> str:
        return (
            f"Memory profiling takes {self.profile_time:.2f} seconds. "
            f"Total non KV cache memory: "
            f"{format_gib(self.non_kv_cache_memory)}GiB; "
            f"torch peak memory increase: "
            f"{format_gib(self.torch_peak_increase)}GiB; "
            f"total consumed (from mem_get_info): "
            f"{format_gib(self.total_consumed)}GiB; "
            f"weights memory: {format_gib(self.weights_memory)}GiB."
        )


@contextlib.contextmanager
def memory_profiling(
    baseline_snapshot: MemorySnapshot,
    weights_memory: int = 0,
) -> Generator[MemoryProfilingResult, None, None]:
    """
    Memory profiling context manager.

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

    The memory used for loading weights (a.) is directly given from the
    argument `weights_memory`.

    The increase of `torch.accelerator.memory_stats()["allocated_bytes.all.peak"]`
    during profiling gives (b.).

    The increase of `non_torch_memory` from creating the current vLLM instance
    until after profiling to get (c.).
    """
    gc.collect()
    torch.accelerator.empty_cache()
    torch.accelerator.reset_peak_memory_stats(baseline_snapshot.device_)

    result = MemoryProfilingResult(
        before_create=baseline_snapshot,
        # the part of memory used for holding the model weights
        weights_memory=weights_memory,
    )

    result.before_profile.measure()

    yield result

    gc.collect()
    torch.accelerator.empty_cache()

    result.after_profile.measure()

    diff_profile = result.after_profile - result.before_profile
    diff_from_create = result.after_profile - result.before_create
    result.torch_peak_increase = diff_profile.torch_peak
    result.non_torch_increase = diff_from_create.non_torch_memory
    result.profile_time = diff_profile.timestamp

    # Measure total consumption via mem_get_info() instead of
    # memory_reserved(), which goes negative when pluggable allocators
    # (e.g. cumem) bypass PyTorch's tracking.
    result.total_consumed = (
        result.before_create.free_memory - result.after_profile.free_memory
    )

    # total_consumed already covers persistent torch allocations; add only the
    # transient peak headroom to avoid double-counting.
    result.transient_peak_headroom = (
        result.after_profile.torch_peak - result.after_profile.torch_allocated
    )
    result.non_kv_cache_memory = result.total_consumed + result.transient_peak_headroom
