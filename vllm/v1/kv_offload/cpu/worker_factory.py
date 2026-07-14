# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.platforms import current_platform
from vllm.v1.kv_offload.base import CanonicalKVCaches, OffloadingWorker
from vllm.v1.kv_offload.cpu.gpu_worker import CPUOffloadingWorker
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion


def is_ascend_platform() -> bool:
    """Return whether the active out-of-tree platform is Ascend NPU."""
    return getattr(current_platform, "device_type", None) == "npu"


def create_cpu_offloading_worker(
    kv_caches: CanonicalKVCaches,
    block_size_factor: int,
    num_cpu_blocks: int,
    mmap_region: SharedOffloadRegion | None = None,
) -> OffloadingWorker:
    """Create the device-specific worker for a CPU-backed offload tier."""
    if is_ascend_platform():
        from vllm.v1.kv_offload.cpu.npu_worker import AscendCPUOffloadingWorker

        return AscendCPUOffloadingWorker(
            kv_caches=kv_caches,
            block_size_factor=block_size_factor,
            num_cpu_blocks=num_cpu_blocks,
            mmap_region=mmap_region,
        )

    return CPUOffloadingWorker(
        kv_caches=kv_caches,
        block_size_factor=block_size_factor,
        num_cpu_blocks=num_cpu_blocks,
        mmap_region=mmap_region,
    )
