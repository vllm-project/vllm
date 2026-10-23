# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-memory admission checks for anonymous AuxOutput SHM arenas."""

import psutil

from vllm.logger import init_logger
from vllm.utils.cpu_resource_utils import (
    get_cgroup_memory_limit,
    get_cgroup_memory_usage,
)

logger = init_logger(__name__)
_METADATA_BYTES_PER_OBJECT = 512


def get_available_host_memory() -> int:
    """Return current system/cgroup headroom for anonymous shared mappings."""
    available: int = psutil.virtual_memory().available
    limit = get_cgroup_memory_limit()
    usage = get_cgroup_memory_usage()
    if limit is not None and usage is not None:
        available = min(available, max(0, limit - usage))
    return available


def validate_shm_budget(
    *,
    num_blocks: int,
    scheduler_block_size: int,
    hash_block_size: int,
    bytes_per_token: int,
    max_bytes: int | None,
    available_host_bytes: int,
    staging_bytes: int,
    local_output_workers: int = 1,
) -> int:
    """Validate all local arenas and return the allocated bytes per worker.

    Args:
        num_blocks: GPU KV blocks per worker.
        scheduler_block_size: Tokens covered by each GPU KV block.
        hash_block_size: Tokens stored in one SHM object.
        bytes_per_token: Sum of all enabled artifact payload widths.
        max_bytes: Configured arena capacity, or None to match KV coverage.
        available_host_bytes: Current system/cgroup memory headroom.
        staging_bytes: Estimated staging memory per output worker.
        local_output_workers: Output workers sharing this host-memory budget.

    Returns:
        Arena capacity rounded down to complete SHM objects.

    Raises:
        ValueError: The configured arenas cannot cover KV blocks or fit in RAM.

    """
    object_bytes = hash_block_size * bytes_per_token
    hashes_per_block = scheduler_block_size // hash_block_size
    required_bytes = num_blocks * scheduler_block_size * bytes_per_token
    arena_bytes = required_bytes if max_bytes is None else max_bytes
    num_objects = arena_bytes // object_bytes
    arena_bytes = num_objects * object_bytes
    metadata_bytes = num_objects * _METADATA_BYTES_PER_OBJECT
    reserve_bytes = max(512 << 20, available_host_bytes // 20)
    worker_headroom = max(
        0,
        (available_host_bytes - reserve_bytes) // local_output_workers - staging_bytes,
    )
    host_objects = worker_headroom // (object_bytes + _METADATA_BYTES_PER_OBJECT)
    max_host_shm_bytes = host_objects * object_bytes
    supported_objects = host_objects
    if max_bytes is not None:
        supported_objects = min(supported_objects, num_objects)
    supported_blocks = supported_objects // hashes_per_block
    host_required = (
        local_output_workers * (arena_bytes + staging_bytes + metadata_bytes)
        + reserve_bytes
    )
    gib = 1024**3
    logger.info(
        "AuxOutput memory budget: available system/cgroup=%.3f GiB, "
        "runtime reserve=%.3f GiB, local output workers=%d; "
        "per worker: required SHM=%.3f GiB, allocated SHM=%.3f GiB, "
        "maximum usable SHM=%.3f GiB, staging=%.3f GiB, "
        "metadata estimate=%.3f GiB; total required SHM=%.3f GiB, "
        "estimated host need=%.3f GiB; KV blocks=%d, maximum supported=%d",
        available_host_bytes / gib,
        reserve_bytes / gib,
        local_output_workers,
        required_bytes / gib,
        arena_bytes / gib,
        max_host_shm_bytes / gib,
        staging_bytes / gib,
        metadata_bytes / gib,
        local_output_workers * required_bytes / gib,
        host_required / gib,
        num_blocks,
        supported_blocks,
    )
    if required_bytes > arena_bytes or host_required > available_host_bytes:
        block_limit = min(num_blocks, supported_blocks)
        if block_limit > 0:
            adjustment = (
                f"Set --num-gpu-blocks-override {block_limit} "
                f"(Python: num_gpu_blocks_override={block_limit}) or lower. "
                "A smaller --max-model-len may also be needed. "
            )
        elif host_objects < hashes_per_block:
            adjustment = (
                "No GPU KV block fits the current host budget. Increase available "
                "RAM/container memory or reduce --max-num-batched-tokens and "
                "--max-num-seqs to reduce staging memory. "
            )
        else:
            adjustment = (
                "The configured aux_output_config.max_bytes cannot hold one "
                "KV block; unset it or raise it within the host-memory limit. "
            )
        raise ValueError(
            f"AuxOutput SHM budget exceeded: Required AuxOutput SHM: "
            f"{required_bytes / gib:.3f} GiB per worker "
            f"({local_output_workers * required_bytes / gib:.3f} GiB total); "
            f"arena capacity={arena_bytes / gib:.3f} GiB per worker; "
            f"available host memory={available_host_bytes / gib:.3f} GiB; "
            f"maximum usable SHM={max_host_shm_bytes / gib:.3f} GiB per worker; "
            f"estimated host need including staging, metadata and reserve="
            f"{host_required / gib:.3f} GiB. Current KV Blocks: {num_blocks}; "
            f"Maximum Supported Blocks: {supported_blocks}. "
            + adjustment
            + "Unset aux_output_config.max_bytes to size the arena from the chosen "
            "KV block count, or adjust it to cover that count within the reported "
            "host limit. Alternatively, lower --kv-cache-memory-bytes or "
            "--gpu-memory-utilization to reduce KV blocks; an explicit "
            "--kv-cache-memory-bytes overrides --gpu-memory-utilization."
        )
    return arena_bytes
