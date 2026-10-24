# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-memory admission checks for anonymous AuxOutput SHM arenas."""

import os

import psutil

from vllm.logger import init_logger
from vllm.utils.cpu_resource_utils import (
    get_cgroup_memory_limit,
    get_cgroup_memory_usage,
)

logger = init_logger(__name__)
_METADATA_BYTES_PER_OBJECT = 512


def _slurm_memory_limit() -> int | None:
    """Return the job's Slurm memory allocation in bytes, or None.

    Slurm exports the request in MiB. Many clusters do not enforce it through a
    cgroup memory controller, so the scheduler quota is the only place the real
    limit is visible; ignoring it lets the admission check pass on a node with
    far less memory than the job may use.
    """
    per_node = os.environ.get("SLURM_MEM_PER_NODE")
    if per_node:
        try:
            return int(per_node) * 1024 * 1024
        except ValueError:
            return None
    per_cpu = os.environ.get("SLURM_MEM_PER_CPU")
    cpus = os.environ.get("SLURM_CPUS_ON_NODE") or os.environ.get("SLURM_CPUS_PER_TASK")
    if per_cpu and cpus:
        try:
            return int(per_cpu) * int(cpus) * 1024 * 1024
        except ValueError:
            return None
    return None


def _cgroup_memory_current() -> int | None:
    """Return this process' cgroup memory usage even when the limit is unset."""
    try:
        with open("/proc/self/cgroup") as fh:
            for line in fh:
                if not line.startswith("0::"):
                    continue
                relative = line.strip().split("::", 1)[1].lstrip("/")
                path = f"/sys/fs/cgroup/{relative}/memory.current"
                with open(path) as current:
                    return int(current.read().strip())
    except (OSError, ValueError):
        pass
    try:
        with open("/sys/fs/cgroup/memory/memory.usage_in_bytes") as usage:
            return int(usage.read().strip())
    except (OSError, ValueError):
        return None


def get_available_host_memory() -> int:
    """Return current system/cgroup headroom for anonymous shared mappings."""
    available: int = psutil.virtual_memory().available
    limit = get_cgroup_memory_limit()
    usage = get_cgroup_memory_usage()
    if limit is not None and usage is not None:
        available = min(available, max(0, limit - usage))
    slurm_limit = _slurm_memory_limit()
    if slurm_limit is not None:
        used = _cgroup_memory_current() or 0
        available = min(available, max(0, slurm_limit - used))
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
    required_context_tokens: int | None = None,
) -> int:
    """Validate all local arenas and return the allocated bytes per worker.

    Args:
        num_blocks: GPU KV blocks per worker.
        scheduler_block_size: Tokens covered by each GPU KV block.
        hash_block_size: Tokens stored in one SHM object.
        bytes_per_token: Sum of all enabled artifact payload widths.
        max_bytes: Configured arena capacity, or None to match KV coverage.
        available_host_bytes: Current system/cgroup/Slurm memory headroom.
        staging_bytes: Estimated staging memory per output worker.
        local_output_workers: Output workers sharing this host-memory budget.
        required_context_tokens: Tokens one request must be able to serve. When
            the host budget only allows fewer KV blocks than that, no
            utilization setting can satisfy both limits and the caller must be
            told so instead of receiving a block count that cannot serve it.

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
        max_context_tokens = supported_blocks * scheduler_block_size
        if (
            required_context_tokens is not None
            and max_context_tokens < required_context_tokens
        ):
            raise ValueError(
                f"AuxOutput SHM budget exceeded: Required AuxOutput SHM: "
                f"{required_bytes / gib:.3f} GiB per worker "
                f"({local_output_workers * required_bytes / gib:.3f} GiB total); "
                f"available host memory={available_host_bytes / gib:.3f} GiB; "
                f"maximum usable SHM={max_host_shm_bytes / gib:.3f} GiB per worker; "
                f"estimated host need including staging, metadata and reserve="
                f"{host_required / gib:.3f} GiB. Current KV Blocks: {num_blocks}. "
                "No feasible AuxOutput configuration: this host budget supports at "
                f"most {max_context_tokens} tokens ({supported_blocks} KV blocks), "
                f"but max_model_len={required_context_tokens}. Increase available "
                "RAM/container memory, reduce --max-model-len, or lower "
                "--max-num-batched-tokens/--max-num-seqs to reduce staging memory."
            )
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
