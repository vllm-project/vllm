# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Capacity validation and host-memory estimates for AuxOutput SHM arenas."""

import psutil

from vllm.logger import init_logger
from vllm.utils.cpu_resource_utils import (
    get_cgroup_memory_limit,
    get_cgroup_memory_usage,
)

logger = init_logger(__name__)
_METADATA_BYTES_PER_OBJECT = 512


def get_available_host_memory() -> int:
    """Return current headroom for anonymous shared mappings.

    Cgroup limit minus usage is conservative: usage includes reclaimable page
    cache, which this startup estimate does not assume will be reclaimed.
    """
    available: int = psutil.virtual_memory().available
    limit = get_cgroup_memory_limit()
    usage = get_cgroup_memory_usage()
    if limit is not None and limit >= 0 and usage is not None and usage >= 0:
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
    min_blocks_for_context: int,
    local_output_workers: int = 1,
    required_context_tokens: int | None = None,
) -> int:
    """Validate KV coverage, warn about host memory and return bytes per worker.

    The host-memory estimate covers the preallocated arena, staging, metadata
    and runtime reserve at startup. It excludes serialized routing payloads,
    background-store queued batches, the transient old-plus-new arena while the
    store grows, and concurrent memory use, so it is not a runtime peak bound.

    Args:
        num_blocks: GPU KV blocks per worker.
        scheduler_block_size: Tokens covered by each GPU KV block.
        hash_block_size: Tokens stored in one SHM object.
        bytes_per_token: Sum of all enabled artifact payload widths.
        max_bytes: Configured arena capacity, or None to match KV coverage.
        available_host_bytes: Current system/cgroup memory headroom.
        staging_bytes: Estimated staging memory per output worker.
        min_blocks_for_context: Minimum device KV pool blocks that can serve one
            request at max_model_len, including the BlockPool null block.
            Smaller block counts must not be suggested as an override.
        local_output_workers: Output workers assumed to share this host-memory
            budget. Workers isolated in separate containers make the estimate
            conservative.
        required_context_tokens: Tokens one request must be able to serve, used
            in the warning text.

    Returns:
        Arena capacity rounded down to complete SHM objects.

    Raises:
        ValueError: The configured arena cannot cover all GPU KV blocks.

    """
    object_bytes = hash_block_size * bytes_per_token
    hashes_per_block = scheduler_block_size // hash_block_size
    required_bytes = num_blocks * scheduler_block_size * bytes_per_token
    arena_bytes = required_bytes if max_bytes is None else max_bytes
    num_objects = arena_bytes // object_bytes
    arena_bytes = num_objects * object_bytes
    gib = 1024**3
    if required_bytes > arena_bytes:
        coverage = (
            "cannot hold one KV block"
            if num_objects < hashes_per_block
            else "cannot cover all GPU KV blocks"
        )
        raise ValueError(
            f"AuxOutput SHM arena {coverage}: required SHM="
            f"{required_bytes / gib:.3f} GiB per worker; arena capacity="
            f"{arena_bytes / gib:.3f} GiB per worker; KV blocks={num_blocks}. "
            "Unset aux_output_config.max_bytes to size the arena from KV "
            f"coverage, or increase it to at least {required_bytes} bytes."
        )
    metadata_bytes = num_objects * _METADATA_BYTES_PER_OBJECT
    reserve_bytes = max(512 << 20, available_host_bytes // 20)
    worker_headroom = max(
        0,
        (available_host_bytes - reserve_bytes) // local_output_workers - staging_bytes,
    )
    host_objects = worker_headroom // (object_bytes + _METADATA_BYTES_PER_OBJECT)
    max_host_shm_bytes = host_objects * object_bytes
    supported_blocks = host_objects // hashes_per_block
    recommended_blocks = (
        supported_blocks
        if min_blocks_for_context <= supported_blocks < num_blocks
        else None
    )
    estimated_startup_need = (
        local_output_workers * (arena_bytes + staging_bytes + metadata_bytes)
        + reserve_bytes
    )
    adjustment = ""
    if estimated_startup_need > available_host_bytes:
        adjustment = (
            " Estimated startup host-memory need exceeds current headroom. "
            "Startup will continue, but AuxOutput may run out of host memory. "
            "This is a conservative startup estimate, not a runtime peak bound; "
            "cgroup usage may include reclaimable page cache. "
        )
        if max_bytes is not None:
            adjustment += (
                "Unset aux_output_config.max_bytes to size the arena from the "
                "chosen KV block count, or resize it to cover that count within "
                "the estimated headroom. "
            )
        max_context_tokens = supported_blocks * scheduler_block_size
        if available_host_bytes <= reserve_bytes + local_output_workers * staging_bytes:
            adjustment += (
                "Staging and runtime reserve exhaust the estimated headroom. "
                "Increase available RAM/container memory or reduce "
                "--max-num-batched-tokens/--max-num-seqs to reduce staging memory."
            )
        elif supported_blocks == 0:
            adjustment += (
                "The estimated headroom cannot fit one GPU KV block. Increase "
                "available RAM/container memory or reduce staging memory."
            )
        elif supported_blocks < min_blocks_for_context:
            if required_context_tokens is not None:
                needed = (
                    f"the {min_blocks_for_context} blocks needed to serve "
                    f"max_model_len={required_context_tokens}"
                )
            else:
                needed = (
                    f"the {min_blocks_for_context} blocks needed for one "
                    "max_model_len request"
                )
            adjustment += (
                f"The conservative estimate supports {supported_blocks} KV blocks "
                f"({max_context_tokens} tokens), fewer than {needed}. Increase "
                "available RAM/container memory, reduce --max-model-len, or lower "
                "--max-num-batched-tokens/--max-num-seqs to reduce staging memory."
            )
        elif recommended_blocks is not None:
            if max_bytes is not None:
                adjustment += "Along with that arena adjustment, consider "
            else:
                adjustment += "Consider "
            adjustment += (
                f"--num-gpu-blocks-override {recommended_blocks} "
                f"(Python: num_gpu_blocks_override={recommended_blocks}) to fit "
                "the conservative estimate. Keep enough KV capacity for "
                "--max-model-len if reducing further. Alternatively, lower "
                "--kv-cache-memory-bytes or --gpu-memory-utilization; an explicit "
                "--kv-cache-memory-bytes overrides --gpu-memory-utilization."
            )
    log_fn = (
        logger.warning if estimated_startup_need > available_host_bytes else logger.info
    )
    log_fn(
        "AuxOutput memory budget (startup estimate): "
        "available system/cgroup=%.3f GiB, "
        "runtime reserve=%.3f GiB, local output workers=%d; "
        "per worker: required SHM=%.3f GiB, allocated SHM=%.3f GiB, "
        "estimated SHM headroom=%.3f GiB, staging=%.3f GiB, "
        "metadata estimate=%.3f GiB; total required SHM=%.3f GiB, "
        "estimated startup need=%.3f GiB; KV blocks=%d, "
        "conservative KV block estimate=%d.%s",
        available_host_bytes / gib,
        reserve_bytes / gib,
        local_output_workers,
        required_bytes / gib,
        arena_bytes / gib,
        max_host_shm_bytes / gib,
        staging_bytes / gib,
        metadata_bytes / gib,
        local_output_workers * required_bytes / gib,
        estimated_startup_need / gib,
        num_blocks,
        supported_blocks,
        adjustment,
    )
    return arena_bytes
