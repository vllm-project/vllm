# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV Cache Observability Utilities for Step-Level Tracing.

This module provides read-only helper functions to extract KV cache metrics
for observability and tracing purposes. It uses ONLY existing exposed interfaces
from KVCacheManager and BlockPool, with no modifications to KV cache behavior.

Design principles:
- Read-only access to existing KV cache state
- Defensive programming: never raises exceptions to caller
- Minimal guaranteed metrics + best-effort optional metrics
- No expensive scans or per-token/per-block iteration beyond existing methods
- No new KV subsystem APIs or Request fields

Guaranteed metrics are always available from current vLLM interfaces.
Optional metrics are best-effort and may return None if not accessible.
"""
from __future__ import annotations

from dataclasses import dataclass

from vllm.logger import init_logger
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass
class PerRequestKVMetrics:
    """Per-request KV cache metrics for observability.

    GUARANTEED fields (always available from existing interfaces):
    - blocks_allocated_gpu: GPU KV blocks allocated to this request
    - blocks_cached_gpu: GPU blocks with prefix cache hits for this request

    OPTIONAL fields (best-effort, may be None if not accessible):
    - blocks_cpu_offload: CPU-offloaded KV blocks (if offload enabled)
    - blocks_disk_offload: Disk-offloaded KV blocks (if offload enabled)
    - effective_prompt_len: Prompt tokens after prefix cache reduction
                            (only if request.num_cached_tokens is valid)

    Contract: Uses ONLY existing exposed interfaces. Never raises exceptions.
    Returns minimal/zero values with None optionals if data unavailable.
    """

    # Guaranteed metrics (GPU)
    blocks_allocated_gpu: int
    blocks_cached_gpu: int

    # Optional metrics (best-effort)
    blocks_cpu_offload: int | None = None
    blocks_disk_offload: int | None = None
    effective_prompt_len: int | None = None


@dataclass
class StepKVSummary:
    """Step-level KV cache summary for batch summary events.

    GUARANTEED fields (always available from BlockPool):
    - blocks_total_gpu: Total GPU KV blocks in pool (excluding null block)
    - blocks_free_gpu: Free GPU KV blocks available for allocation
    - usage_gpu_ratio: KV cache usage ratio (0.0 to 1.0)

    OPTIONAL fields (best-effort):
    - blocks_cpu_offload: Total CPU-offloaded KV blocks (if offload enabled)
    - blocks_disk_offload: Total disk-offloaded KV blocks (if offload enabled)

    Contract: Uses ONLY existing BlockPool methods. Never raises exceptions.
    Returns minimal values with None optionals if data unavailable.
    """

    # Guaranteed metrics (GPU)
    blocks_total_gpu: int
    blocks_free_gpu: int
    usage_gpu_ratio: float

    # Optional metrics (best-effort)
    blocks_cpu_offload: int | None = None
    blocks_disk_offload: int | None = None


def get_per_request_kv_metrics(
    request: Request,
    manager: KVCacheManager,
) -> PerRequestKVMetrics:
    """Extract per-request KV metrics from existing KV cache state.

    This function queries read-only KV cache interfaces to gather metrics
    for a single request. It is defensive and never raises exceptions.

    Args:
        request: The request to get KV metrics for.
        manager: The KVCacheManager instance.

    Returns:
        PerRequestKVMetrics with guaranteed GPU metrics and optional fields.

    Behavior:
        - If manager or coordinator is unavailable: returns zero metrics
        - If request not found in tracking maps: returns zero allocated/cached
        - If offload managers unavailable: optional offload metrics are None
        - If effective_prompt_len not computable: returns None
    """
    try:
        # Get coordinator (handles multiple KV cache groups)
        coordinator = getattr(manager, "coordinator", None)
        if coordinator is None:
            logger.debug(
                "KVCacheManager has no coordinator, returning zero metrics "
                "for request %s",
                request.request_id,
            )
            return PerRequestKVMetrics(blocks_allocated_gpu=0, blocks_cached_gpu=0)

        # Access single_type_managers to get per-request block tracking
        # For simplicity, use first manager (works for most models)
        single_managers = getattr(coordinator, "single_type_managers", ())
        if not single_managers:
            logger.debug(
                "No single_type_managers found, returning zero metrics "
                "for request %s",
                request.request_id,
            )
            return PerRequestKVMetrics(blocks_allocated_gpu=0, blocks_cached_gpu=0)

        # GUARANTEED: Aggregate GPU blocks across all KV cache groups
        # Most models have one group, but some use multiple (e.g., MLA)
        allocated_blocks = sum(
            len(getattr(m, "req_to_blocks", {}).get(request.request_id, []))
            for m in single_managers
        )

        # GUARANTEED: Aggregate GPU blocks cached (prefix cache hits)
        cached_blocks = sum(
            getattr(m, "num_cached_block", {}).get(request.request_id, 0)
            for m in single_managers
        )

        # OPTIONAL: Effective prompt length (only if num_cached_tokens is valid)
        effective_prompt = None
        if hasattr(request, "num_cached_tokens"):
            cached_tokens = getattr(request, "num_cached_tokens", -1)
            # num_cached_tokens is initialized to -1, only compute if set (>= 0)
            if cached_tokens >= 0:
                if cached_tokens < request.num_prompt_tokens:
                    effective_prompt = request.num_prompt_tokens - cached_tokens
                # If cached_tokens >= num_prompt_tokens, full cache hit, no effective prompt needed

        # OPTIONAL: CPU/disk offload metrics
        # Currently not implemented in vLLM v1, but check for future support
        cpu_offload = None
        disk_offload = None
        # Future: if hasattr(manager, 'offload_manager') and manager.offload_manager:
        #     cpu_offload = manager.offload_manager.get_cpu_blocks(request.request_id)
        #     disk_offload = manager.offload_manager.get_disk_blocks(request.request_id)

        return PerRequestKVMetrics(
            blocks_allocated_gpu=allocated_blocks,
            blocks_cached_gpu=cached_blocks,
            blocks_cpu_offload=cpu_offload,
            blocks_disk_offload=disk_offload,
            effective_prompt_len=effective_prompt,
        )

    except Exception as e:
        # Defensive: never raise, return minimal metrics
        logger.debug(
            "Failed to get KV metrics for request %s: %s. "
            "Returning zero metrics.",
            request.request_id,
            e,
        )
        return PerRequestKVMetrics(blocks_allocated_gpu=0, blocks_cached_gpu=0)


def get_step_kv_summary(
    block_pool: BlockPool,
) -> StepKVSummary:
    """Extract step-level KV cache summary from BlockPool.

    This function queries read-only BlockPool state to gather aggregate
    KV cache metrics for the current step. It is defensive and never raises.

    Args:
        block_pool: The BlockPool instance.

    Returns:
        StepKVSummary with guaranteed GPU metrics and optional fields.

    Behavior:
        - If block_pool is None: returns zero/minimal metrics
        - If methods unavailable: returns best-effort fallback values
        - GPU metrics always computed (guaranteed)
        - Offload metrics currently None (not implemented in vLLM v1)

    Note:
        - blocks_total_gpu accounts for the null block (subtracts 1)
        - usage_gpu_ratio matches BlockPool.get_usage() implementation
    """
    try:
        if block_pool is None:
            logger.debug("BlockPool is None, returning zero KV summary metrics")
            return StepKVSummary(
                blocks_total_gpu=0,
                blocks_free_gpu=0,
                usage_gpu_ratio=0.0,
            )

        # GUARANTEED: Total GPU blocks (excluding null block)
        num_gpu_blocks = getattr(block_pool, "num_gpu_blocks", 1)
        try:
            num_gpu_blocks_i = int(num_gpu_blocks)
        except (TypeError, ValueError):
            num_gpu_blocks_i = 1
        blocks_total = max(0, num_gpu_blocks_i - 1)  # Clamp to non-negative

        # GUARANTEED: Free GPU blocks
        blocks_free = 0
        if hasattr(block_pool, "get_num_free_blocks"):
            blocks_free = block_pool.get_num_free_blocks()

        # GUARANTEED: Usage ratio (0.0 to 1.0)
        usage_ratio = 0.0
        if hasattr(block_pool, "get_usage"):
            usage_ratio = block_pool.get_usage()
        elif blocks_total > 0 and hasattr(block_pool, "get_num_free_blocks"):
            # Fallback: compute from free blocks (only if measurable)
            usage_ratio = 1.0 - (blocks_free / blocks_total)
        # else: usage_ratio remains 0.0 (conservative when unmeasurable)

        # Clamp usage to [0.0, 1.0] for safety
        usage_ratio = max(0.0, min(1.0, float(usage_ratio)))

        # OPTIONAL: CPU/disk offload metrics
        # Currently not implemented in vLLM v1
        cpu_offload = None
        disk_offload = None
        # Future: if hasattr(block_pool, 'offload_manager'):
        #     cpu_offload = block_pool.offload_manager.get_total_cpu_blocks()
        #     disk_offload = block_pool.offload_manager.get_total_disk_blocks()

        return StepKVSummary(
            blocks_total_gpu=blocks_total,
            blocks_free_gpu=blocks_free,
            usage_gpu_ratio=usage_ratio,
            blocks_cpu_offload=cpu_offload,
            blocks_disk_offload=disk_offload,
        )

    except Exception as e:
        # Defensive: never raise, return minimal metrics
        logger.debug(
            "Failed to get step KV summary: %s. Returning zero metrics.",
            e,
        )
        return StepKVSummary(
            blocks_total_gpu=0,
            blocks_free_gpu=0,
            usage_gpu_ratio=0.0,
        )
