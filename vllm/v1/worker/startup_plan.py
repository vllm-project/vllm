# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Persist and reuse the memory-profiling result across engine boots.

On startup, vLLM measures how much GPU memory the KV cache can use and
computes the ``--kv-cache-memory`` value that reproduces that allocation.
For a fixed (model, config, hardware, library) combination the result is
deterministic, yet it is re-measured on every boot.

When ``VLLM_STARTUP_PLAN_DIR`` is set, each worker persists that value,
keyed by a fingerprint of everything the value depends on, and later boots
apply it automatically -- skipping the memory-profiling measurement and the
CUDA-graph memory estimation pass -- if and only if the fingerprint matches
and the device has at least as much free memory as when the plan was
recorded. On any mismatch the worker falls back to full profiling, so a
stale plan costs nothing and is never trusted.
"""

import hashlib
import json
import os
import time

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

PLAN_SCHEMA_VERSION = 1


def compute_plan_fingerprint(
    vllm_config: VllmConfig, rank: int, world_size: int
) -> str:
    """Hash everything the profiled KV-cache memory value depends on.

    ``VllmConfig.compute_hash()`` covers the vLLM version and the model,
    cache, parallel, and compilation configs, but deliberately contains no
    device identity (``DeviceConfig.compute_hash`` is empty), so device
    name, total memory, compute capability, and the torch/CUDA build are
    added here. Rank is included because per-rank memory use differs under
    TP/PP. Driver-only changes are not part of the key; the free-memory
    gate at apply time bounds the residual risk.
    """
    capability = current_platform.get_device_capability()
    factors = {
        "schema": PLAN_SCHEMA_VERSION,
        "vllm_config": vllm_config.compute_hash(),
        "device_name": current_platform.get_device_name(),
        "device_total_memory": current_platform.get_device_total_memory(),
        "device_capability": str(capability) if capability else "",
        "torch": torch.__version__,
        "cuda": torch.version.cuda or "",
        "rank": rank,
        "world_size": world_size,
    }
    digest = hashlib.sha256(json.dumps(factors, sort_keys=True).encode()).hexdigest()
    return digest[:16]


def _plan_path(plan_dir: str, fingerprint: str) -> str:
    return os.path.join(
        os.path.expanduser(plan_dir), f"startup_plan_{fingerprint}.json"
    )


def save_startup_plan(
    plan_dir: str,
    fingerprint: str,
    kv_cache_memory_bytes: int,
    free_memory_baseline: int,
) -> None:
    """Atomically persist the plan. Failures are logged, never raised."""
    path = _plan_path(plan_dir, fingerprint)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "schema": PLAN_SCHEMA_VERSION,
            "fingerprint": fingerprint,
            "kv_cache_memory_bytes": int(kv_cache_memory_bytes),
            "free_memory_baseline": int(free_memory_baseline),
            "created_unix": int(time.time()),
        }
        tmp = f"{path}.tmp.{os.getpid()}"
        with open(tmp, "w") as f:
            json.dump(payload, f)
        os.replace(tmp, path)
        logger.info("Saved startup plan to %s", path)
    except OSError as e:
        logger.warning("Failed to save startup plan to %s: %s", path, e)


def load_startup_plan(plan_dir: str, fingerprint: str) -> dict | None:
    """Load a plan for this fingerprint; None if absent or unreadable."""
    path = _plan_path(plan_dir, fingerprint)
    try:
        with open(path) as f:
            plan = json.load(f)
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Ignoring unreadable startup plan %s: %s", path, e)
        return None
    if (
        plan.get("schema") != PLAN_SCHEMA_VERSION
        or plan.get("fingerprint") != fingerprint
    ):
        return None
    return plan


def applicable_kv_cache_memory_bytes(
    plan: dict, current_free_memory: int
) -> int | None:
    """The apply-time OOM-safety gate.

    The recorded value is only valid if the device has at least as much
    free memory now as when the plan was measured (co-tenants, leaked
    allocations, or MIG changes all reduce it). Outside that envelope,
    return None and let the caller re-profile.
    """
    kv_bytes = plan.get("kv_cache_memory_bytes")
    baseline = plan.get("free_memory_baseline")
    if not isinstance(kv_bytes, int) or not isinstance(baseline, int):
        return None
    if kv_bytes <= 0:
        return None
    if current_free_memory < baseline:
        logger.info(
            "Startup plan not applied: current free memory (%.2f GiB) is "
            "below the recorded baseline (%.2f GiB); falling back to full "
            "memory profiling.",
            current_free_memory / (1 << 30),
            baseline / (1 << 30),
        )
        return None
    return kv_bytes


def maybe_apply_startup_plan(
    plan_dir: str,
    vllm_config: VllmConfig,
    rank: int,
    world_size: int,
    current_free_memory: int,
) -> int | None:
    """Load, validate, and return the kv-cache-memory bytes to apply."""
    fingerprint = compute_plan_fingerprint(vllm_config, rank, world_size)
    plan = load_startup_plan(plan_dir, fingerprint)
    if plan is None:
        return None
    kv_bytes = applicable_kv_cache_memory_bytes(plan, current_free_memory)
    if kv_bytes is not None:
        logger.info(
            "Applying persisted startup plan (fingerprint %s): "
            "kv_cache_memory_bytes=%d (%.2f GiB), recorded free-memory "
            "baseline %.2f GiB, current %.2f GiB. Memory profiling will "
            "be skipped.",
            fingerprint,
            kv_bytes,
            kv_bytes / (1 << 30),
            plan["free_memory_baseline"] / (1 << 30),
            current_free_memory / (1 << 30),
        )
    return kv_bytes
