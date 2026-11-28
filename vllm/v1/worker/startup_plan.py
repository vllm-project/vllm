# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Persist and reuse the memory-profiling result across engine boots.

On startup, vLLM measures how much GPU memory the KV cache can use and
computes the `--kv-cache-memory` value that reproduces that allocation.
For a fixed (model, config, hardware, library) combination the result is
deterministic, yet it is re-measured on every boot.

When `VLLM_ENABLE_STARTUP_PLAN=1`, each worker persists that value under
`{VLLM_CACHE_ROOT}/startup_plan/` (regenerable derived state, alongside
the torch.compile cache), keyed by a fingerprint of everything the value
depends on, and later boots apply it automatically -- skipping the
memory-profiling measurement and the CUDA-graph memory estimation pass --
if and only if the fingerprint matches and the device has at least as much
free memory as when the plan was recorded. On any mismatch the worker
falls back to full profiling, so a stale plan costs nothing and is never
trusted.
"""

import hashlib
import json
import os
import re
from typing import TYPE_CHECKING

import torch

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.config.utils import hash_config
from vllm.logger import init_logger
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)

PLAN_SCHEMA_VERSION = 2

# Environment variables that differ between the processes, hosts or pods of one
# deployment (ranks, device lists and rendezvous, pod identity, Kubernetes
# service links) or that carry credentials: they never change what a worker
# measures or tunes.
_ENV_NOT_FACTORS = re.compile(
    r"^(RANK|LOCAL_RANK|GROUP_RANK|ROLE_RANK|WORLD_SIZE|LOCAL_WORLD_SIZE"
    r"|.*_VISIBLE_DEVICES"
    r"|GROUP_WORLD_SIZE|ROLE_WORLD_SIZE|MASTER_ADDR|MASTER_PORT|TORCHELASTIC_.*"
    r"|HOSTNAME|PWD|OLDPWD|SHLVL|_|TERM|POD_.*|NODE_NAME|.*HOST_IP"
    r"|KUBERNETES_.*|.*_SERVICE_(HOST|PORT)(_.*)?|.*_PORT_[0-9]+_(TCP|UDP|SCTP)(_.*)?"
    r"|.*(TOKEN|SECRET|PASSWORD|CREDENTIAL|API_KEY).*)$"
)


def environment_factors() -> dict:
    """The environment a measured or tuned result depends on.

    vLLM's own variables count as in `envs.compile_factors()`. Every other
    variable counts too, except those matched by `_ENV_NOT_FACTORS`: kernels
    and allocators read variables directly, outside `envs.py` (e.g.
    `PYTORCH_CUDA_ALLOC_CONF` sets the allocator's behavior), so any of them
    can change what is measured or which kernel wins.
    """
    return {
        "vllm": sorted((k, repr(v)) for k, v in envs.compile_factors().items()),
        "other": sorted(
            (k, v)
            for k, v in os.environ.items()
            if not k.startswith("VLLM_") and not _ENV_NOT_FACTORS.match(k)
        ),
    }


def memory_sizing_factors(vllm_config: VllmConfig) -> dict:
    """Settings that size the measured memory but that the config's compile
    factors leave out because they do not change compiled graphs."""
    cache = getattr(vllm_config, "cache_config", None)
    scheduler = getattr(vllm_config, "scheduler_config", None)
    return {
        "gpu_memory_utilization": getattr(cache, "gpu_memory_utilization", None),
        "num_gpu_blocks_override": getattr(cache, "num_gpu_blocks_override", None),
        "enable_extensible_kv_cache": getattr(
            cache, "enable_extensible_kv_cache", None
        ),
        "max_num_scheduled_tokens": getattr(
            scheduler, "max_num_scheduled_tokens", None
        ),
        "max_num_encoder_input_tokens": getattr(
            scheduler, "max_num_encoder_input_tokens", None
        ),
        "encoder_cache_size": getattr(scheduler, "encoder_cache_size", None),
    }


def _pp_transport_in_use(worker: "Worker") -> str:
    return "nccl" if worker._pp_transport is None else "copy_engine"


def compute_plan_fingerprint(
    vllm_config: VllmConfig, rank: int, world_size: int, pp_transport: str = "nccl"
) -> str:
    """Hash everything the profiled KV-cache memory value depends on.

    `VllmConfig.compile_factors()` covers the vLLM version and the model,
    cache, parallel, and compilation configs, but deliberately contains no
    device identity (`DeviceConfig.compile_factors` is empty), so device
    name, total memory, compute capability, and the torch/CUDA build are
    added here. The vLLM version is also pinned as an explicit factor so
    version invalidation holds no matter how `compile_factors` evolves.
    Rank is included because per-rank memory use differs under TP/PP.
    Driver-only changes are not part of the key; the free-memory gate at
    apply time bounds the residual risk.
    """
    # Imported here (as VllmConfig.compile_factors does) to avoid a cycle with
    # the top-level vllm package.
    from vllm import __version__ as vllm_version

    capability = current_platform.get_device_capability()
    factors = {
        "schema": PLAN_SCHEMA_VERSION,
        "vllm": vllm_version,
        "vllm_config": hash_config(vllm_config),
        "device_name": current_platform.get_device_name(),
        "device_total_memory": current_platform.get_device_total_memory(),
        "device_capability": str(capability) if capability else "",
        "torch": torch.__version__,
        "cuda": torch.version.cuda or "",
        "rank": rank,
        "world_size": world_size,
        # The PP transport in use changes the memory reserved beside the KV
        # cache, and "auto" may pick either.
        "pp_transport": pp_transport,
        # Sizing settings the compile factors leave out (they do not change
        # compiled graphs), and the environment.
        "sizing": memory_sizing_factors(vllm_config),
        "env": environment_factors(),
    }
    digest = hashlib.sha256(json.dumps(factors, sort_keys=True).encode()).hexdigest()
    return digest[:16]


def _plan_path(fingerprint: str) -> str:
    """Plans are regenerable derived state, so they live under the standard
    vLLM cache root (like the torch.compile cache) and relocate with
    `VLLM_CACHE_ROOT` instead of needing a location knob of their own."""
    # VLLM_CACHE_ROOT is already user-expanded by envs.py.
    return os.path.join(
        envs.VLLM_CACHE_ROOT, "startup_plan", f"startup_plan_{fingerprint}.json"
    )


def _load_plan(fingerprint: str) -> dict | None:
    """Load a plan for this fingerprint; None if absent or unreadable."""
    path = _plan_path(fingerprint)
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


def _applicable_kv_cache_memory_bytes(
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


def maybe_apply_startup_plan(worker: "Worker") -> None:
    """If enabled and `--kv-cache-memory` was not set explicitly, apply a
    persisted plan by setting `worker.cache_config.kv_cache_memory_bytes`.
    No-op unless `VLLM_ENABLE_STARTUP_PLAN=1`."""
    if (
        not envs.VLLM_ENABLE_STARTUP_PLAN
        or worker.cache_config.kv_cache_memory_bytes is not None
    ):
        return
    fingerprint = compute_plan_fingerprint(
        worker.vllm_config,
        worker.rank,
        worker.parallel_config.world_size,
        _pp_transport_in_use(worker),
    )
    plan = _load_plan(fingerprint)
    if plan is None:
        return
    current_free_memory = worker.init_snapshot.free_memory
    kv_bytes = _applicable_kv_cache_memory_bytes(plan, current_free_memory)
    if kv_bytes is None:
        return
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
    worker.cache_config.kv_cache_memory_bytes = kv_bytes


def maybe_save_startup_plan(worker: "Worker", kv_cache_memory_bytes: int) -> None:
    """Atomically persist this boot's profiling result for future boots.
    No-op unless `VLLM_ENABLE_STARTUP_PLAN=1`; failures are logged,
    never raised."""
    if not envs.VLLM_ENABLE_STARTUP_PLAN:
        return
    fingerprint = compute_plan_fingerprint(
        worker.vllm_config,
        worker.rank,
        worker.parallel_config.world_size,
        _pp_transport_in_use(worker),
    )
    path = _plan_path(fingerprint)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "schema": PLAN_SCHEMA_VERSION,
            "fingerprint": fingerprint,
            "kv_cache_memory_bytes": int(kv_cache_memory_bytes),
            "free_memory_baseline": int(worker.init_snapshot.free_memory),
        }
        tmp = f"{path}.tmp.{os.getpid()}"
        with open(tmp, "w") as f:
            json.dump(payload, f)
        os.replace(tmp, path)
        logger.info("Saved startup plan to %s", path)
    except OSError as e:
        logger.warning("Failed to save startup plan to %s: %s", path, e)
