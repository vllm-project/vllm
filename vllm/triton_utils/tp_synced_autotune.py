# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Make Triton autotune choose the same config on every tensor-parallel rank.

Autotune times each candidate config on the local GPU, so ranks can choose different
configs for the same kernel and shape, and results can differ in the last bits. Plain
TP hides this, because the next all-reduce gives every rank the same sum. Code where
ranks compute the same thing on their own with no all-reduce after it (replicated
layers, indexers, per-rank recurrent state) then lets the ranks drift apart.

Under VLLM_TRITON_SYNC_AUTOTUNE_TP=1, when a kernel meets a new shape key, TP rank 0
resolves it as usual (Triton's on-disk autotune cache, or a benchmark) and broadcasts
the chosen config and its timings; the other ranks adopt them without benchmarking or
reading their own disk cache, so a cache that is warm on some ranks only cannot split
them. Sync happens only inside a model forward pass, where all ranks of a TP group
launch the same kernels with the same shapes; a kernel autotuned outside it (or by one
rank only) resolves locally, so it cannot hang. One CPU broadcast per kernel and key.

Layouts where ranks of a TP group run kernels on rank-dependent shapes (expert
parallelism, sequence-parallel MoE, decode/prefill context parallelism) could make one
rank autotune alone and wait forever, so `configure()` turns syncing off for them.
"""

from typing import Any

from vllm.logger import init_logger
from vllm.triton_utils.importing import HAS_TRITON

logger = init_logger(__name__)

_installed: bool = False
# Set by configure() once the parallel layout is known; None until then (no syncing).
_layout_ok: bool | None = None


def is_installed() -> bool:
    """Return whether the TP-synced autotune patch is currently installed."""
    return _installed


def _sync_group() -> Any | None:
    """The TP group to sync on, or None when syncing would be unsafe or useless."""
    import torch.distributed as dist

    from vllm.distributed.parallel_state import (
        get_tp_group,
        model_parallel_is_initialized,
    )
    from vllm.forward_context import is_forward_context_available

    if not _layout_ok:
        return None
    if not (dist.is_initialized() and model_parallel_is_initialized()):
        return None
    if not is_forward_context_available():
        return None
    group = get_tp_group()
    return group if group.world_size > 1 else None


def _resolve_locally(autotuner, original, tuning_key, configs, bench_fn) -> bool:
    """Triton's own path: disk cache if this autotuner uses one, else benchmark."""
    if getattr(autotuner, "_vllm_disk_cache", True):
        return original(autotuner, tuning_key, configs, bench_fn)
    bench_fn()
    return False


def _check_disk_cache(original):
    """Wrap Autotuner.check_disk_cache, where every first-time key is resolved."""

    def check_disk_cache(self, tuning_key, configs, bench_fn):
        group = _sync_group()
        if group is None:
            return _resolve_locally(self, original, tuning_key, configs, bench_fn)
        import torch.distributed as dist

        fingerprint = [str(c) for c in configs]
        is_src = dist.get_rank() == group.ranks[0]
        box: list[Any] = [None]
        used_cache = False
        if is_src:
            used_cache = _resolve_locally(self, original, tuning_key, configs, bench_fn)
            best = self.cache[tuning_key]
            box[0] = (
                fingerprint,
                fingerprint.index(str(best)),
                [self.configs_timings.get(c) for c in configs],
            )
        dist.broadcast_object_list(box, src=group.ranks[0], group=group.cpu_group)
        if is_src:
            return used_cache
        src_fingerprint, best_index, timings = box[0]
        if src_fingerprint != fingerprint:
            # Pruning gave this rank other candidates; nothing to adopt.
            logger.warning_once(
                "TP-synced autotune: rank 0 tuned other configs for %s; tuning here.",
                getattr(self, "fn", self),
            )
            return _resolve_locally(self, original, tuning_key, configs, bench_fn)
        self.cache[tuning_key] = configs[best_index]
        self.configs_timings = dict(zip(configs, timings))
        return True

    return check_disk_cache


def _init(original):
    """Route every first-time key through check_disk_cache, remembering whether the
    autotuner itself wanted the on-disk cache."""
    from triton import knobs

    def __init__(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self._vllm_disk_cache = self.cache_results
        self.cache_results = not knobs.runtime.interpret

    return __init__


def configure(parallel_config: Any) -> None:
    """Enable syncing unless the layout gives TP ranks rank-dependent kernel shapes."""
    global _layout_ok
    reasons = [
        name
        for name, active in (
            ("expert parallelism", parallel_config.enable_expert_parallel),
            ("sequence-parallel MoE", parallel_config.use_sequence_parallel_moe),
            (
                "context parallelism",
                parallel_config.decode_context_parallel_size > 1
                or parallel_config.prefill_context_parallel_size > 1,
            ),
        )
        if active
    ]
    _layout_ok = not reasons
    if reasons and _installed:
        logger.warning(
            "VLLM_TRITON_SYNC_AUTOTUNE_TP is ignored with %s: ranks may autotune "
            "different shapes there, and syncing could hang.",
            ", ".join(reasons),
        )


def install() -> None:
    """Make Triton autotuners resolve new keys on TP rank 0 and share the result.

    Autotuners created before this call are synced only if they use the on-disk
    autotune cache (on by default in vLLM, see env_override.py)."""
    global _installed
    if _installed:
        return
    if not HAS_TRITON:
        return

    import importlib

    Autotuner = importlib.import_module("triton.runtime.autotuner").Autotuner
    Autotuner.__init__ = _init(Autotuner.__init__)
    Autotuner.check_disk_cache = _check_disk_cache(Autotuner.check_disk_cache)
    _installed = True
    logger.info("Triton autotune choices are synced across tensor-parallel ranks.")
