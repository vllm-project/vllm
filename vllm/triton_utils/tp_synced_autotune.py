# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Make Triton autotune choose the same config on every tensor-parallel rank.

Autotune times each candidate config on the local GPU, so ranks can choose different
configs for the same kernel and shape, and results can differ in the last bits. Plain
TP hides this, because the next all-reduce gives every rank the same sum. Code where
ranks compute the same thing on their own with no all-reduce after it (replicated
layers, indexers, per-rank recurrent state) then lets the ranks drift apart.

Under VLLM_TRITON_SYNC_AUTOTUNE_TP=1, inside a model forward pass each autotuner uses an
in-memory cache of its own that holds only keys resolved together with the other TP
ranks. The first time a kernel meets a shape key there, TP rank 0 resolves it as usual
(Triton's on-disk autotune cache, or a benchmark) and broadcasts the chosen config and
its timings; the other ranks adopt them without benchmarking or reading their own disk
cache. So neither a disk cache warm on some ranks only nor a key tuned earlier outside a
forward pass (kept in Triton's regular in-memory cache) can split the ranks. Outside a
forward pass, where ranks need not launch the same kernels, autotune is Triton's own and
cannot hang. One CPU broadcast per kernel and key.

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


_deps: tuple | None = None


def _sync_group() -> Any | None:
    """The TP group to sync on, or None when syncing would be unsafe or useless.

    Runs on every launch of an autotuned kernel, so the imports are resolved once."""
    global _deps
    if not _layout_ok:
        return None
    if _deps is None:
        import torch.distributed as dist

        from vllm.distributed.parallel_state import (
            get_tp_group,
            model_parallel_is_initialized,
        )
        from vllm.forward_context import is_forward_context_available

        _deps = (
            dist,
            model_parallel_is_initialized,
            is_forward_context_available,
            get_tp_group,
        )
    dist, mp_ready, in_forward, get_tp_group = _deps
    if not (in_forward() and dist.is_initialized() and mp_ready()):
        return None
    group = get_tp_group()
    return group if group.world_size > 1 else None


def _resolve_locally(autotuner, original, tuning_key, configs, bench_fn) -> bool:
    """Triton's own path: disk cache if this autotuner uses one, else benchmark."""
    if autotuner._vllm_disk_cache:
        return original(autotuner, tuning_key, configs, bench_fn)
    bench_fn()
    return False


def _check_disk_cache(original):
    """Wrap Autotuner.check_disk_cache, where every first-time key is resolved."""

    def check_disk_cache(self, tuning_key, configs, bench_fn):
        group = _sync_group()
        if group is None:
            return original(self, tuning_key, configs, bench_fn)
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


def _run(original):
    """Wrap Autotuner.run: inside a TP forward pass, look keys up in the synced cache
    and route every miss through check_disk_cache, where rank 0 resolves it for all."""
    from triton import knobs

    def run(self, *args, **kwargs):
        if _sync_group() is None:
            return original(self, *args, **kwargs)
        local_cache, wants_disk_cache = self.cache, self.cache_results
        synced_cache = self.__dict__.setdefault("_vllm_synced_cache", {})
        self._vllm_disk_cache = wants_disk_cache
        self.cache = synced_cache
        self.cache_results = not knobs.runtime.interpret
        try:
            return original(self, *args, **kwargs)
        finally:
            self.cache, self.cache_results = local_cache, wants_disk_cache

    return run


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
    """Make Triton autotuners resolve new keys on TP rank 0 and share the result."""
    global _installed
    if _installed:
        return
    if not HAS_TRITON:
        return

    import importlib

    Autotuner = importlib.import_module("triton.runtime.autotuner").Autotuner
    Autotuner.run = _run(Autotuner.run)
    Autotuner.check_disk_cache = _check_disk_cache(Autotuner.check_disk_cache)
    _installed = True
    logger.info("Triton autotune choices are synced across tensor-parallel ranks.")
