# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Make Triton autotune choose the same config on every tensor-parallel rank.

Autotune times each candidate config on the local GPU, so ranks can choose different
configs for the same kernel and shape, and results can differ in the last bits. Plain
TP hides this, because the next all-reduce gives every rank the same sum. Code where
ranks compute the same thing on their own with no all-reduce after it (replicated
layers, indexers, per-rank recurrent state) then lets the ranks drift apart.

Under VLLM_TRITON_SYNC_AUTOTUNE_TP=1 every rank uses rank 0's benchmark timings, so
the argmin is the same everywhere. Sync happens only inside a model forward pass,
where all ranks of a TP group launch the same kernels with the same shapes; a kernel
autotuned outside it (or by one rank only) keeps its local timings, so it cannot hang.
One CPU broadcast per benchmarked config, and only when a kernel meets a new shape key.

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


def sync_timing(timing: Any) -> Any:
    """Return TP rank 0's timing for this config (identity when not syncing)."""
    group = _sync_group()
    if group is None:
        return timing
    import torch.distributed as dist

    box = [timing]
    dist.broadcast_object_list(box, src=group.ranks[0], group=group.cpu_group)
    return box[0]


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
    """Wrap Autotuner._bench so its timings come from TP rank 0."""
    global _installed
    if _installed:
        return
    if not HAS_TRITON:
        return

    import importlib

    Autotuner = importlib.import_module("triton.runtime.autotuner").Autotuner
    original_bench = Autotuner._bench

    def _bench(self, *args, **kwargs):
        return sync_timing(original_bench(self, *args, **kwargs))

    Autotuner._bench = _bench
    _installed = True
    logger.info("Triton autotune timings are synced across tensor-parallel ranks.")
