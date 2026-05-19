# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared-memory budget helpers for ``@triton.autotune`` config selection.

Triton kernels often ship with autotune configurations tuned for the largest
shared-memory budget Triton supports (228 KiB per block on H100/H200). On
GPUs with a smaller per-block opt-in budget — Turing (~64 KiB), Ampere
A100 (~163 KiB), consumer Blackwell SM_120 RTX 5090 / RTX PRO 6000
(~99 KiB) — the larger configs raise
``triton.runtime.errors.OutOfResources`` at JIT time, killing the worker
mid-cold-load or mid-first-request.

The Triton autotuner accepts a ``prune_configs_by={"early_config_prune":
fn}`` callback. The callback receives the candidate configs and the
runtime kwargs and returns a filtered list. This module ships a generic
helper that builds such a callback from a kernel-specific shmem-byte
estimator, plus a simple ``infer_shmem_budget()`` for the destination
GPU's actual per-block opt-in capability.

Usage
-----
::

    from vllm.triton_utils.shmem_budget import (
        infer_shmem_budget,
        make_shmem_pruner,
    )


    def _est_smem(config, named_args):
        # Author-supplied: return the bytes of shmem the kernel uses for
        # this (config, named_args) pair. Should slightly over-estimate
        # so an off-by-one in Triton's internal allocation is covered by
        # the pruner's safety margin.
        BV = config.kwargs["BV"]
        BT = named_args.get("BT", 64)
        ns = config.num_stages
        # Peak co-resident layout for this hypothetical kernel:
        #   persistent: 4 fp32 tiles of shape [BT, BV]
        #   per stage:  one bf16 [BT, BT] + one bf16 [BT, BV]
        persistent = 4 * (BT * BV * 4)
        per_stage = (BT * BT * 2) + (BT * BV * 2)
        overhead = 4096  # Triton bookkeeping safety
        return persistent + ns * per_stage + overhead


    @triton.autotune(
        configs=[
            triton.Config({"BV": BV}, num_warps=nw, num_stages=ns)
            for nw in [2, 4]
            for ns in [2, 3, 4]
            for BV in [32, 64]
        ],
        key=["H", "K", "V", "BT"],
        prune_configs_by={"early_config_prune": make_shmem_pruner(_est_smem)},
    )
    @triton.jit
    def my_kernel(...): ...

Design notes
~~~~~~~~~~~~

* The pruner is a no-op on GPUs whose budget already accepts every config
  the kernel author shipped (e.g. H100/H200 for kernels tuned at H100
  capability) — preserving existing performance for the common case.
* When the pruner empties the config list (all configs too big), it falls
  back to the *smallest* config rather than raising, with a one-shot
  ``logger.warning``. This is a deliberate liveness-over-perf choice:
  shipping a kernel that cannot launch is strictly worse than shipping
  one that launches with a degraded but valid config. Callers that want
  the strict behaviour can pass ``on_empty="raise"``.
* The estimator is the kernel author's responsibility because only they
  know the actual shmem layout (number of stages, register/shmem split,
  software-pipelined buffers, etc.). The module deliberately does not try
  to introspect kernel IR.
* The budget query caches per-device — multi-GPU systems with mixed
  capabilities (rare but real on some research nodes) still get correct
  pruning because Triton instantiates the autotuner per-device at JIT
  time.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol

from vllm.logger import init_logger

if TYPE_CHECKING:
    import triton

logger = init_logger(__name__)

_BUDGET_CACHE: dict[int, int] = {}
_FALLBACK_BUDGET_BYTES = 49152


def infer_shmem_budget(device: int | None = None) -> int:
    try:
        import torch
    except ImportError:
        return _FALLBACK_BUDGET_BYTES

    if not torch.cuda.is_available():
        return _FALLBACK_BUDGET_BYTES

    dev = device if device is not None else torch.cuda.current_device()
    if dev in _BUDGET_CACHE:
        return _BUDGET_CACHE[dev]

    try:
        props = torch.cuda.get_device_properties(dev)
        budget = int(getattr(props, "shared_memory_per_block_optin", 0))
        if budget <= 0:
            budget = int(getattr(props, "shared_memory_per_block", 0))
        if budget <= 0:
            budget = _FALLBACK_BUDGET_BYTES
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "shmem_budget: could not read per-block opt-in for device %d (%s); "
            "falling back to %d bytes",
            dev,
            exc,
            _FALLBACK_BUDGET_BYTES,
        )
        budget = _FALLBACK_BUDGET_BYTES

    _BUDGET_CACHE[dev] = budget
    return budget


class ShmemPruner(Protocol):
    """Callable returned by :func:`make_shmem_pruner`.

    Triton's autotune calls ``early_config_prune`` with positional
    ``(configs, named_args)`` plus arbitrary keyword arguments (e.g.
    ``num_warps``, ``num_stages`` on newer Triton versions). This Protocol
    documents the contract so callers can pass ``named_args=...`` as a
    keyword and mypy will not complain.
    """

    def __call__(
        self,
        configs: list[triton.Config],
        named_args: dict[str, Any],
        **kwargs: Any,
    ) -> list[triton.Config]: ...


def make_shmem_pruner(
    estimate_shmem_bytes: Callable[[triton.Config, dict[str, Any]], int],
    *,
    safety_margin_bytes: int = 1024,
    on_empty: str = "smallest",
) -> ShmemPruner:
    warned_once: dict[int, bool] = {}

    def _prune(
        configs: list[triton.Config],
        named_args: dict[str, Any],
        **kwargs: Any,  # noqa: ARG001
    ) -> list[triton.Config]:
        budget = infer_shmem_budget() - safety_margin_bytes
        if budget <= 0:
            return configs

        kept: list[tuple[int, triton.Config]] = []
        smallest: tuple[int, triton.Config] | None = None
        for c in configs:
            try:
                est = int(estimate_shmem_bytes(c, named_args))
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "shmem_budget: estimator raised %r for config %r; keeping "
                    "config to preserve upstream behaviour",
                    exc,
                    c,
                )
                kept.append((0, c))
                continue
            if est <= budget:
                kept.append((est, c))
            if smallest is None or est < smallest[0]:
                smallest = (est, c)

        if kept:
            if len(kept) < len(configs):
                # Only log when the pruner actually had work to do — avoids
                # spamming the log for every Triton autotune key on H100 +
                # H200 where every config fits and the pruner is a no-op.
                logger.info(
                    "shmem_budget: kept %d/%d configs within %d-byte budget",
                    len(kept),
                    len(configs),
                    budget + safety_margin_bytes,
                )
            return [c for _, c in kept]

        gpu_id = 0
        try:
            import torch

            gpu_id = torch.cuda.current_device()
        except Exception:  # noqa: BLE001
            pass

        if on_empty == "raise":
            raise RuntimeError(
                "shmem_budget: every autotune config exceeds device budget "
                f"({budget + safety_margin_bytes} bytes)"
            )

        assert smallest is not None
        if not warned_once.get(gpu_id, False):
            logger.warning(
                "shmem_budget: all autotune configs exceed device %d budget "
                "(%d bytes); falling back to smallest config (est %d bytes)",
                gpu_id,
                budget,
                smallest[0],
            )
            warned_once[gpu_id] = True
        return [smallest[1]]

    return _prune


__all__ = ["ShmemPruner", "infer_shmem_budget", "make_shmem_pruner"]
