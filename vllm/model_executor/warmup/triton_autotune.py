# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warmup-time tuning for Triton kernels that read launch configs from JSON
tables.

Every rank reports the items its model needs, the ranks split the union
between them, and every rank saves all results, so all ranks reach every
collective and end up with identical configs.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from abc import ABC, abstractmethod
from collections.abc import Hashable
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import torch

import vllm.envs as envs
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.distributed.parallel_state import GroupCoordinator
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)

Config = dict[str, int]


def get_autotune_cache_dir(table_name: str) -> str:
    from vllm.triton_utils import triton

    root = envs.VLLM_TRITON_AUTOTUNE_CACHE_DIR or os.path.join(
        envs.VLLM_CACHE_ROOT, "triton_autotune"
    )
    version = getattr(triton, "__version__", "unknown")
    return os.path.join(root, f"triton={version}", table_name)


def atomic_write_json(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=os.path.dirname(path),
        prefix=f".{os.path.basename(path)}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=4)
            f.write("\n")
        os.replace(tmp_path, path)
    except BaseException:
        with suppress(OSError):
            os.unlink(tmp_path)
        raise


@dataclass(frozen=True)
class TuningItem:
    table: str
    shape: Hashable
    bucket: int


class TunableConfigTable(ABC):
    """A JSON launch-config table that can fill in missing entries at startup."""

    name: ClassVar[str]

    @abstractmethod
    def pending_items(self, worker: Worker) -> list[TuningItem]:
        """Items this rank's model will look up that have no config yet."""

    @abstractmethod
    def tune(self, item: TuningItem) -> Config | None:
        """Benchmark one item on scratch buffers; None if no candidate works."""

    @abstractmethod
    def commit(self, results: dict[TuningItem, Config]) -> None:
        """Save results and clear the table's lookup caches. Runs on every rank."""


def _all_gather(world: GroupCoordinator, obj: Any) -> list[Any]:
    if world.world_size == 1:
        return [obj]
    gathered: list[Any] = [None] * world.world_size
    torch.distributed.all_gather_object(gathered, obj, group=world.cpu_group)
    return gathered


def run_config_tuning(
    tables: list[TunableConfigTable], worker: Worker, world: GroupCoordinator
) -> dict[TuningItem, Config]:
    by_name = {table.name: table for table in tables}
    local = [item for table in tables for item in table.pending_items(worker)]
    union = {item for items in _all_gather(world, local) for item in items}
    if not union:
        return {}
    todo = sorted(union, key=lambda i: (i.table, repr(i.shape), i.bucket))
    mine = todo[world.rank_in_group :: world.world_size]
    logger.info(
        "Triton autotune: %d items to tune, %d on this rank.", len(todo), len(mine)
    )

    tuned: dict[TuningItem, Config] = {}
    for item in mine:
        # A failure on one rank must not skip the collective below.
        try:
            config = by_name[item.table].tune(item)
        except Exception:
            logger.warning("Triton autotune failed for %s.", item, exc_info=True)
            continue
        if config is not None:
            tuned[item] = config

    results: dict[TuningItem, Config] = {}
    for part in _all_gather(world, tuned):
        results.update(part)
    for table in tables:
        own = {item: cfg for item, cfg in results.items() if item.table == table.name}
        if own:
            table.commit(own)
    return results


def _tables() -> list[TunableConfigTable]:
    from vllm.model_executor.warmup.mamba_ssu_autotune import MambaSSUConfigTable

    return [MambaSSUConfigTable()]


def triton_autotune(worker: Worker) -> None:
    """Tune missing Triton launch configs for the served model before CUDA
    graph capture. Must be called on every rank."""
    if not worker.vllm_config.kernel_config.enable_triton_autotune:
        return
    from vllm.distributed.parallel_state import get_world_group

    start = time.perf_counter()
    results = run_config_tuning(_tables(), worker, get_world_group())
    torch.accelerator.empty_cache()
    if results:
        logger.info(
            "Triton autotune saved %d configs in %.1f s.",
            len(results),
            time.perf_counter() - start,
        )