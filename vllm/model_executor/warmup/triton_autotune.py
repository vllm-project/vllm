# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared orchestration for config-table Triton kernel autotuning."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import torch

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)

TuningConfig = dict[str, Any]


@dataclass(frozen=True)
class TuningItem:
    """One config-table bucket to benchmark.

    ``shape`` identifies the table file while ``bucket`` identifies one entry
    inside it. Shapes must have a stable repr so all ranks sort them equally.
    """

    table: str
    shape: Hashable
    bucket: int


class TunableConfigTable(ABC):
    """Adapter between a config-table kernel family and warmup autotuning."""

    name: ClassVar[str]

    @abstractmethod
    def pending_items(self, worker: Worker) -> list[TuningItem]:
        """Return model-specific table entries which still need tuning."""

    @abstractmethod
    def tune(self, item: TuningItem) -> TuningConfig | None:
        """Benchmark one item, returning its best valid config if one exists."""

    @abstractmethod
    def commit(self, results: Mapping[TuningItem, TuningConfig]) -> None:
        """Persist results and invalidate the table's lookup caches."""


def _item_sort_key(item: TuningItem) -> tuple[str, str, int]:
    return item.table, repr(item.shape), item.bucket


def _merge_pending_items(
    items_by_rank: Iterable[Iterable[TuningItem]],
) -> tuple[TuningItem, ...]:
    """Deduplicate and deterministically order work discovered by all ranks."""
    return tuple(sorted(set().union(*map(set, items_by_rank)), key=_item_sort_key))


def _shard_items(
    items: tuple[TuningItem, ...], rank: int, world_size: int
) -> tuple[TuningItem, ...]:
    """Assign each tuning item to exactly one rank."""
    return items[rank::world_size]


def _merge_results(
    results_by_rank: Iterable[Mapping[TuningItem, TuningConfig]],
) -> dict[TuningItem, TuningConfig]:
    """Combine disjoint per-rank results, rejecting conflicting duplicates."""
    merged: dict[TuningItem, TuningConfig] = {}
    for results in results_by_rank:
        for item, config in results.items():
            if item in merged and merged[item] != config:
                raise ValueError(f"Conflicting autotune results for {item!r}")
            merged[item] = config
    return merged


class TritonAutotuneRegistry:
    """Registry and distributed runner for config-table kernel families."""

    def __init__(self) -> None:
        self._tables: dict[str, TunableConfigTable] = {}

    def register(self, table: TunableConfigTable) -> None:
        name = getattr(table, "name", "")
        if not name:
            raise ValueError("Tunable config tables must define a non-empty name")
        if name in self._tables:
            raise ValueError(f"Triton autotune table {name!r} is already registered")
        self._tables[name] = table

    @property
    def tables(self) -> tuple[TunableConfigTable, ...]:
        return tuple(self._tables[name] for name in sorted(self._tables))

    def _discover(self, worker: Worker) -> list[TuningItem]:
        pending: list[TuningItem] = []
        for table in self.tables:
            try:
                items = list(table.pending_items(worker))
                if any(item.table != table.name for item in items):
                    raise ValueError(
                        f"Table {table.name!r} returned an item for another table"
                    )
                pending.extend(items)
            except Exception:
                # Discovery must not make one rank skip the collectives below.
                logger.exception(
                    "Failed to discover pending Triton configs for table %s.",
                    table.name,
                )
        return pending

    def run(self, worker: Worker) -> None:
        """Discover, shard, tune, and commit missing configs on every rank."""
        if not self._tables:
            return

        from vllm.distributed.parallel_state import get_world_group

        world = get_world_group()
        pending_by_rank: list[list[TuningItem] | None] = [None] * world.world_size
        torch.distributed.all_gather_object(
            pending_by_rank,
            self._discover(worker),
            group=world.cpu_group,
        )
        pending = _merge_pending_items(items or [] for items in pending_by_rank)

        local_results: dict[TuningItem, TuningConfig] = {}
        for item in _shard_items(pending, world.rank_in_group, world.world_size):
            try:
                result = self._tables[item.table].tune(item)
                if result is not None:
                    local_results[item] = result
            except Exception:
                # Every rank must still reach the result collective.
                logger.exception("Triton autotuning failed for %r.", item)

        results_by_rank: list[dict[TuningItem, TuningConfig] | None] = [
            None
        ] * world.world_size
        torch.distributed.all_gather_object(
            results_by_rank,
            local_results,
            group=world.cpu_group,
        )
        results = _merge_results(result or {} for result in results_by_rank)

        for table in self.tables:
            table.commit(
                {
                    item: config
                    for item, config in results.items()
                    if item.table == table.name
                }
            )


TRITON_AUTOTUNE_REGISTRY = TritonAutotuneRegistry()


def register_triton_autotune_table(table: TunableConfigTable) -> None:
    """Register a config-table family for warmup autotuning."""
    TRITON_AUTOTUNE_REGISTRY.register(table)


def triton_autotune(worker: Worker) -> None:
    """Run all registered config-table autotuners."""
    TRITON_AUTOTUNE_REGISTRY.run(worker)
