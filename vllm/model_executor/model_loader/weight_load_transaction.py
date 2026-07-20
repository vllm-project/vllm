# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Transaction-scoped completion for checkpoint-format weight loading."""

from collections.abc import Callable, Hashable, Iterable
from typing import Any, TypeVar

from torch import nn

_LoadedItemT = TypeVar("_LoadedItemT", bound=Hashable)


class WeightLoadTransaction:
    """Aggregate model-specific load results across transport chunks."""

    def __init__(self) -> None:
        self._completions: dict[
            nn.Module, tuple[set[Any], Callable[[set[Any]], None]]
        ] = {}

    def record(
        self,
        module: nn.Module,
        loaded_items: Iterable[_LoadedItemT],
        complete: Callable[[set[_LoadedItemT]], None],
    ) -> None:
        entry = self._completions.get(module)
        if entry is None:
            self._completions[module] = (set(loaded_items), complete)
            return
        entry[0].update(loaded_items)
        self._completions[module] = (entry[0], complete)

    def finish(self) -> None:
        for loaded_items, complete in self._completions.values():
            complete(loaded_items)


def start_weight_load_transaction(model: nn.Module) -> None:
    if get_weight_load_transaction(model) is not None:
        raise RuntimeError("a weight load transaction is already active")
    model._vllm_weight_load_transaction = WeightLoadTransaction()


def finish_weight_load_transaction(model: nn.Module) -> None:
    transaction = get_weight_load_transaction(model)
    if transaction is None:
        raise RuntimeError("no weight load transaction is active")
    try:
        transaction.finish()
    finally:
        abort_weight_load_transaction(model)


def abort_weight_load_transaction(model: nn.Module) -> None:
    model.__dict__.pop("_vllm_weight_load_transaction", None)


def get_weight_load_transaction(model: nn.Module) -> WeightLoadTransaction | None:
    return getattr(model, "_vllm_weight_load_transaction", None)


def complete_weight_load(
    model: nn.Module,
    loaded_items: Iterable[_LoadedItemT],
    complete: Callable[[set[_LoadedItemT]], None],
) -> None:
    """Run completion now, or defer it to the active transaction boundary."""
    transaction = get_weight_load_transaction(model)
    if transaction is None:
        complete(set(loaded_items))
    else:
        transaction.record(model, loaded_items, complete)
