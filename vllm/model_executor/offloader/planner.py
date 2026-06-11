# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Planning utilities for prefetch weight offloading."""

from collections.abc import Iterable
from dataclasses import dataclass

import torch.nn as nn

from vllm.config.offload import PrefetchOffloadSelector
from vllm.model_executor.offloader.selectors import select_module_parameters


@dataclass(frozen=True)
class OffloadUnit:
    """A semantic unit of weights to offload together."""

    module_index: int
    module: nn.Module
    param_names: tuple[str, ...]


@dataclass(frozen=True)
class PrefetchOffloadPlan:
    """Planner output consumed by the prefetch runtime."""

    modules: list[nn.Module]
    units: list[OffloadUnit]


def should_offload_module(
    module_index: int,
    group_size: int,
    num_in_group: int,
) -> bool:
    """Return whether a module falls into the configured offload window."""
    assert group_size > 0, "group_size must be greater than 0"
    assert 0 <= num_in_group <= group_size, (
        "num_in_group must be between 0 and group_size (inclusive)"
    )
    return module_index % group_size >= group_size - num_in_group


def build_prefetch_offload_plan(
    modules: Iterable[nn.Module],
    group_size: int,
    num_in_group: int,
    selectors: set[PrefetchOffloadSelector] | None = None,
    include_names: set[str] | None = None,
) -> PrefetchOffloadPlan:
    """Build a prefetch offload plan from local modules.

    The planner decides which local modules participate in offloading and
    which parameters belong to each offload unit. The prefetch runtime then
    consumes this plan without re-deriving per-module whitelists.
    """
    all_modules = list(modules)
    units: list[OffloadUnit] = []

    for module_index, module in enumerate(all_modules):
        if not should_offload_module(module_index, group_size, num_in_group):
            continue

        param_names = select_module_parameters(
            module,
            selectors=selectors,
            include_names=include_names,
        )
        if not param_names:
            continue

        units.append(
            OffloadUnit(
                module_index=module_index,
                module=module,
                param_names=tuple(param_names),
            )
        )

    return PrefetchOffloadPlan(modules=all_modules, units=units)
