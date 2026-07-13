# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-layer input and output parallel plans."""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum, auto


class LayerType(Enum):
    """Semantic layer types understood by parallel policies."""

    DEFAULT = auto()
    ATTENTION = auto()


class ParallelGroupType(Enum):
    """Runtime process groups that can back a parallel axis."""

    TENSOR = auto()
    ATTENTION_TENSOR = auto()


@dataclass(frozen=True)
class ParallelAxis:
    """A rank's position along one parallel axis."""

    world_size: int
    rank: int
    group: ParallelGroupType

    def __post_init__(self) -> None:
        if self.world_size < 1:
            raise ValueError(f"world_size must be positive, got {self.world_size}")
        if not 0 <= self.rank < self.world_size:
            raise ValueError(f"rank must be in [0, {self.world_size}), got {self.rank}")


@dataclass(frozen=True)
class LayerParallelPlan:
    """Input and output layouts for a layer."""

    input: ParallelAxis
    output: ParallelAxis

    @property
    def reshards_output(self) -> bool:
        return self.input != self.output

    def get_output_size(self, input_size: int) -> int:
        global_size = input_size * self.input.world_size
        if global_size % self.output.world_size != 0:
            raise ValueError(
                "Global input size must be divisible by the output parallel size: "
                f"{global_size=}, output_size={self.output.world_size}"
            )
        return global_size // self.output.world_size


_SINGLE_RANK_AXIS = ParallelAxis(1, 0, ParallelGroupType.TENSOR)
_SINGLE_RANK_PLAN = LayerParallelPlan(
    input=_SINGLE_RANK_AXIS,
    output=_SINGLE_RANK_AXIS,
)
_default_plan = _SINGLE_RANK_PLAN
_layer_plans: dict[LayerType, LayerParallelPlan] = {}


def init_layer_parallel_config(
    default_plan: LayerParallelPlan,
    overrides: Mapping[LayerType, LayerParallelPlan] | None = None,
) -> None:
    """Install the process-local parallel policy after group initialization."""
    global _default_plan, _layer_plans
    _default_plan = default_plan
    _layer_plans = dict(overrides or {})


def get_layer_parallel_config(
    layer_type: LayerType = LayerType.DEFAULT,
) -> LayerParallelPlan:
    """Resolve ``layer_type`` to a fully specified parallel plan."""
    return _layer_plans.get(layer_type, _default_plan)


def clear_layer_parallel_config() -> None:
    """Restore the single-rank default policy."""
    init_layer_parallel_config(_SINGLE_RANK_PLAN)
