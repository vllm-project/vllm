# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sharding metadata for model parameters.

RL frameworks (OpenRLHF, VeRL, TRL, etc.) need to understand how vLLM
shards model weights across GPUs.  This module exposes that information
as a lightweight ``Sharding`` dataclass attached to each weight
parameter at construction time.
"""

import dataclasses
import enum

from vllm.model_executor.utils import set_weight_attrs


class ShardingType(enum.Enum):
    """How a weight tensor is distributed across the tensor-parallel group."""

    REPLICATED = "replicated"
    COLUMN_WISE = "column_wise"
    ROW_WISE = "row_wise"
    VOCAB_PARALLEL = "vocab_parallel"
    EXPERT_PARALLEL = "expert_parallel"
    QKV_PARALLEL = "qkv_parallel"


@dataclasses.dataclass(frozen=True)
class Sharding:
    """Describes the global (unsharded) shape of a parameter and how it is
    partitioned.

    Attributes:
        shape: Global (unsharded) shape of the weight.
        nd_num_shards: Number of shards along each dimension.  Must have the
            same length as *shape*.
        sharding_type: High-level category of the sharding strategy.
    """

    shape: tuple[int, ...]
    nd_num_shards: tuple[int, ...]
    sharding_type: ShardingType

    def __post_init__(self):
        if len(self.shape) != len(self.nd_num_shards):
            raise ValueError(
                f"shape ({len(self.shape)}D) and nd_num_shards "
                f"({len(self.nd_num_shards)}D) must have the same length"
            )
        for i, s in enumerate(self.nd_num_shards):
            if s < 1:
                raise ValueError(f"nd_num_shards[{i}] must be >= 1, got {s}")


def _attach_sharding(param, sharding: Sharding) -> None:
    """Attach a ``Sharding`` descriptor to *param* via ``set_weight_attrs``."""
    set_weight_attrs(param, {"sharding": sharding})
