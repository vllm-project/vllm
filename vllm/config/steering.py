# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pydantic import ConfigDict, Field

from vllm.config.utils import config
from vllm.model_executor.layers.steering import (
    DEFAULT_HOOK_POINT,
    SteeringHookPoint,
)
from vllm.utils.hashing import safe_hash


@config(config=ConfigDict(arbitrary_types_allowed=True))
class SteeringConfig:
    """Configuration for per-request activation steering."""

    max_steering_configs: int = Field(default=4, ge=1)
    """Max number of distinct per-request steering configs in a single batch."""

    steering_hook_points: frozenset[SteeringHookPoint] = Field(
        default=frozenset({DEFAULT_HOOK_POINT}),
    )
    """Which intervention points are active. Each active hook point adds a
    torch.compile splitting op per decoder layer. Default: post_mlp_pre_ln."""

    def compute_hash(self) -> str:
        """
        WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.

        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        factors: list = []
        factors.append(self.max_steering_configs)
        factors.append(sorted(hp.value for hp in self.steering_hook_points))

        hash_str = safe_hash(
            str(factors).encode(), usedforsecurity=False
        ).hexdigest()
        return hash_str
