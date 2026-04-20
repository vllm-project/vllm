# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from collections.abc import Set

import torch
import torch.nn as nn

from vllm.tasks import PoolingTask
from vllm.v1.outputs import PoolerOutput
from vllm.v1.pool.metadata import PoolingMetadata

from .common import PoolingParamsUpdate


class Pooler(nn.Module, ABC):
    """The interface required for all poolers used in pooling models in vLLM."""

    @abstractmethod
    def get_supported_tasks(self) -> Set[PoolingTask]:
        """Determine which pooling tasks are supported."""
        raise NotImplementedError

    def get_pooling_updates(self, task: PoolingTask) -> PoolingParamsUpdate:
        """
        Construct the updated pooling parameters to use for a supported task.
        """
        return PoolingParamsUpdate()

    @abstractmethod
    def forward(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> PoolerOutput:
        raise NotImplementedError


__all__ = ["Pooler"]
