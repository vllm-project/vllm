# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from collections.abc import Set
from typing import TypeAlias

import torch
import torch.nn as nn

from vllm.model_executor.layers.pool.common import PoolingParamsUpdate
from vllm.tasks import PoolingTask
from vllm.v1.pool.metadata import PoolingMetadata

TokenPoolingMethodOutput: TypeAlias = torch.Tensor | list[torch.Tensor]
"""Applicable to pooling strategies that output one token."""

TokenwisePoolingMethodOutput: TypeAlias = list[torch.Tensor] | list[torch.Tensor | None]
"""Applicable to pooling strategies that output multiple tokens."""

TokenwisePoolingMethodOutputItem: TypeAlias = torch.Tensor | None
"""Represents a single element of `TokenwisePoolingMethodOutput`."""

PoolingMethodOutput: TypeAlias = TokenPoolingMethodOutput | TokenwisePoolingMethodOutput


class PoolingMethod(nn.Module, ABC):
    @abstractmethod
    def get_supported_tasks(self) -> Set[PoolingTask]:
        raise NotImplementedError

    def get_pooling_updates(self, task: PoolingTask) -> PoolingParamsUpdate:
        return PoolingParamsUpdate()

    @abstractmethod
    def forward(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> PoolingMethodOutput:
        raise NotImplementedError
