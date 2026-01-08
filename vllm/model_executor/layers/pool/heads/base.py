# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from typing import TypeAlias

import torch
import torch.nn as nn

from vllm.model_executor.layers.pool.methods import PoolingMethodOutput
from vllm.v1.pool.metadata import PoolingMetadata

TokenPoolerHeadOutput: TypeAlias = torch.Tensor | list[torch.Tensor]
"""Applicable to pooling strategies that output one token."""

TokenwisePoolerHeadOutput: TypeAlias = torch.Tensor | None
"""Applicable to pooling strategies that output multiple tokens."""

PoolerHeadOutput: TypeAlias = TokenPoolerHeadOutput | TokenwisePoolerHeadOutput


class PoolerHead(nn.Module, ABC):
    @abstractmethod
    def forward(
        self,
        pooled_data: PoolingMethodOutput,
        pooling_metadata: PoolingMetadata,
    ) -> PoolerHeadOutput:
        raise NotImplementedError
