# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from typing import TypeAlias

import torch
import torch.nn as nn

from vllm.model_executor.layers.pool.methods import (
    TokenPoolingMethodOutput,
    TokenwisePoolingMethodOutputItem,
)
from vllm.pooling_params import PoolingParams
from vllm.v1.pool.metadata import PoolingMetadata

TokenPoolerHeadOutput: TypeAlias = torch.Tensor | list[torch.Tensor]
"""Applicable to pooling strategies that output one token."""

TokenwisePoolerHeadOutput: TypeAlias = torch.Tensor | None
"""Applicable to pooling strategies that output multiple tokens."""


class TokenPoolerHead(nn.Module, ABC):
    @abstractmethod
    def forward(
        self,
        pooled_data: TokenPoolingMethodOutput,
        pooling_metadata: PoolingMetadata,
    ) -> TokenPoolerHeadOutput:
        raise NotImplementedError


class TokenwisePoolerHead(nn.Module, ABC):
    @abstractmethod
    def forward(
        self,
        pooled_data: TokenwisePoolingMethodOutputItem,
        pooling_param: PoolingParams,
    ) -> TokenwisePoolerHeadOutput:
        raise NotImplementedError
