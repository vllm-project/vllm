# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pooling strategies that output one token."""

from .heads import (
    EmbeddingPoolerHead,
    TokenPoolerHead,
    TokenPoolerHeadOutput,
)
from .methods import (
    CLSPool,
    LastPool,
    MeanPool,
    TokenPoolingMethod,
    TokenPoolingMethodOutput,
    get_token_pooling_method,
)
from .poolers import (
    ClassifierPooler,
    SimplePooler,
    TokenPooler,
    TokenPoolerOutput,
    pooler_for_classify,
    pooler_for_embed,
)

__all__ = [
    "TokenPoolerHead",
    "TokenPoolerHeadOutput",
    "EmbeddingPoolerHead",
    "TokenPoolerHead",
    "TokenPoolingMethod",
    "TokenPoolingMethodOutput",
    "CLSPool",
    "LastPool",
    "MeanPool",
    "get_token_pooling_method",
    "TokenPooler",
    "TokenPoolerOutput",
    "ClassifierPooler",
    "SimplePooler",
    "pooler_for_classify",
    "pooler_for_embed",
]
