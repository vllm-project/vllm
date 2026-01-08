# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pooling strategies that output multiple tokens."""

from .heads import (
    TokenClassifierPoolerHead,
    TokenEmbeddingPoolerHead,
    TokenwisePoolerHead,
    TokenwisePoolerHeadOutput,
)
from .methods import (
    AllPool,
    TokenwisePoolingMethod,
    TokenwisePoolingMethodOutput,
    TokenwisePoolingMethodOutputItem,
)
from .poolers import (
    AllPooler,
    StepPooler,
    TokenwisePooler,
    TokenwisePoolerOutput,
    pooler_for_token_classify,
    pooler_for_token_embed,
)

__all__ = [
    "TokenwisePoolerHead",
    "TokenwisePoolerHeadOutput",
    "TokenClassifierPoolerHead",
    "TokenEmbeddingPoolerHead",
    "TokenwisePoolingMethod",
    "TokenwisePoolingMethodOutput",
    "TokenwisePoolingMethodOutputItem",
    "AllPool",
    "TokenwisePooler",
    "TokenwisePoolerOutput",
    "AllPooler",
    "StepPooler",
    "pooler_for_token_classify",
    "pooler_for_token_embed",
]
