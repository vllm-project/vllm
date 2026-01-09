# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Poolers that produce an output aggregating all tokens in the sequence."""

from .heads import (
    EmbeddingPoolerHead,
    SequencePoolerHead,
    SequencePoolerHeadOutput,
)
from .methods import (
    CLSPool,
    LastPool,
    MeanPool,
    SequencePoolingMethod,
    SequencePoolingMethodOutput,
    get_seq_pooling_method,
)
from .poolers import (
    ClassifierPooler,
    SequencePooler,
    SequencePoolerOutput,
    SequencePoolingFn,
    SequencePoolingHeadFn,
    SimplePooler,
    pooler_for_classify,
    pooler_for_embed,
)

__all__ = [
    "SequencePoolerHead",
    "SequencePoolerHeadOutput",
    "EmbeddingPoolerHead",
    "SequencePoolerHead",
    "SequencePoolingMethod",
    "SequencePoolingMethodOutput",
    "CLSPool",
    "LastPool",
    "MeanPool",
    "get_seq_pooling_method",
    "SequencePooler",
    "SequencePoolingFn",
    "SequencePoolingHeadFn",
    "SequencePoolerOutput",
    "ClassifierPooler",
    "SimplePooler",
    "pooler_for_classify",
    "pooler_for_embed",
]
