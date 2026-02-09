# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Poolers that produce an output aggregating all tokens in the sequence."""

from .heads import (
    ClassifierPoolerHead,
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
    SequencePooler,
    SequencePoolerOutput,
    SequencePoolingFn,
    SequencePoolingHeadFn,
    pooler_for_classify,
    pooler_for_embed,
)

__all__ = [
    "SequencePoolerHead",
    "SequencePoolerHeadOutput",
    "ClassifierPoolerHead",
    "EmbeddingPoolerHead",
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
    "pooler_for_classify",
    "pooler_for_embed",
]
