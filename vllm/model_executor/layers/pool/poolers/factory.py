# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.config.pooler import PoolerConfig
from vllm.model_executor.layers.pool.activations import PoolerActivation
from vllm.model_executor.layers.pool.common import ClassifierFn
from vllm.model_executor.layers.pool.heads import (
    EmbeddingPoolerHead,
    TokenClassifierPoolerHead,
    TokenEmbeddingPoolerHead,
)
from vllm.model_executor.layers.pool.methods import get_pooling_method

from .common import SimplePooler
from .token import ClassifierPooler
from .tokenwise import AllPooler, StepPooler


def pooler_for_embed(pooler_config: PoolerConfig):
    pooling = get_pooling_method(pooler_config.get_pooling_type())
    head = EmbeddingPoolerHead()

    return SimplePooler(pooling=pooling, head=head)


def pooler_for_classify(
    pooler_config: PoolerConfig,
    classifier: ClassifierFn | None,
    act_fn: PoolerActivation | str | None = None,
):
    pooling = get_pooling_method(pooler_config.get_pooling_type())

    return ClassifierPooler(pooling=pooling, classifier=classifier, act_fn=act_fn)


def pooler_for_token_embed(pooler_config: PoolerConfig):
    pooling_type = pooler_config.get_pooling_type()
    head = TokenEmbeddingPoolerHead()

    if pooling_type == "ALL":
        return AllPooler(head=head)
    if pooling_type == "STEP":
        return StepPooler(head=head)

    raise NotImplementedError(f"Unsupported method: {pooling_type!r}")


def pooler_for_token_classify(
    pooler_config: PoolerConfig,
    classifier: ClassifierFn | None = None,
    act_fn: PoolerActivation | str | None = None,
):
    pooling_type = pooler_config.get_pooling_type()
    head = TokenClassifierPoolerHead(classifier=classifier, act_fn=act_fn)

    if pooling_type == "ALL":
        return AllPooler(head=head)
    if pooling_type == "STEP":
        return StepPooler(head=head)

    raise NotImplementedError(f"Unsupported method: {pooling_type!r}")
