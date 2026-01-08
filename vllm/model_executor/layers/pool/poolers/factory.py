# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass

from vllm.config.pooler import PoolerConfig, PoolingTypeStr
from vllm.model_executor.layers.pool.activations import PoolerActivation
from vllm.model_executor.layers.pool.common import ClassifierFn
from vllm.model_executor.layers.pool.heads import (
    EmbeddingPoolerHead,
    TokenClassifierPoolerHead,
    TokenEmbeddingPoolerHead,
)
from vllm.model_executor.layers.pool.methods import get_pooling_method
from vllm.tasks import PoolingTask

from .common import SimplePooler
from .token import ClassifierPooler
from .tokenwise import AllPooler, StepPooler


@dataclass(frozen=True)
class ResolvedPoolingConfig:
    pooling_type: PoolingTypeStr
    task: PoolingTask

    @classmethod
    def from_config(
        cls,
        task: PoolingTask,
        pooler_config: PoolerConfig,
    ) -> "ResolvedPoolingConfig":
        assert pooler_config.pooling_type is not None
        return cls(task=task, pooling_type=pooler_config.pooling_type)


def pooler_for_embed(pooler_config: PoolerConfig):
    resolved_config = ResolvedPoolingConfig.from_config(
        task="embed",
        pooler_config=pooler_config,
    )

    pooling = get_pooling_method(resolved_config.pooling_type)
    head = EmbeddingPoolerHead()

    return SimplePooler(pooling=pooling, head=head)


def pooler_for_classify(
    pooler_config: PoolerConfig,
    classifier: ClassifierFn | None,
    act_fn: PoolerActivation | str | None = None,
):
    resolved_config = ResolvedPoolingConfig.from_config(
        task="classify",
        pooler_config=pooler_config,
    )

    pooling = get_pooling_method(resolved_config.pooling_type)

    return ClassifierPooler(
        pooling=pooling,
        classifier=classifier,
        act_fn=act_fn,
    )


def pooler_for_token_embed(pooler_config: PoolerConfig):
    head = TokenEmbeddingPoolerHead()

    if pooler_config.pooling_type == "STEP":
        return StepPooler(head=head)

    return AllPooler(head=head)


def pooler_for_token_classify(
    pooler_config: PoolerConfig,
    classifier: ClassifierFn | None = None,
    act_fn: PoolerActivation | str | None = None,
):
    head = TokenClassifierPoolerHead(classifier=classifier, act_fn=act_fn)

    if pooler_config.pooling_type == "STEP":
        return StepPooler(head=head)

    return AllPooler(head=head)
