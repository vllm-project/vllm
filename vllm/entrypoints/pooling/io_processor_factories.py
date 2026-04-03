# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from vllm.config import ModelConfig
from vllm.entrypoints.chat_utils import ChatTemplateConfig
from vllm.entrypoints.pooling.base.io_processor import PoolingIOProcessor
from vllm.entrypoints.pooling.scoring.io_processor import ScoringIOProcessors
from vllm.entrypoints.pooling.utils import enable_scoring_api
from vllm.renderers import BaseRenderer
from vllm.tasks import SupportedTask


def init_pooling_io_processors(
    supported_tasks: tuple[SupportedTask, ...],
    model_config: ModelConfig,
    renderer: BaseRenderer,
    chat_template_config: ChatTemplateConfig,
) -> dict[str, PoolingIOProcessor]:
    processors: dict[str, type[PoolingIOProcessor]] = {}

    if "classify" in supported_tasks:
        from vllm.entrypoints.pooling.classify.io_processor import ClassifyIOProcessor

        processors["classify"] = ClassifyIOProcessor

    if "token_classify" in supported_tasks:
        from vllm.entrypoints.pooling.classify.io_processor import (
            TokenClassifyIOProcessor,
        )

        processors["token_classify"] = TokenClassifyIOProcessor

    if "embed" in supported_tasks:
        from vllm.entrypoints.pooling.embed.io_processor import EmbedIOProcessor

        processors["embed"] = EmbedIOProcessor

    if "token_embed" in supported_tasks:
        from vllm.entrypoints.pooling.embed.io_processor import TokenEmbedIOProcessor

        processors["token_embed"] = TokenEmbedIOProcessor

    if enable_scoring_api(supported_tasks, model_config):
        score_type = model_config.score_type

        if score_type is not None and score_type in ScoringIOProcessors:
            processors[score_type] = ScoringIOProcessors[score_type]

    if model_config.architecture == "JinaForRanking":
        from vllm.entrypoints.pooling.embed.io_processor import (
            JinaRankingTokenEmbedIOProcessor,
        )

        processors["late-interaction"] = ScoringIOProcessors["jina-reranking-scoring"]
        processors["token_embed"] = JinaRankingTokenEmbedIOProcessor

    return {
        task: processor_cls(
            model_config=model_config,
            renderer=renderer,
            chat_template_config=chat_template_config,
        )
        for task, processor_cls in processors.items()
    }
