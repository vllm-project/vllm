# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.config import VllmConfig
from vllm.entrypoints.chat_utils import ChatTemplateConfig
from vllm.entrypoints.pooling.base.io_processor import PoolingIOProcessor
from vllm.entrypoints.pooling.scoring.io_processor import ScoringIOProcessors
from vllm.entrypoints.pooling.utils import enable_scoring_api
from vllm.plugins.io_processors import has_io_processor
from vllm.renderers import BaseRenderer
from vllm.tasks import SupportedTask


def init_pooling_io_processors(
    supported_tasks: tuple[SupportedTask, ...],
    vllm_config: VllmConfig,
    renderer: BaseRenderer,
    chat_template_config: ChatTemplateConfig,
) -> dict[str, PoolingIOProcessor]:
    model_config = vllm_config.model_config
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

    if "token_embed" in supported_tasks:
        from vllm.entrypoints.pooling.embed.io_processor import TokenEmbedIOProcessor

        processors["token_embed"] = TokenEmbedIOProcessor

    if has_io_processor(
        vllm_config,
        model_config.io_processor_plugin,
    ):
        from vllm.entrypoints.pooling.pooling.io_processor import PluginIOProcessor

        processors["plugin"] = PluginIOProcessor

    if enable_scoring_api(supported_tasks, model_config):
        score_type = model_config.score_type

        if score_type is not None and score_type in ScoringIOProcessors:
            processors[score_type] = ScoringIOProcessors[score_type]

    return {
        task: processor_cls(
            vllm_config=vllm_config,
            renderer=renderer,
            chat_template_config=chat_template_config,
        )
        for task, processor_cls in processors.items()
    }
