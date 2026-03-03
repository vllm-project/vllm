# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from vllm.config import ModelConfig
from vllm.entrypoints.chat_utils import ChatTemplateConfig
from vllm.entrypoints.pooling.base.io_processor import PoolingIOProcessor
from vllm.renderers import BaseRenderer
from vllm.tasks import SupportedTask


def init_pooling_io_processors(
    supported_tasks: tuple[SupportedTask, ...],
    model_config: ModelConfig,
    renderer: BaseRenderer,
    chat_template_config: ChatTemplateConfig,
) -> dict[str, PoolingIOProcessor]:
    pooling_io_processors: dict[str, PoolingIOProcessor] = {}

    if "classify" in supported_tasks:
        from vllm.entrypoints.pooling.classify.io_processor import (
            ClassifyIOProcessor,
        )

        pooling_io_processors["classify"] = ClassifyIOProcessor(
            model_config=model_config,
            renderer=renderer,
            chat_template_config=chat_template_config,
        )

    return pooling_io_processors
