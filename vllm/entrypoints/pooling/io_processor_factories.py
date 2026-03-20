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
    processors: list[tuple[str, type[PoolingIOProcessor]]] = []
    if "classify" in supported_tasks:
        from vllm.entrypoints.pooling.classify.io_processor import ClassifyIOProcessor

        processors.append(("classify", ClassifyIOProcessor))
    if "embed" in supported_tasks:
        from vllm.entrypoints.pooling.embed.io_processor import EmbedIOProcessor

        processors.append(("embed", EmbedIOProcessor))

    return {
        task: processor_cls(
            model_config=model_config,
            renderer=renderer,
            chat_template_config=chat_template_config,
        )
        for task, processor_cls in processors
    }
