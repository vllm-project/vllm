# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock

from vllm.entrypoints.pooling.factories import init_pooling_io_processors
from vllm.entrypoints.pooling.pooling.io_processor import (
    EmbedAndTokenClassifyIOProcessor,
)


def test_init_embed_and_token_classify_io_processor():
    model_config = MagicMock()
    model_config.get_pooling_task.return_value = "embed&token_classify"
    model_config.io_processor_plugin = None
    model_config.hf_config.to_dict.return_value = {}
    model_config.architecture = "BgeM3EmbeddingModel"

    vllm_config = MagicMock(model_config=model_config)
    renderer = MagicMock()
    renderer._executor = MagicMock()
    chat_template_config = MagicMock(
        chat_template=None,
        chat_template_content_format="auto",
        trust_request_chat_template=False,
    )

    processors = init_pooling_io_processors(
        supported_tasks=("embed", "embed&token_classify"),
        vllm_config=vllm_config,
        renderer=renderer,
        chat_template_config=chat_template_config,
    )

    assert processors.keys() == {"embed&token_classify"}
    assert isinstance(
        processors["embed&token_classify"], EmbedAndTokenClassifyIOProcessor
    )
