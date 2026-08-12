# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm.entrypoints.pooling import factories
from vllm.entrypoints.pooling.factories import init_pooling_io_processors
from vllm.entrypoints.pooling.pooling.io_processor import (
    PluginWithIOProcessorPlugins,
)


def _bge_m3_config(io_processor_plugin=None):
    model_config = MagicMock()
    model_config.get_pooling_task.return_value = "embed&token_classify"
    model_config.io_processor_plugin = io_processor_plugin
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
    return vllm_config, renderer, chat_template_config


def test_combined_task_without_plugin_has_no_processor():
    vllm_config, renderer, chat_template_config = _bge_m3_config()

    processors = init_pooling_io_processors(
        supported_tasks=("embed", "embed&token_classify"),
        vllm_config=vllm_config,
        renderer=renderer,
        chat_template_config=chat_template_config,
    )

    assert processors == {}


def test_combined_task_with_plugin_uses_plugin_processor(monkeypatch):
    vllm_config, renderer, chat_template_config = _bge_m3_config("bge_m3_sparse_plugin")
    monkeypatch.setattr(factories, "has_io_processor", lambda *_: True)
    monkeypatch.setattr(
        "vllm.entrypoints.pooling.pooling.io_processor.get_io_processor",
        lambda *_: MagicMock(),
    )

    processors = init_pooling_io_processors(
        supported_tasks=("embed", "embed&token_classify"),
        vllm_config=vllm_config,
        renderer=renderer,
        chat_template_config=chat_template_config,
    )

    assert processors.keys() == {"plugin"}
    assert isinstance(processors["plugin"], PluginWithIOProcessorPlugins)


def test_combined_task_plain_pooling_request_has_actionable_error():
    from vllm.entrypoints.pooling.pooling.serving import ServingPooling

    serving = object.__new__(ServingPooling)
    serving.pooling_task = "embed&token_classify"
    serving.supported_tasks = ("embed", "embed&token_classify")
    serving.io_processors = {"plugin": MagicMock()}
    request = SimpleNamespace(task=None, dimensions=None)

    with pytest.raises(ValueError, match="only available through an IO processor"):
        serving.get_io_processor(request)
