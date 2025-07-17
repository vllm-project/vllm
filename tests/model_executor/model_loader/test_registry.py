# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from torch import nn

from vllm.config import LoadConfig, ModelConfig
from vllm.model_executor.model_loader import get_model_loader
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.registry import ModelLoaderRegistry


class TestModelLoader(BaseModelLoader):

    def __init__(self, load_config: LoadConfig) -> None:
        super().__init__(load_config)

    def download_model(self, model_config: ModelConfig) -> None:
        pass

    def load_weights(self, model: nn.Module,
                     model_config: ModelConfig) -> None:
        pass


class InValidModelLoader:
    pass


@pytest.mark.parametrize(
    "load_format, loader_cls",
    [
        ("test_load_format",
         "tests.model_executor.model_loader.test_registry:TestModelLoader"),
        ("test_load_format", TestModelLoader),
        # Overwrite existing loader
        ("auto", TestModelLoader),
    ])
def test_customized_model_loader(load_format, loader_cls):
    ModelLoaderRegistry.register(
        load_format=load_format,
        loader_cls=loader_cls,
    )
    test_load_config = LoadConfig(load_format=load_format)
    model_loader = get_model_loader(test_load_config)
    assert type(model_loader).__name__ == TestModelLoader.__name__
    assert load_format in ModelLoaderRegistry.get_supported_load_formats()


def test_invalid_model_loader():
    with pytest.raises(TypeError):
        ModelLoaderRegistry.register(
            load_format="test",
            loader_cls=InValidModelLoader,
        )
