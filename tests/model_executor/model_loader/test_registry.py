# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from torch import nn

from vllm.config import LoadConfig, ModelConfig
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.plugins import ExtensionManager


@ExtensionManager.register(base_cls=BaseModelLoader,
                           names=["custom_load_format"])
class CustomModelLoader(BaseModelLoader):

    def __init__(self, load_config: LoadConfig) -> None:
        super().__init__(load_config)

    def download_model(self, model_config: ModelConfig) -> None:
        pass

    def load_weights(self, model: nn.Module,
                     model_config: ModelConfig) -> None:
        pass


def test_register_model_loader():
    assert isinstance(
        ExtensionManager.create(BaseModelLoader, "custom_load_format"),
        CustomModelLoader)
