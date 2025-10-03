# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import pytest
from torch import nn

from vllm.config import DeviceConfig, ModelConfig, VllmConfig
from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader import get_model_loader, register_model_loader
from vllm.model_executor.model_loader.base_loader import BaseModelLoader, DownloadType
from vllm.validation.plugins import (
    ModelType,
    ModelValidationPlugin,
    ModelValidationPluginRegistry,
)


@register_model_loader("custom_load_format")
class CustomModelLoader(BaseModelLoader):
    def __init__(self, load_config: LoadConfig) -> None:
        super().__init__(load_config)
        self.download_type = None

    def download_model(self, model_config: ModelConfig) -> None:
        pass

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        pass

    def set_download_type(self, download_type: DownloadType) -> None:
        """Allow changing download_type"""
        self.download_type = download_type

    def get_download_type(self, model_name_or_path: str) -> DownloadType | None:
        return self.download_type


class MyModelValidator(ModelValidationPlugin):
    def model_validation_needed(self, model_type: ModelType, model_path: str) -> bool:
        return True

    def validate_model(
        self, model_type: ModelType, model_path: str, model: str | None = None
    ) -> None:
        raise BaseException("Model did not validate")


def test_register_model_loader(dist_init):
    load_config = LoadConfig(load_format="custom_load_format")
    custom_model_loader = get_model_loader(load_config)
    assert isinstance(custom_model_loader, CustomModelLoader)

    my_model_validator = MyModelValidator()
    ModelValidationPluginRegistry.register_plugin("test", my_model_validator)

    vllm_config = VllmConfig(
        model_config=ModelConfig(),
        device_config=DeviceConfig("auto"),
        load_config=LoadConfig(),
    )
    with pytest.raises(RuntimeError):
        custom_model_loader.load_model(vllm_config, vllm_config.model_config)

    # have validate_model() called
    custom_model_loader.set_download_type(DownloadType.LOCAL_FILE)

    vllm_config = VllmConfig(
        model_config=ModelConfig(),
        device_config=DeviceConfig("cpu"),
        load_config=LoadConfig(),
    )
    with pytest.raises(BaseException, match="Model did not validate"):
        custom_model_loader.load_model(vllm_config, vllm_config.model_config)
