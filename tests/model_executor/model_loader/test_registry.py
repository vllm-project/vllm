# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from torch import nn

from vllm.config import ModelConfig
from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.model_loader import get_model_loader, register_model_loader
from vllm.model_executor.model_loader.base_loader import BaseModelLoader


@register_model_loader("custom_load_format")
class CustomModelLoader(BaseModelLoader):
    def __init__(self, load_config: LoadConfig) -> None:
        super().__init__(load_config)

    def download_model(self, model_config: ModelConfig) -> None:
        pass

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        pass


def test_register_model_loader():
    load_config = LoadConfig(load_format="custom_load_format")
    assert isinstance(get_model_loader(load_config), CustomModelLoader)


def test_invalid_model_loader():
    with pytest.raises(ValueError):

        @register_model_loader("invalid_load_format")
        class InValidModelLoader:
            pass


def test_auto_load_format_prefers_fastsafetensors_on_cuda_like(monkeypatch):
    loader = DefaultModelLoader(LoadConfig(load_format="auto"))
    monkeypatch.setattr(
        "vllm.model_executor.model_loader.default_loader.list_filtered_repo_files",
        lambda **kwargs: [],
    )
    monkeypatch.setattr(
        "vllm.model_executor.model_loader.default_loader.current_platform.is_cuda_alike",
        lambda: True,
    )

    resolved = loader._resolve_load_format(
        model_name_or_path="facebook/opt-125m",
        revision=None,
    )

    assert resolved == "fastsafetensors"


def test_auto_load_format_keeps_mistral_loader(monkeypatch):
    loader = DefaultModelLoader(LoadConfig(load_format="auto"))
    monkeypatch.setattr(
        "vllm.model_executor.model_loader.default_loader.list_filtered_repo_files",
        lambda **kwargs: ["consolidated.safetensors"],
    )
    monkeypatch.setattr(
        "vllm.model_executor.model_loader.default_loader.current_platform.is_cuda_alike",
        lambda: True,
    )

    resolved = loader._resolve_load_format(
        model_name_or_path="mistralai/Mistral-7B-Instruct-v0.3",
        revision=None,
    )

    assert resolved == "mistral"


def test_auto_load_format_uses_hf_off_cuda_like(monkeypatch):
    loader = DefaultModelLoader(LoadConfig(load_format="auto"))
    monkeypatch.setattr(
        "vllm.model_executor.model_loader.default_loader.list_filtered_repo_files",
        lambda **kwargs: [],
    )
    monkeypatch.setattr(
        "vllm.model_executor.model_loader.default_loader.current_platform.is_cuda_alike",
        lambda: False,
    )

    resolved = loader._resolve_load_format(
        model_name_or_path="facebook/opt-125m",
        revision=None,
    )

    assert resolved == "hf"
