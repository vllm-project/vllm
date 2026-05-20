# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from types import ModuleType, SimpleNamespace

import pytest
from torch import nn

from vllm.config import VllmConfig
from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader import get_model_loader
from vllm.model_executor.model_loader.modelexpress_loader import (
    ModelExpressModelLoader,
)


class FakeModelexpressLoader:
    calls: list[tuple[str, tuple, dict]] = []
    loaded_model: nn.Module

    def __init__(self, load_config: LoadConfig):
        self.load_config = load_config

    def download_model(self, *args, **kwargs):
        self.calls.append(("download_model", args, kwargs))

    def load_weights(self, *args, **kwargs):
        self.calls.append(("load_weights", args, kwargs))

    def load_model(self, *args, **kwargs):
        self.calls.append(("load_model", args, kwargs))
        return self.loaded_model


def _install_fake_modelexpress(monkeypatch):
    FakeModelexpressLoader.calls = []
    FakeModelexpressLoader.loaded_model = nn.Module()

    for name in [
        "modelexpress",
        "modelexpress.engines",
        "modelexpress.engines.vllm",
    ]:
        monkeypatch.setitem(sys.modules, name, ModuleType(name))

    module = ModuleType("modelexpress.engines.vllm.loader")
    setattr(module, "MxModelLoader", FakeModelexpressLoader)
    monkeypatch.setitem(sys.modules, module.__name__, module)


def test_modelexpress_load_format_resolves_to_modelexpress_loader(monkeypatch):
    _install_fake_modelexpress(monkeypatch)

    loader = get_model_loader(LoadConfig(load_format="modelexpress"))

    assert isinstance(loader, ModelExpressModelLoader)


def test_modelexpress_loader_delegates_to_modelexpress(monkeypatch):
    _install_fake_modelexpress(monkeypatch)
    loader = ModelExpressModelLoader(LoadConfig(load_format="modelexpress"))
    model = nn.Module()
    model_config = SimpleNamespace()
    vllm_config = SimpleNamespace()

    loader.download_model(model_config)
    loader.load_weights(model, model_config)
    FakeModelexpressLoader.loaded_model.train()
    result = loader.load_model(
        vllm_config=vllm_config,
        model_config=model_config,
        prefix="model",
    )

    assert result is FakeModelexpressLoader.loaded_model
    assert not result.training
    assert FakeModelexpressLoader.calls == [
        ("download_model", (model_config,), {}),
        ("load_weights", (model, model_config), {}),
        (
            "load_model",
            (),
            {
                "vllm_config": vllm_config,
                "model_config": model_config,
                "prefix": "model",
            },
        ),
    ]


def test_modelexpress_loader_missing_modelexpress_error(monkeypatch):
    import importlib

    def missing_modelexpress(name):
        raise ModuleNotFoundError(name=name)

    monkeypatch.setattr(importlib, "import_module", missing_modelexpress)

    with pytest.raises(ImportError, match="requires the ModelExpress Python package"):
        ModelExpressModelLoader(LoadConfig(load_format="modelexpress"))


def test_modelexpress_loader_preserves_internal_import_errors(monkeypatch):
    import importlib

    def missing_dependency(name):
        raise ModuleNotFoundError(name="not_modelexpress_dependency")

    monkeypatch.setattr(importlib, "import_module", missing_dependency)

    with pytest.raises(ModuleNotFoundError) as exc_info:
        ModelExpressModelLoader(LoadConfig(load_format="modelexpress"))
    assert exc_info.value.name == "not_modelexpress_dependency"


def test_modelexpress_load_format_allows_object_storage_model_weights():
    model_config = SimpleNamespace(
        architecture="UnknownForTest",
        config_updated=False,
        convert_type=None,
        is_hybrid=False,
        model="test-model",
        model_weights="s3://bucket/model",
    )
    vllm_config = object.__new__(VllmConfig)
    vllm_config.model_config = model_config
    vllm_config.load_config = LoadConfig(load_format="modelexpress")

    vllm_config.try_verify_and_update_config()

    assert vllm_config.load_config.load_format == "modelexpress"
