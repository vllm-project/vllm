# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
from torch import nn

import vllm.model_executor.model_loader.reload.inplace as reload_inplace_module
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

    def on_sleep(self, *args, **kwargs):
        self.calls.append(("on_sleep", args, kwargs))

    def on_wake_up(self, *args, **kwargs):
        self.calls.append(("on_wake_up", args, kwargs))

    def reload_model(self, *args, **kwargs):
        self.calls.append(("reload_model", args, kwargs))
        return kwargs["model"]


class FakeModelexpressLoaderWithoutSleepHooks:
    """Mimics an older modelexpress install without the lifecycle hooks."""

    def __init__(self, load_config: LoadConfig):
        self.load_config = load_config

    def download_model(self, *args, **kwargs):
        pass

    def load_weights(self, *args, **kwargs):
        pass

    def load_model(self, *args, **kwargs):
        pass


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
    module.__dict__["MxModelLoader"] = FakeModelexpressLoader
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


def test_modelexpress_loader_delegates_sleep_hooks(monkeypatch):
    _install_fake_modelexpress(monkeypatch)
    loader = ModelExpressModelLoader(LoadConfig(load_format="modelexpress"))

    loader.on_sleep(1)
    loader.on_wake_up(["weights"])

    assert FakeModelexpressLoader.calls == [
        ("on_sleep", (1,), {}),
        ("on_wake_up", (["weights"],), {}),
    ]


def test_modelexpress_loader_tolerates_missing_sleep_hooks(monkeypatch):
    _install_fake_modelexpress(monkeypatch)
    module = sys.modules["modelexpress.engines.vllm.loader"]
    module.__dict__["MxModelLoader"] = FakeModelexpressLoaderWithoutSleepHooks
    loader = ModelExpressModelLoader(LoadConfig(load_format="modelexpress"))

    loader.on_sleep(1)
    loader.on_wake_up(["weights"])


def test_modelexpress_loader_reloads_through_modelexpress(monkeypatch):
    _install_fake_modelexpress(monkeypatch)
    loader = ModelExpressModelLoader(LoadConfig(load_format="modelexpress"))
    model = nn.Module()
    model_config = SimpleNamespace()
    vllm_config = SimpleNamespace()

    loader.reload_model(vllm_config, model_config, model)

    assert FakeModelexpressLoader.calls == [
        (
            "reload_model",
            (),
            {"vllm_config": vllm_config, "model_config": model_config, "model": model},
        ),
    ]


def test_reload_weights_reuses_loader_that_loaded_model(monkeypatch):
    """ModelExpress keeps sleep state on the loader instance, so the reload
    after a level-2 sleep must reach the instance that got `on_sleep`."""
    _install_fake_modelexpress(monkeypatch)
    load_config = LoadConfig(load_format="modelexpress")
    loader = ModelExpressModelLoader(load_config)
    vllm_config = SimpleNamespace(load_config=load_config, model_config=None)
    model = loader.load_model(vllm_config=vllm_config, model_config=None)
    monkeypatch.setattr(
        reload_inplace_module, "get_model_loader", Mock(side_effect=AssertionError)
    )

    reload_inplace_module.reload_weights(
        vllm_config,
        model,
        weights_iterator=None,
        weights_path=None,
        is_checkpoint_format=True,
    )

    assert FakeModelexpressLoader.calls[-1][0] == "reload_model"


def test_modelexpress_loader_without_reload_model_falls_back(monkeypatch):
    _install_fake_modelexpress(monkeypatch)
    module = sys.modules["modelexpress.engines.vllm.loader"]
    module.__dict__["MxModelLoader"] = FakeModelexpressLoaderWithoutSleepHooks
    loader = ModelExpressModelLoader(LoadConfig(load_format="modelexpress"))

    with pytest.raises(NotImplementedError, match="iterating weights"):
        loader.reload_model(SimpleNamespace(), SimpleNamespace(), nn.Module())


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
