# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.model_loader.dummy_loader import DummyModelLoader
from vllm.model_executor.model_loader.load_session import (
    WeightLoadSession,
    get_active_weight_load_session,
)
from vllm.model_executor.model_loader.reload import (
    finalize_layerwise_processing,
    finalize_layerwise_reload,
    initialize_layerwise_reload,
)
from vllm.model_executor.model_loader.reload.layerwise import (
    _layerwise_process,
    record_metadata_for_reloading,
)
from vllm.model_executor.model_loader.reload.types import LayerReloadingInfo
from vllm.model_executor.model_loader.utils import process_weights_after_loading


def test_reload_session_wraps_weight_loading(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "vllm.model_executor.model_loader.reload.layerwise._prepare_layerwise_loading",
        lambda *_: calls.append("prepare"),
    )
    monkeypatch.setattr(
        "vllm.model_executor.model_loader.reload.layerwise._finish_layerwise_loading",
        lambda *_: calls.append("finish"),
    )
    model = nn.Module()
    session = WeightLoadSession(model)

    session.prepare()
    calls.append("load")
    session.finish(Mock(dtype=torch.float32, quantization=None))

    assert calls == ["prepare", "load", "finish"]


def test_quant_processing_runs_once_per_load():
    module = nn.Module()
    module.quant_method = Mock(spec=QuantizeMethodBase)
    model = nn.Module()
    session = WeightLoadSession(model, torch.device("cpu"))

    session.process_quant(module)
    session.process_quant(module)

    module.quant_method.process_weights_after_loading.assert_called_once_with(module)


def test_initial_load_finishes_quant_before_attention(monkeypatch):
    model = nn.Sequential(nn.Linear(1, 1), nn.ReLU())
    session = WeightLoadSession(model, torch.device("cpu"))
    calls = []
    monkeypatch.setattr(
        session, "process_quant", lambda module: calls.append(("quant", module))
    )
    monkeypatch.setattr(
        session,
        "process_attention",
        lambda module, _: calls.append(("attention", module)),
    )
    session.prepare()
    session.finish(Mock(dtype=torch.float32, quantization=None))

    modules = list(model.modules())
    assert calls == [
        *(("quant", module) for module in modules),
        *(("attention", module) for module in modules),
    ]


def test_dummy_online_quant_uses_load_session(monkeypatch):
    session = Mock()
    info = LayerReloadingInfo(({}, {}), torch.device("cpu"))
    info.load_session = session
    monkeypatch.setattr(
        "vllm.model_executor.model_loader.dummy_loader.materialize_layer",
        lambda *_: None,
    )
    monkeypatch.setattr(
        "vllm.model_executor.model_loader.dummy_loader.get_layer_tensors",
        lambda *_: {},
    )
    layer = nn.Module()

    DummyModelLoader._process_online_quant_layer(Mock(), layer, info)

    session.process_quant.assert_called_once_with(layer)


def test_standalone_online_quant_does_not_require_session(monkeypatch):
    process_quant = Mock()
    monkeypatch.setattr(
        "vllm.model_executor.model_loader.reload.layerwise.materialize_layer",
        lambda *_: None,
    )
    monkeypatch.setattr(
        "vllm.model_executor.model_loader.reload.layerwise.get_layer_tensors",
        lambda *_: {},
    )
    monkeypatch.setattr(
        "vllm.model_executor.model_loader.reload.layerwise._process_quant_method",
        process_quant,
    )
    layer = nn.Module()
    info = LayerReloadingInfo(({}, {}), torch.device("cpu"))

    _layerwise_process(layer, info)

    process_quant.assert_called_once_with(layer)


@pytest.mark.parametrize(
    "finalize",
    [finalize_layerwise_processing, finalize_layerwise_reload],
)
def test_legacy_layerwise_api_preserves_storage(finalize):
    model = nn.Linear(2, 2)
    original_weight = model.weight
    original_bias = model.bias
    loaded_weight = torch.full_like(model.weight, 3)
    loaded_bias = torch.full_like(model.bias, 4)
    record_metadata_for_reloading(model)

    initialize_layerwise_reload(model)
    model.weight.weight_loader(model.weight, loaded_weight)
    model.bias.weight_loader(model.bias, loaded_bias)
    finalize(model, None)

    assert model.weight is original_weight
    assert model.bias is original_bias
    assert torch.equal(model.weight, loaded_weight)
    assert torch.equal(model.bias, loaded_bias)


def test_legacy_post_load_hook_still_works_standalone():
    model = nn.Module()
    model.quant_method = Mock(spec=QuantizeMethodBase)
    model.quant_method.uses_meta_device = False

    process_weights_after_loading(
        model,
        Mock(dtype=torch.float32, quantization=None),
        torch.device("cpu"),
    )

    model.quant_method.process_weights_after_loading.assert_called_once_with(model)


def test_initial_quant_processing_uses_device_context(monkeypatch):
    model = nn.Module()
    model.quant_method = Mock(spec=QuantizeMethodBase)
    model.quant_method.uses_meta_device = True
    calls = []

    @contextmanager
    def record_context(module, device):
        calls.append(("enter", module, device))
        yield module
        calls.append(("exit", module, device))

    monkeypatch.setattr(
        "vllm.model_executor.model_loader.utils.device_loading_context",
        record_context,
    )
    monkeypatch.setattr(
        "vllm.model_executor.model_loader.load_session."
        "release_device_memory_under_pressure",
        lambda *_: None,
    )
    session = WeightLoadSession(model, torch.device("cpu"))
    session.prepare()

    session.process_quant(model)
    session.process_quant(model)
    session.abort()

    assert calls == [
        ("enter", model, torch.device("cpu")),
        ("exit", model, torch.device("cpu")),
    ]
    model.quant_method.process_weights_after_loading.assert_called_once_with(model)


def test_prepare_failure_unbinds_session(monkeypatch):
    model = nn.Linear(1, 1)

    def fail_prepare(*_):
        raise RuntimeError("boom")

    monkeypatch.setattr(
        "vllm.model_executor.model_loader.reload.layerwise._prepare_layerwise_loading",
        fail_prepare,
    )
    session = WeightLoadSession(model)

    with pytest.raises(RuntimeError, match="boom"):
        session.prepare()
    assert get_active_weight_load_session(model) is None
