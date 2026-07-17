# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock

import torch
from torch import nn

from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.model_loader.dummy_loader import DummyModelLoader
from vllm.model_executor.model_loader.post_load import WeightLoadSession
from vllm.model_executor.model_loader.reload.types import LayerReloadingInfo


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
    session = WeightLoadSession(nn.Module(), Mock())

    session.prepare()
    calls.append("load")
    session.finish()

    assert calls == ["prepare", "load", "finish"]


def test_quant_processing_runs_once_per_load():
    module = nn.Module()
    module.quant_method = Mock(spec=QuantizeMethodBase)
    session = WeightLoadSession(nn.Module(), Mock(), torch.device("cpu"))

    session.process_quant(module)
    session.process_quant(module)

    module.quant_method.process_weights_after_loading.assert_called_once_with(module)


def test_initial_load_finishes_quant_before_attention(monkeypatch):
    model = nn.Sequential(nn.Linear(1, 1), nn.ReLU())
    session = WeightLoadSession(model, Mock(quantization=None), torch.device("cpu"))
    calls = []
    monkeypatch.setattr(
        session, "process_quant", lambda module: calls.append(("quant", module))
    )
    monkeypatch.setattr(
        session,
        "process_attention",
        lambda module: calls.append(("attention", module)),
    )
    session.finish()

    modules = list(model.modules())
    assert calls == [
        *(("quant", module) for module in modules),
        *(("attention", module) for module in modules),
    ]


def test_dummy_online_quant_uses_load_session(monkeypatch):
    session = Mock()
    info = LayerReloadingInfo(({}, {}), torch.device("cpu"), load_session=session)
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
