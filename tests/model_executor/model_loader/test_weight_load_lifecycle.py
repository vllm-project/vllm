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
from vllm.model_executor.model_loader.reload.torchao_decorator import (
    support_quantized_model_reload_from_hp_weights,
)
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


def test_initial_load_processes_quant_before_finalizing_attention_runtime(monkeypatch):
    model = nn.Sequential(nn.Linear(1, 1), nn.ReLU())
    session = WeightLoadSession(model, torch.device("cpu"))
    calls = []
    monkeypatch.setattr(
        session, "process_quant", lambda module: calls.append(("quant", module))
    )
    monkeypatch.setattr(
        session,
        "finalize_attention_runtime",
        lambda module, _: calls.append(("attention_runtime", module)),
    )
    monkeypatch.setattr(
        "vllm.model_executor.model_loader.utils.POST_LOAD_ATTENTION_TYPES",
        (nn.ReLU,),
    )
    session.prepare()
    session.finish(Mock(dtype=torch.float32, quantization=None))

    modules = list(model.modules())
    assert calls == [
        *(("quant", module) for module in modules),
        ("attention_runtime", model[1]),
    ]


def test_attention_runtime_finalization_runs_once_without_type_discovery():
    class PostLoadModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def process_weights_after_loading(self, act_dtype):
            self.calls += 1

    module = PostLoadModule()
    session = WeightLoadSession(nn.Module(), torch.device("cpu"))

    session.finalize_attention_runtime(module, torch.float32)
    session.finalize_attention_runtime(module, torch.float32)

    assert module.calls == 1


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


def test_legacy_layerwise_api_preserves_storage():
    model = nn.Linear(2, 2)
    original_weight = model.weight
    original_bias = model.bias
    loaded_weight = torch.full_like(model.weight, 3)
    loaded_bias = torch.full_like(model.bias, 4)
    record_metadata_for_reloading(model)

    initialize_layerwise_reload(model)
    model.weight.weight_loader(model.weight, loaded_weight)
    model.bias.weight_loader(model.bias, loaded_bias)
    finalize_layerwise_processing(model, None)

    assert model.weight is original_weight
    assert model.bias is original_bias
    assert torch.equal(model.weight, loaded_weight)
    assert torch.equal(model.bias, loaded_bias)


def test_finalize_layerwise_reload_remains_a_compatibility_alias(monkeypatch):
    finish = Mock()
    monkeypatch.setattr(
        "vllm.model_executor.model_loader.reload.layerwise."
        "finalize_layerwise_processing",
        finish,
    )
    model = nn.Module()
    model_config = Mock()

    finalize_layerwise_reload(model, model_config)

    finish.assert_called_once_with(model, model_config)


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


def test_abort_cleanup_failure_does_not_mask_load_failure(monkeypatch):
    model = nn.Linear(1, 1)
    record_metadata_for_reloading(model)
    session = WeightLoadSession(model)
    session.prepare()
    monkeypatch.setattr(
        "vllm.model_executor.model_loader.reload.layerwise._finish_layerwise_loading",
        Mock(side_effect=ValueError("load failed")),
    )
    monkeypatch.setattr(
        "vllm.model_executor.model_loader.reload.layerwise._abort_layerwise_loading",
        Mock(side_effect=RuntimeError("cleanup failed")),
    )

    with pytest.raises(ValueError, match="load failed"):
        session.finish(None)

    assert get_active_weight_load_session(model) is None


def test_torchao_reload_failure_clears_session():
    model = nn.Linear(1, 1)
    record_metadata_for_reloading(model)
    model._do_torchao_reload = True

    class Loader:
        def __init__(self):
            self.module = model

        @support_quantized_model_reload_from_hp_weights
        def load_weights(self, _weights):
            raise ValueError("load failed")

    with pytest.raises(ValueError, match="load failed"):
        Loader().load_weights([])

    assert get_active_weight_load_session(model) is None
