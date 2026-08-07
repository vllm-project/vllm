# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock

import pytest
import torch

from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.model_loader.reload.modelwise import (
    ModelwiseReloader,
    ModelwiseReloadSession,
    record_modelwise_reload_metadata,
)
from vllm.model_executor.model_loader.utils import process_weights_after_loading


class _Int8Method(QuantizeMethodBase):
    def create_weights(self, layer: torch.nn.Module, *args, **kwargs) -> None:
        raise NotImplementedError

    def apply(self, layer: torch.nn.Module, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        source = layer.weight.detach()
        layer.weight_scale = source.abs().max().reshape(1)
        value = source.round().to(torch.int8)
        layer.weight = torch.nn.Parameter(value, requires_grad=False)


class _ReloadableQuantModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = torch.nn.Module()
        self.layer.weight = torch.nn.Parameter(torch.empty(2))
        self.layer.register_parameter("bias", None)
        self.layer.register_buffer("weight_scale", torch.empty(1))
        self.layer.register_buffer(
            "runtime_cache", torch.tensor([11.0]), persistent=False
        )
        self.layer.quant_method = _Int8Method()

    def load_weights(self, weights):
        loaded = set()
        for name, value in weights:
            self.get_parameter(name).data.copy_(value)
            loaded.add(name)
        return loaded


def _make_runtime_model() -> _ReloadableQuantModel:
    model = _ReloadableQuantModel()
    record_modelwise_reload_metadata(model)
    model.load_weights([("layer.weight", torch.tensor([1.0, 2.0]))])
    process_weights_after_loading(model, Mock(), torch.device("cpu"))
    return model


def test_modelwise_reload_copies_processed_values_to_runtime_storage() -> None:
    model = _make_runtime_model()
    runtime_weight = model.layer.weight
    runtime_ptr = runtime_weight.untyped_storage().data_ptr()
    runtime_scale = model.layer.weight_scale
    runtime_scale_ptr = runtime_scale.untyped_storage().data_ptr()

    loaded = ModelwiseReloader(model, Mock(), torch.device("cpu")).reload(
        [("layer.weight", torch.tensor([3.0, 4.0]))]
    )

    assert loaded == {"layer.weight"}
    assert model.layer.weight is runtime_weight
    assert model.layer.weight.untyped_storage().data_ptr() == runtime_ptr
    assert torch.equal(model.layer.weight, torch.tensor([3, 4], dtype=torch.int8))
    assert model.layer.weight_scale is runtime_scale
    assert model.layer.weight_scale.untyped_storage().data_ptr() == runtime_scale_ptr
    assert torch.equal(model.layer.weight_scale, torch.tensor([4.0]))
    assert model.layer.bias is None


def test_modelwise_reload_restores_runtime_bindings_after_load_failure() -> None:
    model = _make_runtime_model()
    runtime_weight = model.layer.weight
    original = runtime_weight.clone()

    def fail_after_write(weights):
        for name, value in weights:
            model.get_parameter(name).data.copy_(value)
            raise RuntimeError("load failed")

    model.load_weights = fail_after_write

    with pytest.raises(RuntimeError, match="load failed"):
        ModelwiseReloader(model, Mock(), torch.device("cpu")).reload(
            [("layer.weight", torch.tensor([9.0, 9.0]))]
        )

    assert model.layer.weight is runtime_weight
    assert torch.equal(model.layer.weight, original)


def test_modelwise_reload_preserves_tied_parameter_aliases() -> None:
    class TiedModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.left = torch.nn.Module()
            self.right = torch.nn.Module()
            weight = torch.nn.Parameter(torch.zeros(2))
            self.left.weight = weight
            self.right.weight = weight

        def load_weights(self, weights):
            loaded = set()
            for name, value in weights:
                self.get_parameter(name).data.copy_(value)
                loaded.add(name)
            return loaded

    model = TiedModel()
    record_modelwise_reload_metadata(model)
    runtime_weight = model.left.weight

    ModelwiseReloader(model, Mock(), torch.device("cpu")).reload(
        [("left.weight", torch.tensor([5.0, 6.0]))]
    )

    assert model.left.weight is runtime_weight
    assert model.right.weight is runtime_weight
    assert torch.equal(runtime_weight, torch.tensor([5.0, 6.0]))


def test_modelwise_session_processes_only_at_explicit_finish() -> None:
    model = _make_runtime_model()
    runtime_weight = model.layer.weight
    runtime_scale = model.layer.weight_scale
    runtime_cache = model.layer.runtime_cache
    session = ModelwiseReloadSession(model, Mock(), torch.device("cpu"))

    session.start()
    session.load_weights([("layer.weight", torch.tensor([7.0, 8.0]))])

    # The serving tensors remain detached while checkpoint chunks are loaded,
    # and quantization/PWAL has not run merely because a tensor was received.
    assert model.layer.weight is not runtime_weight
    assert model.layer.weight.dtype == torch.float32
    assert model.layer.weight_scale is not runtime_scale
    assert model.layer.runtime_cache is runtime_cache
    assert torch.equal(model.layer.runtime_cache, torch.tensor([11.0]))

    loaded = session.finish()

    assert loaded == {"layer.weight"}
    assert model.layer.weight is runtime_weight
    assert model.layer.weight_scale is runtime_scale
    assert model.layer.runtime_cache is runtime_cache
    assert torch.equal(runtime_weight, torch.tensor([7, 8], dtype=torch.int8))
    assert torch.equal(runtime_scale, torch.tensor([8.0]))


def test_modelwise_session_accepts_multiple_chunks_without_numel_completion() -> None:
    class TwoWeightModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.first = torch.nn.Parameter(torch.zeros(1))
            self.second = torch.nn.Parameter(torch.zeros(3))
            self.pwal_calls = 0

        def load_weights(self, weights):
            loaded = set()
            for name, value in weights:
                self.get_parameter(name).data.copy_(value)
                loaded.add(name)
            return loaded

    model = TwoWeightModel()
    record_modelwise_reload_metadata(model)
    first = model.first
    second = model.second
    session = ModelwiseReloadSession(model, Mock(), torch.device("cpu"))
    session.start()

    # Different tensor sizes and arbitrary chunk boundaries do not trigger a
    # completion heuristic. The caller's finish signal is the only boundary.
    session.load_weights([("second", torch.tensor([2.0, 3.0, 4.0]))])
    session.load_weights([("first", torch.tensor([1.0]))])
    assert model.first is not first
    assert model.second is not second

    assert session.finish() == {"first", "second"}
    assert model.first is first
    assert model.second is second
    assert torch.equal(first, torch.tensor([1.0]))
    assert torch.equal(second, torch.tensor([2.0, 3.0, 4.0]))


def test_modelwise_session_abort_discards_received_weights() -> None:
    model = _make_runtime_model()
    runtime_weight = model.layer.weight
    original = runtime_weight.clone()
    session = ModelwiseReloadSession(model, Mock(), torch.device("cpu"))
    session.start()
    session.load_weights([("layer.weight", torch.tensor([9.0, 9.0]))])

    session.abort()

    assert model.layer.weight is runtime_weight
    assert torch.equal(runtime_weight, original)
