# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock

import pytest
import torch

from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.model_loader.reload.modelwise import (
    ModelwiseReloader,
    ModelwiseReloadSession,
    _audit_meta_bindings_before_pwal,
    record_modelwise_reload_metadata,
)
from vllm.model_executor.model_loader.reload.sharding import (
    capture_rank_sharding,
    install_sharding_recorders,
    uninstall_sharding_recorders,
)
from vllm.model_executor.model_loader.reload.source import observe_weight_sources
from vllm.model_executor.model_loader.utils import process_weights_after_loading
from vllm.model_executor.models.utils import AutoWeightsLoader


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


def _record_initial_load(model, weights) -> None:
    """Load test weights while recording an exact rank-local manifest."""
    install_sharding_recorders(model)
    try:
        with capture_rank_sharding(model, reset=True):
            model.load_weights(observe_weight_sources(weights))
    finally:
        uninstall_sharding_recorders(model)


def test_meta_audit_attributes_skipped_pwal_bindings(monkeypatch) -> None:
    """Residual metas in skipped quant modules remain audited but are safe."""
    debug = Mock()
    monkeypatch.setattr(
        "vllm.model_executor.model_loader.reload.modelwise.logger.debug", debug
    )
    model = torch.nn.Module()
    model.attn = torch.nn.Module()
    model.attn.scale = torch.nn.Parameter(torch.empty(1, device="meta"))
    model.attn.quant_method = _Int8Method()

    findings = _audit_meta_bindings_before_pwal(
        model,
        frozenset(),
        frozenset({"attn"}),
    )

    assert len(findings) == 1
    assert "module=attn" in findings[0]
    assert "quant_owner=attn" in findings[0]
    assert "state=skipped-pwal" in findings[0]
    debug.assert_called_once()
    assert "residual meta audit before PWAL" in debug.call_args.args[0]


def test_meta_audit_rejects_active_pwal_bindings() -> None:
    """A scheduled quant PWAL cannot run while its subtree contains meta."""
    model = torch.nn.Module()
    model.layer = torch.nn.Module()
    model.layer.weight = torch.nn.Parameter(torch.empty(1, device="meta"))
    model.layer.quant_method = _Int8Method()

    with pytest.raises(RuntimeError, match="reachable by scheduled PWAL"):
        _audit_meta_bindings_before_pwal(
            model,
            frozenset({"layer"}),
            frozenset(),
        )


def test_process_weights_skips_deferred_attention_hook(monkeypatch) -> None:
    """Skipped modules must bypass deferred attention PWAL as well as quant PWAL."""
    model = torch.nn.Module()
    model.attn = torch.nn.Module()
    model.attn.process_weights_after_loading = Mock()
    monkeypatch.setattr(
        "vllm.model_executor.model_loader.utils.is_deferred_attention_layer",
        lambda module: module is model.attn,
    )

    process_weights_after_loading(
        model,
        Mock(dtype=torch.float16),
        torch.device("cpu"),
        skip_modules=frozenset({"attn"}),
    )

    model.attn.process_weights_after_loading.assert_not_called()


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


def test_modelwise_session_materializes_siblings_before_loader_snapshots() -> None:
    """Batch loading must not route a later sibling into a stale meta tensor."""

    class SiblingWeightModel(torch.nn.Module):
        """Model whose loader snapshots sibling parameters through AutoWeightsLoader."""

        def __init__(self) -> None:
            """Create two checkpoint parameters owned by the same child module."""
            super().__init__()
            self.layer = torch.nn.Module()
            self.layer.first = torch.nn.Parameter(torch.zeros(1))
            self.layer.second = torch.nn.Parameter(torch.zeros(1))

        def load_weights(self, weights):
            """Load checkpoint tensors through vLLM's recursive loader."""
            return AutoWeightsLoader(self).load_weights(weights)

    model = SiblingWeightModel()
    record_modelwise_reload_metadata(model)
    _record_initial_load(
        model,
        [
            ("layer.first", torch.tensor([1.0])),
            ("layer.second", torch.tensor([2.0])),
        ],
    )
    first = model.layer.first
    second = model.layer.second
    session = ModelwiseReloadSession(model, Mock(), torch.device("cpu"))
    session.start()

    loaded = session.load_weights(
        [
            ("layer.first", torch.tensor([3.0])),
            ("layer.second", torch.tensor([4.0])),
        ]
    )

    assert loaded == {"layer.first", "layer.second"}
    assert session.finish() == {"layer.first", "layer.second"}
    assert model.layer.first is first
    assert model.layer.second is second
    assert torch.equal(first, torch.tensor([3.0]))
    assert torch.equal(second, torch.tensor([4.0]))


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


def test_modelwise_session_materializes_only_started_weights() -> None:
    """A source binds on first use while untouched destinations stay meta."""

    class PlainModel(torch.nn.Module):
        """Two runtime-compatible weights used to exercise direct binding."""

        def __init__(self) -> None:
            """Create weights with observable, model-style weight loaders."""
            super().__init__()
            self.first = torch.nn.Parameter(torch.zeros(2))
            self.second = torch.nn.Parameter(torch.zeros(2))

            def loader(param, value):
                """Copy one complete checkpoint source into its destination."""
                param.data.copy_(value)

            self.first.weight_loader = loader
            self.second.weight_loader = loader

        def load_weights(self, weights):
            """Route named checkpoint sources through their recorded loaders."""
            loaded = set()
            for name, value in weights:
                param = self.get_parameter(name)
                param.weight_loader(param, value)
                loaded.add(name)
            return loaded

    model = PlainModel()
    record_modelwise_reload_metadata(model)
    _record_initial_load(
        model,
        [("first", torch.ones(2)), ("second", torch.ones(2))],
    )
    first = model.first
    second = model.second
    session = ModelwiseReloadSession(model, Mock(), torch.device("cpu"))

    session.start()
    assert model.first.is_meta
    assert model.second.is_meta

    session.load_weights([("first", torch.tensor([3.0, 4.0]))])

    assert not model.first.is_meta
    assert (
        model.first.untyped_storage().data_ptr() == first.untyped_storage().data_ptr()
    )
    assert model.second.is_meta
    session.abort()
    assert model.first is first
    assert model.second is second
    assert torch.equal(first, torch.tensor([3.0, 4.0]))


def test_modelwise_session_runs_pwal_when_module_shards_complete() -> None:
    """PWAL runs immediately after the final shard in its module arrives."""

    class TwoWeightMethod(_Int8Method):
        """Quantize two checkpoint weights as one module-level reload unit."""

        def supports_incremental_pwal(self, layer: torch.nn.Module) -> bool:
            """Allow the test method to run after its two inputs arrive."""
            return True

        def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
            """Convert both weights to runtime INT8 format and derive scale."""
            layer.pwal_calls += 1
            source = torch.cat((layer.first, layer.second))
            layer.weight_scale = source.abs().max().reshape(1)
            layer.first = torch.nn.Parameter(
                layer.first.round().to(torch.int8), requires_grad=False
            )
            layer.second = torch.nn.Parameter(
                layer.second.round().to(torch.int8), requires_grad=False
            )

    class QuantModel(torch.nn.Module):
        """Minimal quantized model whose PWAL requires two source weights."""

        def __init__(self) -> None:
            """Create checkpoint-format bindings and attach the test method."""
            super().__init__()
            self.layer = torch.nn.Module()
            self.layer.first = torch.nn.Parameter(torch.zeros(1))
            self.layer.second = torch.nn.Parameter(torch.zeros(1))
            self.layer.register_buffer("weight_scale", torch.zeros(1))
            self.layer.pwal_calls = 0
            self.layer.quant_method = TwoWeightMethod()

            def loader(param, value):
                """Copy one complete checkpoint source into staging storage."""
                param.data.copy_(value)

            self.layer.first.weight_loader = loader
            self.layer.second.weight_loader = loader

        def load_weights(self, weights):
            """Route named checkpoint sources through their recorded loaders."""
            loaded = set()
            for name, value in weights:
                param = self.get_parameter(name)
                param.weight_loader(param, value)
                loaded.add(name)
            return loaded

    model = QuantModel()
    record_modelwise_reload_metadata(model)
    _record_initial_load(
        model,
        [("layer.first", torch.ones(1)), ("layer.second", torch.ones(1))],
    )
    process_weights_after_loading(model, Mock(), torch.device("cpu"))
    first = model.layer.first
    second = model.layer.second
    scale = model.layer.weight_scale
    session = ModelwiseReloadSession(model, Mock(), torch.device("cpu"))

    session.start()
    session.load_weights([("layer.first", torch.tensor([5.0]))])
    assert model.layer.pwal_calls == 1
    assert not model.layer.first.is_meta
    assert model.layer.second.is_meta

    session.load_weights([("layer.second", torch.tensor([7.0]))])
    assert model.layer.pwal_calls == 2
    assert model.layer.first is first
    assert model.layer.second is second
    assert model.layer.weight_scale is scale
    assert torch.equal(first, torch.tensor([5], dtype=torch.int8))
    assert torch.equal(second, torch.tensor([7], dtype=torch.int8))
    assert torch.equal(scale, torch.tensor([7.0]))

    assert session.finish() == {"layer.first", "layer.second"}
    assert model.layer.pwal_calls == 2


def test_modelwise_session_preserves_model_order_for_completed_pwal() -> None:
    """Completed modules wait for earlier PWAL modules in model order."""

    calls = []

    class OrderedMethod(_Int8Method):
        """Record PWAL order while preserving the checkpoint tensor schema."""

        def supports_incremental_pwal(self, layer: torch.nn.Module) -> bool:
            """Allow the test method to run after its input arrives."""
            return True

        def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
            """Record the layer name and leave its runtime-compatible weight."""
            calls.append(layer.label)

    class OrderedModel(torch.nn.Module):
        """Two quantized layers whose checkpoint sources arrive in reverse."""

        def __init__(self) -> None:
            """Create layers in the required PWAL order."""
            super().__init__()
            for name in ("first", "second"):
                layer = torch.nn.Module()
                layer.label = name
                layer.weight = torch.nn.Parameter(torch.zeros(1))
                layer.quant_method = OrderedMethod()

                def loader(param, value):
                    """Copy a complete checkpoint source into its parameter."""
                    param.data.copy_(value)

                layer.weight.weight_loader = loader
                setattr(self, name, layer)

        def load_weights(self, weights):
            """Route checkpoint sources through their parameter loaders."""
            loaded = set()
            for name, value in weights:
                param = self.get_parameter(name)
                param.weight_loader(param, value)
                loaded.add(name)
            return loaded

    model = OrderedModel()
    record_modelwise_reload_metadata(model)
    _record_initial_load(
        model,
        [("first.weight", torch.ones(1)), ("second.weight", torch.ones(1))],
    )
    process_weights_after_loading(model, Mock(), torch.device("cpu"))
    calls.clear()
    session = ModelwiseReloadSession(model, Mock(), torch.device("cpu"))
    session.start()

    session.load_weights([("second.weight", torch.tensor([2.0]))])
    assert calls == []
    session.load_weights([("first.weight", torch.tensor([1.0]))])
    assert calls == ["first", "second"]
    session.finish()
