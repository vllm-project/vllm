# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Layerwise reload invariants for checkpoint-backed attention sinks.

The defect these tests guard against is a model ``load_weights`` that writes a
sink with a bare ``copy_``. Such a write is invisible to
``online_process_loader``, so during a live update it lands on the temporary
parameter that the reload materializes and is then discarded when the saved
kernel tensor is put back. Testing a mock ``weight_loader`` invocation would not
catch that, so these tests drive the real layerwise reload lifecycle.

The tests drive a real layerwise reload and check the resulting sink values,
padding, and parameter storage.
"""

from collections.abc import Iterable
from types import SimpleNamespace

import pytest
import torch
from vllm.model_executor.model_loader.attention_sink import load_padded_attn_sink

from vllm.model_executor.model_loader.reload import layerwise
from vllm.model_executor.model_loader.reload.layerwise import (
    finalize_layerwise_reload,
    initialize_layerwise_reload,
    record_metadata_for_reloading,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

PADDED_HEADS = 8
LOCAL_HEADS = 4
GLOBAL_HEADS = 16
PROJ_SIZE = 256
# The fake layers only require dtype during deferred attention finalization.
MODEL_CONFIG = SimpleNamespace(dtype=torch.float32)


def _make_padded_sink() -> torch.nn.Parameter:
    sink = torch.nn.Parameter(
        torch.full((PADDED_HEADS,), -float("inf"), dtype=torch.float32),
        requires_grad=False,
    )
    sink.weight_loader = default_weight_loader
    return sink


def _checkpoint_a() -> list[tuple[str, torch.Tensor]]:
    return [
        ("sink", torch.arange(GLOBAL_HEADS, dtype=torch.float32)),
        ("proj", torch.arange(PROJ_SIZE, dtype=torch.float32)),
    ]


def _checkpoint_b() -> list[tuple[str, torch.Tensor]]:
    return [
        ("sink", -torch.arange(GLOBAL_HEADS, dtype=torch.float32) - 1.0),
        ("proj", -torch.arange(PROJ_SIZE, dtype=torch.float32) - 1.0),
    ]


def _expected_local(weights: list[tuple[str, torch.Tensor]], start: int, end: int):
    sink = next(w for n, w in weights if n == "sink")
    return sink[start:end]


class _SinkLayer(torch.nn.Module):
    """Minimal layer with the sink loading shape the models use."""

    def __init__(self) -> None:
        super().__init__()
        self.sink = _make_padded_sink()
        self.proj = torch.nn.Parameter(torch.zeros(PROJ_SIZE))
        self.proj.weight_loader = default_weight_loader
        # Sinks are consumed by attention backends, which defer processing.
        self.process_weights_after_loading = lambda dtype: None

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
        head_start: int,
        head_end: int,
    ) -> set[str]:
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if "sink" in name:
                param = params_dict[name]
                load_padded_attn_sink(param, loaded_weight, head_start, head_end)
                loaded_params.add(name)
                continue
            weight_loader = getattr(
                params_dict[name], "weight_loader", default_weight_loader
            )
            weight_loader(params_dict[name], loaded_weight)
            loaded_params.add(name)
        return loaded_params


class _ExactShapeSinkLayer(torch.nn.Module):
    """Local runtime sink shape used by HY-V4, MiMo-V2, and GPT-OSS."""

    def __init__(self) -> None:
        super().__init__()
        self.sink = torch.nn.Parameter(torch.zeros(LOCAL_HEADS), requires_grad=False)
        self.sink.weight_loader = default_weight_loader

    def load_weights(
        self, loaded_weight: torch.Tensor, head_start: int, head_end: int
    ) -> None:
        param = self.sink
        local_weight = loaded_weight[head_start:head_end]
        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        weight_loader(param, local_weight)


def _live_reload(layer, head_start: int, head_end: int, weights) -> None:
    """Drive the real layerwise reload lifecycle for one weight update."""
    record_metadata_for_reloading(layer)
    initialize_layerwise_reload(layer)
    layer.load_weights(weights, head_start, head_end)
    finalize_layerwise_reload(layer, MODEL_CONFIG)


@pytest.mark.parametrize("tp_rank", [0, 1])
def test_padded_sink_live_update_reaches_kernel_tensor(tp_rank: int):
    """A live update must land in kernel tensor storage with padding intact."""
    head_start = LOCAL_HEADS * tp_rank
    head_end = head_start + LOCAL_HEADS

    layer = _SinkLayer()
    layer.load_weights(_checkpoint_a(), head_start, head_end)
    kernel_ptr = layer.sink.data_ptr()

    _live_reload(layer, head_start, head_end, _checkpoint_b())

    expected = _expected_local(_checkpoint_b(), head_start, head_end)
    assert torch.equal(layer.sink.data[:LOCAL_HEADS], expected)
    assert torch.isneginf(layer.sink.data[LOCAL_HEADS:]).all()
    assert layer.sink.data_ptr() == kernel_ptr


def test_padded_sink_padding_tail_not_polluted_after_reload():
    """Polluted padding must be restored to -inf by the live update itself."""
    layer = _SinkLayer()
    layer.load_weights(_checkpoint_a(), 0, LOCAL_HEADS)
    with torch.no_grad():
        layer.sink.data[LOCAL_HEADS:] = 0.0

    _live_reload(layer, 0, LOCAL_HEADS, _checkpoint_b())

    assert torch.isneginf(layer.sink.data[LOCAL_HEADS:]).all()


def test_padded_sink_survives_repeated_reloads():
    """Reloading twice keeps loading through the restored original loader."""
    layer = _SinkLayer()
    layer.load_weights(_checkpoint_a(), 0, LOCAL_HEADS)
    kernel_ptr = layer.sink.data_ptr()

    _live_reload(layer, 0, LOCAL_HEADS, _checkpoint_b())
    _live_reload(layer, 0, LOCAL_HEADS, _checkpoint_a())

    expected = _expected_local(_checkpoint_a(), 0, LOCAL_HEADS)
    assert torch.equal(layer.sink.data[:LOCAL_HEADS], expected)
    assert torch.isneginf(layer.sink.data[LOCAL_HEADS:]).all()
    assert layer.sink.data_ptr() == kernel_ptr


@pytest.mark.parametrize("tp_rank", [0, 1])
def test_padded_sink_deferred_attention_finalize(monkeypatch, tp_rank: int):
    """Deferred attention replays the sink at finalize into stable storage."""
    head_start = LOCAL_HEADS * tp_rank
    head_end = head_start + LOCAL_HEADS
    layer = _SinkLayer()
    layer.load_weights(_checkpoint_a(), head_start, head_end)
    kernel_ptr = layer.sink.data_ptr()
    post_load_dtypes = []
    layer.process_weights_after_loading = post_load_dtypes.append
    monkeypatch.setattr(
        layerwise, "is_deferred_attention_layer", lambda candidate: candidate is layer
    )

    record_metadata_for_reloading(layer)
    initialize_layerwise_reload(layer)
    layer.load_weights(_checkpoint_b(), head_start, head_end)
    assert not post_load_dtypes  # Deferred until _finalize_attention_layer.
    finalize_layerwise_reload(layer, MODEL_CONFIG)

    assert post_load_dtypes == [torch.float32]
    assert torch.equal(
        layer.sink.data[:LOCAL_HEADS],
        _expected_local(_checkpoint_b(), head_start, head_end),
    )
    assert torch.isneginf(layer.sink.data[LOCAL_HEADS:]).all()
    assert layer.sink.data_ptr() == kernel_ptr


@pytest.mark.parametrize("tp_rank", [0, 1])
def test_exact_shape_sink_live_reload(tp_rank: int):
    """A TP-local sink without padding also reloads into stable storage."""
    head_start = LOCAL_HEADS * tp_rank
    head_end = head_start + LOCAL_HEADS
    checkpoint_a = _checkpoint_a()[0][1]
    checkpoint_b = _checkpoint_b()[0][1]
    layer = _ExactShapeSinkLayer()
    layer.load_weights(checkpoint_a, head_start, head_end)
    kernel_ptr = layer.sink.data_ptr()

    _live_reload(layer, head_start, head_end, checkpoint_b)

    assert torch.equal(layer.sink.data, checkpoint_b[head_start:head_end])
    assert layer.sink.data_ptr() == kernel_ptr


def _make_param(padded_heads: int = PADDED_HEADS) -> torch.nn.Parameter:
    return torch.nn.Parameter(
        torch.full((padded_heads,), -float("inf"), dtype=torch.float32),
        requires_grad=False,
    )


def _checkpoint() -> torch.Tensor:
    return torch.arange(GLOBAL_HEADS, dtype=torch.float32)


@pytest.mark.parametrize("tp_rank", [0, 1, 2, 3])
def test_load_padded_attn_sink_writes_local_heads_and_keeps_padding_at_neginf(
    tp_rank: int,
):
    """Every rank takes its own heads and keeps the padded tail disabled."""
    param = _make_param()
    loaded_weight = _checkpoint()
    head_start = LOCAL_HEADS * tp_rank
    head_end = head_start + LOCAL_HEADS

    load_padded_attn_sink(param, loaded_weight, head_start, head_end)

    expected_local = loaded_weight[head_start:head_end]
    assert torch.equal(param.data[:LOCAL_HEADS], expected_local)
    assert torch.isneginf(param.data[LOCAL_HEADS:]).all()


def test_load_padded_attn_sink_resets_polluted_padding_tail():
    """A reload must restore the padding tail even if it was polluted."""
    param = _make_param()
    with torch.no_grad():
        param.data[LOCAL_HEADS:] = 0.0

    load_padded_attn_sink(param, _checkpoint(), 0, LOCAL_HEADS)

    assert torch.isneginf(param.data[LOCAL_HEADS:]).all()


def test_load_padded_attn_sink_preserves_parameter_storage():
    """The helper must load in place so kernel tensor references stay valid."""
    param = _make_param()
    ptr = param.data.data_ptr()

    load_padded_attn_sink(param, _checkpoint(), 0, LOCAL_HEADS)

    assert param.data.data_ptr() == ptr


def test_load_padded_attn_sink_uses_current_weight_loader():
    """The load must be routed through the parameter's current loader.

    Layerwise reload replaces ``weight_loader`` with ``online_process_loader``;
    reading it dynamically is what makes a live update observable.
    """
    param = _make_param()
    calls = []

    def recording_loader(p, w):
        calls.append((p, w))

    param.weight_loader = recording_loader

    load_padded_attn_sink(param, _checkpoint(), 0, LOCAL_HEADS)

    assert len(calls) == 1
    called_param, called_weight = calls[0]
    assert called_param is param
    # The loader receives the full padded runtime representation so that the
    # element counts match and the replay re-establishes the padding.
    assert called_weight.shape == param.shape
    assert torch.equal(called_weight[:LOCAL_HEADS], _checkpoint()[:LOCAL_HEADS])
    assert torch.isneginf(called_weight[LOCAL_HEADS:]).all()


def test_load_padded_attn_sink_degenerate_head_range():
    """A rank with no heads keeps the whole sink disabled and stays loadable."""
    param = _make_param()

    load_padded_attn_sink(param, _checkpoint(), 0, 0)

    assert torch.isneginf(param.data).all()


def test_load_padded_attn_sink_builds_runtime_weight_in_param_dtype():
    """The reconstructed tensor must match the parameter, not the checkpoint.

    Padding is written by the load, so a dtype mismatch between the checkpoint
    sink and the runtime parameter must be resolved before the load.
    """
    param = _make_param().to(torch.bfloat16)
    loader_inputs = []
    param.weight_loader = lambda p, w: loader_inputs.append(w)

    load_padded_attn_sink(param, _checkpoint(), 0, LOCAL_HEADS)

    assert loader_inputs[0].dtype == torch.bfloat16
    assert torch.equal(
        loader_inputs[0][:LOCAL_HEADS], _checkpoint()[:LOCAL_HEADS].to(torch.bfloat16)
    )
    assert torch.isneginf(loader_inputs[0][LOCAL_HEADS:]).all()


def test_load_padded_attn_sink_rejects_head_range_wider_than_param():
    """A rank claiming more heads than the padded parameter must fail loudly.

    Silently truncating would load the wrong heads, and the online loader
    rejects a source larger than its destination anyway.
    """
    param = _make_param(padded_heads=2)

    with pytest.raises(ValueError, match="does not fit the runtime sink parameter"):
        load_padded_attn_sink(param, _checkpoint(), 0, LOCAL_HEADS)
