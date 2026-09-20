# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Layerwise reload invariants for checkpoint-backed attention sinks.

The defect these tests guard against is a model ``load_weights`` that writes a
sink with a bare ``copy_``. Such a write is invisible to
``online_process_loader``, so during a live update it lands on the temporary
parameter that the reload materializes and is then discarded when the saved
kernel tensor is put back. Testing a mock ``weight_loader`` invocation would not
catch that, so these tests drive the real layerwise reload lifecycle.

``test_direct_copy_loses_the_update_and_corrupts_padding`` pins the failure mode
itself, and ``test_padded_sink_live_update_reaches_kernel_tensor`` pins the fix.
"""

from collections.abc import Iterable

import pytest
import torch
from vllm.models.deepseek_v4.sink import load_padded_attn_sink

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
    """Minimal layer with the sink loading shape the models use.

    ``use_helper`` selects between the fixed load and the historical direct
    copy, so one fixture can pin both the bug and the fix.
    """

    def __init__(self, use_helper: bool) -> None:
        super().__init__()
        self.use_helper = use_helper
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
                if self.use_helper:
                    load_padded_attn_sink(param, loaded_weight, head_start, head_end)
                else:
                    narrow_weight = loaded_weight[head_start:head_end]
                    param[: narrow_weight.shape[0]].copy_(narrow_weight)
                loaded_params.add(name)
                continue
            weight_loader = getattr(
                params_dict[name], "weight_loader", default_weight_loader
            )
            weight_loader(params_dict[name], loaded_weight)
            loaded_params.add(name)
        return loaded_params


def _live_reload(layer, head_start: int, head_end: int, weights) -> None:
    """Drive the real layerwise reload lifecycle for one weight update."""
    record_metadata_for_reloading(layer)
    initialize_layerwise_reload(layer)
    layer.load_weights(weights, head_start, head_end)
    finalize_layerwise_reload(layer, None)


@pytest.mark.parametrize("tp_rank", [0, 1])
def test_direct_copy_loses_the_update_and_corrupts_padding(tp_rank: int):
    """Pin the defect: a bare copy_ never reaches the kernel tensor.

    This is the behaviour the fix removes. With a direct copy the live update
    is written into the parameter instance the reload materialized, while the
    saved kernel tensor is restored afterwards, so the kernel keeps a
    freshly-allocated tensor whose sink values were never written at all.
    """
    head_start = LOCAL_HEADS * tp_rank
    head_end = head_start + LOCAL_HEADS

    layer = _SinkLayer(use_helper=False)
    layer.load_weights(_checkpoint_a(), head_start, head_end)
    kernel_ptr = layer.sink.data_ptr()

    _live_reload(layer, head_start, head_end, _checkpoint_b())

    expected = _expected_local(_checkpoint_b(), head_start, head_end)
    # The pointer is stable, but the values were never written.
    assert layer.sink.data_ptr() == kernel_ptr
    assert not torch.equal(layer.sink.data[:LOCAL_HEADS], expected)


@pytest.mark.parametrize("tp_rank", [0, 1])
def test_padded_sink_live_update_reaches_kernel_tensor(tp_rank: int):
    """A live update must land in kernel tensor storage with padding intact."""
    head_start = LOCAL_HEADS * tp_rank
    head_end = head_start + LOCAL_HEADS

    layer = _SinkLayer(use_helper=True)
    layer.load_weights(_checkpoint_a(), head_start, head_end)
    kernel_ptr = layer.sink.data_ptr()

    _live_reload(layer, head_start, head_end, _checkpoint_b())

    expected = _expected_local(_checkpoint_b(), head_start, head_end)
    assert torch.equal(layer.sink.data[:LOCAL_HEADS], expected)
    assert torch.isneginf(layer.sink.data[LOCAL_HEADS:]).all()
    assert layer.sink.data_ptr() == kernel_ptr


def test_padded_sink_padding_tail_not_polluted_after_reload():
    """Polluted padding must be restored to -inf by the live update itself."""
    layer = _SinkLayer(use_helper=True)
    layer.load_weights(_checkpoint_a(), 0, LOCAL_HEADS)
    with torch.no_grad():
        layer.sink.data[LOCAL_HEADS:] = 0.0

    _live_reload(layer, 0, LOCAL_HEADS, _checkpoint_b())

    assert torch.isneginf(layer.sink.data[LOCAL_HEADS:]).all()


def test_padded_sink_survives_repeated_reloads():
    """Reloading twice keeps loading through the restored original loader."""
    layer = _SinkLayer(use_helper=True)
    layer.load_weights(_checkpoint_a(), 0, LOCAL_HEADS)
    kernel_ptr = layer.sink.data_ptr()

    _live_reload(layer, 0, LOCAL_HEADS, _checkpoint_b())
    _live_reload(layer, 0, LOCAL_HEADS, _checkpoint_a())

    expected = _expected_local(_checkpoint_a(), 0, LOCAL_HEADS)
    assert torch.equal(layer.sink.data[:LOCAL_HEADS], expected)
    assert torch.isneginf(layer.sink.data[LOCAL_HEADS:]).all()
    assert layer.sink.data_ptr() == kernel_ptr


def test_sink_load_is_buffered_during_live_update():
    """The sink load must be recorded, since the framework replays that buffer.

    Only a loader-mediated load is buffered; a bare ``copy_`` leaves the
    framework with nothing to replay. This is the mechanism the fix restores.
    """
    from vllm.model_executor.model_loader.reload.layerwise import get_layerwise_info

    layer = _SinkLayer(use_helper=True)
    layer.load_weights(_checkpoint_a(), 0, LOCAL_HEADS)

    record_metadata_for_reloading(layer)
    initialize_layerwise_reload(layer)

    # Stop before the layer is finalized so the buffer is still observable.
    info = get_layerwise_info(layer)
    info.loaded_weights.clear()
    info.load_numel = 0
    layer.load_weights([("sink", _checkpoint_b()[0][1])], 0, LOCAL_HEADS)

    assert [name for name, _ in info.loaded_weights] == ["sink"]
    assert info.load_numel > 0

    finalize_layerwise_reload(layer, None)


def test_padded_sink_loads_through_current_loader_when_meta():
    """On a meta parameter the helper must use the current wrapped loader."""
    from vllm.model_executor.model_loader.reload.layerwise import get_layerwise_info

    layer = _SinkLayer(use_helper=True)
    layer.load_weights(_checkpoint_a(), 0, LOCAL_HEADS)

    record_metadata_for_reloading(layer)
    initialize_layerwise_reload(layer)

    params_dict = dict(layer.named_parameters())
    assert params_dict["sink"].is_meta
    assert params_dict["sink"].weight_loader.__name__ == "online_process_loader"

    info = get_layerwise_info(layer)
    info.loaded_weights.clear()
    info.load_numel = 0
    layer.load_weights([("sink", _checkpoint_b()[0][1])], 0, LOCAL_HEADS)

    assert [name for name, _ in info.loaded_weights] == ["sink"]
    finalize_layerwise_reload(layer, None)
