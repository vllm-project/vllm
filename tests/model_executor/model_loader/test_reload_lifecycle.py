# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`weight_update_active` tells a model's `load_weights` whether it runs in
a weight update, where one call may hold only part of the checkpoint, or in a
cold start, where completeness checks must run."""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.model_loader.reload import (
    finish_reload,
    record_metadata_for_reloading,
    start_reload,
    weight_update_active,
)
from vllm.model_executor.model_loader.reload.layerwise import max_layer_reload_bytes
from vllm.model_executor.model_loader.utils import process_weights_after_loading
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

_MODEL_CONFIG = SimpleNamespace(
    word_embeddings_untied_by_checkpoint=False, quantization=None, dtype=torch.float32
)
_DEVICE = torch.device("cpu")


class _TinyModel(torch.nn.Module):
    """Records `weight_update_active` at every weight it loads."""

    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(8, 4)
        self.proj = torch.nn.Linear(4, 4, bias=False)
        self.observed: list[bool] = []

    def load_weights(self, weights):
        params = dict(self.named_parameters())
        for name, weight in weights:
            self.observed.append(weight_update_active())
            param = params[name]
            getattr(param, "weight_loader", default_weight_loader)(param, weight)
        return set(params)


def _checkpoint(scale: float = 1.0) -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(0)
    return {
        "embed.weight": torch.rand(8, 4, generator=gen) * scale,
        "proj.weight": torch.rand(4, 4, generator=gen) * scale,
    }


@pytest.mark.parametrize("mode", ["direct", "layerwise"], ids=["direct", "layerwise"])
def test_active_only_between_start_and_finish(monkeypatch, mode):
    monkeypatch.setenv("VLLM_RELOAD_MODE", mode)
    model = _TinyModel()
    record_metadata_for_reloading(model)  # as the loader does at cold start
    model.load_weights(_checkpoint().items())
    process_weights_after_loading(model, _MODEL_CONFIG, _DEVICE)
    assert model.observed == [False, False]  # cold start keeps its checks
    assert not weight_update_active()

    model.observed.clear()
    start_reload(model)
    assert weight_update_active()
    model.load_weights(_checkpoint(1.01).items())
    assert model.observed == [True, True]
    finish_reload(model, _MODEL_CONFIG, _DEVICE)
    assert not weight_update_active()

    torch.testing.assert_close(model.proj.weight, _checkpoint(1.01)["proj.weight"])


class _Outer(torch.nn.Module):
    """A wrapper like Step5ForConditionalGeneration around its language model."""

    def __init__(self):
        super().__init__()
        self.language_model = _TinyModel()

    def load_weights(self, weights):
        return self.language_model.load_weights(
            (name.removeprefix("language_model."), w) for name, w in weights
        )


@pytest.mark.parametrize("mode", ["direct", "layerwise"], ids=["direct", "layerwise"])
def test_nested_module_sees_the_update(monkeypatch, mode):
    monkeypatch.setenv("VLLM_RELOAD_MODE", mode)
    model = _Outer()
    record_metadata_for_reloading(model)
    model.load_weights(
        (f"language_model.{name}", w) for name, w in _checkpoint().items()
    )
    process_weights_after_loading(model, _MODEL_CONFIG, _DEVICE)
    inner = model.language_model
    inner.observed.clear()

    start_reload(model)
    # one weight per call, as a bucketed update sends them
    for name, w in _checkpoint(1.01).items():
        model.load_weights([(f"language_model.{name}", w)])
    finish_reload(model, _MODEL_CONFIG, _DEVICE)

    assert inner.observed == [True, True]
    assert not weight_update_active()


class _CPUOffloadedEngram(torch.nn.Module):
    """A host table whose loader and post-load hook must keep its storage."""

    _skip_layerwise_reload = True

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(8, 4), requires_grad=False)
        self.weight_scale_inv = torch.nn.Parameter(
            torch.zeros(8, 1), requires_grad=False
        )
        self.loads = 0
        self.finalized: list[dict[str, torch.Tensor]] = []
        for param in self.parameters():
            param.weight_loader = self.weight_loader

    def weight_loader(self, param, loaded_weight):
        self.loads += 1
        param.data.copy_(loaded_weight)

    def process_weights_after_loading(self):
        self.finalized.append(
            {name: param.clone() for name, param in self.named_parameters()}
        )


@pytest.mark.parametrize("mode", ["layerwise", "direct"])
def test_cpu_offloaded_table_keeps_storage_and_loader_across_updates(monkeypatch, mode):
    """A -> B -> A updates load host weights once and still finalize every layer."""
    monkeypatch.setenv("VLLM_RELOAD_MODE", mode)
    table = _CPUOffloadedEngram()
    linear = torch.nn.Linear(4, 4, bias=False)
    model = torch.nn.Sequential(table, linear)
    record_metadata_for_reloading(model)
    assert max_layer_reload_bytes(table) == 0
    assert max_layer_reload_bytes(model) == linear.weight.nbytes
    process_weights_after_loading(model, _MODEL_CONFIG, _DEVICE)
    table.finalized.clear()
    originals = {
        name: (param, param.data_ptr(), param.weight_loader)
        for name, param in table.named_parameters()
    }
    linear_weight = linear.weight

    for update, value in enumerate((1, 2, 1), start=1):
        start_reload(model)
        if mode == "layerwise":
            assert linear.weight.is_meta
            assert linear.weight.weight_loader.__name__ == "online_process_loader"

        expected = {}
        for name, param in table.named_parameters():
            assert param.device.type == "cpu"
            assert param.weight_loader is originals[name][2]
            expected[name] = torch.full_like(param, value)
            calls_before = table.loads
            param.weight_loader(param, expected[name])
            assert table.loads == calls_before + 1
            assert torch.equal(param, expected[name])
        new_linear_weight = torch.full_like(linear_weight, value)
        getattr(linear.weight, "weight_loader", default_weight_loader)(
            linear.weight, new_linear_weight
        )
        finish_reload(model, _MODEL_CONFIG, _DEVICE)

        assert table.loads == update * len(originals)
        assert len(table.finalized) == update
        for name, param in table.named_parameters():
            original, pointer, loader = originals[name]
            assert param is original
            assert param.data_ptr() == pointer
            assert param.weight_loader is loader
            assert torch.equal(param, expected[name])
            assert torch.equal(table.finalized[-1][name], expected[name])
        assert linear.weight is linear_weight
        assert torch.equal(linear.weight, new_linear_weight)
