# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Direct weight reload: checkpoint weights are written into the live parameters."""

from types import SimpleNamespace

import pytest
import torch

from vllm.config.weight_transfer import WeightTransferConfig
from vllm.model_executor.model_loader.reload import (
    finish_reload,
    record_metadata_for_reloading,
    start_reload,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader


class _TinyModel(torch.nn.Module):
    """Two linear layers loaded through the real `AutoWeightsLoader`."""

    def __init__(self, bias: bool = False):
        super().__init__()
        self.a = torch.nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
        self.b = torch.nn.Linear(4, 4, bias=bias, dtype=torch.bfloat16)
        for p in self.parameters():
            p.weight_loader = default_weight_loader

    def load_weights(self, weights):
        from vllm.model_executor.models.utils import AutoWeightsLoader

        return AutoWeightsLoader(self).load_weights(weights)


def _payload(model: torch.nn.Module, value: float) -> list[tuple[str, torch.Tensor]]:
    return [(name, torch.full_like(p, value)) for name, p in model.named_parameters()]


def _assert_all(model: torch.nn.Module, value: float) -> None:
    for name, param in model.named_parameters():
        assert torch.equal(param, torch.full_like(param, value)), name


def test_layerwise_is_the_default():
    model = _TinyModel()
    record_metadata_for_reloading(model)
    payload = _payload(model, 1.0)
    start_reload(model)
    assert model.a.weight.is_meta
    model.load_weights(payload)
    finish_reload(model, None)


def test_an_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="unknown reload mode"):
        start_reload(_TinyModel(), "eager")


def test_direct_reload_writes_into_the_live_parameters():
    """The parameters never leave the device and keep their storage, so a
    loader's `copy_` lands where a CUDA graph reads."""
    model = _TinyModel()
    identity = {n: (p, p.data_ptr()) for n, p in model.named_parameters()}

    for value in (3.0, 6.0):
        start_reload(model, "direct")
        assert not any(p.is_meta for p in model.parameters())
        model.load_weights(_payload(model, value))
        finish_reload(model, None)
        _assert_all(model, value)
        for name, param in model.named_parameters():
            assert param is identity[name][0]
            assert param.data_ptr() == identity[name][1]


def test_torchao_models_are_refused():
    """A torchao `load_weights` drives a layerwise reload of its own, which is
    post-processing by definition."""
    model = _TinyModel()
    model._do_torchao_reload = True
    with pytest.raises(RuntimeError, match="torchao"):
        start_reload(model, "direct")


def test_nested_start_is_rejected():
    model = _TinyModel()
    start_reload(model, "direct")
    with pytest.raises(RuntimeError, match="already in progress"):
        start_reload(model, "direct")
    finish_reload(model, None)


def test_finish_completes_the_mode_that_was_started():
    """An engine whose config changes between start and finish must not
    finalize a direct update through the layerwise path or vice versa."""
    model = _TinyModel()
    record_metadata_for_reloading(model)
    start_reload(model, "direct")
    model.load_weights(_payload(model, 3.0))
    finish_reload(model, None)
    assert "_direct_reload_live" not in model.__dict__

    payload = _payload(model, 4.0)
    start_reload(model, "layerwise")
    model.load_weights(payload)
    finish_reload(model, None)
    assert not any(p.is_meta for p in model.parameters())
    _assert_all(model, 4.0)


@pytest.mark.parametrize(
    "rebind",
    [
        lambda m: setattr(m.a.weight, "data", torch.full_like(m.a.weight, 5.0)),
        lambda m: setattr(m.a.weight, "data", m.a.weight.data.t()),  # same ptr
        lambda m: setattr(m.a, "scale", torch.ones(1)),  # buffer
    ],
    ids=["storage", "layout", "buffer"],
)
def test_a_rebinding_load_is_caught_at_finish(rebind):
    """No check before the load can see a model's own loader replace a tensor,
    so the outcome is checked at finish."""
    model = _TinyModel()
    model.a.register_buffer("scale", torch.ones(1), persistent=False)
    start_reload(model, "direct")
    rebind(model)
    with pytest.raises(RuntimeError, match="relocated a\\."):
        finish_reload(model, None)


@pytest.mark.parametrize(
    ("engine_module", "engine_name"),
    [
        ("nccl_engine", "NCCLWeightTransferEngine"),
        ("ipc_engine", "IPCWeightTransferEngine"),
    ],
)
def test_engines_pass_reload_mode_from_their_config(
    monkeypatch, engine_module, engine_name
):
    import importlib

    import vllm.model_executor.model_loader.reload as reload_pkg

    mod = importlib.import_module(f"vllm.distributed.weight_transfer.{engine_module}")
    engine_cls = getattr(mod, engine_name)
    calls: list = []
    monkeypatch.setattr(reload_pkg, "start_reload", lambda m, mode: calls.append(mode))
    monkeypatch.setattr(reload_pkg, "finish_reload", lambda m, cfg: calls.append("fin"))
    fake = SimpleNamespace(
        model=torch.nn.Module(),
        model_config=None,
        config=WeightTransferConfig(reload_mode="direct"),
        _packed_importer=SimpleNamespace(close=lambda: None),
    )
    engine_cls.start_weight_update(fake)
    engine_cls.finish_weight_update(fake)
    assert calls == ["direct", "fin"]
