# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn

from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.model_loader import utils as loader_utils
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.utils import log_layer_backends
from vllm.model_inspection import (
    LayerBackend,
    _format_layer_names,
    collect_layer_backends,
    format_layer_backends,
)

pytestmark = pytest.mark.cpu_test


class Proj(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
        self.quant_method = UnquantizedLinearMethod()


def test_layer_names_collapse_indices_and_leaves():
    names = [
        f"model.layers.{i}.self_attn.{proj}"
        for i in range(4)
        for proj in ("qkv_proj", "o_proj")
    ]
    assert _format_layer_names([*names, "lm_head"]) == (
        "model.layers.[0-3].self_attn.{qkv_proj,o_proj}, lm_head"
    )
    # Scattered indices would not fit on a line: show the span and the count.
    scattered = [f"layers.{i}.attn" for i in (3, 7, 11, 15, 19)]
    assert _format_layer_names(scattered) == "layers.[3-19 (5)].attn"


def test_backend_summary_groups_layers_by_backend():
    """Layers that selected the same backend share one line; a model without
    such layers has no summary."""

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = Proj()
            self.norm = nn.LayerNorm(4)

    model = nn.ModuleDict({"layers": nn.ModuleList(Block() for _ in range(3))})

    backends = collect_layer_backends(model)
    assert backends == [
        LayerBackend(
            kind="linear",
            backend="unquantized",
            details=(("w", "float32"),),
            modules=("layers.0.proj", "layers.1.proj", "layers.2.proj"),
        )
    ]
    assert format_layer_backends(backends) == (
        "  layer  | count | backend                 | modules\n"
        "  linear |     3 | unquantized [w=float32] | layers.[0-2].proj"
    )
    assert collect_layer_backends(nn.Linear(4, 4)) == []


def test_backend_summary_failure_does_not_fail_the_load():
    """The summary runs on every model load, so a layer it cannot describe
    (e.g. from an out-of-tree plugin) must not raise."""

    class Opaque(nn.Module):
        def __init__(self):
            super().__init__()
            self.quant_method = 0  # not a quant method: has no `__dict__`

    with pytest.raises(TypeError):
        collect_layer_backends(Opaque())
    log_layer_backends(Opaque(), SimpleNamespace(model="opaque"))
    # Nor may an unexpected model config, as tests pass stand-ins for it.
    log_layer_backends(nn.ModuleDict({"proj": Proj()}), SimpleNamespace())


def test_backends_are_logged_when_loading_fails(monkeypatch):
    """Selections are made before the weights load, so a failed load still
    reports them, and the original error propagates unchanged."""

    class FailingLoader(BaseModelLoader):
        def download_model(self, model_config): ...

        def create_model(self, vllm_config, model_config, prefix=""):
            return nn.ModuleDict({"proj": Proj()})

        def load_weights(self, model, model_config):
            raise RuntimeError("out of memory")

    logger = Mock()
    monkeypatch.setattr(loader_utils, "logger", logger)
    vllm_config = SimpleNamespace(
        device_config=SimpleNamespace(device="cpu"),
        load_config=SimpleNamespace(device=None),
    )
    model_config = SimpleNamespace(dtype=torch.float32, model="org/toy")

    with pytest.raises(RuntimeError, match="out of memory") as error:
        FailingLoader(vllm_config.load_config).load_model(vllm_config, model_config)

    assert error.value.__context__ is None
    (logged,) = [
        call.args
        for call in logger.info_once.call_args_list
        if call.args[0].startswith("Backends selected")
    ]
    assert logged[1:4] == ("ModuleDict", "org/toy", " before loading failed")
    assert "unquantized [w=float32] | proj" in logged[4]
