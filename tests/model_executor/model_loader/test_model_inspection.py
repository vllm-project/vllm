# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn as nn

from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.model_loader.base_loader import log_backend_summary
from vllm.model_inspection import _format_layer_names, format_backend_summary

pytestmark = pytest.mark.cpu_test


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

    class Proj(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(1))
            self.quant_method = UnquantizedLinearMethod()

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = Proj()
            self.norm = nn.LayerNorm(4)

    model = nn.ModuleDict({"layers": nn.ModuleList(Block() for _ in range(3))})

    assert format_backend_summary(model) == (
        "  layer  | count | backend               | modules\n"
        "  linear |     3 | unquantized [float32] | layers.[0-2].proj"
    )
    assert format_backend_summary(nn.Linear(4, 4)) is None


def test_backend_summary_failure_does_not_fail_the_load():
    """The summary runs on every model load, so a layer it cannot describe
    (e.g. from an out-of-tree plugin) must not raise."""

    class Opaque(nn.Module):
        def __init__(self):
            super().__init__()
            self.quant_method = 0  # not a quant method: has no `__dict__`

    with pytest.raises(TypeError):
        format_backend_summary(Opaque())
    log_backend_summary(Opaque())
