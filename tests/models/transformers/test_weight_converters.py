# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for how the backend applies Transformers' `WeightConverter`s."""

from types import SimpleNamespace

import torch
import torch.nn as nn
from transformers.core_model_loading import (
    Chunk,
    Concatenate,
    MergeModulelist,
    WeightConverter,
)

from vllm.model_executor.models.transformers.base import Base
from vllm.model_executor.models.utils import WeightsMapper


class FakeBackend(nn.Module):
    """Holds the parameters named as the backend would, and its converters."""

    def __init__(self, param_names: list[str], weight_converters: list):
        super().__init__()
        for name in param_names:
            module = self
            *parents, leaf = name.split(".")
            for parent in parents:
                if not hasattr(module, parent):
                    module.add_module(parent, nn.Module())
                module = getattr(module, parent)
            module.register_parameter(leaf, nn.Parameter(torch.empty(0)))
        self.weight_converters = weight_converters
        self.config = SimpleNamespace(get_text_config=lambda: self.config)

    _convert_weights = Base._convert_weights


def convert(backend, weights, stack_mapper=None):
    stack_mapper = stack_mapper or WeightsMapper()
    return dict(backend._convert_weights(iter(weights.items()), stack_mapper))


def test_split():
    kv = torch.arange(8.0).reshape(4, 2)
    backend = FakeBackend(
        ["model.layers.0.attn.k_proj.weight", "model.layers.0.attn.v_proj.weight"],
        [
            WeightConverter(
                "attn.kv_proj.weight",
                ["attn.k_proj.weight", "attn.v_proj.weight"],
                operations=[Chunk(dim=0)],
            )
        ],
    )
    other = torch.ones(1)
    out = convert(
        backend,
        {"model.layers.0.attn.kv_proj.weight": kv, "model.norm.weight": other},
    )
    k, v = kv.chunk(2)
    assert out.keys() == {
        "model.layers.0.attn.k_proj.weight",
        "model.layers.0.attn.v_proj.weight",
        "model.norm.weight",
    }
    torch.testing.assert_close(out["model.layers.0.attn.k_proj.weight"], k)
    torch.testing.assert_close(out["model.layers.0.attn.v_proj.weight"], v)
    assert out["model.norm.weight"] is other


def test_split_into_stacked_params():
    # The fusers stack k_proj and v_proj into a parameter the converter doesn't name
    stack_mapper = WeightsMapper(
        orig_to_new_stacked={".k_proj": (".kv_proj", "k"), ".v_proj": (".kv_proj", "v")}
    )
    backend = FakeBackend(
        ["model.attn.kv_proj.weight"],
        [
            WeightConverter(
                "attn.key_value.weight",
                ["attn.k_proj.weight", "attn.v_proj.weight"],
                operations=[Chunk(dim=0)],
            )
        ],
    )
    out = convert(
        backend, {"model.attn.key_value.weight": torch.zeros(4, 2)}, stack_mapper
    )
    assert out.keys() == {"model.attn.k_proj.weight", "model.attn.v_proj.weight"}


def test_merge_waits_for_all_sources():
    backend = FakeBackend(
        ["model.attn.qkv_proj.weight"],
        [
            WeightConverter(
                ["attn.q_proj.weight", "attn.k_proj.weight", "attn.v_proj.weight"],
                "attn.qkv_proj.weight",
                operations=[Concatenate(dim=0)],
            )
        ],
    )
    q, k, v = torch.zeros(2, 2), torch.ones(1, 2), torch.full((1, 2), 2.0)
    stream = backend._convert_weights(
        iter(
            [
                ("model.attn.q_proj.weight", q),
                ("model.attn.k_proj.weight", k),
                ("model.attn.v_proj.weight", v),
            ]
        ),
        WeightsMapper(),
    )
    ((name, qkv),) = list(stream)
    assert name == "model.attn.qkv_proj.weight"
    torch.testing.assert_close(qkv, torch.cat([q, k, v]))


def test_converter_without_matching_param_is_skipped():
    # e.g. experts which vLLM loads into its own MoE layers instead
    backend = FakeBackend(
        ["model.mlp.experts.w13_weight"],
        [
            WeightConverter(
                "mlp.experts.*.gate_proj.weight",
                "mlp.experts.gate_up_proj",
                operations=[MergeModulelist(dim=0)],
            )
        ],
    )
    weights = {
        f"model.mlp.experts.{i}.gate_proj.weight": torch.full((1, 2), float(i))
        for i in range(2)
    }
    out = convert(backend, weights)
    assert out.keys() == weights.keys()
    for name, tensor in weights.items():
        assert out[name] is tensor
