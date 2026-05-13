# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for model adapter weight loading (adapters.py)."""

import pytest
import torch

from vllm.model_executor.models.adapters import _create_pooling_model_cls
from vllm.model_executor.models.utils import AutoWeightsLoader, StageMissingLayer

pytestmark = pytest.mark.cpu_test


class SimpleInnerModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Linear(4, 8, bias=False)
        self.layer0 = torch.nn.Linear(8, 8, bias=False)
        self.layer1 = torch.nn.Linear(8, 8, bias=False)
        self.norm = torch.nn.Linear(8, 4, bias=False)

    def load_weights(self, weights):
        params = dict(self.named_parameters())
        loaded = set()
        for name, tensor in weights:
            if name in params:
                params[name].data.copy_(tensor)
                loaded.add(name)
        return loaded


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = SimpleInnerModel()
        self.lm_head = torch.nn.Linear(8, 16, bias=False)

    def load_weights(self, weights):
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)


class PackedWeightInnerModel(torch.nn.Module):
    """Remaps q_proj/k_proj into a fused qkv_proj (Qwen2/Llama pattern)."""

    def __init__(self):
        super().__init__()
        self.qkv_proj = torch.nn.Linear(4, 16, bias=False)
        self.out = torch.nn.Linear(8, 4, bias=False)

    def load_weights(self, weights):
        params = dict(self.named_parameters())
        loaded = set()
        for name, tensor in weights:
            if name == "q_proj.weight":
                params["qkv_proj.weight"].data[:8].copy_(tensor)
                loaded.add("qkv_proj.weight")
            elif name == "k_proj.weight":
                params["qkv_proj.weight"].data[8:].copy_(tensor)
                loaded.add("qkv_proj.weight")
            elif name in params:
                params[name].data.copy_(tensor)
                loaded.add(name)
        return loaded


class PackedWeightModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = PackedWeightInnerModel()
        self.lm_head = torch.nn.Linear(4, 8, bias=False)

    def load_weights(self, weights):
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)


def _buffer_reusing_iterator(weight_dict):
    """Yield weights through a shared buffer overwritten each step.

    Mimics ``runai_model_streamer`` with ``RUNAI_STREAMER_MEMORY_LIMIT=0``.
    """
    buf = None
    for name, tensor in weight_dict.items():
        if buf is None or buf.numel() < tensor.numel():
            buf = torch.empty(tensor.numel(), dtype=tensor.dtype)
        view = buf[: tensor.numel()].view(tensor.shape)
        view.copy_(tensor)
        yield name, view


def _make_pooling_model(base_cls=SimpleModel):
    PoolingModel = _create_pooling_model_cls(base_cls)
    model = base_cls()
    model.__class__ = PoolingModel
    model.lm_head = StageMissingLayer("output", model.lm_head)
    return model


def _make_reference_weights():
    torch.manual_seed(42)
    return {
        "model.embed.weight": torch.randn(8, 4),
        "model.layer0.weight": torch.randn(8, 8),
        "model.layer1.weight": torch.randn(8, 8),
        "model.norm.weight": torch.randn(4, 8),
        "lm_head.weight": torch.randn(16, 8),
    }


def _make_packed_reference_weights():
    torch.manual_seed(42)
    return {
        "model.q_proj.weight": torch.randn(8, 4),
        "model.k_proj.weight": torch.randn(8, 4),
        "model.out.weight": torch.randn(4, 8),
        "lm_head.weight": torch.randn(8, 4),
    }


def _load_and_compare(model, ref, expected):
    for p in model.parameters():
        p.data.zero_()
    model.load_weights(_buffer_reusing_iterator(ref))
    for name, param in model.named_parameters():
        assert torch.equal(param.data, expected[name]), name


def test_pooling_load_weights_with_buffer_reuse():
    """Ensure ModelForPooling.load_weights works with buffer-reusing iterators."""
    ref = _make_reference_weights()

    ground_truth = SimpleModel()
    ground_truth.load_weights(ref.items())
    expected = {n: p.data.clone() for n, p in ground_truth.named_parameters()}

    _load_and_compare(_make_pooling_model(), ref, expected)


def test_pooling_load_weights_clones_probed_weights():
    """Ensure probed weights survive buffer reuse during packed remapping."""
    ref = _make_packed_reference_weights()

    ground_truth = PackedWeightModel()
    ground_truth.load_weights(ref.items())
    expected = {n: p.data.clone() for n, p in ground_truth.named_parameters()}

    _load_and_compare(_make_pooling_model(PackedWeightModel), ref, expected)
