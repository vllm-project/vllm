# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for model adapter weight loading (adapters.py)."""

import pytest
import torch
from transformers import Gemma3Config, PretrainedConfig, Qwen2Config

from vllm.model_executor.models.adapters import (
    _create_pooling_model_cls,
    _resolve_num_labels,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    StageMissingLayer,
    WeightsMapper,
)

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


class _LanguageModelInner(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Linear(4, 8, bias=False)
        self.layer0 = torch.nn.Linear(8, 8, bias=False)


class _LanguageModelWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _LanguageModelInner()


class ModelWithWeightsMapper(torch.nn.Module):
    """Stand-in for models whose keys only align after hf_to_vllm_mapper.

    Checkpoint keys like ``model.language_model.*`` never match
    ``""`` / ``model.`` + name against ``language_model.model.*`` params,
    so the generic pooling prefix probe would otherwise scan and clone the
    entire iterator.
    """

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.language_model.": "language_model.model.",
        }
    )

    def __init__(self):
        super().__init__()
        self.language_model = _LanguageModelWrapper()
        self.lm_head = torch.nn.Linear(8, 16, bias=False)

    def load_weights(self, weights):
        params = dict(self.named_parameters())
        loaded = set()
        self.seen_names = []
        for name, tensor in weights:
            self.seen_names.append(name)
            mapped = self.hf_to_vllm_mapper._map_name(name)
            if mapped is not None and mapped in params:
                params[mapped].data.copy_(tensor)
                loaded.add(mapped)
        return loaded


def test_pooling_skips_prefix_probe_when_hf_to_vllm_mapper_present(monkeypatch):
    """Mapper-owned models must not clone the full checkpoint via the probe."""
    ref = {
        "model.language_model.embed.weight": torch.randn(8, 4),
        "model.language_model.layer0.weight": torch.randn(8, 8),
        "model.language_model.extra0.weight": torch.randn(8, 8),
        "model.language_model.extra1.weight": torch.randn(8, 8),
        "model.language_model.extra2.weight": torch.randn(8, 8),
    }

    # Sanity: none of these keys match the generic "" / "model." probe.
    model = _make_pooling_model(ModelWithWeightsMapper)
    params = dict(model.named_parameters())
    for name in ref:
        assert name not in params
        assert f"model.{name}" not in params

    clone_count = 0
    original_clone = torch.Tensor.clone

    def counting_clone(self, *args, **kwargs):
        nonlocal clone_count
        clone_count += 1
        return original_clone(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "clone", counting_clone)

    loaded = model.load_weights(iter(ref.items()))

    assert clone_count == 0
    assert model.seen_names == list(ref.keys())
    assert loaded == {
        "language_model.model.embed.weight",
        "language_model.model.layer0.weight",
    }


def _composite_config(outer_labels=None, inner_labels=None):
    """Build a multimodal-style config whose text_config is a separate object."""
    config = Gemma3Config(text_config={"num_hidden_layers": 1})
    if outer_labels is not None:
        config.num_labels = outer_labels
    if inner_labels is not None:
        config.get_text_config().num_labels = inner_labels
    return config


def test_resolve_num_labels_text_only_config():
    config = Qwen2Config(num_labels=7)
    assert config.get_text_config() is config
    assert _resolve_num_labels(config, config.get_text_config()) == 7


def test_resolve_num_labels_defaults_when_undeclared():
    config = _composite_config()
    assert (
        _resolve_num_labels(config, config.get_text_config())
        == PretrainedConfig().num_labels
    )


def test_resolve_num_labels_declared_on_outer_config():
    """Multimodal checkpoints keep id2label/problem_type on the top-level config."""
    config = _composite_config(outer_labels=20)
    assert config.get_text_config().num_labels == PretrainedConfig().num_labels
    assert _resolve_num_labels(config, config.get_text_config()) == 20


def test_resolve_num_labels_declared_on_text_config():
    """Overrides written into text_config keep working."""
    config = _composite_config(inner_labels=5)
    assert _resolve_num_labels(config, config.get_text_config()) == 5


def test_resolve_num_labels_outer_wins_when_both_declared():
    config = _composite_config(outer_labels=20, inner_labels=5)
    assert _resolve_num_labels(config, config.get_text_config()) == 20
