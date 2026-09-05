# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for model adapter weight loading (adapters.py)."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors.torch import save_file
from transformers import Gemma3Config, PretrainedConfig, Qwen2Config

from vllm.model_executor.models import adapters as adapters_module
from vllm.model_executor.models.adapters import (
    _create_pooling_model_cls,
    _load_dense_weights,
    _load_st_projector,
    _resolve_num_labels,
    as_seq_cls_model,
)
from vllm.model_executor.models.interfaces import SupportsCrossEncoding
from vllm.model_executor.models.interfaces_base import (
    VllmModelForPooling,
    get_score_type,
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
    ``""`` / ``model.`` + raw name against ``language_model.model.*`` params.
    The pooling prefix probe must consult the mapper so it can early-exit.
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


def test_pooling_prefix_probe_uses_hf_to_vllm_mapper(monkeypatch):
    """Mapper lets the prefix probe early-exit without cloning the full ckpt.

    Checkpoint keys only align after ``hf_to_vllm_mapper``. The probe must use
    the mapped name for membership, keep forwarding original names to the
    parent loader, and stop after the first hit instead of cloning everything.
    """
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

    # First key maps onto a param, so the probe clones once and breaks.
    assert clone_count == 1
    assert clone_count < len(ref)
    # Parent still receives original (unmapped, unprefixed) checkpoint names.
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


def test_load_current_sentence_transformers_dense_module(tmp_path):
    dense_path = tmp_path / "2_Dense"
    dense_path.mkdir()
    (tmp_path / "modules.json").write_text(
        json.dumps(
            [
                {
                    "path": "2_Dense",
                    "type": "sentence_transformers.base.modules.dense.Dense",
                }
            ]
        ),
        encoding="utf-8",
    )
    dense_config = {
        "in_features": 4,
        "out_features": 1,
        "bias": False,
        "activation_function": "torch.nn.modules.linear.Identity",
    }
    (dense_path / "config.json").write_text(
        json.dumps(dense_config),
        encoding="utf-8",
    )
    score_weight = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    save_file({"linear.weight": score_weight}, dense_path / "model.safetensors")

    projector = _load_st_projector(
        SimpleNamespace(
            model=str(tmp_path),
            revision=None,
            hf_token=None,
            head_dtype=torch.float32,
        )
    )

    assert isinstance(projector, torch.nn.Sequential)
    assert isinstance(projector[0], torch.nn.Linear)
    assert isinstance(projector[1], torch.nn.Identity)
    torch.testing.assert_close(projector[0].weight, score_weight)
    torch.testing.assert_close(
        projector(torch.ones(2, 4)),
        torch.full((2, 1), 10.0),
    )


@pytest.mark.parametrize(
    ("configured_bias", "saved_bias"),
    [(True, False), (False, True)],
)
def test_dense_loader_rejects_bias_mismatch(
    tmp_path,
    configured_bias,
    saved_bias,
):
    dense_path = tmp_path / "2_Dense"
    dense_path.mkdir()
    state_dict = {"linear.weight": torch.ones(1, 4)}
    if saved_bias:
        state_dict["linear.bias"] = torch.ones(1)
    save_file(state_dict, dense_path / "model.safetensors")

    loaded = _load_dense_weights(
        torch.nn.Linear(4, 1, bias=configured_bias),
        "2_Dense",
        SimpleNamespace(model=str(tmp_path), revision=None, hf_token=None),
    )

    assert not loaded


class ExistingEmbeddingModel(torch.nn.Module, VllmModelForPooling):
    is_pooling_model = True

    def __init__(self, *, vllm_config, prefix="", **kwargs):
        super().__init__()
        self.backbone = torch.nn.Linear(4, 4, bias=False)
        self.pooler = torch.nn.Linear(4, 2, bias=False)

    def embed_input_ids(self, input_ids):
        return input_ids

    def forward(self, input_ids, positions):
        return self.backbone(input_ids)

    def load_weights(self, weights):
        loaded = set()
        for name, weight in weights:
            if name == "backbone.weight":
                with torch.no_grad():
                    self.backbone.weight.copy_(weight)
                loaded.add(name)
        return loaded


class NativeCrossEncoder(ExistingEmbeddingModel, SupportsCrossEncoding):
    pass


def test_sequence_classification_preserves_native_cross_encoder():
    assert as_seq_cls_model(NativeCrossEncoder) is NativeCrossEncoder


def test_sequence_classification_replaces_existing_embedding_pooler(
    monkeypatch,
    tmp_path,
):
    from vllm.model_executor.layers.pooler import DispatchPooler
    from vllm.model_executor.model_loader.reload import (
        finalize_layerwise_reload,
        initialize_layerwise_reload,
        record_metadata_for_reloading,
    )
    from vllm.model_executor.model_loader.weight_utils import default_weight_loader

    dense_config = {
        "folder": "2_Dense",
        "in_features": 4,
        "out_features": 1,
        "bias": False,
        "activation_function": "torch.nn.modules.linear.Identity",
        "module_output_name": "scores",
    }
    modules_path = tmp_path / "modules.json"
    dense_module = {
        "path": dense_config["folder"],
        "type": "sentence_transformers.base.modules.dense.Dense",
    }
    modules = [
        {
            "path": "",
            "type": "sentence_transformers.base.modules.transformer.Transformer",
        },
        {
            "path": "1_Pooling",
            "type": (
                "sentence_transformers.sentence_transformer.modules.pooling.Pooling"
            ),
        },
        dense_module,
    ]
    modules_path.write_text(json.dumps(modules), encoding="utf-8")
    (tmp_path / "config_sentence_transformers.json").write_text(
        json.dumps(
            {
                "model_type": "CrossEncoder",
                "activation_fn": "torch.nn.Sigmoid",
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "sentence_bert_config.json").write_text(
        json.dumps({"transformer_task": "feature-extraction"}),
        encoding="utf-8",
    )
    pooling_path = tmp_path / "1_Pooling"
    pooling_path.mkdir()
    (pooling_path / "config.json").write_text(
        json.dumps({"pooling_mode": "mean", "include_prompt": True}),
        encoding="utf-8",
    )
    dense_path = tmp_path / dense_config["folder"]
    dense_path.mkdir()
    (dense_path / "config.json").write_text(
        json.dumps(
            {key: value for key, value in dense_config.items() if key != "folder"}
        ),
        encoding="utf-8",
    )

    initial_score = torch.nn.Sequential(torch.nn.Linear(4, 1, bias=False))
    reload_weight = torch.full((1, 4), 2.0)
    with torch.no_grad():
        initial_score[0].weight.zero_()
    model_config = SimpleNamespace(
        model=str(tmp_path),
        revision=None,
        hf_token=None,
        hf_config=Qwen2Config(num_labels=1),
        pooler_config=object(),
        get_hidden_size=lambda: 4,
        head_dtype=torch.float32,
        dtype=torch.float32,
    )
    vllm_config = SimpleNamespace(model_config=model_config, quant_config=None)

    monkeypatch.setattr(
        adapters_module,
        "_load_st_projector",
        lambda *_args, **_kwargs: initial_score,
    )

    loaded_folders = []

    def load_dense(linear, folder, _model_config):
        loaded_folders.append(folder)
        weight_loader = getattr(
            linear.weight,
            "weight_loader",
            default_weight_loader,
        )
        weight_loader(linear.weight, reload_weight)
        return True

    monkeypatch.setattr(adapters_module, "_load_dense_weights", load_dense)
    replacement_pooler = torch.nn.Identity()
    monkeypatch.setattr(
        DispatchPooler,
        "for_seq_cls",
        lambda _pooler_config, *, classifier: replacement_pooler,
    )

    model_cls = as_seq_cls_model(ExistingEmbeddingModel)
    model = model_cls(vllm_config=vllm_config)

    assert get_score_type(model_cls) == "cross-encoder"
    assert model.pooler is replacement_pooler
    assert model.score is initial_score
    assert not any(
        name.startswith("pooler.weight") for name, _ in model.named_parameters()
    )

    loaded = model.load_weights(
        [("backbone.weight", torch.ones_like(model.backbone.weight))]
    )
    assert loaded == {"backbone.weight", "score.0.weight"}
    torch.testing.assert_close(model.score[0].weight, reload_weight)

    from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

    DefaultModelLoader.track_weights_loading(object(), model, loaded)

    record_metadata_for_reloading(model)
    reload_weight.fill_(7.0)
    reloaded_dense_path = tmp_path / "head"
    reloaded_dense_path.mkdir()
    (reloaded_dense_path / "config.json").write_text(
        (dense_path / "config.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    dense_module["path"] = "head"
    modules_path.write_text(json.dumps(modules), encoding="utf-8")
    initialize_layerwise_reload(model)
    reloaded = model.load_weights([])
    finalize_layerwise_reload(model, model_config)

    assert reloaded == {"score.0.weight"}
    assert loaded_folders == ["2_Dense", "head"]
    torch.testing.assert_close(model.score[0].weight, reload_weight)


def test_sentence_transformers_cross_encoder_pooling_order():
    from vllm.config import PoolerConfig, set_current_vllm_config
    from vllm.model_executor.layers.pooler.seqwise import pooler_for_classify
    from vllm.pooling_params import PoolingParams
    from vllm.v1.pool.metadata import PoolingMetadata, PoolingStates

    hf_config = Qwen2Config(num_labels=1)
    hf_config.sentence_transformers = {
        "activation_fn": "torch.nn.modules.activation.Sigmoid"
    }
    pooler_config = PoolerConfig(seq_pooling_type="MEAN", use_activation=True)
    model_config = SimpleNamespace(
        head_dtype=torch.float32,
        hf_config=hf_config,
        pooler_config=pooler_config,
    )
    vllm_config = SimpleNamespace(model_config=model_config)

    score = torch.nn.Sequential(
        torch.nn.Linear(3, 1, bias=True),
        torch.nn.Tanh(),
    )
    with torch.no_grad():
        score[0].weight.copy_(torch.tensor([[0.2, -0.4, 0.6]]))
        score[0].bias.copy_(torch.tensor([0.1]))

    hidden_states = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0],
            [-1.0, 0.0, 1.0],
            [3.0, 1.0, -1.0],
        ]
    )
    pooling_metadata = PoolingMetadata(
        prompt_lens=torch.tensor([2, 2], dtype=torch.int32),
        prompt_token_ids=None,
        prompt_token_ids_cpu=None,
        pooling_params=[
            PoolingParams(task="classify", use_activation=True),
            PoolingParams(task="classify", use_activation=True),
        ],
        pooling_states=[PoolingStates(), PoolingStates()],
    )
    pooling_metadata.build_pooling_cursor(
        np.array([2, 2]),
        torch.tensor([2, 2], dtype=torch.int32),
        torch.device("cpu"),
    )

    with set_current_vllm_config(vllm_config):
        pooler = pooler_for_classify(pooler_config, classifier=score)
    actual = pooler(hidden_states, pooling_metadata)

    mean_pooled = torch.stack(
        [hidden_states[:2].mean(dim=0), hidden_states[2:].mean(dim=0)]
    )
    expected = torch.sigmoid(score(mean_pooled))
    torch.testing.assert_close(actual, expected)
