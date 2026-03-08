# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.models import ernie as ernie_models


class _RecordingLoader:
    instances: list["_RecordingLoader"] = []

    def __init__(self, _module, *, skip_prefixes=None, **_kwargs):
        self.skip_prefixes = skip_prefixes or []
        self.mapped_names: list[str] = []
        type(self).instances.append(self)

    def load_weights(self, weights, mapper=None):
        weight_names = [name for name, _ in weights]
        self.mapped_names = (
            mapper.apply_list(weight_names) if mapper is not None else weight_names
        )
        return {"mapped.sentinel"}


def test_ernie_embedding_load_weights_maps_base_checkpoint_names(monkeypatch):
    monkeypatch.setattr(ernie_models, "AutoWeightsLoader", _RecordingLoader)
    _RecordingLoader.instances.clear()

    model = object.__new__(ernie_models.ErnieEmbeddingModel)
    loaded = model.load_weights(
        [("embeddings.LayerNorm.gamma", torch.ones(4, dtype=torch.float16))]
    )

    loader = _RecordingLoader.instances[-1]
    assert loader.skip_prefixes == ["lm_head.", "cls."]
    assert loader.mapped_names == ["model.embeddings.LayerNorm.weight"]
    assert loaded == {"mapped.sentinel"}


def test_ernie_embedding_load_weights_maps_ernie_prefix_and_legacy_suffix(
    monkeypatch,
):
    monkeypatch.setattr(ernie_models, "AutoWeightsLoader", _RecordingLoader)
    _RecordingLoader.instances.clear()

    model = object.__new__(ernie_models.ErnieEmbeddingModel)
    loaded = model.load_weights(
        [("ernie.embeddings.LayerNorm.beta", torch.zeros(4, dtype=torch.float16))]
    )

    loader = _RecordingLoader.instances[-1]
    assert loader.skip_prefixes == ["lm_head.", "cls."]
    assert loader.mapped_names == ["model.embeddings.LayerNorm.bias"]
    assert loaded == {"mapped.sentinel"}


def test_ernie_sequence_classification_load_weights_maps_bert_prefix(
    monkeypatch,
):
    monkeypatch.setattr(ernie_models, "AutoWeightsLoader", _RecordingLoader)
    _RecordingLoader.instances.clear()

    model = object.__new__(ernie_models.ErnieForSequenceClassification)
    loaded = model.load_weights(
        [("bert.embeddings.LayerNorm.gamma", torch.ones(4, dtype=torch.float16))]
    )

    loader = _RecordingLoader.instances[-1]
    assert loader.skip_prefixes == ["cls.", "lm_head."]
    assert loader.mapped_names == ["ernie.embeddings.LayerNorm.weight"]
    assert loaded == {"mapped.sentinel"}


def test_ernie_token_classification_load_weights_maps_legacy_suffix(
    monkeypatch,
):
    monkeypatch.setattr(ernie_models, "AutoWeightsLoader", _RecordingLoader)
    _RecordingLoader.instances.clear()

    model = object.__new__(ernie_models.ErnieForTokenClassification)
    loaded = model.load_weights(
        [("ernie.embeddings.LayerNorm.beta", torch.zeros(4, dtype=torch.float16))]
    )

    loader = _RecordingLoader.instances[-1]
    assert loader.skip_prefixes == ["cls.", "lm_head."]
    assert loader.mapped_names == ["ernie.embeddings.LayerNorm.bias"]
    assert loaded == {"mapped.sentinel"}
