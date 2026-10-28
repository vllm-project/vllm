# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The `get_language_model` cache must not own the models it keys on.

A strong-keyed module-level dict pins every model ever loaded in the process
for the life of the interpreter. In-process teardown
(``VLLM_ENABLE_V1_MULTIPROCESSING=0``) then keeps the weights -- and the KV
cache aliased onto the model's layers -- resident after engine shutdown.
"""

import gc
import weakref

import pytest
import torch
import torch.nn as nn

from vllm.model_executor.models.interfaces import (
    SupportsMultiModal,
    _language_model_by_module,
)

pytestmark = pytest.mark.cpu_test


class _LanguageModel(nn.Module):
    def embed_input_ids(self, input_ids):  # pragma: no cover - never called
        raise NotImplementedError


class _MultiModalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = _LanguageModel()


def test_cache_is_weak_keyed():
    assert isinstance(_language_model_by_module, weakref.WeakKeyDictionary)


def test_cached_entry_does_not_pin_the_model():
    model = _MultiModalModel()
    _language_model_by_module[model] = model.language_model
    ref = weakref.ref(model)

    del model
    gc.collect()

    assert ref() is None, (
        "the get_language_model cache is keeping the model alive; in-process "
        "engine shutdown will not release its weights or KV cache"
    )


def test_cached_entry_is_dropped_with_its_key():
    model = _MultiModalModel()
    _language_model_by_module[model] = model.language_model
    assert model in _language_model_by_module

    del model
    gc.collect()

    assert not any(isinstance(k, _MultiModalModel) for k in _language_model_by_module)


@pytest.mark.parametrize("marked", [False, True])
@pytest.mark.parametrize("text_model", ["clip", "siglip"])
def test_text_tower_without_loader_hook_is_found(text_model, marked, caplog):
    from vllm.model_executor.models.clip import CLIPTextTransformer
    from vllm.model_executor.models.siglip import SiglipTextTransformer

    cls = CLIPTextTransformer if text_model == "clip" else SiglipTextTransformer
    # Only token embeddings are needed to exercise text tower discovery.
    tower = cls.__new__(cls)
    nn.Module.__init__(tower)
    tower.embeddings = nn.Module()
    tower.embeddings.token_embedding = nn.Embedding(4, 2)
    model = nn.Module()
    model.text_model = tower
    model._language_model_names = ["text_model"] if marked else []

    caplog.clear()
    language_model = SupportsMultiModal.get_language_model(model)

    assert language_model is tower
    input_ids = torch.tensor([0, 1])
    torch.testing.assert_close(
        language_model.embed_input_ids(input_ids),
        tower.embeddings.token_embedding.weight[:2],
    )
    assert not caplog.records


@pytest.mark.parametrize("marked", [False, True])
@pytest.mark.parametrize("method", ["embed_input_ids", "forward"])
def test_language_model_discovery_rejects_noncallable_methods(method, marked):
    model = _MultiModalModel()
    model._language_model_names = ["language_model"] if marked else []
    setattr(model.language_model, method, None)

    with pytest.raises(NotImplementedError, match="No language model found"):
        SupportsMultiModal.get_language_model(model)
