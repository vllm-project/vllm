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
import torch.nn as nn

from vllm.model_executor.models.interfaces import _language_model_by_module

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
