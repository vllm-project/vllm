# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model caches must not keep models alive after in-process teardown.

Module-level caches can pin every model loaded in the process for the life of
the interpreter. In-process teardown (``VLLM_ENABLE_V1_MULTIPROCESSING=0``)
then keeps weights and other model-owned allocations resident after shutdown.
"""

import gc
import weakref
from types import MethodType

import pytest
import torch
import torch.nn as nn

from vllm.model_executor.models.interfaces import _language_model_by_module
from vllm.model_executor.models.qwen2_5_vl import Qwen2_5_VisionTransformer
from vllm.utils.cache import LRUCache

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


def test_qwen2_5_vl_rope_cache_does_not_pin_model():
    model = object.__new__(Qwen2_5_VisionTransformer)
    model._rope_by_thw_cache = LRUCache(capacity=1024)
    model.get_window_index_thw = MethodType(
        lambda self, t, h, w: (torch.arange(t * h * w), torch.tensor([t * h * w])),
        model,
    )
    model.rotary_pos_emb_thw = MethodType(
        lambda self, t, h, w: (
            torch.zeros(t * h * w, 1, 2),
            torch.zeros(t * h * w, 1, 2),
        ),
        model,
    )

    first = model.get_rope_by_thw(1, 2, 2)
    assert model.get_rope_by_thw(1, 2, 2) is first

    ref = weakref.ref(model)
    del model, first
    gc.collect()

    assert ref() is None, (
        "the Qwen2.5-VL RoPE cache is keeping the model alive after shutdown"
    )
