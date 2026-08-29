# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model caches must not keep models alive after in-process teardown.

Module-level caches can pin every model loaded in the process for the life of
the interpreter. In-process teardown (``VLLM_ENABLE_V1_MULTIPROCESSING=0``)
then keeps weights and other model-owned allocations resident after shutdown.
"""

import gc
import weakref

import pytest
import torch.nn as nn
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
    Qwen2_5_VLVisionConfig,
)
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig

from vllm.model_executor.models.interfaces import _language_model_by_module
from vllm.model_executor.models.qwen2_5_vl import Qwen2_5_VisionTransformer
from vllm.model_executor.models.qwen3_vl import Qwen3_VisionTransformer
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


def test_qwen2_5_vl_rope_cache_does_not_pin_model(
    monkeypatch: pytest.MonkeyPatch, default_vllm_config
):
    monkeypatch.setattr(
        "vllm.model_executor.models.qwen2_5_vl.is_vit_use_data_parallel",
        lambda: True,
    )
    monkeypatch.setattr(
        "vllm.model_executor.models.qwen2_5_vl.Qwen2_5_VisionPatchMerger",
        lambda **_: nn.Identity(),
    )
    model = Qwen2_5_VisionTransformer(
        Qwen2_5_VLVisionConfig(
            depth=0,
            hidden_size=8,
            intermediate_size=8,
            num_heads=1,
            in_channels=3,
            patch_size=2,
            spatial_merge_size=1,
            temporal_patch_size=1,
            window_size=2,
            out_hidden_size=8,
            fullatt_block_indexes=[],
        )
    )
    assert isinstance(model._rope_by_thw_cache, LRUCache)

    first = model.get_rope_by_thw(1, 2, 2)
    assert model.get_rope_by_thw(1, 2, 2) is first

    ref = weakref.ref(model)
    del model, first
    gc.collect()

    assert ref() is None, (
        "the Qwen2.5-VL RoPE cache is keeping the model alive after shutdown"
    )


def test_qwen3_vl_rope_cache_is_released_with_model(
    monkeypatch: pytest.MonkeyPatch, default_vllm_config
):
    monkeypatch.setattr(
        "vllm.model_executor.models.qwen3_vl.is_vit_use_data_parallel",
        lambda: True,
    )
    monkeypatch.setattr(
        "vllm.model_executor.models.qwen3_vl.Qwen3_VisionPatchMerger",
        lambda **_: nn.Identity(),
    )
    model = Qwen3_VisionTransformer(
        Qwen3VLVisionConfig(
            depth=0,
            hidden_size=8,
            intermediate_size=8,
            num_heads=1,
            in_channels=3,
            patch_size=2,
            spatial_merge_size=1,
            temporal_patch_size=1,
            out_hidden_size=8,
            num_position_embeddings=4,
            deepstack_visual_indexes=[],
        )
    )
    assert isinstance(model._rot_pos_ids_cache, LRUCache)

    first = model.rot_pos_ids(2, 2, 1)
    assert model.rot_pos_ids(2, 2, 1) is first

    model_ref = weakref.ref(model)
    tensor_ref = weakref.ref(first)
    del model, first
    gc.collect()

    assert model_ref() is None
    assert tensor_ref() is None, (
        "the Qwen3-VL RoPE cache outlives the model after shutdown"
    )
