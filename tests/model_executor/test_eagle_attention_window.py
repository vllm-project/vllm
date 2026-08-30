# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch
from transformers import PretrainedConfig

from vllm.model_executor.layers.attention.attention import Attention
from vllm.model_executor.models.llama_eagle3 import _configure_draft_attention_window
from vllm.v1.attention.backend import AttentionType
from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec


class _TestAttentionBackend:
    @staticmethod
    def customize_spec(spec):
        return spec

    @staticmethod
    def get_supported_kernel_block_sizes():
        return [128]

    @staticmethod
    def is_mla():
        return False


def _get_cache_spec(*, retain_full_kv_cache: bool):
    layer = SimpleNamespace(
        attn_type=AttentionType.DECODER,
        kv_cache_dtype="auto",
        kv_cache_torch_dtype=torch.float16,
        head_size=128,
        head_size_v=128,
        num_kv_heads=8,
        sliding_window=32768,
        retain_full_kv_cache=retain_full_kv_cache,
        attn_backend=_TestAttentionBackend,
    )
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(
            block_size=128,
            skip_page_size_padded=None,
        )
    )
    return Attention.get_kv_cache_spec(layer, vllm_config)


def test_eagle3_draft_attention_window_configures_all_layers():
    config = PretrainedConfig(num_hidden_layers=2)
    retain_full_kv_cache = _configure_draft_attention_window(config, 32768)

    assert config.sliding_window == 32768
    assert config.layer_types == ["sliding_attention", "sliding_attention"]
    assert retain_full_kv_cache


def test_eagle3_draft_attention_window_retains_full_kv_cache():
    cache_spec = _get_cache_spec(retain_full_kv_cache=True)

    assert isinstance(cache_spec, FullAttentionSpec)
    assert cache_spec.sliding_window == 32768


def test_default_sliding_attention_keeps_windowed_kv_cache():
    cache_spec = _get_cache_spec(retain_full_kv_cache=False)

    assert isinstance(cache_spec, SlidingWindowSpec)
    assert cache_spec.sliding_window == 32768


def test_eagle3_draft_attention_window_none_preserves_config():
    config = PretrainedConfig(num_hidden_layers=1)
    config.sliding_window = None
    config.layer_types = ["full_attention"]

    assert not _configure_draft_attention_window(config, None)
    assert config.sliding_window is None
    assert config.layer_types == ["full_attention"]
