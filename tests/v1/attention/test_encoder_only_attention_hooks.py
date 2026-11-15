# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hooks that let a model (in-tree or a plugin) own its encoder attention:
`EncoderOnlyAttention(attn_backend=...)`, `FlexAttentionMetadataBuilder.
metadata_cls` and `register_model_config`."""

import pytest
import torch

from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_standard_kv_cache_spec,
    create_vllm_config,
    registered_attention_layers,
)
from vllm.model_executor.layers.attention import encoder_only_attention
from vllm.model_executor.models.config import (
    MODELS_CONFIG_MAP,
    VerifyAndUpdateConfig,
    register_model_config,
)
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.flex_attention import (
    FlexAttentionBackend,
    FlexAttentionMetadata,
    FlexAttentionMetadataBuilder,
)

pytestmark = pytest.mark.cpu_test


def test_encoder_only_attention_takes_a_backend(monkeypatch):
    def no_selection(*args, **kwargs):
        raise AssertionError("a given backend must not be re-selected")

    seen = {}

    def record(self, **kwargs):
        seen.update(kwargs)

    monkeypatch.setattr(encoder_only_attention, "get_attn_backend", no_selection)
    monkeypatch.setattr(encoder_only_attention.Attention, "__init__", record)
    encoder_only_attention.EncoderOnlyAttention(
        num_heads=2, head_size=64, scale=1.0, attn_backend=FlexAttentionBackend
    )
    # Wrapped like a selected backend: non-causal metadata, same impl.
    backend = seen["attn_backend"]
    assert issubclass(backend, FlexAttentionBackend)
    assert backend is encoder_only_attention.create_encoder_only_attention_backend(
        FlexAttentionBackend
    )
    assert seen["attn_type"] == AttentionType.ENCODER_ONLY


def _only_self(b, h, q_idx, kv_idx):
    return q_idx == kv_idx


class _OwnMetadata(FlexAttentionMetadata):
    def get_mask_mod(self):
        return _only_self


class _OwnBuilder(FlexAttentionMetadataBuilder):
    metadata_cls = _OwnMetadata


def test_flex_builder_builds_its_metadata_cls():
    vllm_config = create_vllm_config(max_model_len=256, block_size=16)
    device = torch.device("cpu")
    layer_names = ["layer.0"]
    with registered_attention_layers(
        vllm_config, layer_names, AttentionType.ENCODER_ONLY, None
    ):
        builder = _OwnBuilder(
            create_standard_kv_cache_spec(vllm_config),
            layer_names,
            vllm_config,
            device,
        )
        metadata = builder.build(
            common_prefix_len=0,
            common_attn_metadata=create_common_attn_metadata(
                BatchSpec(seq_lens=[8, 5], query_lens=[8, 5]),
                16,
                device,
                arange_block_indices=True,
            ),
        )
    assert type(metadata) is _OwnMetadata
    # The block mask is built from the subclass's mask.
    assert metadata.block_mask is not None
    assert metadata.block_mask.mask_mod is _only_self


class _HookA(VerifyAndUpdateConfig):
    pass


class _HookB(VerifyAndUpdateConfig):
    pass


def test_register_model_config(monkeypatch):
    monkeypatch.setitem(MODELS_CONFIG_MAP, "PluginArchForTest", _HookA)
    MODELS_CONFIG_MAP.pop("PluginArchForTest")
    register_model_config("PluginArchForTest", _HookA)
    assert MODELS_CONFIG_MAP["PluginArchForTest"] is _HookA
    register_model_config("PluginArchForTest", _HookA)  # idempotent
    with pytest.raises(ValueError, match="already has"):
        register_model_config("PluginArchForTest", _HookB)
