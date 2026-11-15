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
from vllm.v1.attention.backends import flex_attention
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


@pytest.mark.parametrize("model_cls", ["BertModel", "BertPoolingModel"])
def test_bert_layers_receive_the_backend(monkeypatch, dist_init, model_cls):
    from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config
    from vllm.model_executor.models import bert

    seen = []

    def record(self, *args, attn_backend=None, **kwargs):
        torch.nn.Module.__init__(self)
        seen.append(attn_backend)

    monkeypatch.setattr(encoder_only_attention.EncoderOnlyAttention, "__init__", record)
    vllm_config = VllmConfig(
        model_config=ModelConfig(
            "sentence-transformers/all-MiniLM-L6-v2", runner="pooling"
        )
    )
    with set_current_vllm_config(vllm_config):
        getattr(bert, model_cls)(
            vllm_config=vllm_config, prefix="model", attn_backend=FlexAttentionBackend
        )
    # Every encoder layer gets the model's backend.
    assert len(seen) == vllm_config.model_config.hf_config.num_hidden_layers
    assert all(backend is FlexAttentionBackend for backend in seen)


def _only_self(b, h, q_idx, kv_idx):
    return q_idx == kv_idx


class _OwnMetadata(FlexAttentionMetadata):
    def get_mask_mod(self):
        return _only_self


class _OwnBuilder(FlexAttentionMetadataBuilder):
    metadata_cls = _OwnMetadata


def test_flex_builder_builds_its_metadata_cls(monkeypatch):
    # CPU test: build the block mask eagerly, without torch.compile.
    monkeypatch.setattr(
        flex_attention, "create_block_mask_compiled", flex_attention.create_block_mask
    )
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
    # Restored (absent) after the test.
    monkeypatch.delitem(MODELS_CONFIG_MAP, "PluginArchForTest", raising=False)
    register_model_config("PluginArchForTest", _HookA)
    assert MODELS_CONFIG_MAP["PluginArchForTest"] is _HookA
    register_model_config("PluginArchForTest", _HookA)  # idempotent
    with pytest.raises(ValueError, match="already has"):
        register_model_config("PluginArchForTest", _HookB)
