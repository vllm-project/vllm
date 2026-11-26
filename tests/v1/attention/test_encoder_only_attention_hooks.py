# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hooks that let a model (in-tree or a plugin) own its encoder attention:
`EncoderOnlyAttention(attn_backend=...)`, `FlexAttentionMetadataBuilder.
_make_metadata` and `register_model_config`."""

from dataclasses import dataclass

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
from vllm.model_executor.models import register_model_config
from vllm.model_executor.models.config import (
    MODELS_CONFIG_MAP,
    VerifyAndUpdateConfig,
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


def test_given_backend_logs_a_warning(monkeypatch, caplog_vllm):
    from vllm.config import AttentionConfig, VllmConfig, set_current_vllm_config
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    monkeypatch.setattr(
        encoder_only_attention.Attention, "__init__", lambda self, **kwargs: None
    )
    vllm_config = VllmConfig(
        attention_config=AttentionConfig(backend=AttentionBackendEnum.TRITON_ATTN)
    )
    with set_current_vllm_config(vllm_config):
        encoder_only_attention.EncoderOnlyAttention(
            num_heads=2, head_size=64, scale=1.0, attn_backend=FlexAttentionBackend
        )
    # It names the backend that is used and the one it overrides.
    assert "FlexAttentionBackend" in caplog_vllm.text
    assert "--attention-backend: TRITON_ATTN" in caplog_vllm.text


# A model's own backend with the same class name as a stock one, defined in
# another module (here: this test module).
_SameNameFlexBackend = type(
    "FlexAttentionBackend",
    (FlexAttentionBackend,),
    {"__module__": __name__, "__qualname__": "FlexAttentionBackend"},
)


def test_wrapped_backends_keep_their_identity():
    stock = encoder_only_attention.create_encoder_only_attention_backend(
        FlexAttentionBackend
    )
    own = encoder_only_attention.create_encoder_only_attention_backend(
        _SameNameFlexBackend
    )
    # Attention groups are keyed by `full_cls_name()`: the two must not share one.
    assert stock.full_cls_name() == (
        FlexAttentionBackend.__module__,
        "EncoderOnlyAttention_FlexAttentionBackend",
    )
    assert own.full_cls_name() == (
        __name__,
        "EncoderOnlyAttention_FlexAttentionBackend",
    )
    assert stock.full_cls_name() != own.full_cls_name()


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
    def _make_metadata(self, **kwargs):
        return _OwnMetadata(**kwargs)


@dataclass
class _LimitMetadata(FlexAttentionMetadata):
    kv_limit: int = 0

    def get_mask_mod(self):
        kv_limit = self.kv_limit

        def below_limit(b, h, q_idx, kv_idx):
            return kv_idx < kv_limit

        return below_limit


class _LimitBuilder(FlexAttentionMetadataBuilder):
    # Per-builder data the mask reads, set on each instance.
    kv_limit = 0

    def _make_metadata(self, **kwargs):
        return _LimitMetadata(**kwargs, kv_limit=self.kv_limit)


def _build(builder_cls, monkeypatch, configure=None):
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
        builder = builder_cls(
            create_standard_kv_cache_spec(vllm_config),
            layer_names,
            vllm_config,
            device,
        )
        if configure is not None:
            configure(builder)
        return builder.build(
            common_prefix_len=0,
            common_attn_metadata=create_common_attn_metadata(
                BatchSpec(seq_lens=[8, 5], query_lens=[8, 5]),
                16,
                device,
                arange_block_indices=True,
            ),
        )


def test_flex_builder_builds_its_own_metadata(monkeypatch):
    metadata = _build(_OwnBuilder, monkeypatch)
    assert type(metadata) is _OwnMetadata
    # The block mask is built from the subclass's mask.
    assert metadata.block_mask is not None
    assert metadata.block_mask.mask_mod is _only_self


def test_flex_builders_keep_their_own_data(monkeypatch):
    def limit(value):
        def configure(builder):
            builder.kv_limit = value

        return configure

    # Two builders of one class, each with its own data for its mask.
    first = _build(_LimitBuilder, monkeypatch, limit(2))
    second = _build(_LimitBuilder, monkeypatch, limit(5))
    q_idx, kv_idx = torch.tensor(0), torch.tensor(3)
    for metadata, kv_limit in ((first, 2), (second, 5)):
        assert type(metadata) is _LimitMetadata
        assert metadata.kv_limit == kv_limit
        assert metadata.block_mask is not None
        allowed = metadata.block_mask.mask_mod(0, 0, q_idx, kv_idx)
        assert bool(allowed) == (kv_limit > 3)


class _HookA(VerifyAndUpdateConfig):
    pass


class _HookB(VerifyAndUpdateConfig):
    pass


def test_register_model_config(monkeypatch, caplog_vllm):
    # Restored (absent) after the test.
    monkeypatch.delitem(MODELS_CONFIG_MAP, "PluginArchForTest", raising=False)
    register_model_config("PluginArchForTest", _HookA)
    assert MODELS_CONFIG_MAP["PluginArchForTest"] is _HookA
    register_model_config("PluginArchForTest", _HookA)  # idempotent, no warning
    assert "already has the config hook" not in caplog_vllm.text
    # Like `ModelRegistry.register_model`: a later registration overwrites.
    register_model_config("PluginArchForTest", _HookB)
    assert MODELS_CONFIG_MAP["PluginArchForTest"] is _HookB
    assert "already has the config hook _HookA" in caplog_vllm.text
