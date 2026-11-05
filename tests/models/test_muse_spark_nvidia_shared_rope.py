# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch
from torch import nn

import vllm.model_executor.layers.rotary_embedding as rotary_embedding
import vllm.models.muse_spark.nvidia.model as nvidia_model


def test_target_layers_share_fp32_rope_cache(monkeypatch, default_vllm_config):
    # Keep these CPU tables out of the get_rope cache that later GPU tests use.
    monkeypatch.setattr(rotary_embedding, "_ROPE_DICT", {})
    monkeypatch.setattr(nvidia_model, "get_tensor_model_parallel_world_size", lambda: 1)
    for name in (
        "MuseSpark1xQKVOGateParallelLinear",
        "QKVParallelLinear",
        "RowParallelLinear",
        "Attention",
    ):
        monkeypatch.setattr(nvidia_model, name, lambda *args, **kwargs: nn.Identity())
    monkeypatch.setattr(nvidia_model, "supports_fused_kv_insert", lambda attn: True)
    config = SimpleNamespace(
        hidden_size=128,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        layer_types=["sliding_attention"] * 2,
        layer_rope_theta=[5e5] * 2,
        sliding_window=2048,
        rms_norm_eps=1e-5,
        post_norm_eps=1e-5,
        qk_scale_factor=1.0,
    )
    first, second = (
        nvidia_model.MuseSpark1xAttention(
            config=config,
            max_position_embeddings=4096,
            prefix=f"model.layers.{i}.self_attn",
        )
        for i in range(2)
    )

    assert first.rotary_emb is second.rotary_emb
    assert first.rotary_emb.cos_sin_cache.dtype == torch.float32
