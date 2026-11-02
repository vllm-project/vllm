# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch
from torch import nn

import vllm.models.muse_spark.amd.model as amd_model


def test_target_layers_share_fp32_rope_cache(monkeypatch, default_vllm_config):
    monkeypatch.setattr(amd_model, "get_tensor_model_parallel_world_size", lambda: 1)
    for name in ("MuseSpark1xQKVOGateParallelLinear", "RowParallelLinear", "Attention"):
        monkeypatch.setattr(amd_model, name, lambda *args, **kwargs: nn.Identity())
    monkeypatch.setattr(amd_model, "supports_fused_kv_insert", lambda attn: True)
    config = SimpleNamespace(
        hidden_size=128,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        layer_types=["sliding_attention"] * 2,
        layer_rope_theta=[5e5] * 2,
        sliding_window=2048,
        rms_norm_eps=1e-5,
        qk_scale_factor=1.0,
    )
    first, second = (
        amd_model.MuseSpark1xAttention(
            config=config,
            max_position_embeddings=4096,
            prefix=f"model.layers.{i}.self_attn",
        )
        for i in range(2)
    )

    assert first.rotary_emb is second.rotary_emb
    assert first.rotary_emb.cos_sin_cache.dtype == torch.float32
