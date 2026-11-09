# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.rotary_embedding.ernie45_vl_rope import (
    Ernie4_5_VLRotaryEmbedding,
)
from vllm.model_executor.models.ernie45_vl_moe import Ernie4_5_VLMoeAttention

MODEL = "baidu/ERNIE-4.5-VL-28B-A3B-PT"
MAX_POSITION_EMBEDDINGS = 131072


@pytest.mark.parametrize("max_model_len", [32768, 262144])
def test_layers_share_one_mrope_cache(dist_init, monkeypatch, max_model_len):
    monkeypatch.setenv("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")
    config = VllmConfig(
        model_config=ModelConfig(
            model=MODEL, trust_remote_code=True, max_model_len=max_model_len
        )
    )
    with set_current_vllm_config(config):
        first, second = (
            Ernie4_5_VLMoeAttention(
                hidden_size=256,
                num_heads=2,
                num_kv_heads=1,
                rope_parameters={"rope_theta": 500000},
                head_dim=128,
                max_position_embeddings=MAX_POSITION_EMBEDDINGS,
                prefix=f"model.layers.{i}.self_attn",
            )
            for i in range(2)
        )
        reference = Ernie4_5_VLRotaryEmbedding(
            head_size=128,
            rotary_dim=128,
            max_position_embeddings=MAX_POSITION_EMBEDDINGS,
            base=500000,
            is_neox_style=False,
            dtype=torch.get_default_dtype(),
            mrope_section=[22, 22, 20],
        )
    num_rows = max(MAX_POSITION_EMBEDDINGS, max_model_len)

    assert first.rotary_emb is second.rotary_emb
    assert first.rotary_emb.cos_sin_cache.shape[0] == num_rows
    assert torch.equal(
        first.rotary_emb.cos_sin_cache.cpu(), reference.cos_sin_cache[:num_rows].cpu()
    )
