# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import regex as re
import torch

from vllm.config import CacheConfig, ModelConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.models.hy_v4_flash.nvidia.attention import HYV4FlashMLAAttention
from vllm.models.hy_v4_flash.nvidia.swa import FlashSWABackend
from vllm.platforms import current_platform
from vllm.transformers_utils.configs.hy_v4 import HYV4Config
from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager
from vllm.v1.kv_cache_interface import MLAAttentionSpec

NUM_HEADS, NOPE, ROPE, V_DIM, KV_LORA_RANK = 8, 128, 64, 128, 512
SLIDING_WINDOW = 512


def _flash_swa_layer(cache_config: CacheConfig) -> HYV4FlashMLAAttention:
    kv_b_proj = ColumnParallelLinear(
        KV_LORA_RANK, NUM_HEADS * (NOPE + V_DIM), bias=False
    )
    return HYV4FlashMLAAttention(
        num_heads=NUM_HEADS,
        scale=1.0,
        qk_nope_head_dim=NOPE,
        qk_rope_head_dim=ROPE,
        v_head_dim=V_DIM,
        q_lora_rank=512,
        kv_lora_rank=KV_LORA_RANK,
        kv_b_proj=kv_b_proj,
        cache_config=cache_config,
        prefix="model.layers.0.self_attn.attn",
        attn_backend=FlashSWABackend,
        use_sparse=True,
        sliding_window=SLIDING_WINDOW,
    )


@pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="HY V4 Flash sparse MLA runs on SM100",
)
@pytest.mark.parametrize("kv_cache_dtype", ["fp8", "fp8_e4m3", "fp8_ds_mla"])
def test_flash_swa_layer_serves_fp8_ds_mla(
    dist_init, workspace_init, tmp_path, kv_cache_dtype: str
):
    """HY V4 Flash window layers run the paged fp8_ds_mla cache that base
    sliding-window MLA rejects, so fp8 aliases must yield a sliding-window
    fp8_ds_mla KV cache."""
    hf_config = HYV4Config(
        hidden_size=256,
        num_hidden_layers=1,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_HEADS,
        q_lora_rank=128,
        max_position_embeddings=8192,
        layer_types=["deepseek_sparse_attention"],
    )
    hf_config.architectures = ["HYV4ForCausalLM"]
    hf_config.save_pretrained(tmp_path)
    vllm_config = VllmConfig(
        model_config=ModelConfig(
            model=str(tmp_path), dtype="bfloat16", skip_tokenizer_init=True
        ),
        cache_config=CacheConfig(cache_dtype=kv_cache_dtype),
    )
    with set_current_vllm_config(vllm_config), torch.device("cuda"):
        layer = _flash_swa_layer(vllm_config.cache_config)

    layer_config = layer.get_kv_cache_config(vllm_config)
    assert layer_config.manager_cls is SlidingWindowManager
    assert isinstance(layer_config.spec, MLAAttentionSpec)
    assert layer_config.spec.cache_dtype_str == "fp8_ds_mla"


def test_flash_swa_layer_rejects_nvfp4_ds_mla(dist_init, default_vllm_config):
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Sliding-window MLA does not support kv_cache_dtype=nvfp4_ds_mla."
        ),
    ):
        _flash_swa_layer(CacheConfig(cache_dtype="nvfp4_ds_mla"))
