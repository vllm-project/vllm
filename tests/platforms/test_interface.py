# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from transformers import Qwen3NextConfig

from vllm.config import CacheConfig, ModelConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.platforms import current_platform
from vllm.platforms.interface import Platform

pytestmark = [
    pytest.mark.skip_global_cleanup,
    pytest.mark.skipif(
        not current_platform.is_cuda(), reason="Requires FlashInfer CUDA"
    ),
]


class _LayerwiseKVConfig(Fp8Config):
    def has_layerwise_kv_cache(self):
        return True


@pytest.fixture
def hybrid_config(tmp_path, dist_init, monkeypatch):
    from vllm.v1.attention.backends.flashinfer import FlashInferBackend

    monkeypatch.delenv("VLLM_KV_CACHE_LAYOUT", raising=False)
    Qwen3NextConfig(
        architectures=["Qwen3NextForCausalLM"],
        hidden_size=128,
        head_dim=128,
        num_attention_heads=1,
        num_key_value_heads=1,
        num_hidden_layers=2,
        layer_types=["linear_attention", "full_attention"],
        linear_num_key_heads=1,
        linear_num_value_heads=1,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
    ).save_pretrained(tmp_path)
    config = VllmConfig(
        cache_config=CacheConfig(block_size=64, mamba_cache_mode="align"),
        quant_config=_LayerwiseKVConfig(),
    )
    config.model_config = ModelConfig(
        model=str(tmp_path), max_model_len=128, skip_tokenizer_init=True
    )
    with set_current_vllm_config(config), torch.device("cuda"):
        Attention(
            num_heads=1,
            head_size=128,
            scale=128**-0.5,
            cache_config=config.cache_config,
            attn_backend=FlashInferBackend,
            prefix="model.layers.1.self_attn.attn",
        )
    return config


@pytest.mark.parametrize("skip_layers", [[], ["0"]])
def test_layerwise_hybrid_keeps_backend_block_size(hybrid_config, skip_layers):
    config = hybrid_config
    config.cache_config.kv_cache_dtype_skip_layers = skip_layers

    Platform.update_block_size_for_backend(config)

    assert config.cache_config.block_size == 64
    assert config.cache_config.mamba_block_size == 64
    assert config.cache_config.mamba_page_size_padded is None


@pytest.mark.parametrize("layout", ["LBHNC", "BLHNC"])
def test_layerwise_hybrid_honors_explicit_layout(hybrid_config, layout):
    config = hybrid_config
    config.cache_config.kv_cache_layout = layout

    Platform.update_block_size_for_backend(config)

    if layout == "BLHNC":
        assert config.cache_config.block_size == 64
        assert config.cache_config.mamba_page_size_padded is None
    else:
        assert config.cache_config.block_size > 64
        assert config.cache_config.mamba_page_size_padded is not None


def test_layerwise_hybrid_rejects_unsupported_manager_block_size(hybrid_config):
    hybrid_config.cache_config.block_size = 48

    with pytest.raises(
        ValueError, match="block size 48.*FLASHINFER.*Omit --block-size"
    ):
        Platform.update_block_size_for_backend(hybrid_config)


def test_layerwise_hybrid_without_recipe_keeps_legacy_alignment(hybrid_config):
    config = hybrid_config
    config.quant_config = Fp8Config()

    Platform.update_block_size_for_backend(config)

    assert config.cache_config.block_size > 64
    assert config.cache_config.mamba_page_size_padded is not None
