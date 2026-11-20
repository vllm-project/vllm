# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3's fused prefill runs a plain fp8 KV cache only with an fp8 query.

Without one the engine used to start, report ready, and then fail at the first
prefill. Building the attention layer must refuse the configuration instead.
"""

import pytest
import torch

from tests.v1.attention.utils import create_vllm_config
from vllm.config import set_current_vllm_config
from vllm.model_executor.layers.attention import mla_attention
from vllm.models.kimi_k3.nvidia.mla import MultiHeadLatentAttention
from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig
from vllm.utils.torch_utils import set_default_torch_dtype
from vllm.v1.attention.backends.mla.prefill.registry import MLAPrefillBackendEnum


@pytest.fixture
def fresh_prefill_query_support():
    # Cached per process from whichever config is current on the first call.
    mla_attention.backend_supports_prefill_query_quantization.cache_clear()
    yield
    mla_attention.backend_supports_prefill_query_quantization.cache_clear()


@pytest.mark.parametrize(
    ("use_prefill_query_quantization", "mla_prefill_backend"),
    [
        # The fp8 cache alone.
        (False, None),
        # The flag with a prefill backend that cannot take an fp8 query.
        (True, MLAPrefillBackendEnum.FLASH_ATTN),
    ],
)
def test_fp8_kv_cache_without_fp8_prefill_query_fails_at_init(
    dist_init,
    fresh_prefill_query_support,
    use_prefill_query_quantization: bool,
    mla_prefill_backend: MLAPrefillBackendEnum | None,
):
    # DeepSeek-V2-Lite has K3's MLA head dims, which prefill backend selection
    # reads from the model config.
    vllm_config = create_vllm_config(
        model_name="deepseek-ai/DeepSeek-V2-Lite", dtype=torch.bfloat16
    )
    vllm_config.cache_config.cache_dtype = "fp8"
    attention_config = vllm_config.attention_config
    attention_config.use_prefill_query_quantization = use_prefill_query_quantization
    attention_config.mla_prefill_backend = mla_prefill_backend

    with (
        set_current_vllm_config(vllm_config),
        set_default_torch_dtype(torch.bfloat16),
        pytest.raises(
            ValueError,
            match=r"use_prefill_query_quantization.*mla_prefill_backend.*TRTLLM_RAGGED",
        ),
    ):
        MultiHeadLatentAttention(
            config=KimiLinearConfig(),
            hidden_size=256,
            num_heads=2,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,
            q_lora_rank=64,
            kv_lora_rank=512,
            cache_config=vllm_config.cache_config,
            prefix="model.layers.0.self_attn",
        )
