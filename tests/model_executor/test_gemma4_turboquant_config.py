# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from vllm.model_executor.models.config import Gemma4Config
from vllm.v1.attention.backends.registry import AttentionBackendEnum


def _make_vllm_config(cache_dtype: str, backend=None):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(
                head_dim=256,
                global_head_dim=512,
            ),
        ),
        cache_config=SimpleNamespace(cache_dtype=cache_dtype),
        attention_config=SimpleNamespace(backend=backend),
    )


def test_gemma4_heterogeneous_heads_select_turboquant_for_tq_cache():
    vllm_config = _make_vllm_config("turboquant_4bit_nc")

    Gemma4Config.verify_and_update_config(vllm_config)

    assert vllm_config.attention_config.backend is AttentionBackendEnum.TURBOQUANT


def test_gemma4_heterogeneous_heads_select_triton_for_auto_cache():
    vllm_config = _make_vllm_config("auto")

    Gemma4Config.verify_and_update_config(vllm_config)

    assert vllm_config.attention_config.backend is AttentionBackendEnum.TRITON_ATTN


def test_gemma4_heterogeneous_heads_keeps_explicit_backend():
    vllm_config = _make_vllm_config(
        "turboquant_4bit_nc",
        backend=AttentionBackendEnum.FLASH_ATTN,
    )

    Gemma4Config.verify_and_update_config(vllm_config)

    assert vllm_config.attention_config.backend is AttentionBackendEnum.FLASH_ATTN
