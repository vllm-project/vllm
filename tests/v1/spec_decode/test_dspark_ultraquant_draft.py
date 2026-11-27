# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Draft selection when the DSpark target KV cache is UltraQuant.

The default draft kernel is AITER FA on the UltraQuant layout, reading a
sink (4 tokens, rounded up to whole KV pages) plus the last 2048 tokens.
"""

from types import SimpleNamespace

import pytest

from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.worker.gpu.spec_decode.dspark.utils import (
    _resolve_draft_kv_cache_dtype,
    _resolve_dspark_attention_backend,
)

pytestmark = pytest.mark.cpu_test


def _draft(model_type: str) -> SimpleNamespace:
    return SimpleNamespace(hf_config=SimpleNamespace(model_type=model_type))


def test_qwen_ultraquant_target_selects_windowed_draft():
    backend = _resolve_dspark_attention_backend(
        _draft("qwen3"),
        None,
        AttentionBackendEnum.ULTRAQUANT,
        "ultraquant_4bit",
    )
    assert backend is AttentionBackendEnum.ROCM_AITER_FA_DSPARK


def test_explicit_draft_backend_is_kept():
    backend = _resolve_dspark_attention_backend(
        _draft("qwen3"),
        AttentionBackendEnum.TRITON_ATTN,
        AttentionBackendEnum.ULTRAQUANT,
        "ultraquant_4bit",
    )
    assert backend is AttentionBackendEnum.TRITON_ATTN


def test_non_ultraquant_target_keeps_platform_default():
    backend = _resolve_dspark_attention_backend(
        _draft("qwen3"),
        None,
        AttentionBackendEnum.ROCM_AITER_FA,
        "fp8",
    )
    assert backend is None


def test_deepseek_draft_keeps_the_target_backend():
    backend = _resolve_dspark_attention_backend(
        _draft("deepseek_v4"),
        None,
        AttentionBackendEnum.ULTRAQUANT,
        "ultraquant_4bit",
    )
    assert backend is AttentionBackendEnum.ULTRAQUANT


def test_ultraquant_target_draft_defaults_to_auto_cache():
    # The measured UltraQuant DSpark config: 4-bit target, unquantized draft.
    assert _resolve_draft_kv_cache_dtype(None, "ultraquant_4bit", False) == "auto"


def test_explicit_draft_kv_cache_dtype_wins():
    assert _resolve_draft_kv_cache_dtype("fp8", "ultraquant_4bit", False) == "fp8"
    assert _resolve_draft_kv_cache_dtype("auto", "fp8", False) == "auto"


def test_deepseek_draft_keeps_the_target_cache_dtype():
    assert _resolve_draft_kv_cache_dtype(None, "ultraquant_4bit", True) is None


def test_fp8_target_draft_inherits_the_target_cache_dtype():
    assert _resolve_draft_kv_cache_dtype(None, "fp8", False) is None
