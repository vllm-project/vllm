# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, call, patch

import torch

from vllm.config.attention import AttentionConfig
from vllm.v1.attention.backends.mla.prefill import aiter_flash_attn as mod


class _FakeAiterModule(ModuleType):
    flash_attn_varlen_func: MagicMock


class _FakeMhaModule(ModuleType):
    kimi_k3_fp8_prefill_gfx942: MagicMock
    supports_kimi_k3_fp8_prefill_gfx942: MagicMock


class _FakePerHeadModule(ModuleType):
    dynamic_per_head_quant_fp8: MagicMock


def _fake_aiter_modules() -> dict[str, ModuleType]:
    aiter = _FakeAiterModule("aiter")
    aiter.flash_attn_varlen_func = MagicMock()

    ops = ModuleType("aiter.ops")
    triton = ModuleType("aiter.ops.triton")
    attention = ModuleType("aiter.ops.triton.attention")
    mha = _FakeMhaModule("aiter.ops.triton.attention.mha")
    mha.kimi_k3_fp8_prefill_gfx942 = MagicMock()
    mha.supports_kimi_k3_fp8_prefill_gfx942 = MagicMock(return_value=True)

    quant = ModuleType("aiter.ops.triton.quant")
    per_head = _FakePerHeadModule("aiter.ops.triton.quant.per_head")
    per_head.dynamic_per_head_quant_fp8 = MagicMock()

    return {
        "aiter": aiter,
        "aiter.ops": ops,
        "aiter.ops.triton": triton,
        "aiter.ops.triton.attention": attention,
        "aiter.ops.triton.attention.mha": mha,
        "aiter.ops.triton.quant": quant,
        "aiter.ops.triton.quant.per_head": per_head,
    }


def _make_backend() -> mod.AiterFlashAttnPrefillBackend:
    config = SimpleNamespace(
        model_config=SimpleNamespace(dtype=torch.bfloat16),
        attention_config=AttentionConfig(),
    )
    return mod.AiterFlashAttnPrefillBackend(
        num_heads=12,
        scale=1.0,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        vllm_config=config,
    )


def test_fp8_prefill_has_single_opt_in(monkeypatch):
    monkeypatch.setenv("VLLM_ROCM_KIMI_K3_FP8_PREFILL", "1")
    monkeypatch.setenv("VLLM_ROCM_KIMI_K3_FP8_PREFILL_V", "0")
    monkeypatch.setenv("VLLM_ROCM_KIMI_K3_FP8_PREFILL_MIN_CONTEXT", "9999999")

    with patch.dict(sys.modules, _fake_aiter_modules()):
        backend = _make_backend()

    assert backend._fp8_prefill_enabled
    assert not hasattr(backend, "_fp8_v_enabled")
    assert not hasattr(backend, "_fp8_prefill_min_context")

    metadata = SimpleNamespace(
        chunked_context=SimpleNamespace(context_lens_list=[786432])
    )
    backend.prepare_metadata(metadata)
    assert backend._fp8_prefill_active


def test_fp8_prefill_always_quantizes_v():
    backend = object.__new__(mod.AiterFlashAttnPrefillBackend)
    backend.scale = 1.0
    backend._fp8_prefill_active = True
    backend._fp8_q_cache = None

    q, k, v = MagicMock(), MagicMock(), MagicMock()
    q_fp8, k_fp8, v_fp8 = MagicMock(), MagicMock(), MagicMock()
    q_descale, k_descale, v_descale = MagicMock(), MagicMock(), MagicMock()
    out, lse = MagicMock(), MagicMock()

    backend._quantize_q_once = MagicMock(return_value=(q_fp8, q_descale))
    backend._quantize_per_head = MagicMock(
        side_effect=[(k_fp8, k_descale), (v_fp8, v_descale)]
    )
    backend._fp8_prefill_func = MagicMock(return_value=(out, lse))

    chunk = SimpleNamespace(
        query_start_loc=MagicMock(),
        cu_seq_lens=MagicMock(),
        max_query_len=1,
        max_seq_len=786432,
    )
    fp8_dtype = torch.float8_e4m3fn

    with patch.object(mod.current_platform, "fp8_dtype", return_value=fp8_dtype):
        assert backend.run_prefill_context_chunk(chunk, q, k, v) == (out, lse)

    backend._quantize_q_once.assert_called_once_with(q, "", None, None, False)
    assert backend._quantize_per_head.call_args_list == [
        call(k, fp8_dtype),
        call(v, fp8_dtype),
    ]
    backend._fp8_prefill_func.assert_called_once_with(
        q=q_fp8,
        k=k_fp8,
        v=v_fp8,
        cu_seqlens_q=chunk.query_start_loc,
        cu_seqlens_k=chunk.cu_seq_lens,
        max_seqlen_q=chunk.max_query_len,
        max_seqlen_k=chunk.max_seq_len,
        softmax_scale=backend.scale,
        causal=False,
        descale_q=q_descale,
        descale_k=k_descale,
        descale_v=v_descale,
    )
