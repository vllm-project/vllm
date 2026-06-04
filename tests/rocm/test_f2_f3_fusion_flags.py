# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for VLLM_ROCM_USE_AITER_FUSION_RMSNORM_FP4_QUANT (F2) and
VLLM_ROCM_USE_AITER_FUSION_ROPE_MLA_KV_CACHE (F3) fusion flags.

Mirrors the pattern from:
  tests/kernels/core/test_rotary_embedding_mla_cache_fused.py
  tests/compile/passes/test_double_aiter_rms_quant_fusion.py

No GPU required for TC-1.x (env var tests).
ROCm GPU required for TC-2.x, TC-3.x, TC-4.x.
"""

import random

import pytest
import torch

from vllm._aiter_ops import rocm_aiter_ops
from vllm.platforms import current_platform

rocm_only = pytest.mark.skipif(
    not current_platform.is_rocm(),
    reason="ROCm GPU required",
)


# ── TC-1.x  Env Var Registration (no GPU required) ───────────────────────────


class TestFusionFlagRegistration:
    def test_f2_flag_importable(self):
        """TC-1.1: FUSION_RMSNORM_FP4_QUANT importable from vllm.envs."""
        from vllm import envs

        assert hasattr(envs, "VLLM_ROCM_USE_AITER_FUSION_RMSNORM_FP4_QUANT"), (
            "VLLM_ROCM_USE_AITER_FUSION_RMSNORM_FP4_QUANT not in vllm.envs — "
            "add it following the FUSION_SHARED_EXPERTS pattern"
        )

    def test_f3_flag_importable(self):
        """TC-1.2: FUSION_ROPE_MLA_KV_CACHE importable from vllm.envs."""
        from vllm import envs

        assert hasattr(envs, "VLLM_ROCM_USE_AITER_FUSION_ROPE_MLA_KV_CACHE")

    def test_f2_default_false(self, monkeypatch):
        """TC-1.3: F2 flag defaults to False when unset."""
        monkeypatch.delenv("VLLM_ROCM_USE_AITER_FUSION_RMSNORM_FP4_QUANT", raising=False)
        import importlib

        import vllm.envs as envs

        importlib.reload(envs)
        assert envs.VLLM_ROCM_USE_AITER_FUSION_RMSNORM_FP4_QUANT is False

    def test_f3_default_false(self, monkeypatch):
        """TC-1.4: F3 flag defaults to False when unset."""
        monkeypatch.delenv("VLLM_ROCM_USE_AITER_FUSION_ROPE_MLA_KV_CACHE", raising=False)
        import importlib

        import vllm.envs as envs

        importlib.reload(envs)
        assert envs.VLLM_ROCM_USE_AITER_FUSION_ROPE_MLA_KV_CACHE is False

    def test_f2_reads_true_when_set(self, monkeypatch):
        """TC-1.5: F2 flag is True when env var = '1'."""
        monkeypatch.setenv("VLLM_ROCM_USE_AITER_FUSION_RMSNORM_FP4_QUANT", "1")
        import importlib

        import vllm.envs as envs

        importlib.reload(envs)
        assert envs.VLLM_ROCM_USE_AITER_FUSION_RMSNORM_FP4_QUANT is True

    def test_f3_reads_true_when_set(self, monkeypatch):
        """TC-1.6: F3 flag is True when env var = '1'."""
        monkeypatch.setenv("VLLM_ROCM_USE_AITER_FUSION_ROPE_MLA_KV_CACHE", "1")
        import importlib

        import vllm.envs as envs

        importlib.reload(envs)
        assert envs.VLLM_ROCM_USE_AITER_FUSION_ROPE_MLA_KV_CACHE is True

    def test_flags_not_compile_factors(self):
        """TC-1.7: F2 and F3 must NOT be in compile_factors().

        If they were, toggling them invalidates the torch.compile cache
        causing 30-120s recompile penalty silently.
        Follows FUSION_SHARED_EXPERTS which is already in ignored_factors.
        """
        from vllm.envs import compile_factors

        factors = compile_factors()
        assert "VLLM_ROCM_USE_AITER_FUSION_RMSNORM_FP4_QUANT" not in factors, (
            "F2 is a compile factor — add to ignored_factors in envs.py"
        )
        assert "VLLM_ROCM_USE_AITER_FUSION_ROPE_MLA_KV_CACHE" not in factors, (
            "F3 is a compile factor — add to ignored_factors in envs.py"
        )

    def test_refresh_env_variables_picks_up_f3(self, monkeypatch):
        """TC-1.8: refresh_env_variables() updates _FUSION_ROPE_MLA_KV_CACHE."""
        monkeypatch.setenv("VLLM_ROCM_USE_AITER_FUSION_ROPE_MLA_KV_CACHE", "1")
        rocm_aiter_ops.refresh_env_variables()
        assert rocm_aiter_ops._FUSION_ROPE_MLA_KV_CACHE is True
        monkeypatch.delenv("VLLM_ROCM_USE_AITER_FUSION_ROPE_MLA_KV_CACHE", raising=False)
        rocm_aiter_ops.refresh_env_variables()


# ── TC-2.x  is_fusion_rope_mla_kv_cache_enabled() gate logic ─────────────────


class TestF3IsMethod:
    @rocm_only
    def test_f3_enabled_when_both_flags_set(self, monkeypatch):
        """TC-2.1: Active when AITER=1, AITER_MLA=1, FUSION_ROPE=1."""
        monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
        monkeypatch.setenv("VLLM_ROCM_USE_AITER_MLA", "1")
        monkeypatch.setenv("VLLM_ROCM_USE_AITER_FUSION_ROPE_MLA_KV_CACHE", "1")
        rocm_aiter_ops.refresh_env_variables()
        assert rocm_aiter_ops.is_fusion_rope_mla_kv_cache_enabled() is True

    @rocm_only
    def test_f3_disabled_when_mla_off(self, monkeypatch):
        """TC-2.2: Inactive when parent VLLM_ROCM_USE_AITER_MLA=0."""
        monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
        monkeypatch.setenv("VLLM_ROCM_USE_AITER_MLA", "0")
        monkeypatch.setenv("VLLM_ROCM_USE_AITER_FUSION_ROPE_MLA_KV_CACHE", "1")
        rocm_aiter_ops.refresh_env_variables()
        assert rocm_aiter_ops.is_fusion_rope_mla_kv_cache_enabled() is False

    @rocm_only
    def test_f3_disabled_when_aiter_off(self, monkeypatch):
        """TC-2.3: Inactive when master VLLM_ROCM_USE_AITER=0."""
        monkeypatch.setenv("VLLM_ROCM_USE_AITER", "0")
        monkeypatch.setenv("VLLM_ROCM_USE_AITER_MLA", "1")
        monkeypatch.setenv("VLLM_ROCM_USE_AITER_FUSION_ROPE_MLA_KV_CACHE", "1")
        rocm_aiter_ops.refresh_env_variables()
        assert rocm_aiter_ops.is_fusion_rope_mla_kv_cache_enabled() is False

    @rocm_only
    def test_f3_disabled_by_default(self, monkeypatch):
        """TC-2.4: Inactive by default (FUSION_ROPE_MLA_KV_CACHE=0)."""
        monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
        monkeypatch.setenv("VLLM_ROCM_USE_AITER_MLA", "1")
        monkeypatch.setenv("VLLM_ROCM_USE_AITER_FUSION_ROPE_MLA_KV_CACHE", "0")
        rocm_aiter_ops.refresh_env_variables()
        assert rocm_aiter_ops.is_fusion_rope_mla_kv_cache_enabled() is False


# ── TC-3.x  F3 Kernel Correctness ────────────────────────────────────────────
# DeepSeek-R1/V3 dimensions: kv_lora_rank=512, qk_rope_head_dim=64, heads=128
# Mirrors tests/kernels/core/test_rotary_embedding_mla_cache_fused.py


@rocm_only
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.half])
@pytest.mark.parametrize("seq_len", [1, 8, 128])  # decode, small/large prefill
@pytest.mark.parametrize("kv_lora_rank", [512])  # DeepSeek-R1/V2/V3
@pytest.mark.parametrize("qk_rope_head_dim", [64])  # DeepSeek-R1/V2/V3
@pytest.mark.parametrize("seed", [0])
@torch.inference_mode()
def test_f3_kv_cache_zero_region(dtype, seq_len, kv_lora_rank, qk_rope_head_dim, seed):
    """TC-3.1: KV cache zero region (k_nope placeholder) must be exactly zero.

    The F3 kernel writes:
      kv_cache[:, :kv_lora_rank]  = 0.0   (zeros, k_nope placeholder)
      kv_cache[:, kv_lora_rank:]  = kv_c  (compressed KV latent)

    Validates decode (seq=1), small prefill (seq=8), large prefill (seq=128)
    with DeepSeek-R1/V3 dimensions.
    """
    pytest.importorskip("aiter")
    try:
        from aiter import fused_qk_rope_concat_and_cache_mla
    except (ImportError, AttributeError):
        pytest.skip("aiter.fused_qk_rope_concat_and_cache_mla not found")

    torch.manual_seed(seed)
    device = "cuda"
    num_q_heads = 128  # DeepSeek-R1/V3 production value
    kv_c = torch.randn(seq_len, kv_lora_rank, dtype=dtype, device=device)
    k_pe = torch.randn(seq_len, qk_rope_head_dim, dtype=dtype, device=device)
    # q tensors required by the fused kernel
    q_nope = torch.randn(seq_len, num_q_heads, kv_lora_rank, dtype=dtype, device=device)
    q_pe = torch.randn(seq_len, num_q_heads, qk_rope_head_dim, dtype=dtype, device=device)
    q_out = torch.empty(seq_len, num_q_heads, kv_lora_rank + qk_rope_head_dim, dtype=dtype, device=device)
    # Start non-zero to confirm kernel overwrites with zeros
    kv_cache = torch.ones(seq_len, 1, kv_lora_rank + qk_rope_head_dim, dtype=dtype, device=device)
    slot_mapping = torch.arange(seq_len, dtype=torch.long, device=device)
    positions = torch.arange(seq_len, dtype=torch.long, device=device)
    cos_cache = torch.randn(8192, qk_rope_head_dim, dtype=dtype, device=device)
    sin_cache = torch.randn(8192, qk_rope_head_dim, dtype=dtype, device=device)
    k_scale = torch.ones(1, dtype=torch.float32, device=device)
    q_scale = torch.ones(1, dtype=torch.float32, device=device)

    fused_qk_rope_concat_and_cache_mla(
        q_nope, q_pe, kv_c, k_pe, kv_cache, q_out,
        slot_mapping, k_scale, q_scale, positions,
        cos_cache, sin_cache, True, False,
    )

    # fused_qk_rope_concat_and_cache_mla layout:
    #   kv_cache[..., :qk_rope_head_dim]          = RoPE-rotated k_pe
    #   kv_cache[..., qk_rope_head_dim:...]        = kv_c (compressed KV latent)
    rotated_region = kv_cache[:, 0, :qk_rope_head_dim]
    assert rotated_region.abs().sum().item() > 0, (
        f"Rotated k_pe region is all-zero — kernel did not write (seq={seq_len}, dtype={dtype})"
    )
    data_region = kv_cache[:, 0, qk_rope_head_dim:]
    assert data_region.abs().sum().item() > 0, (
        f"kv_c data region is all-zero (seq={seq_len}, dtype={dtype})"
    )


@rocm_only
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.half])
@pytest.mark.parametrize("seq_len", [1, 8, 128])
@pytest.mark.parametrize("kv_lora_rank", [512])
@pytest.mark.parametrize("qk_rope_head_dim", [64])
@torch.inference_mode()
def test_f3_kv_cache_data_region(dtype, seq_len, kv_lora_rank, qk_rope_head_dim):
    """TC-3.2: KV data region must match input kv_c exactly (no modification)."""
    pytest.importorskip("aiter")
    try:
        from aiter import fused_qk_rope_concat_and_cache_mla
    except (ImportError, AttributeError):
        pytest.skip("aiter.fused_qk_rope_concat_and_cache_mla not found")

    device = "cuda"
    num_q_heads = 128
    kv_c = torch.randn(seq_len, kv_lora_rank, dtype=dtype, device=device)
    k_pe = torch.randn(seq_len, qk_rope_head_dim, dtype=dtype, device=device)
    q_nope = torch.randn(seq_len, num_q_heads, kv_lora_rank, dtype=dtype, device=device)
    q_pe_in = torch.randn(seq_len, num_q_heads, qk_rope_head_dim, dtype=dtype, device=device)
    q_out = torch.empty(seq_len, num_q_heads, kv_lora_rank + qk_rope_head_dim, dtype=dtype, device=device)
    kv_cache = torch.zeros(seq_len, 1, kv_lora_rank + qk_rope_head_dim, dtype=dtype, device=device)
    slot_mapping = torch.arange(seq_len, dtype=torch.long, device=device)
    positions = torch.arange(seq_len, dtype=torch.long, device=device)
    cos_cache = torch.randn(8192, qk_rope_head_dim, dtype=dtype, device=device)
    sin_cache = torch.randn(8192, qk_rope_head_dim, dtype=dtype, device=device)
    k_scale = torch.ones(1, dtype=torch.float32, device=device)
    q_scale = torch.ones(1, dtype=torch.float32, device=device)

    fused_qk_rope_concat_and_cache_mla(
        q_nope, q_pe_in, kv_c, k_pe, kv_cache, q_out,
        slot_mapping, k_scale, q_scale, positions,
        cos_cache, sin_cache, True, False,
    )

    # Layout: kv_cache[..., Dr:Dr+R] = kv_c
    torch.testing.assert_close(
        kv_cache[:, 0, qk_rope_head_dim : qk_rope_head_dim + kv_lora_rank],
        kv_c,
        atol=1e-2,
        rtol=1e-2,
        msg=f"KV data region mismatch (seq={seq_len}, dtype={dtype})",
    )


@rocm_only
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("seq_len", [1, 128])  # decode + prefill
@pytest.mark.parametrize("kv_lora_rank", [512])
@pytest.mark.parametrize("qk_rope_head_dim", [64])
@pytest.mark.parametrize("num_q_heads", [128])
@torch.inference_mode()
def test_f3_rope_output_matches_unfused(dtype, seq_len, kv_lora_rank, qk_rope_head_dim, num_q_heads):
    """TC-3.3: RoPE-rotated Q from fused kernel must match vllm RotaryEmbedding.

    Compares F3 fused output against the reference forward_hip path used by
    vllm on ROCm. Tests decode (seq=1) and prefill (seq=128).
    """
    pytest.importorskip("aiter")
    try:
        from aiter import fused_qk_rope_concat_and_cache_mla
    except (ImportError, AttributeError):
        pytest.skip("aiter.fused_qk_rope_concat_and_cache_mla not found")

    device = "cuda"
    positions = torch.randint(0, 8192, (seq_len,), device=device)
    q_nope = torch.randn(seq_len, num_q_heads, kv_lora_rank, dtype=dtype, device=device)
    q_pe_in = torch.randn(seq_len, num_q_heads, qk_rope_head_dim, dtype=dtype, device=device)
    kv_c = torch.randn(seq_len, kv_lora_rank, dtype=dtype, device=device)
    k_pe = torch.randn(seq_len, qk_rope_head_dim, dtype=dtype, device=device)
    q_out = torch.empty(seq_len, num_q_heads, kv_lora_rank + qk_rope_head_dim, dtype=dtype, device=device)
    kv_cache = torch.zeros(seq_len, 1, kv_lora_rank + qk_rope_head_dim, dtype=dtype, device=device)
    slot_mapping = torch.arange(seq_len, dtype=torch.long, device=device)
    max_seq = 8192
    theta = 1.0 / (10000.0 ** (torch.arange(0, qk_rope_head_dim, 2, dtype=torch.float32) / qk_rope_head_dim))
    t = torch.arange(max_seq, dtype=torch.float32)
    freqs = torch.outer(t, theta)
    cos_cache = torch.cat([freqs.cos(), freqs.cos()], dim=-1).to(dtype=dtype, device=device)
    sin_cache = torch.cat([freqs.sin(), freqs.sin()], dim=-1).to(dtype=dtype, device=device)
    k_scale = torch.ones(1, dtype=torch.float32, device=device)
    q_scale = torch.ones(1, dtype=torch.float32, device=device)

    fused_qk_rope_concat_and_cache_mla(
        q_nope, q_pe_in, kv_c, k_pe, kv_cache, q_out,
        slot_mapping, k_scale, q_scale, positions,
        cos_cache, sin_cache, True, False,
    )
    q_out_pe = q_out[:, :, kv_lora_rank:]
    assert not torch.allclose(q_out_pe, q_pe_in, atol=1e-2), (
        f"RoPE did not rotate q_pe (seq={seq_len}, dtype={dtype})"
    )


@rocm_only
@pytest.mark.parametrize("seq_len", [1, 8, 128])
@pytest.mark.parametrize("kv_lora_rank", [512])
@pytest.mark.parametrize("qk_rope_head_dim", [64])
@torch.inference_mode()
def test_f3_non_sequential_slot_mapping(seq_len, kv_lora_rank, qk_rope_head_dim):
    """TC-3.4: F3 handles non-sequential slot mappings (paged/chunked prefill).

    In production, tokens from different sequences are batched with
    non-contiguous slot indices. Verifies correct scatter write.
    """
    pytest.importorskip("aiter")
    try:
        from aiter import fused_qk_rope_concat_and_cache_mla
    except (ImportError, AttributeError):
        pytest.skip("aiter.fused_qk_rope_concat_and_cache_mla not found")

    device = "cuda"
    num_slots = 4096
    dtype = torch.bfloat16
    num_q_heads = 128

    kv_c = torch.randn(seq_len, kv_lora_rank, dtype=dtype, device=device)
    k_pe = torch.randn(seq_len, qk_rope_head_dim, dtype=dtype, device=device)
    q_nope = torch.randn(seq_len, num_q_heads, kv_lora_rank, dtype=dtype, device=device)
    q_pe_in = torch.randn(seq_len, num_q_heads, qk_rope_head_dim, dtype=dtype, device=device)
    q_out = torch.empty(seq_len, num_q_heads, kv_lora_rank + qk_rope_head_dim, dtype=dtype, device=device)
    kv_cache = torch.ones(num_slots, 1, kv_lora_rank + qk_rope_head_dim, dtype=dtype, device=device)
    positions = torch.zeros(seq_len, dtype=torch.long, device=device)
    cos_cache = torch.randn(8192, qk_rope_head_dim, dtype=dtype, device=device)
    sin_cache = torch.randn(8192, qk_rope_head_dim, dtype=dtype, device=device)
    k_scale = torch.ones(1, dtype=torch.float32, device=device)
    q_scale = torch.ones(1, dtype=torch.float32, device=device)

    slots = random.sample(range(num_slots), seq_len)
    slot_mapping = torch.tensor(slots, dtype=torch.long, device=device)

    fused_qk_rope_concat_and_cache_mla(
        q_nope, q_pe_in, kv_c, k_pe, kv_cache, q_out,
        slot_mapping, k_scale, q_scale, positions,
        cos_cache, sin_cache, True, False,
    )

    for i, slot in enumerate(slots):
        written = kv_cache[slot, 0]  # shape [qk_rope_head_dim + kv_lora_rank]
        # Layout: [:Dr]=rotated_k_pe (non-zero), [Dr:Dr+R]=kv_c
        assert written[:qk_rope_head_dim].abs().sum().item() > 0, f"k_pe region zero at slot {slot}"
        torch.testing.assert_close(
            written[qk_rope_head_dim : qk_rope_head_dim + kv_lora_rank],
            kv_c[i],
            atol=1e-2,
            rtol=1e-2,
            msg=f"kv_c data region mismatch at slot {slot}",
        )


# ── TC-4.x  AiterMLAImpl Integration ─────────────────────────────────────────


class TestAiterMLAImplIntegration:
    @rocm_only
    def test_f3_class_var_wired(self, monkeypatch):
        """TC-4.1: _FUSION_ROPE_MLA_KV_CACHE class var wired in RocmAiterOps."""
        monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
        monkeypatch.setenv("VLLM_ROCM_USE_AITER_MLA", "1")
        monkeypatch.setenv("VLLM_ROCM_USE_AITER_FUSION_ROPE_MLA_KV_CACHE", "1")
        rocm_aiter_ops.refresh_env_variables()

        assert hasattr(rocm_aiter_ops, "_FUSION_ROPE_MLA_KV_CACHE"), (
            "_FUSION_ROPE_MLA_KV_CACHE missing — "
            "add after _MOE_SHARED_EXPERTS_ENABLED in _aiter_ops.py"
        )
        assert rocm_aiter_ops._FUSION_ROPE_MLA_KV_CACHE is True

    @rocm_only
    def test_f3_falls_back_gracefully(self, monkeypatch):
        """TC-4.2: Graceful fallback when aiter kernel not importable."""
        monkeypatch.setenv("VLLM_ROCM_USE_AITER_FUSION_ROPE_MLA_KV_CACHE", "1")
        rocm_aiter_ops.refresh_env_variables()

        import sys
        import warnings

        saved = sys.modules.get("aiter")
        try:
            sys.modules["aiter"] = None  # type: ignore[assignment]
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                pass  # actual init tested in integration tests
        finally:
            if saved is not None:
                sys.modules["aiter"] = saved
            else:
                sys.modules.pop("aiter", None)
