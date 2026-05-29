# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for MLA KV cache NaN debug instrumentation (VLLM_DEBUG_MLA_CACHE)."""

import pytest
import torch

try:
    from vllm import _custom_ops as ops
except ImportError:
    pytest.skip(
        "Could not import vllm._custom_ops. (pip install -e .)",
        allow_module_level=True,
    )

CUDA_AVAILABLE = torch.accelerator.device_count() >= 1

KV_LORA_RANK = 512
PE_DIM = 64
BLOCK_SIZE = 1
ENTRY_SIZE = KV_LORA_RANK + PE_DIM


def _make_inputs(num_tokens, dtype=torch.bfloat16, device="cuda"):
    kv_c = torch.randn(num_tokens, KV_LORA_RANK, dtype=dtype, device=device)
    k_pe = torch.randn(num_tokens, PE_DIM, dtype=dtype, device=device)
    kv_cache = torch.zeros(
        num_tokens,
        BLOCK_SIZE,
        ENTRY_SIZE,
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)
    scale = torch.tensor([1.0], dtype=torch.float32, device=device)
    nan_count = torch.zeros(1, dtype=torch.int32, device=device)
    return kv_c, k_pe, kv_cache, slot_mapping, scale, nan_count


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="Need CUDA device")
class TestDebugMetaNanDetection:
    def test_detects_nans_in_kv_c(self):
        """NaNs injected into kv_c should increment nan_count."""
        kv_c, k_pe, kv_cache, slot_mapping, scale, nan_count = _make_inputs(32)
        kv_c[0, 0] = float("nan")
        kv_c[5, 10] = float("nan")
        kv_c[31, KV_LORA_RANK - 1] = float("nan")

        ops.concat_and_cache_mla(
            kv_c,
            k_pe,
            kv_cache,
            slot_mapping,
            "fp8",
            scale,
            nan_count,
        )
        torch.accelerator.synchronize()
        assert nan_count.sum().item() >= 3

    def test_detects_nans_in_k_pe(self):
        """NaNs injected into k_pe should increment nan_count."""
        kv_c, k_pe, kv_cache, slot_mapping, scale, nan_count = _make_inputs(16)
        k_pe[0, 0] = float("nan")
        k_pe[15, PE_DIM - 1] = float("nan")

        ops.concat_and_cache_mla(
            kv_c,
            k_pe,
            kv_cache,
            slot_mapping,
            "fp8",
            scale,
            nan_count,
        )
        torch.accelerator.synchronize()
        assert nan_count.sum().item() >= 2

    def test_clean_input_stays_zero(self):
        """No NaNs → nan_count stays zero."""
        kv_c, k_pe, kv_cache, slot_mapping, scale, nan_count = _make_inputs(32)

        ops.concat_and_cache_mla(
            kv_c,
            k_pe,
            kv_cache,
            slot_mapping,
            "fp8",
            scale,
            nan_count,
        )
        torch.accelerator.synchronize()
        assert nan_count[0].item() == 0

    def test_none_nan_count_no_crash(self):
        """Passing None for nan_count should not crash."""
        kv_c, k_pe, kv_cache, slot_mapping, scale, _ = _make_inputs(32)
        kv_c[0, 0] = float("nan")

        ops.concat_and_cache_mla(
            kv_c,
            k_pe,
            kv_cache,
            slot_mapping,
            "fp8",
            scale,
            None,
        )
        torch.accelerator.synchronize()

    def test_accumulates_across_calls(self):
        """nan_count accumulates across multiple kernel calls."""
        kv_c, k_pe, kv_cache, slot_mapping, scale, nan_count = _make_inputs(8)
        kv_c[0, 0] = float("nan")

        ops.concat_and_cache_mla(
            kv_c,
            k_pe,
            kv_cache,
            slot_mapping,
            "fp8",
            scale,
            nan_count,
        )
        first = nan_count.sum().item()
        assert first >= 1

        ops.concat_and_cache_mla(
            kv_c,
            k_pe,
            kv_cache,
            slot_mapping,
            "fp8",
            scale,
            nan_count,
        )
        torch.accelerator.synchronize()
        assert nan_count.sum().item() >= first * 2

    def test_zero_resets_counter(self):
        """nan_count.zero_() should reset the counter."""
        kv_c, k_pe, kv_cache, slot_mapping, scale, nan_count = _make_inputs(8)
        kv_c[0, 0] = float("nan")

        ops.concat_and_cache_mla(
            kv_c,
            k_pe,
            kv_cache,
            slot_mapping,
            "fp8",
            scale,
            nan_count,
        )
        assert nan_count.sum().item() >= 1

        nan_count.zero_()
        assert nan_count[0].item() == 0

        kv_c_clean = torch.randn_like(kv_c)
        ops.concat_and_cache_mla(
            kv_c_clean,
            k_pe,
            kv_cache,
            slot_mapping,
            "fp8",
            scale,
            nan_count,
        )
        torch.accelerator.synchronize()
        assert nan_count[0].item() == 0

    def test_padded_slots_ignored(self):
        """Tokens with slot_mapping=-1 (padding) should not trigger NaN counts."""
        kv_c, k_pe, kv_cache, slot_mapping, scale, nan_count = _make_inputs(4)
        kv_c[2, :] = float("nan")
        slot_mapping[2] = -1

        ops.concat_and_cache_mla(
            kv_c,
            k_pe,
            kv_cache,
            slot_mapping,
            "fp8",
            scale,
            nan_count,
        )
        torch.accelerator.synchronize()
        assert nan_count[0].item() == 0

    def test_auto_dtype_detects_nans(self):
        """kv_cache_dtype='auto' also detects NaNs when debug is enabled."""
        num_tokens = 8
        kv_c = torch.randn(
            num_tokens,
            KV_LORA_RANK,
            dtype=torch.bfloat16,
            device="cuda",
        )
        k_pe = torch.randn(
            num_tokens,
            PE_DIM,
            dtype=torch.bfloat16,
            device="cuda",
        )
        kv_cache = torch.zeros(
            num_tokens,
            BLOCK_SIZE,
            ENTRY_SIZE,
            dtype=torch.bfloat16,
            device="cuda",
        )
        slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device="cuda")
        scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")
        nan_count = torch.zeros(1, dtype=torch.int32, device="cuda")

        kv_c[0, 0] = float("nan")

        ops.concat_and_cache_mla(
            kv_c,
            k_pe,
            kv_cache,
            slot_mapping,
            "auto",
            scale,
            nan_count,
        )
        torch.accelerator.synchronize()
        assert nan_count[0].item() > 0


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="Need CUDA device")
class TestEnvVarGating:
    def test_flag_off(self, monkeypatch):
        """VLLM_DEBUG_MLA_CACHE=0 → evaluates to False."""
        monkeypatch.setenv("VLLM_DEBUG_MLA_CACHE", "0")
        import importlib

        import vllm.envs

        importlib.reload(vllm.envs)
        assert vllm.envs.VLLM_DEBUG_MLA_CACHE is False

    def test_flag_on(self, monkeypatch):
        """VLLM_DEBUG_MLA_CACHE=1 → evaluates to True."""
        monkeypatch.setenv("VLLM_DEBUG_MLA_CACHE", "1")
        import importlib

        import vllm.envs

        importlib.reload(vllm.envs)
        assert vllm.envs.VLLM_DEBUG_MLA_CACHE is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
