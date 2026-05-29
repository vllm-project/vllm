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


class TestPhaseTracking:
    """Unit tests for IterationStats NaN phase labeling."""

    @staticmethod
    def _make_output(layers: dict[str, int], ts: float = 1000.0):
        from vllm.v1.engine import EngineCoreOutput

        first = next(iter(layers)) if layers else None
        return EngineCoreOutput(
            request_id="req-test",
            new_token_ids=[1],
            kv_cache_nans_per_layer=layers,
            kv_cache_nan_timestamp=ts,
            kv_cache_nan_first_layer=first,
        )

    @staticmethod
    def _make_stats():
        from vllm.v1.metrics.stats import IterationStats

        return IterationStats()

    @staticmethod
    def _make_req_stats():
        from vllm.v1.metrics.stats import RequestStateStats

        return RequestStateStats(arrival_time=0.0)

    def test_prefill_labeled(self, monkeypatch):
        monkeypatch.setenv("VLLM_DEBUG_MLA_CACHE", "1")
        import importlib

        import vllm.envs

        importlib.reload(vllm.envs)

        stats = self._make_stats()
        output = self._make_output({"model.layers.0.self_attn": 5})
        stats.update_from_output(
            output,
            engine_core_timestamp=0.0,
            is_prefilling=True,
            req_stats=self._make_req_stats(),
            lora_states=None,
            lora_name=None,
        )
        assert ("model.layers.0.self_attn", "prefill") in stats.kv_cache_nans
        assert stats.kv_cache_nans[("model.layers.0.self_attn", "prefill")] == 5

    def test_decode_labeled(self, monkeypatch):
        monkeypatch.setenv("VLLM_DEBUG_MLA_CACHE", "1")
        import importlib

        import vllm.envs

        importlib.reload(vllm.envs)

        stats = self._make_stats()
        output = self._make_output({"model.layers.0.self_attn": 3})
        stats.update_from_output(
            output,
            engine_core_timestamp=0.0,
            is_prefilling=False,
            req_stats=self._make_req_stats(),
            lora_states=None,
            lora_name=None,
        )
        assert ("model.layers.0.self_attn", "decode") in stats.kv_cache_nans
        assert stats.kv_cache_nans[("model.layers.0.self_attn", "decode")] == 3

    def test_origin_set_to_first_seen(self, monkeypatch):
        """Origin is locked to the first (layer, phase) that reports NaNs."""
        monkeypatch.setenv("VLLM_DEBUG_MLA_CACHE", "1")
        import importlib

        import vllm.envs

        importlib.reload(vllm.envs)

        stats = self._make_stats()

        out_prefill = self._make_output(
            {"model.layers.0.self_attn": 2, "model.layers.1.self_attn": 3},
        )
        stats.update_from_output(
            out_prefill,
            engine_core_timestamp=0.0,
            is_prefilling=True,
            req_stats=self._make_req_stats(),
            lora_states=None,
            lora_name=None,
        )
        assert stats.kv_cache_nan_origin == (
            "model.layers.0.self_attn",
            "prefill",
        )

        out_decode = self._make_output(
            {"model.layers.1.self_attn": 1},
        )
        stats.update_from_output(
            out_decode,
            engine_core_timestamp=0.0,
            is_prefilling=False,
            req_stats=self._make_req_stats(),
            lora_states=None,
            lora_name=None,
        )
        # Origin should NOT change -- first seen wins.
        assert stats.kv_cache_nan_origin == (
            "model.layers.0.self_attn",
            "prefill",
        )

    def test_timestamp_recorded(self, monkeypatch):
        monkeypatch.setenv("VLLM_DEBUG_MLA_CACHE", "1")
        import importlib

        import vllm.envs

        importlib.reload(vllm.envs)

        stats = self._make_stats()
        output = self._make_output(
            {"model.layers.0.self_attn": 1},
            ts=12345.678,
        )
        stats.update_from_output(
            output,
            engine_core_timestamp=0.0,
            is_prefilling=True,
            req_stats=self._make_req_stats(),
            lora_states=None,
            lora_name=None,
        )
        key = ("model.layers.0.self_attn", "prefill")
        assert key in stats.kv_cache_nan_first_seen
        assert stats.kv_cache_nan_first_seen[key] == 12345.678


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="Need CUDA device")
class TestKernelToStatsOrigin:
    """Kernel -> IterationStats: verify first-layer origin detection."""

    def test_two_layers_prefill_origin_is_first(self, monkeypatch):
        """Call kernel for two layers with NaNs, verify origin is layer 0."""
        monkeypatch.setenv("VLLM_DEBUG_MLA_CACHE", "1")
        import importlib
        import time

        import vllm.envs

        importlib.reload(vllm.envs)

        from vllm.v1.engine import EngineCoreOutput
        from vllm.v1.metrics.stats import IterationStats, RequestStateStats

        kv_c0, k_pe0, kv_cache0, slots0, scale0, cnt0 = _make_inputs(16)
        kv_c0[0, 0] = float("nan")
        ops.concat_and_cache_mla(kv_c0, k_pe0, kv_cache0, slots0, "fp8", scale0, cnt0)

        kv_c1, k_pe1, kv_cache1, slots1, scale1, cnt1 = _make_inputs(16)
        kv_c1[0, 0] = float("nan")
        ops.concat_and_cache_mla(kv_c1, k_pe1, kv_cache1, slots1, "fp8", scale1, cnt1)
        torch.accelerator.synchronize()

        layer_nans = {
            "model.layers.0.self_attn": int(cnt0[0].item()),
            "model.layers.1.self_attn": int(cnt1[0].item()),
        }
        assert layer_nans["model.layers.0.self_attn"] >= 1
        assert layer_nans["model.layers.1.self_attn"] >= 1

        ts = time.time()
        output = EngineCoreOutput(
            request_id="req-test",
            new_token_ids=[1],
            kv_cache_nans_per_layer=layer_nans,
            kv_cache_nan_timestamp=ts,
            kv_cache_nan_first_layer=next(iter(layer_nans)),
        )

        stats = IterationStats()
        stats.update_from_output(
            output,
            engine_core_timestamp=0.0,
            is_prefilling=True,
            req_stats=RequestStateStats(arrival_time=0.0),
            lora_states=None,
            lora_name=None,
        )

        assert stats.kv_cache_nan_origin == (
            "model.layers.0.self_attn",
            "prefill",
        )
        assert ("model.layers.0.self_attn", "prefill") in stats.kv_cache_nans
        assert ("model.layers.1.self_attn", "prefill") in stats.kv_cache_nans


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
