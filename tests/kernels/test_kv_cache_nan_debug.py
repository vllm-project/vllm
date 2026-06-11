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


def _run_kernel(kv_c, k_pe, kv_cache, slot_mapping, scale, nan_count, dtype="fp8"):
    ops.concat_and_cache_mla(
        kv_c,
        k_pe,
        kv_cache,
        slot_mapping,
        dtype,
        scale,
        nan_count,
    )
    torch.accelerator.synchronize()
    return nan_count.sum().item()


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="Need CUDA device")
class TestDebugMetaNanDetection:
    @pytest.mark.parametrize("field", ["kv_c", "k_pe"])
    def test_detects_nans(self, field):
        kv_c, k_pe, kv_cache, slots, scale, cnt = _make_inputs(16)
        (kv_c if field == "kv_c" else k_pe)[0, 0] = float("nan")
        assert _run_kernel(kv_c, k_pe, kv_cache, slots, scale, cnt) >= 1

    def test_clean_input_stays_zero(self):
        kv_c, k_pe, kv_cache, slots, scale, cnt = _make_inputs(32)
        assert _run_kernel(kv_c, k_pe, kv_cache, slots, scale, cnt) == 0

    def test_none_nan_count_no_crash(self):
        kv_c, k_pe, kv_cache, slots, scale, _ = _make_inputs(32)
        kv_c[0, 0] = float("nan")
        ops.concat_and_cache_mla(
            kv_c,
            k_pe,
            kv_cache,
            slots,
            "fp8",
            scale,
            None,
        )
        torch.accelerator.synchronize()

    def test_padded_slots_ignored(self):
        kv_c, k_pe, kv_cache, slots, scale, cnt = _make_inputs(4)
        kv_c[2, :] = float("nan")
        slots[2] = -1
        assert _run_kernel(kv_c, k_pe, kv_cache, slots, scale, cnt) == 0


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="Need CUDA device")
class TestEnvVarGating:
    def test_flag_off(self, monkeypatch):
        monkeypatch.setenv("VLLM_DEBUG_MLA_CACHE", "0")
        import importlib

        import vllm.envs

        importlib.reload(vllm.envs)
        assert vllm.envs.VLLM_DEBUG_MLA_CACHE is False

    def test_flag_on(self, monkeypatch):
        monkeypatch.setenv("VLLM_DEBUG_MLA_CACHE", "1")
        import importlib

        import vllm.envs

        importlib.reload(vllm.envs)
        assert vllm.envs.VLLM_DEBUG_MLA_CACHE is True


@pytest.fixture
def _enable_debug_env(monkeypatch):
    monkeypatch.setenv("VLLM_DEBUG_MLA_CACHE", "1")
    import importlib

    import vllm.envs

    importlib.reload(vllm.envs)


class TestPhaseTracking:
    @staticmethod
    def _run(layers, is_prefilling=True, ts=1000.0):
        from vllm.v1.engine import EngineCoreOutput
        from vllm.v1.metrics.stats import IterationStats, RequestStateStats

        first = next(iter(layers)) if layers else None
        output = EngineCoreOutput(
            request_id="req-test",
            new_token_ids=[1],
            kv_cache_nans_per_layer=layers,
            kv_cache_nan_timestamp=ts,
            kv_cache_nan_first_layer=first,
        )
        stats = IterationStats()
        stats.update_from_output(
            output,
            engine_core_timestamp=0.0,
            is_prefilling=is_prefilling,
            req_stats=RequestStateStats(arrival_time=0.0),
            lora_states=None,
            lora_name=None,
        )
        return stats

    def test_prefill_labeled(self, _enable_debug_env):
        stats = self._run({"model.layers.0.self_attn": 5})
        assert stats.kv_cache_nans[("model.layers.0.self_attn", "prefill")] == 5

    def test_decode_labeled(self, _enable_debug_env):
        stats = self._run(
            {"model.layers.0.self_attn": 3},
            is_prefilling=False,
        )
        assert stats.kv_cache_nans[("model.layers.0.self_attn", "decode")] == 3

    def test_origin_locked_to_first_seen(self, _enable_debug_env):
        from vllm.v1.engine import EngineCoreOutput
        from vllm.v1.metrics.stats import IterationStats, RequestStateStats

        stats = IterationStats()
        layers = {
            "model.layers.0.self_attn": 2,
            "model.layers.1.self_attn": 3,
        }
        out = EngineCoreOutput(
            request_id="req-test",
            new_token_ids=[1],
            kv_cache_nans_per_layer=layers,
            kv_cache_nan_timestamp=1000.0,
            kv_cache_nan_first_layer=next(iter(layers)),
        )
        stats.update_from_output(
            out,
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

        out2 = EngineCoreOutput(
            request_id="req-test",
            new_token_ids=[1],
            kv_cache_nans_per_layer={"model.layers.1.self_attn": 1},
            kv_cache_nan_timestamp=1001.0,
            kv_cache_nan_first_layer="model.layers.1.self_attn",
        )
        stats.update_from_output(
            out2,
            engine_core_timestamp=0.0,
            is_prefilling=False,
            req_stats=RequestStateStats(arrival_time=0.0),
            lora_states=None,
            lora_name=None,
        )
        assert stats.kv_cache_nan_origin == (
            "model.layers.0.self_attn",
            "prefill",
        )


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="Need CUDA device")
class TestKernelToStatsOrigin:
    def test_two_layers_prefill_origin_is_first(self, _enable_debug_env):
        import time

        from vllm.v1.engine import EngineCoreOutput
        from vllm.v1.metrics.stats import IterationStats, RequestStateStats

        kv_c0, k_pe0, cache0, slots0, scale0, cnt0 = _make_inputs(16)
        kv_c0[0, 0] = float("nan")
        _run_kernel(kv_c0, k_pe0, cache0, slots0, scale0, cnt0)

        kv_c1, k_pe1, cache1, slots1, scale1, cnt1 = _make_inputs(16)
        kv_c1[0, 0] = float("nan")
        _run_kernel(kv_c1, k_pe1, cache1, slots1, scale1, cnt1)

        layer_nans = {
            "model.layers.0.self_attn": int(cnt0[0].item()),
            "model.layers.1.self_attn": int(cnt1[0].item()),
        }
        output = EngineCoreOutput(
            request_id="req-test",
            new_token_ids=[1],
            kv_cache_nans_per_layer=layer_nans,
            kv_cache_nan_timestamp=time.time(),
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
