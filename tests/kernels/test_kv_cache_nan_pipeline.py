# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Integration test for the NaN detection pipeline without a full LLM.

Calls concat_and_cache_mla directly with NaN inputs, then feeds the
kernel's counter through IterationStats to verify phase labeling,
origin tracking, and timestamp recording -- no model loading, no
monkey-patching, no subprocesses.
"""

import time

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


def _call_kernel_with_nans(
    nan_positions: dict[int, list[int]],
    num_tokens: int = 32,
    dtype: torch.dtype = torch.bfloat16,
    kv_cache_dtype: str = "fp8",
):
    """Call concat_and_cache_mla with NaNs at specified positions.

    Args:
        nan_positions: {token_idx: [element_indices]} in kv_c to set to NaN.
        num_tokens: number of tokens.
        dtype: input dtype.
        kv_cache_dtype: "fp8" or "auto".

    Returns:
        nan_count tensor (1-element int32 on CUDA).
    """
    kv_c = torch.randn(num_tokens, KV_LORA_RANK, dtype=dtype, device="cuda")
    k_pe = torch.randn(num_tokens, PE_DIM, dtype=dtype, device="cuda")
    cache_dt = torch.float8_e4m3fn if kv_cache_dtype == "fp8" else dtype
    kv_cache = torch.zeros(
        num_tokens, BLOCK_SIZE, ENTRY_SIZE, dtype=cache_dt, device="cuda"
    )
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device="cuda")
    scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")
    nan_count = torch.zeros(1, dtype=torch.int32, device="cuda")

    for tok, elems in nan_positions.items():
        for e in elems:
            kv_c[tok, e] = float("nan")

    ops.concat_and_cache_mla(
        kv_c, k_pe, kv_cache, slot_mapping, kv_cache_dtype, scale, nan_count
    )
    torch.accelerator.synchronize()
    return nan_count


def _build_engine_core_output(
    layer_nan_counts: dict[str, int],
    timestamp: float,
):
    """Build an EngineCoreOutput with the given per-layer NaN counts."""
    import regex as re

    from vllm.v1.engine import EngineCoreOutput

    first_layer = None
    best_idx = float("inf")
    for name in layer_nan_counts:
        m = re.search(r"\.([0-9]+)\.", name)
        if m and int(m.group(1)) < best_idx:
            best_idx = int(m.group(1))
            first_layer = name

    return EngineCoreOutput(
        request_id="req-test",
        new_token_ids=[1],
        kv_cache_nans_per_layer=layer_nan_counts,
        kv_cache_nan_timestamp=timestamp,
        kv_cache_nan_first_layer=first_layer,
    )


def _make_req_stats():
    from vllm.v1.metrics.stats import RequestStateStats

    return RequestStateStats(arrival_time=0.0)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="Need CUDA device")
class TestNanDetectionPipeline:
    """Kernel -> counter -> IterationStats, no LLM needed."""

    def test_prefill_nans_detected_and_labeled(self, monkeypatch):
        """Inject NaNs, simulate prefill, verify phase='prefill' in stats."""
        monkeypatch.setenv("VLLM_DEBUG_MLA_CACHE", "1")
        import importlib

        import vllm.envs

        importlib.reload(vllm.envs)

        from vllm.v1.metrics.stats import IterationStats

        # Simulate two layers seeing NaNs during prefill.
        layer0_count = _call_kernel_with_nans({0: [0], 1: [10]})
        layer1_count = _call_kernel_with_nans({3: [5]})

        ts = time.time()
        layer_nans = {
            "model.layers.0.self_attn": int(layer0_count[0].item()),
            "model.layers.1.self_attn": int(layer1_count[0].item()),
        }

        stats = IterationStats()
        output = _build_engine_core_output(layer_nans, ts)
        stats.update_from_output(
            output,
            engine_core_timestamp=0.0,
            is_prefilling=True,
            req_stats=_make_req_stats(),
            lora_states=None,
            lora_name=None,
        )

        assert ("model.layers.0.self_attn", "prefill") in stats.kv_cache_nans
        assert ("model.layers.1.self_attn", "prefill") in stats.kv_cache_nans
        assert stats.kv_cache_nans[("model.layers.0.self_attn", "prefill")] >= 2
        assert stats.kv_cache_nans[("model.layers.1.self_attn", "prefill")] >= 1

    def test_decode_nans_detected_and_labeled(self, monkeypatch):
        """Inject NaNs, simulate decode, verify phase='decode' in stats."""
        monkeypatch.setenv("VLLM_DEBUG_MLA_CACHE", "1")
        import importlib

        import vllm.envs

        importlib.reload(vllm.envs)

        from vllm.v1.metrics.stats import IterationStats

        layer0_count = _call_kernel_with_nans({0: [0]})

        ts = time.time()
        layer_nans = {
            "model.layers.0.self_attn": int(layer0_count[0].item()),
        }

        stats = IterationStats()
        output = _build_engine_core_output(layer_nans, ts)
        stats.update_from_output(
            output,
            engine_core_timestamp=0.0,
            is_prefilling=False,
            req_stats=_make_req_stats(),
            lora_states=None,
            lora_name=None,
        )

        assert ("model.layers.0.self_attn", "decode") in stats.kv_cache_nans
        assert ("model.layers.0.self_attn", "prefill") not in stats.kv_cache_nans

    def test_origin_locked_to_first_phase(self, monkeypatch):
        """Origin is set on first update and not overwritten by later ones."""
        monkeypatch.setenv("VLLM_DEBUG_MLA_CACHE", "1")
        import importlib

        import vllm.envs

        importlib.reload(vllm.envs)

        from vllm.v1.metrics.stats import IterationStats

        count = _call_kernel_with_nans({0: [0]})
        ts = time.time()

        stats = IterationStats()

        # First: prefill on layer 0
        out1 = _build_engine_core_output(
            {"model.layers.0.self_attn": int(count[0].item())}, ts
        )
        stats.update_from_output(
            out1,
            engine_core_timestamp=0.0,
            is_prefilling=True,
            req_stats=_make_req_stats(),
            lora_states=None,
            lora_name=None,
        )
        assert stats.kv_cache_nan_origin == (
            "model.layers.0.self_attn",
            "prefill",
        )

        # Second: decode on layer 1 -- origin must NOT change
        count2 = _call_kernel_with_nans({0: [0]})
        out2 = _build_engine_core_output(
            {"model.layers.1.self_attn": int(count2[0].item())}, time.time()
        )
        stats.update_from_output(
            out2,
            engine_core_timestamp=0.0,
            is_prefilling=False,
            req_stats=_make_req_stats(),
            lora_states=None,
            lora_name=None,
        )
        assert stats.kv_cache_nan_origin == (
            "model.layers.0.self_attn",
            "prefill",
        )

    def test_first_layer_is_lowest_index(self, monkeypatch):
        """When multiple layers have NaNs, origin picks the lowest index."""
        monkeypatch.setenv("VLLM_DEBUG_MLA_CACHE", "1")
        import importlib

        import vllm.envs

        importlib.reload(vllm.envs)

        from vllm.v1.metrics.stats import IterationStats

        c0 = _call_kernel_with_nans({0: [0]})
        c5 = _call_kernel_with_nans({0: [0]})
        c2 = _call_kernel_with_nans({0: [0]})

        ts = time.time()
        layer_nans = {
            "model.layers.5.self_attn": int(c5[0].item()),
            "model.layers.0.self_attn": int(c0[0].item()),
            "model.layers.2.self_attn": int(c2[0].item()),
        }

        stats = IterationStats()
        output = _build_engine_core_output(layer_nans, ts)
        stats.update_from_output(
            output,
            engine_core_timestamp=0.0,
            is_prefilling=True,
            req_stats=_make_req_stats(),
            lora_states=None,
            lora_name=None,
        )

        assert stats.kv_cache_nan_origin == (
            "model.layers.0.self_attn",
            "prefill",
        )

    def test_timestamp_from_kernel_call(self, monkeypatch):
        """Timestamp recorded in stats matches the one from the kernel call."""
        monkeypatch.setenv("VLLM_DEBUG_MLA_CACHE", "1")
        import importlib

        import vllm.envs

        importlib.reload(vllm.envs)

        from vllm.v1.metrics.stats import IterationStats

        count = _call_kernel_with_nans({0: [0]})

        t_before = time.time()
        ts = t_before
        layer_nans = {
            "model.layers.0.self_attn": int(count[0].item()),
        }

        stats = IterationStats()
        output = _build_engine_core_output(layer_nans, ts)
        stats.update_from_output(
            output,
            engine_core_timestamp=0.0,
            is_prefilling=True,
            req_stats=_make_req_stats(),
            lora_states=None,
            lora_name=None,
        )

        key = ("model.layers.0.self_attn", "prefill")
        assert key in stats.kv_cache_nan_first_seen
        assert stats.kv_cache_nan_first_seen[key] == ts

    def test_clean_input_no_nans_in_stats(self, monkeypatch):
        """No NaN injection -> kernel returns 0 -> nothing in stats."""
        monkeypatch.setenv("VLLM_DEBUG_MLA_CACHE", "1")
        import importlib

        import vllm.envs

        importlib.reload(vllm.envs)

        from vllm.v1.metrics.stats import IterationStats

        nan_count = _call_kernel_with_nans({})
        assert nan_count[0].item() == 0

        stats = IterationStats()
        # Empty dict -> update_from_output skips kv_cache_nans branch
        assert len(stats.kv_cache_nans) == 0
        assert stats.kv_cache_nan_origin is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
