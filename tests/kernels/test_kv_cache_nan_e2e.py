# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end test for MLA KV cache NaN detection and Prometheus publishing.

Validates that NaNs injected during FP8 KV cache conversion are detected
by the CUDA kernel and published as Prometheus counters by the model runner.

Requires: CUDA, vLLM built with custom ops.
"""

from unittest.mock import patch

import pytest
import torch

from vllm import _custom_ops as ops

CUDA_AVAILABLE = torch.accelerator.device_count() >= 1

_orig_concat_and_cache_mla = ops.concat_and_cache_mla


def _nan_injecting_concat_and_cache_mla(
    kv_c,
    k_pe,
    kv_cache,
    slot_mapping,
    kv_cache_dtype,
    scale,
    num_kv_cache_nan_insertions=None,
):
    """Wrapper that injects NaNs into kv_c before calling the real kernel."""
    kv_c = kv_c.clone()
    n = min(4, kv_c.shape[0])
    kv_c[:n, 0] = float("nan")
    _orig_concat_and_cache_mla(
        kv_c,
        k_pe,
        kv_cache,
        slot_mapping,
        kv_cache_dtype,
        scale,
        num_kv_cache_nan_insertions,
    )


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="Need CUDA device")
class TestKVCacheNanE2E:
    """End-to-end: NaN injection -> kernel detection -> Prometheus counter."""

    @pytest.fixture(autouse=True)
    def _setup_env(self, monkeypatch):
        monkeypatch.setenv("VLLM_DEBUG_MLA_CACHE", "1")
        monkeypatch.setenv("VLLM_USE_FLASHINFER_SAMPLER", "0")
        monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    def test_prometheus_counters_populated(self):
        """NaN injection -> kernel -> model runner -> Prometheus counters."""
        from prometheus_client import REGISTRY

        with patch.object(
            ops, "concat_and_cache_mla", _nan_injecting_concat_and_cache_mla
        ):
            from vllm import LLM, SamplingParams

            llm = LLM(
                model="deepseek-ai/DeepSeek-V2-Lite",
                # Only load 2 layers to keep the test fast.
                hf_overrides={"num_hidden_layers": 2},
                kv_cache_dtype="fp8",
                enforce_eager=True,
                load_format="dummy",
                num_gpu_blocks_override=32,
                max_model_len=128,
                max_num_seqs=2,
                disable_log_stats=False,
                attention_config={"backend": "TRITON_MLA"},
                kernel_config={"moe_backend": "triton"},
            )
            llm.generate(
                ["Hello world"],
                SamplingParams(temperature=0, max_tokens=2),
            )

        # Collect Prometheus samples
        prom_total = 0
        prom_layers = 0
        for metric in REGISTRY.collect():
            if "kv_cache_nans" not in metric.name:
                continue
            for sample in metric.samples:
                if sample.name.endswith("_total") and sample.value > 0:
                    prom_layers += 1
                    prom_total += sample.value

        assert prom_total > 0, (
            f"Expected NaN counts in Prometheus, got total={prom_total}"
        )
        assert prom_layers >= 2, (
            f"Expected NaNs in at least 2 layers, got {prom_layers}"
        )

        # nan counter should be zeroed (model runner resets after read)
        ec = llm.llm_engine.engine_core.engine_core
        mr = ec.model_executor.driver_worker.model_runner
        sfc = mr.compilation_config.static_forward_context
        for name, layer in sfc.items():
            nc = getattr(layer, "num_kv_cache_nan_insertions", None)
            if nc is not None:
                assert nc.sum().item() == 0, f"nan counter not zeroed for {name}"

    def test_no_nans_no_counters(self):
        """Clean weights should produce zero NaN counters."""
        from prometheus_client import REGISTRY

        from vllm import LLM, SamplingParams

        llm = LLM(
            model="deepseek-ai/DeepSeek-V2-Lite",
            kv_cache_dtype="fp8",
            enforce_eager=True,
            load_format="dummy",
            num_gpu_blocks_override=32,
            max_model_len=128,
            max_num_seqs=2,
            disable_log_stats=False,
            attention_config={"backend": "TRITON_MLA"},
            kernel_config={"moe_backend": "triton"},
        )
        llm.generate(
            ["Test prompt"],
            SamplingParams(temperature=0, max_tokens=1),
        )

        prom_total = 0
        for metric in REGISTRY.collect():
            if "kv_cache_nans" not in metric.name:
                continue
            for sample in metric.samples:
                if sample.name.endswith("_total"):
                    prom_total += sample.value

        assert prom_total == 0, (
            f"No NaN injection but got {prom_total} NaN(s) in Prometheus"
        )
