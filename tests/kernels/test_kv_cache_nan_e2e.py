# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E2E test for MLA KV cache NaN detection and Prometheus publishing."""

import os
import subprocess
import sys
import tempfile
import textwrap
from unittest.mock import patch

import pytest
import torch

from vllm import _custom_ops as ops

METRIC_NAME = "vllm:kv_cache_nans"
SAMPLE_NAME = "vllm:kv_cache_nans_total"
ORIGIN_METRIC = "vllm:kv_cache_nan_origin"
TIMESTAMP_METRIC = "vllm:kv_cache_nan_first_seen_timestamp"

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
    kv_c = kv_c.clone()
    kv_c[: min(4, kv_c.shape[0]), 0] = float("nan")
    _orig_concat_and_cache_mla(
        kv_c,
        k_pe,
        kv_cache,
        slot_mapping,
        kv_cache_dtype,
        scale,
        num_kv_cache_nan_insertions,
    )


@pytest.mark.slow_test
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="Need CUDA device")
class TestKVCacheNanE2E:
    @pytest.fixture(autouse=True)
    def _setup_env(self, monkeypatch):
        monkeypatch.setenv("VLLM_DEBUG_MLA_CACHE", "1")
        monkeypatch.setenv("VLLM_USE_FLASHINFER_SAMPLER", "0")
        monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    def test_prometheus_counters_populated(self):
        import time

        from prometheus_client import REGISTRY

        with patch.object(
            ops, "concat_and_cache_mla", _nan_injecting_concat_and_cache_mla
        ):
            from vllm import LLM, SamplingParams

            llm = LLM(
                model="deepseek-ai/DeepSeek-V2-Lite",
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
                compilation_config={"cudagraph_mode": "full"},
            )
            t_before = time.time()
            llm.generate(
                ["Hello world"],
                SamplingParams(temperature=0, max_tokens=2),
            )
            t_after = time.time()

        prom_total = prom_layers = 0
        phases_seen: set[str] = set()
        layers_seen: set[str] = set()
        for metric in REGISTRY.collect():
            if metric.name != METRIC_NAME:
                continue
            for sample in metric.samples:
                if sample.name == SAMPLE_NAME and sample.value > 0:
                    prom_layers += 1
                    prom_total += sample.value
                    phases_seen.add(sample.labels["phase"])
                    layers_seen.add(sample.labels["layer"])

        assert prom_total > 0
        assert prom_layers >= 2
        assert phases_seen <= {"prefill", "decode"}
        assert "prefill" in phases_seen

        origin_entries = []
        for metric in REGISTRY.collect():
            if metric.name != ORIGIN_METRIC:
                continue
            for sample in metric.samples:
                if sample.value == 1:
                    origin_entries.append(sample.labels)

        assert len(origin_entries) == 1
        assert origin_entries[0]["phase"] in {"prefill", "decode"}
        assert origin_entries[0]["layer"] in layers_seen

        timestamps = []
        for metric in REGISTRY.collect():
            if metric.name != TIMESTAMP_METRIC:
                continue
            for sample in metric.samples:
                if sample.value > 0:
                    timestamps.append(sample.value)

        assert len(timestamps) > 0
        for ts in timestamps:
            assert t_before <= ts <= t_after


@pytest.mark.slow_test
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="Need CUDA device")
class TestKVCacheNanMultiprocessing:
    _SITECUSTOMIZE = textwrap.dedent("""\
        import importlib, importlib.abc, importlib.machinery, os, sys

        if os.environ.get("_VLLM_TEST_INJECT_KV_NANS") == "1":
            class _Finder(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path, target=None):
                    if fullname != "vllm._custom_ops":
                        return None
                    sys.meta_path.remove(self)
                    try:
                        spec = importlib.util.find_spec(fullname)
                    finally:
                        sys.meta_path.insert(0, self)
                    if spec is None:
                        return None
                    return importlib.machinery.ModuleSpec(
                        fullname, _Loader(spec), origin=spec.origin,
                        is_package=spec.submodule_search_locations is not None)

            class _Loader(importlib.abc.Loader):
                def __init__(self, s):
                    self._s = s

                def create_module(self, spec):
                    return None

                def exec_module(self, module):
                    self._s.loader.exec_module(module)
                    if hasattr(module, "concat_and_cache_mla"):
                        _orig = module.concat_and_cache_mla

                        def _p(kv_c, k_pe, kv_cache, slot_mapping,
                               kv_cache_dtype, scale,
                               num_kv_cache_nan_insertions=None):
                            import torch
                            kv_c = kv_c.clone()
                            kv_c[:min(4, kv_c.shape[0]), 0] = float("nan")
                            _orig(kv_c, k_pe, kv_cache, slot_mapping,
                                  kv_cache_dtype, scale,
                                  num_kv_cache_nan_insertions)

                        module.concat_and_cache_mla = _p

            sys.meta_path.insert(0, _Finder())
    """)

    _DRIVER_SCRIPT = textwrap.dedent("""\
        import os
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "1"
        os.environ["VLLM_DEBUG_MLA_CACHE"] = "1"
        os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

        from vllm import LLM, SamplingParams
        from prometheus_client import REGISTRY

        llm = LLM(
            model="deepseek-ai/DeepSeek-V2-Lite",
            hf_overrides={"num_hidden_layers": 2},
            kv_cache_dtype="fp8", enforce_eager=True,
            load_format="dummy", num_gpu_blocks_override=32,
            max_model_len=128, max_num_seqs=2,
            disable_log_stats=False,
            attention_config={"backend": "TRITON_MLA"},
            kernel_config={"moe_backend": "triton"},
            compilation_config={"cudagraph_mode": "full"})
        llm.generate(
            ["Hello world"], SamplingParams(temperature=0, max_tokens=2))

        nan_total = nan_layers = origin_count = 0
        phases = set()
        for m in REGISTRY.collect():
            if m.name == "vllm:kv_cache_nans":
                for s in m.samples:
                    if s.name == "vllm:kv_cache_nans_total" and s.value > 0:
                        nan_total += s.value
                        nan_layers += 1
                        phases.add(s.labels["phase"])
            elif m.name == "vllm:kv_cache_nan_origin":
                for s in m.samples:
                    if s.value == 1:
                        origin_count += 1

        assert nan_total > 0, f"Expected NaN counts, got {nan_total}"
        assert nan_layers >= 2, f"Expected >= 2 layers, got {nan_layers}"
        assert "prefill" in phases, f"Expected prefill, got {phases}"
        assert origin_count == 1, f"Expected 1 origin, got {origin_count}"
    """)

    def test_nan_detection_with_multiprocessing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            site_path = os.path.join(tmpdir, "sitecustomize.py")
            driver_path = os.path.join(tmpdir, "driver.py")
            with open(site_path, "w") as f:
                f.write(self._SITECUSTOMIZE)
            with open(driver_path, "w") as f:
                f.write(self._DRIVER_SCRIPT)

            env = os.environ.copy()
            env["PYTHONPATH"] = tmpdir + os.pathsep + env.get("PYTHONPATH", "")
            env["_VLLM_TEST_INJECT_KV_NANS"] = "1"

            result = subprocess.run(
                [sys.executable, driver_path],
                env=env,
                capture_output=True,
                text=True,
                timeout=600,
            )
            assert result.returncode == 0, (
                f"rc={result.returncode}\n"
                f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )
