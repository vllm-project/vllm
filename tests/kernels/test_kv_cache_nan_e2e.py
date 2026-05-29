# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end test for MLA KV cache NaN detection and Prometheus publishing.

Validates that NaNs injected during FP8 KV cache conversion are detected
by the CUDA kernel and published as Prometheus counters by the model runner.

Requires: CUDA, vLLM built with custom ops.
"""

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
        """NaN injection -> kernel -> model runner -> Prometheus counters.

        Verifies:
        - NaN counts are published per (layer, phase)
        - Phase labels are from {"prefill", "decode"}
        - Origin gauge identifies the first (layer, phase) with NaNs
        - First-seen timestamp gauge is set to a recent wall-clock time
        - Per-layer counters are zeroed after the model runner reads them
        """
        import time

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
                compilation_config={"cudagraph_mode": "full"},
            )
            t_before = time.time()
            llm.generate(
                ["Hello world"],
                SamplingParams(temperature=0, max_tokens=2),
            )
            t_after = time.time()

        # --- NaN counter ---
        prom_total = 0
        prom_layers = 0
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

        assert prom_total > 0, (
            f"Expected NaN counts in Prometheus, got total={prom_total}"
        )
        assert prom_layers >= 2, (
            f"Expected NaNs in at least 2 layers, got {prom_layers}"
        )
        assert phases_seen <= {"prefill", "decode"}, (
            f"Unexpected phase labels: {phases_seen}"
        )
        assert "prefill" in phases_seen, f"Expected 'prefill' phase, got {phases_seen}"

        # --- Origin gauge: exactly one (rank, layer, phase) set to 1 ---
        origin_entries: list[dict[str, str]] = []
        for metric in REGISTRY.collect():
            if metric.name != ORIGIN_METRIC:
                continue
            for sample in metric.samples:
                if sample.value == 1:
                    origin_entries.append(sample.labels)

        assert len(origin_entries) == 1, (
            f"Expected exactly 1 origin entry, got {len(origin_entries)}: "
            f"{origin_entries}"
        )
        origin = origin_entries[0]
        assert origin["phase"] in {"prefill", "decode"}
        assert origin["layer"] in layers_seen

        # --- First-seen timestamp gauge ---
        timestamps: list[float] = []
        for metric in REGISTRY.collect():
            if metric.name != TIMESTAMP_METRIC:
                continue
            for sample in metric.samples:
                if sample.value > 0:
                    timestamps.append(sample.value)

        assert len(timestamps) > 0, "Expected first-seen timestamp(s)"
        for ts in timestamps:
            assert t_before <= ts <= t_after, (
                f"Timestamp {ts} outside [{t_before}, {t_after}]"
            )

        # --- Counter reset ---
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
            compilation_config={"cudagraph_mode": "full"},
        )
        llm.generate(
            ["Test prompt"],
            SamplingParams(temperature=0, max_tokens=1),
        )

        prom_total = 0
        for metric in REGISTRY.collect():
            if metric.name != METRIC_NAME:
                continue
            for sample in metric.samples:
                if sample.name == SAMPLE_NAME:
                    prom_total += sample.value

        assert prom_total == 0, (
            f"No NaN injection but got {prom_total} NaN(s) in Prometheus"
        )


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="Need CUDA device")
class TestKVCacheNanCUDAGraph:
    """Verify NaN detection works when CUDA graphs are enabled."""

    @pytest.fixture(autouse=True)
    def _setup_env(self, monkeypatch):
        monkeypatch.setenv("VLLM_DEBUG_MLA_CACHE", "1")
        monkeypatch.setenv("VLLM_USE_FLASHINFER_SAMPLER", "0")
        monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    def test_nan_detected_under_cudagraph(self):
        """NaN counter tensor is baked into the graph and survives replay."""

        with patch.object(
            ops, "concat_and_cache_mla", _nan_injecting_concat_and_cache_mla
        ):
            from vllm import LLM, SamplingParams

            llm = LLM(
                model="deepseek-ai/DeepSeek-V2-Lite",
                hf_overrides={"num_hidden_layers": 2},
                kv_cache_dtype="fp8",
                load_format="dummy",
                num_gpu_blocks_override=32,
                max_model_len=128,
                max_num_seqs=2,
                disable_log_stats=False,
                attention_config={"backend": "TRITON_MLA"},
                kernel_config={"moe_backend": "triton"},
                compilation_config={"cudagraph_mode": "full"},
            )
            # max_tokens=4 to run multiple decode steps through the graph.
            llm.generate(
                ["Hello world"],
                SamplingParams(temperature=0, max_tokens=4),
            )

        # Verify NaN counters were populated despite cudagraph replay.
        ec = llm.llm_engine.engine_core.engine_core
        mr = ec.model_executor.driver_worker.model_runner
        sfc = mr.compilation_config.static_forward_context

        layers_with_debug = 0
        for name, layer in sfc.items():
            nc = getattr(layer, "num_kv_cache_nan_insertions", None)
            if nc is not None:
                layers_with_debug += 1
                # Counter was zeroed by model runner after reading.
                assert nc.sum().item() == 0, f"nan counter not zeroed for {name}"

        assert layers_with_debug >= 2, (
            f"Expected >= 2 layers with debug tensor, got {layers_with_debug}"
        )

        # Check Prometheus saw NaNs.
        from prometheus_client import REGISTRY

        prom_total = 0
        for metric in REGISTRY.collect():
            if metric.name != METRIC_NAME:
                continue
            for sample in metric.samples:
                if sample.name == SAMPLE_NAME and sample.value > 0:
                    prom_total += sample.value

        assert prom_total > 0, (
            f"Expected NaN counts under cudagraph, got total={prom_total}"
        )


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="Need CUDA device")
class TestKVCacheNanMultiprocessing:
    """Verify NaN detection works with VLLM_ENABLE_V1_MULTIPROCESSING=1.

    Monkey-patching concat_and_cache_mla in the test process does not
    propagate to vLLM worker subprocesses (spawn mode). We work around
    this by writing a sitecustomize.py that patches the op at import
    time in every Python process, then running the test as a subprocess.
    """

    _SITECUSTOMIZE = textwrap.dedent("""\
        import importlib
        import importlib.abc
        import importlib.machinery
        import os
        import sys

        if os.environ.get("_VLLM_TEST_INJECT_KV_NANS") == "1":

            class _NanPatchFinder(importlib.abc.MetaPathFinder):
                \"\"\"Intercept import of vllm._custom_ops to inject NaNs.\"\"\"

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
                        fullname,
                        _NanPatchLoader(spec),
                        origin=spec.origin,
                        is_package=spec.submodule_search_locations is not None,
                    )

            class _NanPatchLoader(importlib.abc.Loader):
                def __init__(self, original_spec):
                    self._spec = original_spec

                def create_module(self, spec):
                    return None

                def exec_module(self, module):
                    self._spec.loader.exec_module(module)
                    if hasattr(module, "concat_and_cache_mla"):
                        _orig = module.concat_and_cache_mla

                        def _patched(
                            kv_c,
                            k_pe,
                            kv_cache,
                            slot_mapping,
                            kv_cache_dtype,
                            scale,
                            num_kv_cache_nan_insertions=None,
                        ):
                            import torch  # noqa: F811

                            kv_c = kv_c.clone()
                            n = min(4, kv_c.shape[0])
                            kv_c[:n, 0] = float("nan")
                            _orig(
                                kv_c,
                                k_pe,
                                kv_cache,
                                slot_mapping,
                                kv_cache_dtype,
                                scale,
                                num_kv_cache_nan_insertions,
                            )

                        module.concat_and_cache_mla = _patched

            sys.meta_path.insert(0, _NanPatchFinder())
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
        llm.generate(
            ["Hello world"],
            SamplingParams(temperature=0, max_tokens=2),
        )

        nan_total = 0
        nan_layers = 0
        phases = set()
        for metric in REGISTRY.collect():
            if metric.name != "vllm:kv_cache_nans":
                continue
            for sample in metric.samples:
                if sample.name == "vllm:kv_cache_nans_total" and sample.value > 0:
                    nan_total += sample.value
                    nan_layers += 1
                    phases.add(sample.labels["phase"])

        origin_count = 0
        for metric in REGISTRY.collect():
            if metric.name != "vllm:kv_cache_nan_origin":
                continue
            for sample in metric.samples:
                if sample.value == 1:
                    origin_count += 1

        assert nan_total > 0, f"Expected NaN counts, got {nan_total}"
        assert nan_layers >= 2, f"Expected >= 2 layers with NaNs, got {nan_layers}"
        assert "prefill" in phases, f"Expected prefill phase, got {phases}"
        assert origin_count == 1, f"Expected 1 origin entry, got {origin_count}"
        print(f"PASS: nan_total={nan_total} layers={nan_layers} phases={phases}")
    """)

    def test_nan_detection_with_multiprocessing(self):
        """Full NaN injection test with VLLM_ENABLE_V1_MULTIPROCESSING=1.

        Uses sitecustomize.py to patch concat_and_cache_mla in all
        subprocesses, including the vLLM engine-core worker.
        """
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
                f"Multiprocessing NaN test failed (rc={result.returncode}).\n"
                f"STDOUT:\n{result.stdout}\n"
                f"STDERR:\n{result.stderr}"
            )
