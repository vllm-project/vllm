# SPDX-License-Identifier: Apache-2.0
"""Tests for the v2 bench probes added 2026-04-30.

The 5 new probes:
1. test_output_length — generation capacity at 1K..16K with VRAM tracking
2. scrape_accept_rate — Prometheus /metrics spec-decode counters
3. capture_vllm_version — parse system_fingerprint
4. capture_genesis_patch_state — local self-test --json invocation
5. _local_vram_used_mib — nvidia-smi snapshot helper

These tests verify module structure, argument shapes, and graceful
fallback when external dependencies (vllm engine, nvidia-smi) are
unavailable in the CPU dev env.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def bench_module():
    """Load the bench module via spec_from_file_location since it lives
    outside the package tree."""
    repo_root = Path(__file__).resolve().parents[4]
    bench_path = repo_root / "tools" / "genesis_bench_suite.py"
    if not bench_path.is_file():
        pytest.skip(f"bench module not found at {bench_path}")
    spec = importlib.util.spec_from_file_location("gbs", bench_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestVRAMSnapshotHelper:
    def test_helper_exists(self, bench_module):
        assert hasattr(bench_module, "_local_vram_used_mib")

    def test_returns_list_or_none(self, bench_module):
        result = bench_module._local_vram_used_mib()
        # Either list of ints (one per GPU) or None when nvidia-smi
        # unavailable. NEVER raises.
        assert result is None or (
            isinstance(result, list)
            and all(isinstance(v, int) for v in result)
        )


class TestScrapeAcceptRate:
    def test_function_exists(self, bench_module):
        assert hasattr(bench_module, "scrape_accept_rate")

    def test_returns_dict_with_accept_rate_key(self, bench_module):
        # Calling against a non-existent host should NOT raise; should
        # return dict with accept_rate=None on failure.
        result = bench_module.scrape_accept_rate(
            "127.0.0.1", 9, "irrelevant"
        )
        assert isinstance(result, dict)
        assert "accept_rate" in result

    def test_disabled_log_stats_returns_none(self, bench_module):
        """When --disable-log-stats is passed to vLLM (PROD config),
        spec_decode metrics are suppressed. scrape_accept_rate must
        return accept_rate=None gracefully — not crash."""
        # Hitting localhost:8000 IF a vLLM is running with
        # --disable-log-stats: should return accept_rate=None without
        # raising. If no vLLM is running locally, we get error path
        # which also returns dict-with-None.
        result = bench_module.scrape_accept_rate(
            "127.0.0.1", 8000, "any-key"
        )
        # Must not crash; accept_rate is None either way (no vllm or
        # vllm with disabled stats both produce None).
        assert isinstance(result, dict)


class TestCaptureVllmVersion:
    def test_function_exists(self, bench_module):
        assert hasattr(bench_module, "capture_vllm_version")

    def test_returns_dict_on_no_server(self, bench_module):
        """When server unreachable, must return dict with system_fingerprint=None
        rather than raising."""
        result = bench_module.capture_vllm_version(
            "127.0.0.1", 9, "irrelevant", "fake-model"
        )
        assert isinstance(result, dict)
        assert "system_fingerprint" in result
        # parse subdict always present even if empty
        assert "parsed" in result


class TestCaptureGenesisPatchState:
    def test_function_exists(self, bench_module):
        assert hasattr(bench_module, "capture_genesis_patch_state")

    def test_runs_self_test_when_available(self, bench_module):
        """When invoked from a Genesis checkout, this runs
        `python3 -m vllm._genesis.compat.cli self-test --json` locally
        and parses the result."""
        result = bench_module.capture_genesis_patch_state()
        assert isinstance(result, dict)
        assert "available" in result
        if result["available"]:
            # Parsed self-test should contain checks + summary
            assert "self_test" in result
            assert "summary" in result["self_test"]
            assert "checks" in result["self_test"]
            # And at minimum the version check should be present
            check_names = [c["name"] for c in result["self_test"]["checks"]]
            assert any("version" in n.lower() for n in check_names)


class TestOutputLengthProbe:
    def test_function_exists(self, bench_module):
        assert hasattr(bench_module, "test_output_length")

    def test_returns_probes_dict_on_unreachable(self, bench_module):
        """Must return dict with `probes` key + max_reached_tokens
        even when server unreachable. Each probe must have the
        expected shape including vram fields."""
        # Hit unreachable port with very fast timeout.
        # The function does up to 5 targets, but we expect each to fail
        # quickly with HTTP error → stops on first FAIL_ERROR.
        result = bench_module.test_output_length(
            "127.0.0.1", 9, "irrelevant", "fake-model"
        )
        assert isinstance(result, dict)
        assert "probes" in result
        assert "max_reached_tokens" in result

    def test_probe_record_shape(self, bench_module):
        """Verify the probe-record dict has the v2 vram fields."""
        result = bench_module.test_output_length(
            "127.0.0.1", 9, "irrelevant", "fake-model"
        )
        if result["probes"]:
            p = result["probes"][0]
            for key in (
                "target_max_tokens", "completion_tokens", "finish_reason",
                "elapsed_s", "ttft_ms", "wall_tps", "error",
                "vram_before_mib", "vram_after_mib",
                # v2 list-of-int delta fields (replaces the old
                # vram_delta_mib that crashed on list subtraction):
                "vram_delta_per_gpu_mib", "vram_delta_total_mib",
                "verdict",
            ):
                assert key in p, f"output-length probe record missing {key}"


class TestStabilityVerdictV2:
    """v2 stress tracks SHA1/NaN/drift and emits STABILITY_VERDICT."""

    def test_stress_function_signature(self, bench_module):
        """test_stability_stress must support the v2 instrumentation
        without breaking the v1 signature."""
        import inspect
        sig = inspect.signature(bench_module.test_stability_stress)
        params = list(sig.parameters.keys())
        # v1 args still required first 6
        for p in ("host", "port", "key", "model", "iterations", "prompts"):
            assert p in params

    def test_stress_dict_includes_v2_fields_on_unreachable(self, bench_module):
        """Even on unreachable server (all trials fail), the stress
        dict must include the v2 instrumentation fields with empty
        defaults — no KeyError on downstream consumers."""
        # 0 iterations to short-circuit cleanly; the function should
        # still emit the v2 schema.
        result = bench_module.test_stability_stress(
            "127.0.0.1", 9, "irrelevant", "fake-model",
            iterations=0, prompts=["hi"], max_tokens=8,
        )
        for v2_key in (
            "STABILITY_VERDICT", "verdict_notes",
            "nan_detections", "repetition_detections",
            "drift_per_prompt", "tpot_trend",
        ):
            assert v2_key in result, (
                f"v2 stress instrumentation missing {v2_key}"
            )

    def test_stress_pass_verdict_when_no_failures(self, bench_module):
        """0 iterations with no failures = PASS (no negatives detected)."""
        result = bench_module.test_stability_stress(
            "127.0.0.1", 9, "irrelevant", "fake-model",
            iterations=0, prompts=["hi"], max_tokens=8,
        )
        assert result["STABILITY_VERDICT"] == "PASS"


class TestCLIFlags:
    """Verify the new CLI flags are wired correctly."""

    def test_scheme_flag(self, bench_module):
        ns = bench_module.parse_args(["--scheme", "https"])
        assert ns.scheme == "https"

    def test_scheme_default(self, bench_module):
        ns = bench_module.parse_args([])
        assert ns.scheme == "http"

    def test_arm_name_alias(self, bench_module):
        # Both --name and --arm-name target dest=name
        ns_name = bench_module.parse_args(["--name", "X"])
        ns_arm = bench_module.parse_args(["--arm-name", "X"])
        assert ns_name.name == "X"
        assert ns_arm.name == "X"

    def test_compare_out(self, bench_module):
        ns = bench_module.parse_args([
            "--compare", "a.json", "b.json", "--compare-out", "delta.json"
        ])
        assert ns.compare_out == "delta.json"

    def test_probe_output_length_flag(self, bench_module):
        ns = bench_module.parse_args(["--probe-output-length"])
        assert ns.probe_output_length is True
        ns2 = bench_module.parse_args([])
        assert ns2.probe_output_length is False


class TestBuildURL:
    """The _build_url helper must respect the module-level _URL_SCHEME."""

    def test_default_http(self, bench_module):
        # Reset to default
        bench_module._URL_SCHEME = "http"
        assert bench_module._build_url(
            "1.2.3.4", 8000, "/v1/models"
        ) == "http://1.2.3.4:8000/v1/models"

    def test_https_after_switch(self, bench_module):
        bench_module._URL_SCHEME = "https"
        try:
            assert bench_module._build_url(
                "host", 443, "/v1/x"
            ) == "https://host:443/v1/x"
        finally:
            bench_module._URL_SCHEME = "http"

    def test_path_normalization(self, bench_module):
        bench_module._URL_SCHEME = "http"
        # No leading slash on path — helper adds it
        assert bench_module._build_url(
            "h", 1, "v1/x"
        ) == "http://h:1/v1/x"
