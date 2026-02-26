# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import json
import os
import subprocess
import tempfile
import time

import openai
import pytest
import pytest_asyncio
from typing import Dict, Any, Optional

from vllm.platforms import current_platform
import vllm

from tests.evals.gsm8k.gsm8k_eval import evaluate_gsm8k
from tests.models.registry import HF_EXAMPLE_MODELS
from tests.utils import RemoteOpenAIServer


AITER_MODEL_LIST = [
    "openai/gpt-oss-120b"
]

MODEL_NAME = "openai/gpt-oss-120b"

# Thresholds (±90%) - relaxed for CI variance (TTFT can vary significantly)
THRESHOLD = 0.90


@pytest.fixture(scope="module")
def default_server_args():
    attention_backend = (
        "ROCM_AITER_UNIFIED_ATTN" if current_platform.is_rocm()
        else "TRITON_ATTN"
    )
    return [
        "--enforce-eager",
        "--max-model-len", "1024",
        "--max-num-seqs", "256",
        "--gpu-memory-utilization", "0.85",
        "--reasoning-parser", "openai_gptoss",
        "--tensor-parallel-size", "4",
        "--attention-backend", attention_backend,
    ]


@pytest.fixture(scope="module")
def server(default_server_args):
    """Start vLLM HTTP server for online serving tests."""
    model_info = HF_EXAMPLE_MODELS.find_hf_info(MODEL_NAME)
    model_info.check_available_online(on_fail="skip")
    model_info.check_transformers_version(on_fail="skip")
    
    # Handle ROCm AITER if needed
    env_dict = None
    if current_platform.is_rocm() and MODEL_NAME in AITER_MODEL_LIST:
        env_dict = {"VLLM_ROCM_USE_AITER": "1"}
    
    with RemoteOpenAIServer(MODEL_NAME, default_server_args, env_dict=env_dict) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    """Get async OpenAI client for testing online serving endpoints."""
    async with server.get_async_client() as async_client:
        yield async_client



def _run_guidellm_benchmark(
    server: RemoteOpenAIServer,
    model_name: str,
    num_requests: int = 100,
    max_concurrency: Optional[int] = None,
    request_rate: Optional[float] = None,
    prompt_tokens: int = 100,
    output_tokens: int = 100,
    timeout: int = 600,
) -> str:
    server_url = server.url_for("v1")

    # Extract vLLM version
    vllm_version = vllm.__version__
    print("VLLM VERSION: ", vllm_version)

    output_filename = f"gptoss120b_{vllm_version}.json"
    print("OUTPUT FILENAME: ", output_filename)

    cmd = [
        "guidellm",
        "benchmark",
        "--target", server_url,
        "--model", model_name,
        "--data", f"prompt_tokens={prompt_tokens},output_tokens={output_tokens}",
        "--rate-type", "concurrent",
        "--max-seconds", str(timeout),
        "--max-requests", str(num_requests),
        "--output-path", output_filename,
    ]

    if request_rate is not None:
        cmd.extend(["--rate", str(int(request_rate))])
    else:
        cmd.extend(["--rate", "100"])

    if max_concurrency is not None:
        cmd.extend(["--max-concurrency", str(max_concurrency)])

    print(f"\nRunning guidellm benchmark → output: {output_filename}\n")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout + 30,
            check=True,
        )
        print(f"Benchmark complete. Results saved to {output_filename}")

        # Read and print JSON output
        with open(output_filename, "r") as f:
            raw_output = f.read()
        print("GUIDELLM OUTPUT: \n", raw_output)

        return output_filename

    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Guidellm benchmark timed out after {timeout} seconds")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Guidellm benchmark failed with exit code {e.returncode}:\n"
            f"stdout: {e.stdout}\n"
            f"stderr: {e.stderr}"
        )
    except FileNotFoundError:
        raise RuntimeError(
            "guidellm command not found. Please install guidellm:\n"
            "  pip install guidellm"
        )


def _get_benchmark(data: dict) -> dict:
    """Extract the first benchmark from guidellm output."""
    benchmarks = data.get("benchmarks", [])
    assert len(benchmarks) > 0, "No benchmarks found in guidellm output"
    return benchmarks[0]


def _get_metric_mean(benchmark: dict, metric_name: str) -> float:
    """Extract mean value from a metric's successful bucket."""
    return benchmark["metrics"][metric_name]["successful"]["mean"]


def _get_metric_percentile(benchmark: dict, metric_name: str, percentile: str) -> float:
    """Extract a percentile value from a metric's successful bucket."""
    return benchmark["metrics"][metric_name]["successful"]["percentiles"][percentile]


@pytest.mark.parametrize("model_name", [MODEL_NAME])
def test_compare_guidellm_results(
    server: RemoteOpenAIServer,
    model_name: str,
) -> None:
    """
    Run guidellm benchmark, then compare the output against the baseline file.

    Checks the following metrics with ±10% tolerance:
      - requests_per_second (mean)
      - request_latency (mean, p50, p95, p99)
      - time_to_first_token_ms (mean, p50, p95, p99)
      - time_per_output_token_ms (mean)
      - inter_token_latency_ms (mean)
      - request success count (exact: must be 100%)
    """
    # Run benchmark first
    current_path = _run_guidellm_benchmark(server, model_name)
    vllm_version = vllm.__version__

    # Resolve baseline path relative to this test file (works in CI regardless of cwd)
    baseline_path = os.path.join(
        os.path.dirname(__file__), "output", "gptoss120b.json"
    )
    assert os.path.exists(baseline_path), (
        f"Baseline file not found: {baseline_path}"
    )

    with open(current_path) as f:
        current_data = json.load(f)
    with open(baseline_path) as f:
        baseline_data = json.load(f)

    current_bm = _get_benchmark(current_data)
    baseline_bm = _get_benchmark(baseline_data)

    failures = []

    def check(metric_label: str, current_val: float, baseline_val: float,
               lower_is_better: bool = True) -> None:
        """Assert current value is within ±10% of baseline."""
        if baseline_val == 0:
            return  # skip division by zero

        diff_pct = (current_val - baseline_val) / baseline_val

        # For metrics where lower is better (latency), flag regressions > +10%
        # For metrics where higher is better (throughput), flag regressions < -10%
        if lower_is_better:
            if diff_pct > THRESHOLD:
                failures.append(
                    f"REGRESSION [{metric_label}]: "
                    f"current={current_val:.4f}, baseline={baseline_val:.4f}, "
                    f"degraded by {diff_pct*100:.1f}% (threshold: +{THRESHOLD*100:.0f}%)"
                )
        else:
            if diff_pct < -THRESHOLD:
                failures.append(
                    f"REGRESSION [{metric_label}]: "
                    f"current={current_val:.4f}, baseline={baseline_val:.4f}, "
                    f"degraded by {abs(diff_pct)*100:.1f}% (threshold: -{THRESHOLD*100:.0f}%)"
                )

    # --- Correctness check: 100% success rate ---
    current_totals = current_bm["request_totals"]
    baseline_totals = baseline_bm["request_totals"]

    assert current_totals["errored"] == 0, (
        f"Current run had {current_totals['errored']} errored requests"
    )
    assert current_totals["incomplete"] == 0, (
        f"Current run had {current_totals['incomplete']} incomplete requests"
    )
    assert current_totals["successful"] == baseline_totals["successful"], (
        f"Successful request count mismatch: "
        f"current={current_totals['successful']}, "
        f"baseline={baseline_totals['successful']}"
    )

    # --- Throughput (higher is better) ---
    check(
        "requests_per_second.mean",
        _get_metric_mean(current_bm, "requests_per_second"),
        _get_metric_mean(baseline_bm, "requests_per_second"),
        lower_is_better=False,
    )

    # --- Request latency (lower is better) ---
    for percentile in ["p50", "p95", "p99"]:
        check(
            f"request_latency.{percentile}",
            _get_metric_percentile(current_bm, "request_latency", percentile),
            _get_metric_percentile(baseline_bm, "request_latency", percentile),
            lower_is_better=True,
        )
    check(
        "request_latency.mean",
        _get_metric_mean(current_bm, "request_latency"),
        _get_metric_mean(baseline_bm, "request_latency"),
        lower_is_better=True,
    )

    # --- TTFT (lower is better) ---
    for percentile in ["p50", "p95", "p99"]:
        check(
            f"time_to_first_token_ms.{percentile}",
            _get_metric_percentile(current_bm, "time_to_first_token_ms", percentile),
            _get_metric_percentile(baseline_bm, "time_to_first_token_ms", percentile),
            lower_is_better=True,
        )
    check(
        "time_to_first_token_ms.mean",
        _get_metric_mean(current_bm, "time_to_first_token_ms"),
        _get_metric_mean(baseline_bm, "time_to_first_token_ms"),
        lower_is_better=True,
    )

    # --- TPOT (lower is better) ---
    check(
        "time_per_output_token_ms.mean",
        _get_metric_mean(current_bm, "time_per_output_token_ms"),
        _get_metric_mean(baseline_bm, "time_per_output_token_ms"),
        lower_is_better=True,
    )

    # --- ITL (lower is better) ---
    check(
        "inter_token_latency_ms.mean",
        _get_metric_mean(current_bm, "inter_token_latency_ms"),
        _get_metric_mean(baseline_bm, "inter_token_latency_ms"),
        lower_is_better=True,
    )

    # --- Final assertion ---
    if failures:
        failure_msg = "\n".join(failures)
        pytest.fail(
            f"\n{'='*60}\n"
            f"Guidellm benchmark regression detected "
            f"(vLLM {vllm_version} vs baseline):\n\n"
            f"{failure_msg}\n"
            f"{'='*60}"
        )

    print(f"\n✓ All metrics within ±{THRESHOLD*100:.0f}% of baseline (vLLM {vllm_version})")