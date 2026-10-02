# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GSM8K evaluation using vLLM server and isolated GSM8K script.
Replacement for lm-eval-harness with better performance and control.

Usage:
pytest -s -v tests/evals/gsm8k/test_gsm8k_correctness.py \
    --config-list-file=configs/models-small.txt
"""

import shlex

import pytest
import requests
import yaml

from tests.utils import RemoteOpenAIServer
from vllm.platforms import current_platform

from .gsm8k_eval import evaluate_gsm8k

DEFAULT_STARTUP_MAX_WAIT_SECONDS = 1200


def run_gsm8k_eval(eval_config: dict, server_url: str) -> dict:
    """Run GSM8K evaluation using our isolated script."""
    # Extract host and port from server URL
    if "://" in server_url:
        server_url = server_url.split("://")[1]

    host_port = server_url.split("/")[0]  # Remove path if present
    if ":" in host_port:
        host, p = host_port.split(":")
        port = int(p)
    else:
        host = host_port
        port = 8000

    # Add http:// prefix if not present
    if not host.startswith("http"):
        host = f"http://{host}"

    # Run GSM8K evaluation
    request_timeout_seconds = eval_config.get("request_timeout_seconds", 600)
    if current_platform.is_rocm():
        request_timeout_seconds = eval_config.get(
            "rocm_request_timeout_seconds", request_timeout_seconds
        )

    results = evaluate_gsm8k(
        num_questions=eval_config["num_questions"],
        num_shots=eval_config["num_fewshot"],
        max_tokens=eval_config.get("max_tokens", 256),
        model=eval_config["model_name"],
        use_chat_completions=eval_config.get("use_chat_completions", False),
        host=host,
        port=port,
        temperature=eval_config.get("temperature", 0.0),
        seed=eval_config.get("seed", 42),
        request_timeout_seconds=request_timeout_seconds,
        gen_prefix=eval_config.get("gen_prefix", ""),
        max_concurrency=eval_config.get("max_concurrency"),
    )

    return results


def get_acceptance_length(server_url: str) -> float:
    """Mean tokens emitted per verification step, from the server's counters.

    1.0 means every draft was rejected (speculation bought nothing); the
    theoretical maximum is 1 + num_speculative_tokens.
    """
    response = requests.get(f"{server_url.rstrip('/').removesuffix('/v1')}/metrics")
    response.raise_for_status()
    counters: dict[str, float] = {}
    for line in response.text.splitlines():
        if line.startswith("vllm:spec_decode_num_"):
            name, _, value = line.partition(" ")
            counters[name.split("{")[0]] = float(value)

    num_drafts = counters.get("vllm:spec_decode_num_drafts_total", 0.0)
    num_accepted = counters.get("vllm:spec_decode_num_accepted_tokens_total", 0.0)
    assert num_drafts > 0, (
        "no drafts recorded; speculative decoding did not run for this config"
    )
    return 1.0 + num_accepted / num_drafts


def test_gsm8k_correctness(config_filename):
    """Test GSM8K correctness for a given model configuration."""
    eval_config = yaml.safe_load(config_filename.read_text(encoding="utf-8"))

    if (
        not current_platform.is_cuda()
        and "Qwen3-30B-A3B-MXFP4A16" in eval_config["model_name"]
    ):
        pytest.skip(
            "Skipping Qwen3-30B-A3B-MXFP4A16 on non-CUDA platforms. "
            "Marlin kernels are not supported."
        )

    if (
        not current_platform.is_cuda()
        and "gemma-4-E4B-it-qat-mobile-ct" in eval_config["model_name"]
    ):
        pytest.skip(
            "Skipping gemma-4-E4B-it-qat-mobile-ct on non-CUDA platforms. "
            "Its W2A16 (uint2b2) scheme has no kernel outside CUDA."
        )

    # TODO(akaratza): Enable DeepSeek-V3.2 and DeepSeek-R1 on ROCm platforms
    if current_platform.is_rocm() and (
        "deepseek-ai/DeepSeek-V3.2" in eval_config["model_name"]
        or "deepseek-ai/DeepSeek-R1" in eval_config["model_name"]
    ):
        pytest.skip(
            "Skipping DeepSeek-V3.2 and DeepSeek-R1 on ROCm platforms "
            "due to agent pool disk space issues and pod evictions."
        )
    if current_platform.is_rocm() and (
        "Qwen3.5-35B-A3B-MXFP4-AITER-TP2" in config_filename.name
    ):
        from vllm.platforms.rocm import on_gfx950

        if not on_gfx950():
            pytest.skip(
                "Skipping Qwen3.5-35B-A3B-MXFP4-AITER-TP2 on non-GFX950 platforms. "
                "The quantization scheme is not supported on non-GFX950 platforms."
            )
    # Parse server arguments from config (use shlex to handle quoted strings)
    server_args_str = eval_config.get("server_args", "")
    server_args = shlex.split(server_args_str) if server_args_str else []

    # Add standard server arguments
    server_args.extend(
        [
            "--trust-remote-code",
            "--disable-uvicorn-access-log",
        ]
    )

    startup_max_wait_seconds = eval_config.get(
        "startup_max_wait_seconds", DEFAULT_STARTUP_MAX_WAIT_SECONDS
    )
    env_dict = dict(eval_config.get("env") or {})
    env_dict["VLLM_ENGINE_READY_TIMEOUT_S"] = str(int(startup_max_wait_seconds))

    print(f"Starting GSM8K evaluation for model: {eval_config['model_name']}")
    print(f"Expected metric threshold: {eval_config['accuracy_threshold']}")
    print(f"Number of questions: {eval_config['num_questions']}")
    print(f"Number of few-shot examples: {eval_config['num_fewshot']}")
    request_timeout_seconds = eval_config.get("request_timeout_seconds", 600)
    if current_platform.is_rocm():
        request_timeout_seconds = eval_config.get(
            "rocm_request_timeout_seconds", request_timeout_seconds
        )
    print(f"Request timeout: {request_timeout_seconds}s")
    print(f"Startup max wait: {startup_max_wait_seconds}s")
    print(f"Server args: {' '.join(server_args)}")
    print(f"Environment variables: {env_dict}")

    # Launch server and run evaluation
    with RemoteOpenAIServer(
        eval_config["model_name"],
        server_args,
        env_dict=env_dict,
        max_wait_seconds=startup_max_wait_seconds,
    ) as remote_server:
        server_url = remote_server.url_for("v1")
        print(f"Server started at: {server_url}")

        results = run_gsm8k_eval(eval_config, server_url)

        measured_metric = results["accuracy"]
        expected_metric = eval_config["accuracy_threshold"]
        tol = eval_config.get("tolerance", 0.08)

        print(f"GSM8K Results for {eval_config['model_name']}:")
        print(f"  Measured metric: {measured_metric:.4f}")
        print(f"  Expected metric: {expected_metric:.4f}")
        print(f"  Tolerance: {tol:.4f}")
        print(f"  Questions: {results['num_questions']}")
        print(f"  Invalid rate: {results['invalid_rate']:.3f}")
        print(f"  Latency: {results['latency']:.1f}s")
        print(f"  QPS: {results['questions_per_second']:.1f}")

        assert measured_metric >= expected_metric - tol, (
            f"GSM8K metric too low: {measured_metric:.4f} < "
            f"{expected_metric:.4f} - {tol:.4f} = {expected_metric - tol:.4f}"
        )

        # Speculative configs additionally assert that drafts are actually
        # landing: accuracy alone passes even when every draft is rejected.
        min_acceptance_length = eval_config.get("min_acceptance_length")
        if min_acceptance_length is not None:
            acceptance_length = get_acceptance_length(server_url)
            print(f"  Mean acceptance length: {acceptance_length:.3f}")
            print(f"  Minimum acceptance length: {min_acceptance_length:.3f}")
            assert acceptance_length >= min_acceptance_length, (
                f"Acceptance length too low: {acceptance_length:.3f} < "
                f"{min_acceptance_length:.3f}"
            )

        print(f"✅ GSM8K test passed for {eval_config['model_name']}")
