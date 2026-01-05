# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
GSM8K evaluation using vLLM server and isolated GSM8K script.
Replacement for lm-eval-harness with better performance and control.

Usage:
pytest -s -v tests/evals/gsm8k/test_gsm8k_correctness.py \
    --config-list-file=configs/models-small.txt
"""

import shlex

import yaml

from tests.utils import RemoteOpenAIServer

from .gsm8k_eval import evaluate_gsm8k

TOL = 0.08  # Absolute tolerance for accuracy comparison


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
    results = evaluate_gsm8k(
        num_questions=eval_config["num_questions"],
        num_shots=eval_config["num_fewshot"],
        host=host,
        port=port,
    )

    return results


def test_gsm8k_correctness(config_filename):
    """Test GSM8K correctness for a given model configuration."""
    eval_config = yaml.safe_load(config_filename.read_text(encoding="utf-8"))

    # Parse server arguments from config (use shlex to handle quoted strings)
    server_args_str = eval_config.get("server_args", "")
    server_args = shlex.split(server_args_str) if server_args_str else []

    # Add standard server arguments
    server_args.extend(
        [
            "--trust-remote-code",
        ]
    )

    env_dict = eval_config.get("env", None)

    print(f"Starting GSM8K evaluation for model: {eval_config['model_name']}")
    print(f"Expected metric threshold: {eval_config['accuracy_threshold']}")
    print(f"Number of questions: {eval_config['num_questions']}")
    print(f"Number of few-shot examples: {eval_config['num_fewshot']}")
    print(f"Server args: {' '.join(server_args)}")
    print(f"Environment variables: {env_dict}")

    # Launch server and run evaluation
    with RemoteOpenAIServer(
        eval_config["model_name"],
        server_args,
        env_dict=env_dict,
        max_wait_seconds=eval_config.get("startup_max_wait_seconds", 600),
    ) as remote_server:
        server_url = remote_server.url_for("v1")
        print(f"Server started at: {server_url}")

        results = run_gsm8k_eval(eval_config, server_url)

        measured_metric = results["accuracy"]
        expected_metric = eval_config["accuracy_threshold"]

        print(f"GSM8K Results for {eval_config['model_name']}:")
        print(f"  Measured metric: {measured_metric:.4f}")
        print(f"  Expected metric: {expected_metric:.4f}")
        print(f"  Tolerance: {TOL:.4f}")
        print(f"  Questions: {results['num_questions']}")
        print(f"  Invalid rate: {results['invalid_rate']:.3f}")
        print(f"  Latency: {results['latency']:.1f}s")
        print(f"  QPS: {results['questions_per_second']:.1f}")

        # Verify metric is within tolerance
        assert measured_metric >= expected_metric - TOL, (
            f"GSM8K metric too low: {measured_metric:.4f} < "
            f"{expected_metric:.4f} - {TOL:.4f} = {expected_metric - TOL:.4f}"
        )

        print(f"✅ GSM8K test passed for {eval_config['model_name']}")
