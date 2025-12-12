# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
GSM8K evaluation using vLLM server and isolated GSM8K script.
Replacement for lm-eval-harness with better performance and control.

Usage:
pytest -s -v test_gsm8k_correctness.py \
    --config-list-file=configs/models-small.txt \
    --tp-size=1
"""

import yaml

from tests.utils import RemoteOpenAIServer

from .gsm8k_eval import evaluate_gsm8k

RTOL = 0.08  # Relative tolerance for accuracy comparison


def launch_gsm8k_eval(eval_config, server_url, tp_size):
    """Launch GSM8K evaluation using our isolated script."""
    # Extract host and port from server URL
    if "://" in server_url:
        server_url = server_url.split("://")[1]

    host_port = server_url.split("/")[0]  # Remove path if present
    if ":" in host_port:
        host, port = host_port.split(":")
        port = int(port)
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


def test_gsm8k_correctness_param(config_filename, tp_size):
    """Test GSM8K correctness for a given model configuration."""
    eval_config = yaml.safe_load(config_filename.read_text(encoding="utf-8"))

    # Server arguments
    server_args = [
        "--max-model-len",
        str(eval_config.get("max_model_len", 4096)),
        "--enforce-eager",
        "--trust-remote-code",
        "--tensor-parallel-size",
        str(tp_size),
    ]

    env_dict = eval_config.get("env", None)

    # Launch server and run evaluation
    with RemoteOpenAIServer(
        eval_config["model_name"], server_args, env_dict=env_dict, max_wait_seconds=480
    ) as remote_server:
        server_url = remote_server.url_for("v1")

        results = launch_gsm8k_eval(eval_config, server_url, tp_size)

        # Check accuracy against threshold
        measured_accuracy = results["accuracy"]
        expected_accuracy = eval_config["accuracy_threshold"]

        print(f"GSM8K Results for {eval_config['model_name']}:")
        print(f"  Accuracy: {measured_accuracy:.3f}")
        print(f"  Expected: {expected_accuracy:.3f}")
        print(f"  Questions: {results['num_questions']}")
        print(f"  Invalid rate: {results['invalid_rate']:.3f}")
        print(f"  Latency: {results['latency']:.1f}s")
        print(f"  QPS: {results['questions_per_second']:.1f}")

        # Verify accuracy is within tolerance
        assert measured_accuracy >= expected_accuracy - RTOL, (
            f"Accuracy too low: {measured_accuracy:.3f} < "
            f"{expected_accuracy:.3f} - {RTOL:.3f}"
        )

        print(f"✅ GSM8K test passed for {eval_config['model_name']}")
