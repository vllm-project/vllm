# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
GPQA evaluation using vLLM server and GPT-OSS evaluation package.

Usage:
pytest -s -v tests/evals/gpt_oss/test_gpqa_correctness.py \
    --model openai/gpt-oss-20b \
    --metric 0.58 \
    --server-args "--tensor-parallel-size 2"
"""

import subprocess
import sys

import regex as re

from tests.utils import RemoteOpenAIServer

TOL = 0.05  # Absolute tolerance for accuracy comparison


def run_gpqa_eval(model_name: str, base_url: str) -> float:
    """Run GPQA evaluation using the gpt-oss evaluation package."""

    # Build the command to run the evaluation
    cmd = [
        sys.executable,
        "-m",
        "gpt_oss.evals",
        "--eval",
        "gpqa",
        "--model",
        model_name,
        "--reasoning-effort",
        "low",
        "--base-url",
        base_url,
        "--n-threads",
        "200",
    ]

    try:
        # Run the evaluation
        result = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            timeout=1800,  # 30 minute timeout
            env={"OPENAI_API_KEY": "dummy"},
        )

        print("Evaluation process output:\n", result.stdout)

        # Parse the output to extract the score
        match = re.search(r"'metric':\s*([\d.]+)", result.stdout)
        if match:
            return float(match.group(1))

        # If we still can't find it, raise an error
        raise ValueError(
            f"Could not parse score from evaluation output:\n{result.stdout}"
        )

    except subprocess.TimeoutExpired as e:
        raise RuntimeError("Evaluation timed out") from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Evaluation failed with exit code {e.returncode}:\n"
            f"stdout: {e.stdout}\nstderr: {e.stderr}"
        ) from e


def test_gpqa_correctness(request):
    """Test GPQA correctness for GPT-OSS model."""

    # Get command line arguments
    model_name = request.config.getoption("--model")
    expected_metric = request.config.getoption("--metric")
    server_args_str = request.config.getoption("--server-args")

    # Parse server arguments
    server_args = []
    if server_args_str:
        server_args = server_args_str.split()

    # Add standard server arguments
    server_args.extend(
        [
            "--trust-remote-code",
        ]
    )

    print(f"Starting GPQA evaluation for model: {model_name}")
    print(f"Expected metric threshold: {expected_metric}")
    print(f"Server args: {' '.join(server_args)}")

    # Launch server and run evaluation
    with RemoteOpenAIServer(
        model_name, server_args, max_wait_seconds=1800
    ) as remote_server:
        base_url = remote_server.url_for("v1")
        print(f"Server started at: {base_url}")

        measured_metric = run_gpqa_eval(model_name, base_url)

        print(f"GPQA Results for {model_name}:")
        print(f"  Measured metric: {measured_metric:.4f}")
        print(f"  Expected metric: {expected_metric:.4f}")
        print(f"  Tolerance: {TOL:.4f}")

        # Verify metric is within tolerance
        assert measured_metric >= expected_metric - TOL, (
            f"GPQA metric too low: {measured_metric:.4f} < "
            f"{expected_metric:.4f} - {TOL:.4f} = {expected_metric - TOL:.4f}"
        )

        print(f"✅ GPQA test passed for {model_name}")
