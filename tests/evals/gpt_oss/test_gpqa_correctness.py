# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
GPQA evaluation using vLLM server and GPT-OSS evaluation package.

Usage:
pytest -s -v tests/evals/gpt_oss/test_gpqa_correctness.py \
    --config-list-file=configs/models-h200.txt
"""

import os
import shlex
import subprocess
import sys
import urllib.request
from pathlib import Path

import regex as re
import yaml

from tests.utils import RemoteOpenAIServer

TOL = 0.05  # Absolute tolerance for accuracy comparison

# Path to tiktoken encoding files
TIKTOKEN_DATA_DIR = Path(__file__).parent / "data"

# Tiktoken encoding files to download
TIKTOKEN_FILES = {
    "cl100k_base.tiktoken": "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken",
    "o200k_base.tiktoken": "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken",
}


def ensure_tiktoken_files():
    """Download tiktoken encoding files if they don't exist."""
    TIKTOKEN_DATA_DIR.mkdir(parents=True, exist_ok=True)

    for filename, url in TIKTOKEN_FILES.items():
        filepath = TIKTOKEN_DATA_DIR / filename
        if not filepath.exists():
            print(f"Downloading {filename} from {url}...")
            urllib.request.urlretrieve(url, filepath)
            print(f"  Downloaded to {filepath}")
        else:
            print(f"  {filename} already exists.")


def run_gpqa_eval(model_name: str, base_url: str, reasoning_effort: str) -> float:
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
        reasoning_effort,
        "--base-url",
        base_url,
        "--n-threads",
        "200",
    ]

    try:
        # Set up environment for the evaluation subprocess
        # Inherit current environment and add required variables
        eval_env = os.environ.copy()
        eval_env["OPENAI_API_KEY"] = "dummy"

        # Run the evaluation
        result = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            timeout=1800,  # 30 minute timeout
            env=eval_env,
        )

        print("Evaluation process stdout:\n", result.stdout)
        print("Evaluation process stderr:\n", result.stderr)
        print(f"Evaluation process return code: {result.returncode}")

        if result.returncode != 0:
            raise RuntimeError(
                f"Evaluation failed with exit code {result.returncode}:\n"
                f"stdout: {result.stdout}\nstderr: {result.stderr}"
            )

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


def test_gpqa_correctness(config_filename):
    """Test GPQA correctness for a given model configuration."""
    # Ensure tiktoken files are downloaded
    ensure_tiktoken_files()

    # Verify tiktoken files exist
    for filename in TIKTOKEN_FILES:
        filepath = TIKTOKEN_DATA_DIR / filename
        assert filepath.exists(), f"Tiktoken file not found: {filepath}"

    eval_config = yaml.safe_load(config_filename.read_text(encoding="utf-8"))

    # Parse server arguments from config (use shlex to handle quoted strings)
    server_args_str = eval_config.get("server_args", "")
    server_args = shlex.split(server_args_str) if server_args_str else []

    # Add standard server arguments
    server_args.extend(
        [
            "--trust-remote-code",
            "--enforce-eager",
            "--disable-uvicorn-access-log",
        ]
    )

    # Build server environment with tiktoken path and any config-specified vars
    server_env = {"TIKTOKEN_ENCODINGS_BASE": str(TIKTOKEN_DATA_DIR)}
    if eval_config.get("env"):
        server_env.update(eval_config["env"])

    reasoning_effort = eval_config.get("reasoning_effort", "low")

    print(f"Starting GPQA evaluation for model: {eval_config['model_name']}")
    print(f"Expected metric threshold: {eval_config['metric_threshold']}")
    print(f"Reasoning effort: {reasoning_effort}")
    print(f"Server args: {' '.join(server_args)}")
    print(f"Server environment variables: {server_env}")

    # Launch server and run evaluation
    with RemoteOpenAIServer(
        eval_config["model_name"],
        server_args,
        env_dict=server_env,
        max_wait_seconds=eval_config.get("startup_max_wait_seconds", 1800),
    ) as remote_server:
        base_url = remote_server.url_for("v1")
        print(f"Server started at: {base_url}")

        measured_metric = run_gpqa_eval(
            eval_config["model_name"], base_url, reasoning_effort
        )
        expected_metric = eval_config["metric_threshold"]

        print(f"GPQA Results for {eval_config['model_name']}:")
        print(f"  Measured metric: {measured_metric:.4f}")
        print(f"  Expected metric: {expected_metric:.4f}")
        print(f"  Tolerance: {TOL:.4f}")

        # Verify metric is within tolerance
        assert measured_metric >= expected_metric - TOL, (
            f"GPQA metric too low: {measured_metric:.4f} < "
            f"{expected_metric:.4f} - {TOL:.4f} = {expected_metric - TOL:.4f}"
        )

        print(f"GPQA test passed for {eval_config['model_name']}")
