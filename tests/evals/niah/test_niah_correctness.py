# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Needle-in-a-Haystack (NIAH) long-context evaluation.

Tests KV cache quantization quality at various context lengths by inserting
a retrievable fact at different positions and measuring recall accuracy.

Usage:
    pytest -s -v tests/evals/niah/test_niah_correctness.py \
        --config-list-file=configs/models-turboquant.txt
"""

import asyncio
import shlex

import pytest
import yaml
from transformers import AutoTokenizer

from tests.utils import RemoteOpenAIServer

from .niah_eval import evaluate_niah


def test_niah_correctness(config_filename):
    eval_config = yaml.safe_load(config_filename.read_text(encoding="utf-8"))

    server_args_str = eval_config.get("server_args", "")
    server_args = shlex.split(server_args_str) if server_args_str else []
    server_args.extend(["--trust-remote-code", "--disable-uvicorn-access-log"])

    model_name = eval_config["model_name"]
    context_lengths = eval_config.get("context_lengths", [4096, 8192, 16384, 32768])
    depths = eval_config.get("depths", [0.0, 0.25, 0.5, 0.75, 1.0])
    accuracy_threshold = eval_config.get("accuracy_threshold", 0.8)
    num_trials = eval_config.get("num_trials", 3)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    print(f"Starting NIAH evaluation for model: {model_name}")
    print(f"Context lengths: {context_lengths}")
    print(f"Depths: {depths}")
    print(f"Trials per cell: {num_trials}")
    print(f"Accuracy threshold: {accuracy_threshold}")
    print(f"Server args: {' '.join(server_args)}")

    with RemoteOpenAIServer(
        model_name,
        server_args,
        max_wait_seconds=eval_config.get("startup_max_wait_seconds", 600),
    ) as remote_server:
        url = remote_server.url_for("health")
        host = url.rsplit("/health", 1)[0]
        if ":" in host.rsplit("//", 1)[-1]:
            base, port_str = host.rsplit(":", 1)
            port = int(port_str)
            host = base
        else:
            port = 8000

        results = asyncio.run(
            evaluate_niah(
                host=host,
                port=port,
                model=model_name,
                context_lengths=context_lengths,
                depths=depths,
                tokenizer=tokenizer,
                num_trials=num_trials,
            )
        )

    overall = results["overall_accuracy"]
    print(f"\nNIAH Results for {model_name}:")
    print(f"  Overall accuracy: {overall:.3f}")
    print(f"  Threshold: {accuracy_threshold}")
    print(f"  Total probes: {results['total']}")
    print(f"  Correct: {results['correct']}")

    assert overall >= accuracy_threshold, (
        f"NIAH accuracy {overall:.3f} below threshold {accuracy_threshold} "
        f"for {model_name}"
    )
    print(f"✅ NIAH test passed for {model_name}")
