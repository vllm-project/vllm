# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Serve a small model on CPU and run a short `vllm bench serve` against it."""

import subprocess

import pytest

from tests.utils import RemoteOpenAIServer
from vllm.platforms import current_platform

pytestmark = [pytest.mark.cpu_model, pytest.mark.skip_global_cleanup]

if not current_platform.is_cpu():
    pytest.skip("skipping CPU-only tests", allow_module_level=True)


@pytest.mark.parametrize("model", ["Qwen/Qwen3-0.6B", "Qwen/Qwen3.5-0.8B"])
def test_serve_and_bench(model: str):
    with RemoteOpenAIServer(
        model, ["--max-model-len", "2048"], max_wait_seconds=600
    ) as server:
        subprocess.run(
            [
                "vllm",
                "bench",
                "serve",
                "--backend",
                "vllm",
                "--dataset-name",
                "random",
                "--model",
                model,
                "--num-prompts",
                "20",
                "--endpoint",
                "/v1/completions",
                "--host",
                server.host,
                "--port",
                str(server.port),
            ],
            check=True,
        )
