# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
AITER SmolLM3 release tests for AMD ROCm.

Loads each model, runs health check, and verifies inference with a single
completion. Runs only on AMD ROCm with AITER enabled.
"""

from http import HTTPStatus

import openai
import pytest
import pytest_asyncio
import requests

from tests.models.registry import HF_EXAMPLE_MODELS
from tests.utils import RemoteOpenAIServer
from vllm.platforms import current_platform

AITER_MODEL_LIST = [
    "HuggingFaceTB/SmolLM3-3B",
    "RedHatAI/SmolLM3-3B-FP8-dynamic",
]

pytestmark = [
    pytest.mark.skipif(
        not current_platform.is_rocm(),
        reason="AITER SmolLM3 tests require AMD ROCm",
    ),
]


@pytest.fixture(params=AITER_MODEL_LIST, ids=lambda m: m.split("/")[-1])
def model_name(request):
    return request.param


@pytest.fixture
def default_server_args():
    """Server args for small SmolLM3 models on ROCm with AITER."""
    return [
        "--enforce-eager",
        "--max-model-len",
        "1024",
        "--max-num-seqs",
        "256",
        "--gpu-memory-utilization",
        "0.85",
        "--tensor-parallel-size",
        "1",
        "--attention-backend",
        "ROCM_AITER_UNIFIED_ATTN",
    ]


@pytest.fixture
def server(default_server_args, model_name):
    """Start vLLM HTTP server for the given model."""
    model_info = HF_EXAMPLE_MODELS.find_hf_info(model_name)
    model_info.check_available_online(on_fail="skip")
    model_info.check_transformers_version(on_fail="skip")

    env_dict = {"VLLM_ROCM_USE_AITER": "1"}

    with RemoteOpenAIServer(
        model_name, default_server_args, env_dict=env_dict
    ) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    """Get async OpenAI client for testing."""
    async with server.get_async_client() as async_client:
        yield async_client


@pytest.mark.asyncio
async def test_load_and_health_check(
    server: RemoteOpenAIServer, client: openai.AsyncOpenAI, model_name: str
):
    """
    Load model, verify health endpoint, and run one completion.

    Server startup already waits for /health. This test explicitly
    checks /health and runs a single completion to verify inference.
    """
    # Health check
    response = requests.get(server.url_for("health"))
    assert response.status_code == HTTPStatus.OK, (
        f"Health check failed: {response.status_code}"
    )

    # Single completion to verify inference
    completion = await client.completions.create(
        model=model_name,
        prompt="Hello, how are you?",
        max_tokens=32,
        temperature=0.0,
    )

    assert completion.choices is not None and len(completion.choices) == 1
    choice = completion.choices[0]
    assert choice.text is not None and len(choice.text) > 0
    assert completion.usage is not None
    assert completion.usage.completion_tokens > 0
