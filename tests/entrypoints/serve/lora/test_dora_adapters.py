# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import openai  # use the official client for correctness check
import pytest
import pytest_asyncio

from tests.utils import RemoteOpenAIServer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DORA_ADAPTER_NAME = "qwen25-dora"
DYNAMIC_DORA_ADAPTER_NAME = "qwen25-dora-dynamic"
DORA_ADAPTER_REPO = "Dhanushkumaramk/healthcare-qwen2.5-0.5B-dora"


@pytest.fixture(scope="session")
def qwen25_05b_dora_files():
    """Download Qwen2.5 DoRA files once per test session."""
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=DORA_ADAPTER_REPO)


@pytest.fixture(scope="module")
def server_with_dora(qwen25_05b_dora_files):
    lora_module = {
        "name": DORA_ADAPTER_NAME,
        "path": qwen25_05b_dora_files,
        "base_model_name": MODEL_NAME,
    }

    args = [
        # use half precision for speed and memory savings in CI environment
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "2048",
        "--enforce-eager",
        # lora config below
        "--enable-lora",
        "--lora-modules",
        json.dumps(lora_module),
        "--max-lora-rank",
        "16",
        "--max-loras",
        "2",
        "--max-cpu-loras",
        "2",
        "--max-num-seqs",
        "16",
    ]

    # Enable the /v1/load_lora_adapter endpoint.
    envs = {"VLLM_ALLOW_RUNTIME_LORA_UPDATING": "True"}

    with RemoteOpenAIServer(MODEL_NAME, args, env_dict=envs) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server_with_dora):
    async with server_with_dora.get_async_client() as async_client:
        yield async_client


@pytest.mark.asyncio
async def test_static_dora_lineage_and_completion(
    client: openai.AsyncOpenAI,
    qwen25_05b_dora_files,
):
    models = await client.models.list()
    models = models.data
    served_model = models[0]
    dora_model = next(model for model in models if model.id == DORA_ADAPTER_NAME)
    assert served_model.id == MODEL_NAME
    assert served_model.root == MODEL_NAME
    assert served_model.parent is None
    assert dora_model.id == DORA_ADAPTER_NAME
    assert dora_model.root == qwen25_05b_dora_files
    assert dora_model.parent == MODEL_NAME

    completion = await client.completions.create(
        model=DORA_ADAPTER_NAME,
        prompt="The capital of France is",
        max_tokens=4,
    )
    assert completion.choices


@pytest.mark.asyncio
async def test_dynamic_dora_lineage_and_completion(
    client: openai.AsyncOpenAI,
    qwen25_05b_dora_files,
):
    response = await client.post(
        "load_lora_adapter",
        cast_to=str,
        body={
            "lora_name": DYNAMIC_DORA_ADAPTER_NAME,
            "lora_path": qwen25_05b_dora_files,
        },
    )
    assert "success" in response.lower()

    models = await client.models.list()
    dynamic_dora_model = next(
        model for model in models.data if model.id == DYNAMIC_DORA_ADAPTER_NAME
    )
    assert dynamic_dora_model.root == qwen25_05b_dora_files
    assert dynamic_dora_model.parent == MODEL_NAME

    completion = await client.completions.create(
        model=DYNAMIC_DORA_ADAPTER_NAME,
        prompt="The capital of France is",
        max_tokens=4,
    )
    assert completion.choices
