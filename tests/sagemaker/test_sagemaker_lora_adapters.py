# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import openai  # use the official client for correctness check
import pytest
import pytest_asyncio
import requests

from ..utils import RemoteOpenAIServer

# any model with a chat template should work here
MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"
# technically this needs Mistral-7B-v0.1 as base, but we're not testing
# generation quality here


@pytest.fixture(scope="session")
def zephyr_lora_files():
    """Download zephyr LoRA files once per test session."""
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id="typeof/zephyr-7b-beta-lora")


@pytest.fixture(scope="module", params=[True])
def server_with_lora_modules_json(request, zephyr_lora_files):
    # Define the json format LoRA module configurations
    lora_module_1 = {
        "name": "zephyr-lora",
        "path": zephyr_lora_files,
        "base_model_name": MODEL_NAME,
    }

    args = [
        # use half precision for speed and memory savings in CI environment
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "8192",
        "--enforce-eager",
        # lora config below
        "--enable-lora",
        "--lora-modules",
        json.dumps(lora_module_1),
        "--max-lora-rank",
        "64",
        "--max-cpu-loras",
        "2",
        "--max-num-seqs",
        "64",
    ]

    # Enable the /v1/load_lora_adapter endpoint
    envs = {"VLLM_ALLOW_RUNTIME_LORA_UPDATING": "True"}

    with RemoteOpenAIServer(MODEL_NAME, args, env_dict=envs) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server_with_lora_modules_json):
    async with server_with_lora_modules_json.get_async_client() as async_client:
        yield async_client


@pytest.mark.asyncio
async def test_sagemaker_load_adapter_happy_path(
    client: openai.AsyncOpenAI,
    server_with_lora_modules_json: RemoteOpenAIServer,
    zephyr_lora_files,
):
    # The SageMaker standards library creates a POST /adapters endpoint
    # that maps to the load_lora_adapter handler with request shape:
    # {"lora_name": "body.name", "lora_path": "body.src"}
    load_response = requests.post(
        server_with_lora_modules_json.url_for("adapters"),
        json={"name": "zephyr-lora-sagemaker", "src": zephyr_lora_files},
    )
    load_response.raise_for_status()

    models = await client.models.list()
    models = models.data
    dynamic_lora_model = models[-1]
    assert dynamic_lora_model.root == zephyr_lora_files
    assert dynamic_lora_model.parent == MODEL_NAME
    assert dynamic_lora_model.id == "zephyr-lora-sagemaker"


@pytest.mark.asyncio
async def test_sagemaker_unload_adapter_happy_path(
    client: openai.AsyncOpenAI,
    server_with_lora_modules_json: RemoteOpenAIServer,
    zephyr_lora_files,
):
    # First, load an adapter
    adapter_name = "zephyr-lora-sagemaker-unload"
    load_response = requests.post(
        server_with_lora_modules_json.url_for("adapters"),
        json={"name": adapter_name, "src": zephyr_lora_files},
    )
    load_response.raise_for_status()

    # Verify it's in the models list
    models = await client.models.list()
    adapter_ids = [model.id for model in models.data]
    assert adapter_name in adapter_ids

    # Now unload it using DELETE /adapters/{adapter_name}
    # The SageMaker standards maps this to unload_lora_adapter with:
    # {"lora_name": "path_params.adapter_name"}
    unload_response = requests.delete(
        server_with_lora_modules_json.url_for("adapters", adapter_name),
    )
    unload_response.raise_for_status()

    # Verify it's no longer in the models list
    models = await client.models.list()
    adapter_ids = [model.id for model in models.data]
    assert adapter_name not in adapter_ids


@pytest.mark.asyncio
async def test_sagemaker_load_adapter_not_found(
    server_with_lora_modules_json: RemoteOpenAIServer,
):
    load_response = requests.post(
        server_with_lora_modules_json.url_for("adapters"),
        json={"name": "nonexistent-adapter", "src": "/path/does/not/exist"},
    )
    assert load_response.status_code == 404


@pytest.mark.asyncio
async def test_sagemaker_load_adapter_invalid_files(
    server_with_lora_modules_json: RemoteOpenAIServer,
    tmp_path,
):
    invalid_files = tmp_path / "invalid_adapter"
    invalid_files.mkdir()
    (invalid_files / "adapter_config.json").write_text("not valid json")

    load_response = requests.post(
        server_with_lora_modules_json.url_for("adapters"),
        json={"name": "invalid-adapter", "src": str(invalid_files)},
    )
    assert load_response.status_code == 400


@pytest.mark.asyncio
async def test_sagemaker_unload_nonexistent_adapter(
    server_with_lora_modules_json: RemoteOpenAIServer,
):
    # Attempt to unload an adapter that doesn't exist
    unload_response = requests.delete(
        server_with_lora_modules_json.url_for("adapters", "nonexistent-adapter-name"),
    )
    assert unload_response.status_code in (400, 404)


@pytest.mark.asyncio
async def test_sagemaker_invocations_with_adapter(
    server_with_lora_modules_json: RemoteOpenAIServer,
    zephyr_lora_files,
):
    # First, load an adapter via SageMaker endpoint
    adapter_name = "zephyr-lora-invoke-test"
    load_response = requests.post(
        server_with_lora_modules_json.url_for("adapters"),
        json={"name": adapter_name, "src": zephyr_lora_files},
    )
    load_response.raise_for_status()

    # Now test the /invocations endpoint with the adapter
    invocation_response = requests.post(
        server_with_lora_modules_json.url_for("invocations"),
        headers={
            "X-Amzn-SageMaker-Adapter-Identifier": adapter_name,
        },
        json={
            "prompt": "Hello, how are you?",
            "max_tokens": 10,
        },
    )
    invocation_response.raise_for_status()
    invocation_output = invocation_response.json()

    # Verify we got a valid completion response
    assert "choices" in invocation_output
    assert len(invocation_output["choices"]) > 0
    assert "text" in invocation_output["choices"][0]


@pytest.mark.asyncio
async def test_sagemaker_multiple_adapters_load_unload(
    client: openai.AsyncOpenAI,
    server_with_lora_modules_json: RemoteOpenAIServer,
    zephyr_lora_files,
):
    adapter_names = [f"sagemaker-adapter-{i}" for i in range(5)]

    # Load all adapters
    for adapter_name in adapter_names:
        load_response = requests.post(
            server_with_lora_modules_json.url_for("adapters"),
            json={"name": adapter_name, "src": zephyr_lora_files},
        )
        load_response.raise_for_status()

    # Verify all are in the models list
    models = await client.models.list()
    adapter_ids = [model.id for model in models.data]
    for adapter_name in adapter_names:
        assert adapter_name in adapter_ids

    # Unload all adapters
    for adapter_name in adapter_names:
        unload_response = requests.delete(
            server_with_lora_modules_json.url_for("adapters", adapter_name),
        )
        unload_response.raise_for_status()

    # Verify all are removed from models list
    models = await client.models.list()
    adapter_ids = [model.id for model in models.data]
    for adapter_name in adapter_names:
        assert adapter_name not in adapter_ids
