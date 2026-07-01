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
COMPLETION_PROMPT = "The capital of France is"


async def _complete_text(client: openai.AsyncOpenAI, model: str) -> str:
    completion = await client.completions.create(
        model=model,
        prompt=COMPLETION_PROMPT,
        max_tokens=4,
        temperature=0,
    )
    assert completion.choices
    return completion.choices[0].text


async def _unload_lora_adapter(client: openai.AsyncOpenAI, lora_name: str) -> None:
    await client.post(
        "unload_lora_adapter",
        cast_to=str,
        body={"lora_name": lora_name},
    )


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
        "float16",
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
        "4",
        "--max-cpu-loras",
        "4",
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
async def test_static_dora_lineage(
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


@pytest.mark.asyncio
async def test_dynamic_dora_load_unload_reload_and_lineage(
    client: openai.AsyncOpenAI,
    qwen25_05b_dora_files,
):
    adapter_name = DYNAMIC_DORA_ADAPTER_NAME
    loaded = False
    try:
        response = await client.post(
            "load_lora_adapter",
            cast_to=str,
            body={
                "lora_name": adapter_name,
                "lora_path": qwen25_05b_dora_files,
            },
        )
        assert "success" in response.lower()
        loaded = True

        models = await client.models.list()
        dynamic_dora_model = next(
            model for model in models.data if model.id == adapter_name
        )
        assert dynamic_dora_model.root == qwen25_05b_dora_files
        assert dynamic_dora_model.parent == MODEL_NAME

        await _complete_text(client, adapter_name)

        response = await client.post(
            "unload_lora_adapter",
            cast_to=str,
            body={"lora_name": adapter_name},
        )
        assert "success" in response.lower()
        loaded = False

        models = await client.models.list()
        assert adapter_name not in {model.id for model in models.data}

        response = await client.post(
            "load_lora_adapter",
            cast_to=str,
            body={"lora_name": adapter_name, "lora_path": qwen25_05b_dora_files},
        )
        assert "success" in response.lower()
        loaded = True
    finally:
        if loaded:
            await _unload_lora_adapter(client, adapter_name)
