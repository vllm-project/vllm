# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import openai  # use the official client for correctness check
import pytest
import pytest_asyncio

from ...utils import RemoteOpenAIServer

# any model with a chat template should work here
MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"
# technically this needs Mistral-7B-v0.1 as base, but we're not testing
# generation quality here


@pytest.fixture(scope="module")
def server(zephyr_lora_files):
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
        f"zephyr-lora={zephyr_lora_files}",
        "--max-lora-rank",
        "64",
        "--max-cpu-loras",
        "2",
        "--max-num-seqs",
        "128",
    ]

    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client() as async_client:
        yield async_client


@pytest.mark.asyncio
async def test_check_models(client: openai.AsyncOpenAI, zephyr_lora_files):
    models = await client.models.list()
    models = models.data
    served_model = models[0]
    lora_models = models[1:]
    assert served_model.id == MODEL_NAME
    assert served_model.root == MODEL_NAME
    assert all(lora_model.root == zephyr_lora_files for lora_model in lora_models)
    assert lora_models[0].id == "zephyr-lora"
