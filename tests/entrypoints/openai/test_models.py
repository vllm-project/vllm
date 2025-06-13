# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import openai  # use the official client for correctness check
import pytest
import pytest_asyncio
# downloading lora to test lora requests
from huggingface_hub import snapshot_download

from ...utils import RemoteOpenAIServer

# any model with a chat template should work here
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LORA_NAME = "jashing/tinyllama-colorist-lora"


@pytest.fixture(scope="module")
def lora_files():
    return snapshot_download(repo_id=LORA_NAME)


@pytest.fixture(scope="module")
def server(lora_files):
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
        f"lora={lora_files}",
        f"lora2={lora_files}",
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
async def test_check_models(client: openai.AsyncOpenAI, lora_files):
    models = await client.models.list()
    models = models.data
    served_model = models[0]
    lora_models = models[1:]
    assert served_model.id == MODEL_NAME
    assert served_model.root == MODEL_NAME
    assert all(lora_model.root == lora_files for lora_model in lora_models)
    assert lora_models[0].id == "lora"
    assert lora_models[1].id == "lora2"
