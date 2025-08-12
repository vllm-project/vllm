import json

import openai  # use the official client for correctness check
import pytest
import pytest_asyncio
# downloading lora to test lora requests
from huggingface_hub import snapshot_download

from ...utils import RemoteOpenAIServer

# any model with a chat template should work here
MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"
# technically this needs Mistral-7B-v0.1 as base, but we're not testing
# generation quality here
LORA_NAME = "typeof/zephyr-7b-beta-lora"


@pytest.fixture(scope="module")
def zephyr_lora_files():
    return snapshot_download(repo_id=LORA_NAME)


@pytest.fixture(scope="module")
def server_with_lora_modules_json(zephyr_lora_files):
    # Define the json format LoRA module configurations
    lora_module_1 = {
        "name": "zephyr-lora",
        "path": zephyr_lora_files,
        "base_model_name": MODEL_NAME
    }

    lora_module_2 = {
        "name": "zephyr-lora2",
        "path": zephyr_lora_files,
        "base_model_name": MODEL_NAME
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
        json.dumps(lora_module_2),
        "--max-lora-rank",
        "64",
        "--max-cpu-loras",
        "2",
        "--max-num-seqs",
        "64",
    ]

    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client_for_lora_lineage(server_with_lora_modules_json):
    async with server_with_lora_modules_json.get_async_client(
    ) as async_client:
        yield async_client


@pytest.mark.asyncio
async def test_check_lora_lineage(client_for_lora_lineage: openai.AsyncOpenAI,
                                  zephyr_lora_files):
    models = await client_for_lora_lineage.models.list()
    models = models.data
    served_model = models[0]
    lora_models = models[1:]
    assert served_model.id == MODEL_NAME
    assert served_model.root == MODEL_NAME
    assert served_model.parent is None
    assert all(lora_model.root == zephyr_lora_files
               for lora_model in lora_models)
    assert all(lora_model.parent == MODEL_NAME for lora_model in lora_models)
    assert lora_models[0].id == "zephyr-lora"
    assert lora_models[1].id == "zephyr-lora2"
