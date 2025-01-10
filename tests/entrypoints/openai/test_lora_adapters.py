import asyncio
import json
import shutil
from contextlib import suppress

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

    # Enable the /v1/load_lora_adapter endpoint
    envs = {"VLLM_ALLOW_RUNTIME_LORA_UPDATING": "True"}

    with RemoteOpenAIServer(MODEL_NAME, args, env_dict=envs) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server_with_lora_modules_json):
    async with server_with_lora_modules_json.get_async_client(
    ) as async_client:
        yield async_client


@pytest.mark.asyncio
async def test_static_lora_lineage(client: openai.AsyncOpenAI,
                                   zephyr_lora_files):
    models = await client.models.list()
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


@pytest.mark.asyncio
async def test_dynamic_lora_lineage(client: openai.AsyncOpenAI,
                                    zephyr_lora_files):

    response = await client.post("load_lora_adapter",
                                 cast_to=str,
                                 body={
                                     "lora_name": "zephyr-lora-3",
                                     "lora_path": zephyr_lora_files
                                 })
    # Ensure adapter loads before querying /models
    assert "success" in response

    models = await client.models.list()
    models = models.data
    dynamic_lora_model = models[-1]
    assert dynamic_lora_model.root == zephyr_lora_files
    assert dynamic_lora_model.parent == MODEL_NAME
    assert dynamic_lora_model.id == "zephyr-lora-3"


@pytest.mark.asyncio
async def test_dynamic_lora_not_found(client: openai.AsyncOpenAI):
    with pytest.raises(openai.NotFoundError):
        await client.post("load_lora_adapter",
                          cast_to=str,
                          body={
                              "lora_name": "notfound",
                              "lora_path": "/not/an/adapter"
                          })


@pytest.mark.asyncio
async def test_dynamic_lora_invalid_files(client: openai.AsyncOpenAI,
                                          tmp_path):
    invalid_files = tmp_path / "invalid_files"
    invalid_files.mkdir()
    (invalid_files / "adapter_config.json").write_text("this is not json")

    with pytest.raises(openai.BadRequestError):
        await client.post("load_lora_adapter",
                          cast_to=str,
                          body={
                              "lora_name": "invalid-json",
                              "lora_path": str(invalid_files)
                          })


@pytest.mark.asyncio
async def test_dynamic_lora_invalid_lora_rank(client: openai.AsyncOpenAI,
                                              tmp_path, zephyr_lora_files):
    invalid_rank = tmp_path / "invalid_rank"

    # Copy adapter from zephyr_lora_files to invalid_rank
    shutil.copytree(zephyr_lora_files, invalid_rank)

    with open(invalid_rank / "adapter_config.json") as f:
        adapter_config = json.load(f)

    print(adapter_config)

    # assert False

    # Change rank to invalid value
    adapter_config["r"] = 1024
    with open(invalid_rank / "adapter_config.json", "w") as f:
        json.dump(adapter_config, f)

    with pytest.raises(openai.BadRequestError,
                       match="is greater than max_lora_rank"):
        await client.post("load_lora_adapter",
                          cast_to=str,
                          body={
                              "lora_name": "invalid-json",
                              "lora_path": str(invalid_rank)
                          })


@pytest.mark.asyncio
async def test_multiple_lora_adapters(client: openai.AsyncOpenAI, tmp_path,
                                      zephyr_lora_files):
    """Validate that many loras can be dynamically registered and inferenced 
    with concurrently"""

    # This test file configures the server with --max-cpu-loras=2 and this test
    # will concurrently load 10 adapters, so it should flex the LRU cache
    async def load_and_run_adapter(adapter_name: str):
        await client.post("load_lora_adapter",
                          cast_to=str,
                          body={
                              "lora_name": adapter_name,
                              "lora_path": str(zephyr_lora_files)
                          })
        for _ in range(3):
            await client.completions.create(
                model=adapter_name,
                prompt=["Hello there", "Foo bar bazz buzz"],
                max_tokens=5,
            )

    lora_tasks = []
    for i in range(10):
        lora_tasks.append(
            asyncio.create_task(load_and_run_adapter(f"adapter_{i}")))

    results, _ = await asyncio.wait(lora_tasks)

    for r in results:
        assert not isinstance(r, Exception), f"Got exception {r}"


@pytest.mark.asyncio
async def test_loading_invalid_adapters_does_not_break_others(
        client: openai.AsyncOpenAI, tmp_path, zephyr_lora_files):

    invalid_files = tmp_path / "invalid_files"
    invalid_files.mkdir()
    (invalid_files / "adapter_config.json").write_text("this is not json")

    stop_good_requests_event = asyncio.Event()

    async def run_good_requests(client):
        # Run chat completions requests until event set

        results = []

        while not stop_good_requests_event.is_set():
            try:
                batch = await client.completions.create(
                    model="zephyr-lora",
                    prompt=["Hello there", "Foo bar bazz buzz"],
                    max_tokens=5,
                )
                results.append(batch)
            except Exception as e:
                results.append(e)

        return results

    # Create task to run good requests
    good_task = asyncio.create_task(run_good_requests(client))

    # Run a bunch of bad adapter loads
    for _ in range(25):
        with suppress(openai.NotFoundError):
            await client.post("load_lora_adapter",
                              cast_to=str,
                              body={
                                  "lora_name": "notfound",
                                  "lora_path": "/not/an/adapter"
                              })
    for _ in range(25):
        with suppress(openai.BadRequestError):
            await client.post("load_lora_adapter",
                              cast_to=str,
                              body={
                                  "lora_name": "invalid",
                                  "lora_path": str(invalid_files)
                              })

    # Ensure all the running requests with lora adapters succeeded
    stop_good_requests_event.set()
    results = await good_task
    for r in results:
        assert not isinstance(r, Exception), f"Got exception {r}"

    # Ensure we can load another adapter and run it
    await client.post("load_lora_adapter",
                      cast_to=str,
                      body={
                          "lora_name": "valid",
                          "lora_path": zephyr_lora_files
                      })
    await client.completions.create(
        model="valid",
        prompt=["Hello there", "Foo bar bazz buzz"],
        max_tokens=5,
    )
