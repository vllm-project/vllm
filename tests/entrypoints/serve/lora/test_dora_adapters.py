# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import json
import shutil
from contextlib import suppress

import openai  # use the official client for correctness check
import pytest
import pytest_asyncio
from safetensors.torch import load_file, save_file

from tests.lora.utils import convert_dora_checkpoint_to_lora
from tests.utils import RemoteOpenAIServer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DORA_ADAPTER_NAME = "qwen25-dora"
DYNAMIC_DORA_ADAPTER_NAME = "qwen25-dora-dynamic"
LORA_ADAPTER_NAME = "qwen25-lora-from-dora"
DORA_ADAPTER_REPO = "Dhanushkumaramk/healthcare-qwen2.5-0.5B-dora"
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


async def _unload_lora_adapter_if_exists(
    client: openai.AsyncOpenAI, lora_name: str
) -> None:
    with suppress(openai.NotFoundError):
        await _unload_lora_adapter(client, lora_name)


@pytest.fixture(scope="session")
def qwen25_05b_dora_files():
    """Download Qwen2.5 DoRA files once per test session."""
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=DORA_ADAPTER_REPO)


@pytest.fixture(scope="module")
def qwen25_05b_lora_files(tmp_path_factory, qwen25_05b_dora_files):
    lora_dir = tmp_path_factory.mktemp("qwen25_05b_lora_from_dora")
    return convert_dora_checkpoint_to_lora(qwen25_05b_dora_files, lora_dir)


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
        prompt=COMPLETION_PROMPT,
        max_tokens=4,
    )
    assert completion.choices


@pytest.mark.asyncio
async def test_dynamic_dora_lineage_and_completion(
    client: openai.AsyncOpenAI,
    qwen25_05b_dora_files,
):
    loaded = False
    try:
        response = await client.post(
            "load_lora_adapter",
            cast_to=str,
            body={
                "lora_name": DYNAMIC_DORA_ADAPTER_NAME,
                "lora_path": qwen25_05b_dora_files,
            },
        )
        assert "success" in response.lower()
        loaded = True

        models = await client.models.list()
        dynamic_dora_model = next(
            model for model in models.data if model.id == DYNAMIC_DORA_ADAPTER_NAME
        )
        assert dynamic_dora_model.root == qwen25_05b_dora_files
        assert dynamic_dora_model.parent == MODEL_NAME

        await _complete_text(client, DYNAMIC_DORA_ADAPTER_NAME)
    finally:
        if loaded:
            await _unload_lora_adapter(client, DYNAMIC_DORA_ADAPTER_NAME)


@pytest.mark.asyncio
async def test_concurrent_base_lora_and_dora_requests(
    client: openai.AsyncOpenAI,
    qwen25_05b_lora_files,
):
    loaded = False
    try:
        response = await client.post(
            "load_lora_adapter",
            cast_to=str,
            body={
                "lora_name": LORA_ADAPTER_NAME,
                "lora_path": qwen25_05b_lora_files,
                "load_inplace": True,
            },
        )
        assert "success" in response.lower()
        loaded = True

        models = await client.models.list()
        model_ids = {model.id for model in models.data}
        assert MODEL_NAME in model_ids
        assert DORA_ADAPTER_NAME in model_ids
        assert LORA_ADAPTER_NAME in model_ids

        async def complete(model: str):
            return await client.completions.create(
                model=model,
                prompt=COMPLETION_PROMPT,
                max_tokens=4,
            )

        completions = await asyncio.gather(
            complete(MODEL_NAME),
            complete(LORA_ADAPTER_NAME),
            complete(DORA_ADAPTER_NAME),
        )
        for completion in completions:
            assert completion.choices
    finally:
        if loaded:
            await _unload_lora_adapter(client, LORA_ADAPTER_NAME)


@pytest.mark.asyncio
async def test_dora_unload_reload_and_replace_clears_slot_state(
    client: openai.AsyncOpenAI,
    qwen25_05b_dora_files,
    qwen25_05b_lora_files,
):
    adapter_name = "replaceable-dora"
    clean_adapter_name = "clean-lora-from-dora"

    try:
        response = await client.post(
            "load_lora_adapter",
            cast_to=str,
            body={"lora_name": adapter_name, "lora_path": qwen25_05b_dora_files},
        )
        assert "success" in response.lower()
        await _complete_text(client, adapter_name)

        response = await client.post(
            "unload_lora_adapter",
            cast_to=str,
            body={"lora_name": adapter_name},
        )
        assert "success" in response.lower()
        models = await client.models.list()
        assert adapter_name not in {model.id for model in models.data}

        response = await client.post(
            "load_lora_adapter",
            cast_to=str,
            body={"lora_name": adapter_name, "lora_path": qwen25_05b_dora_files},
        )
        assert "success" in response.lower()
        await _complete_text(client, adapter_name)

        response = await client.post(
            "load_lora_adapter",
            cast_to=str,
            body={
                "lora_name": adapter_name,
                "lora_path": qwen25_05b_lora_files,
                "load_inplace": True,
            },
        )
        assert "success" in response.lower()
        replaced_lora_text = await _complete_text(client, adapter_name)

        response = await client.post(
            "load_lora_adapter",
            cast_to=str,
            body={
                "lora_name": clean_adapter_name,
                "lora_path": qwen25_05b_lora_files,
            },
        )
        assert "success" in response.lower()
        assert replaced_lora_text == await _complete_text(client, clean_adapter_name)
    finally:
        for lora_name in (adapter_name, clean_adapter_name):
            await _unload_lora_adapter_if_exists(client, lora_name)


@pytest.mark.asyncio
async def test_multiple_dora_adapters_concurrently(
    client: openai.AsyncOpenAI,
    qwen25_05b_dora_files,
):
    async def load_and_run_adapter(adapter_name: str):
        loaded = False
        try:
            response = await client.post(
                "load_lora_adapter",
                cast_to=str,
                body={"lora_name": adapter_name, "lora_path": qwen25_05b_dora_files},
            )
            assert "success" in response.lower()
            loaded = True
            for _ in range(2):
                await _complete_text(client, adapter_name)
        finally:
            if loaded:
                await _unload_lora_adapter(client, adapter_name)

    tasks = [
        asyncio.create_task(load_and_run_adapter(f"dynamic-dora-{i}"))
        for i in range(3)
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for result in results:
        assert not isinstance(result, Exception), f"Got exception {result}"


@pytest.mark.asyncio
async def test_invalid_dora_adapter_does_not_break_loaded_dora(
    client: openai.AsyncOpenAI,
    tmp_path,
    qwen25_05b_dora_files,
):
    invalid_files = tmp_path / "invalid_dora_missing_magnitude"
    shutil.copytree(qwen25_05b_dora_files, invalid_files)

    weights_path = invalid_files / "adapter_model.safetensors"
    tensors = load_file(weights_path)
    tensors = {
        name: tensor
        for name, tensor in tensors.items()
        if not name.endswith("lora_magnitude_vector")
    }
    save_file(tensors, weights_path)

    for idx in range(3):
        with pytest.raises(
            openai.InternalServerError, match="missing lora_magnitude_vector"
        ):
            await client.post(
                "load_lora_adapter",
                cast_to=str,
                body={
                    "lora_name": f"invalid-dora-{idx}",
                    "lora_path": str(invalid_files),
                },
            )

    completion = await client.completions.create(
        model=DORA_ADAPTER_NAME,
        prompt=COMPLETION_PROMPT,
        max_tokens=4,
    )
    assert completion.choices
    models = await client.models.list()
    model_ids = {model.id for model in models.data}
    for idx in range(3):
        assert f"invalid-dora-{idx}" not in model_ids
