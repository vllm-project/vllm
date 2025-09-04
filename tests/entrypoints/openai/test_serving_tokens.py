# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import httpx
import pytest
import pytest_asyncio
from transformers import AutoTokenizer

from vllm.config import ModelConfig

from ...utils import RemoteOpenAIServer

MODEL_NAME = "Qwen/Qwen3-0.6B"
GEN_ENDPOINT = "/v1/generate"


def get_vocab_size(model_name):
    config = ModelConfig(
        model=model_name,
        seed=0,
        dtype="bfloat16",
    )
    return config.get_vocab_size()


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)


@pytest.fixture(scope="module")
def server():
    args = [
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "1024",
        "--enforce-eager",
    ]

    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server: RemoteOpenAIServer):
    transport = httpx.AsyncHTTPTransport(
        uds=server.uds) if server.uds else None
    headers = {"Authorization": f"Bearer {server.DUMMY_API_KEY}"}
    async with httpx.AsyncClient(
            transport=transport,
            base_url=server.url_root,
            timeout=600,
            headers=headers,
    ) as c:
        yield c


@pytest.mark.asyncio
async def test_generate_endpoint(client):
    payload = {
        "model": MODEL_NAME,
        "token_ids": [1, 2, 3],
        "sampling_params": {
            "max_tokens": 5
        },
        "stream": False,
    }
    resp = await client.post(GEN_ENDPOINT, json=payload)
    resp.raise_for_status()
    data = resp.json()
    assert "choices" in data


@pytest.mark.asyncio
async def test_same_response_as_chat_completions(client, tokenizer):
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": "How many countries are in the EU?"
        },
    ]
    token_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=False,  # default with Qwen3
    )
    payload = {
        "model": MODEL_NAME,
        "token_ids": token_ids,
        "sampling_params": {
            "max_tokens": 24,
            "temperature": 0.2
        },
        "stream": False,
    }
    generate_resp = await client.post(GEN_ENDPOINT, json=payload)
    generate_data = generate_resp.json()
    generate_res = tokenizer.decode(generate_data["choices"][0]["token_ids"],
                                    skip_special_tokens=True)

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": 24,
        "temperature": 0.2,
        "stream": False,
        "chat_template_kwargs": dict(enable_thinking=False)
    }
    completions_resp = await client.post("/v1/chat/completions", json=payload)
    completions_data = completions_resp.json()
    completions_res = completions_data["choices"][0]["message"]["content"]

    assert generate_res == completions_res
