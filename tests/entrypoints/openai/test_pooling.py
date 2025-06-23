# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import base64

import numpy as np
import pytest
import requests

from tests.models.utils import check_embeddings_close
from vllm.entrypoints.openai.protocol import PoolingResponse
from vllm.transformers_utils.tokenizer import get_tokenizer

from ...utils import RemoteOpenAIServer

MODEL_NAME = "jason9693/Qwen2.5-1.5B-apeach"
DUMMY_CHAT_TEMPLATE = """{% for message in messages %}{{message['role'] + ': ' + message['content'] + '\\n'}}{% endfor %}"""  # noqa: E501


@pytest.fixture(scope="module")
def server():
    args = [
        "--task",
        "classify",
        # use half precision for speed and memory savings in CI environment
        "--dtype",
        "bfloat16",
        "--enforce-eager",
        "--max-model-len",
        "8192",
        "--chat-template",
        DUMMY_CHAT_TEMPLATE,
    ]

    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_single_pooling(server: RemoteOpenAIServer, model_name: str):
    input_texts = [
        "The chef prepared a delicious meal.",
    ]

    # test single pooling
    response = requests.post(
        server.url_for("pooling"),
        json={
            "model": model_name,
            "input": input_texts,
            "encoding_format": "float"
        },
    )
    response.raise_for_status()
    poolings = PoolingResponse.model_validate(response.json())

    assert poolings.id is not None
    assert len(poolings.data) == 1
    assert len(poolings.data[0].data) == 2
    assert poolings.usage.completion_tokens == 0
    assert poolings.usage.prompt_tokens == 7
    assert poolings.usage.total_tokens == 7

    # test using token IDs
    input_tokens = [1, 1, 1, 1, 1]
    response = requests.post(
        server.url_for("pooling"),
        json={
            "model": model_name,
            "input": input_tokens,
            "encoding_format": "float"
        },
    )
    response.raise_for_status()
    poolings = PoolingResponse.model_validate(response.json())

    assert poolings.id is not None
    assert len(poolings.data) == 1
    assert len(poolings.data[0].data) == 2
    assert poolings.usage.completion_tokens == 0
    assert poolings.usage.prompt_tokens == 5
    assert poolings.usage.total_tokens == 5


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_batch_pooling(server: RemoteOpenAIServer, model_name: str):
    # test list[str]
    input_texts = [
        "The cat sat on the mat.", "A feline was resting on a rug.",
        "Stars twinkle brightly in the night sky."
    ]
    response = requests.post(
        server.url_for("pooling"),
        json={
            "model": model_name,
            "input": input_texts,
            "encoding_format": "float"
        },
    )
    response.raise_for_status()
    poolings = PoolingResponse.model_validate(response.json())

    assert poolings.id is not None
    assert len(poolings.data) == 3
    assert len(poolings.data[0].data) == 2
    assert poolings.usage.completion_tokens == 0
    assert poolings.usage.prompt_tokens == 25
    assert poolings.usage.total_tokens == 25

    # test list[list[int]]
    input_tokens = [[4, 5, 7, 9, 20], [15, 29, 499], [24, 24, 24, 24, 24],
                    [25, 32, 64, 77]]
    response = requests.post(
        server.url_for("pooling"),
        json={
            "model": model_name,
            "input": input_tokens,
            "encoding_format": "float"
        },
    )
    response.raise_for_status()
    poolings = PoolingResponse.model_validate(response.json())

    assert poolings.id is not None
    assert len(poolings.data) == 4
    assert len(poolings.data[0].data) == 2
    assert poolings.usage.completion_tokens == 0
    assert poolings.usage.prompt_tokens == 17
    assert poolings.usage.total_tokens == 17


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_conversation_pooling(server: RemoteOpenAIServer,
                                    model_name: str):
    messages = [{
        "role": "user",
        "content": "The cat sat on the mat.",
    }, {
        "role": "assistant",
        "content": "A feline was resting on a rug.",
    }, {
        "role": "user",
        "content": "Stars twinkle brightly in the night sky.",
    }]

    chat_response = requests.post(
        server.url_for("pooling"),
        json={
            "model": model_name,
            "messages": messages,
            "encoding_format": "float",
        },
    )
    chat_response.raise_for_status()
    chat_poolings = PoolingResponse.model_validate(chat_response.json())

    tokenizer = get_tokenizer(tokenizer_name=model_name, tokenizer_mode="fast")
    prompt = tokenizer.apply_chat_template(
        messages,
        chat_template=DUMMY_CHAT_TEMPLATE,
        add_generation_prompt=True,
        continue_final_message=False,
        tokenize=False,
    )
    completions_response = requests.post(
        server.url_for("pooling"),
        json={
            "model": model_name,
            "input": prompt,
            "encoding_format": "float",
            # To be consistent with chat
            "add_special_tokens": False,
        },
    )
    completions_response.raise_for_status()
    completion_poolings = PoolingResponse.model_validate(
        completions_response.json())

    assert chat_poolings.id is not None
    assert completion_poolings.id is not None
    assert chat_poolings.created <= completion_poolings.created
    assert chat_poolings.model_dump(
        exclude={"id", "created"}) == (completion_poolings.model_dump(
            exclude={"id", "created"}))


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_batch_base64_pooling(server: RemoteOpenAIServer,
                                    model_name: str):
    input_texts = [
        "Hello my name is",
        "The best thing about vLLM is that it supports many different models"
    ]

    float_response = requests.post(
        server.url_for("pooling"),
        json={
            "input": input_texts,
            "model": model_name,
            "encoding_format": "float",
        },
    )
    float_response.raise_for_status()
    responses_float = PoolingResponse.model_validate(float_response.json())

    base64_response = requests.post(
        server.url_for("pooling"),
        json={
            "input": input_texts,
            "model": model_name,
            "encoding_format": "base64",
        },
    )
    base64_response.raise_for_status()
    responses_base64 = PoolingResponse.model_validate(base64_response.json())

    decoded_responses_base64_data = []
    for data in responses_base64.data:
        decoded_responses_base64_data.append(
            np.frombuffer(base64.b64decode(data.data),
                          dtype="float32").tolist())

    check_embeddings_close(
        embeddings_0_lst=[d.data for d in responses_float.data],
        embeddings_1_lst=decoded_responses_base64_data,
        name_0="float32",
        name_1="base64")

    # Default response is float32 decoded from base64 by OpenAI Client
    default_response = requests.post(
        server.url_for("pooling"),
        json={
            "input": input_texts,
            "model": model_name,
        },
    )
    default_response.raise_for_status()
    responses_default = PoolingResponse.model_validate(default_response.json())

    check_embeddings_close(
        embeddings_0_lst=[d.data for d in responses_default.data],
        embeddings_1_lst=[d.data for d in responses_default.data],
        name_0="float32",
        name_1="base64")
