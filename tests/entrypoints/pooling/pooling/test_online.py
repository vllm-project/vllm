# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import base64
import json

import numpy as np
import pytest
import requests
import torch

from tests.models.utils import check_embeddings_close
from tests.utils import RemoteOpenAIServer
from vllm.entrypoints.pooling.pooling.protocol import PoolingResponse
from vllm.tokenizers import get_tokenizer
from vllm.utils.serial_utils import (
    EMBED_DTYPE_TO_TORCH_DTYPE,
    ENDIANNESS,
    MetadataItem,
    binary2tensor,
    decode_pooling_output,
)

MODEL_NAME = "internlm/internlm2-1_8b-reward"
DUMMY_CHAT_TEMPLATE = """{% for message in messages %}{{message['role'] + ': ' + message['content'] + '\\n'}}{% endfor %}"""  # noqa: E501


@pytest.fixture(scope="module")
def server():
    args = [
        "--runner",
        "pooling",
        # use half precision for speed and memory savings in CI environment
        "--dtype",
        "bfloat16",
        "--enforce-eager",
        "--max-model-len",
        "512",
        "--chat-template",
        DUMMY_CHAT_TEMPLATE,
        "--trust-remote-code",
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
        json={"model": model_name, "input": input_texts, "encoding_format": "float"},
    )
    response.raise_for_status()
    poolings = PoolingResponse.model_validate(response.json())

    assert poolings.id is not None
    assert len(poolings.data) == 1
    assert len(poolings.data[0].data) == 8
    assert poolings.usage.completion_tokens == 0
    assert poolings.usage.prompt_tokens == 8
    assert poolings.usage.total_tokens == 8

    # test using token IDs
    input_tokens = [1, 1, 1, 1, 1]
    response = requests.post(
        server.url_for("pooling"),
        json={"model": model_name, "input": input_tokens, "encoding_format": "float"},
    )
    response.raise_for_status()
    poolings = PoolingResponse.model_validate(response.json())

    assert poolings.id is not None
    assert len(poolings.data) == 1
    assert len(poolings.data[0].data) == 5
    assert poolings.usage.completion_tokens == 0
    assert poolings.usage.prompt_tokens == 5
    assert poolings.usage.total_tokens == 5


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_batch_pooling(server: RemoteOpenAIServer, model_name: str):
    # test list[str]
    input_texts = [
        "The cat sat on the mat.",
        "A feline was resting on a rug.",
        "Stars twinkle brightly in the night sky.",
    ]
    response = requests.post(
        server.url_for("pooling"),
        json={"model": model_name, "input": input_texts, "encoding_format": "float"},
    )
    response.raise_for_status()
    poolings = PoolingResponse.model_validate(response.json())

    assert poolings.id is not None
    assert len(poolings.data) == 3
    assert len(poolings.data[0].data) == 8
    assert poolings.usage.completion_tokens == 0
    assert poolings.usage.prompt_tokens == 29
    assert poolings.usage.total_tokens == 29

    # test list[list[int]]
    input_tokens = [
        [4, 5, 7, 9, 20],
        [15, 29, 499],
        [24, 24, 24, 24, 24],
        [25, 32, 64, 77],
    ]
    response = requests.post(
        server.url_for("pooling"),
        json={"model": model_name, "input": input_tokens, "encoding_format": "float"},
    )
    response.raise_for_status()
    poolings = PoolingResponse.model_validate(response.json())

    assert poolings.id is not None
    assert len(poolings.data) == 4
    assert len(poolings.data[0].data) == 5
    assert poolings.usage.completion_tokens == 0
    assert poolings.usage.prompt_tokens == 17
    assert poolings.usage.total_tokens == 17


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_conversation_pooling(server: RemoteOpenAIServer, model_name: str):
    messages = [
        {
            "role": "user",
            "content": "The cat sat on the mat.",
        },
        {
            "role": "assistant",
            "content": "A feline was resting on a rug.",
        },
        {
            "role": "user",
            "content": "Stars twinkle brightly in the night sky.",
        },
    ]

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

    tokenizer = get_tokenizer(tokenizer_name=model_name, trust_remote_code=True)
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
    completion_poolings = PoolingResponse.model_validate(completions_response.json())

    assert chat_poolings.id is not None
    assert completion_poolings.id is not None
    assert chat_poolings.created <= completion_poolings.created
    assert chat_poolings.model_dump(exclude={"id", "created"}) == (
        completion_poolings.model_dump(exclude={"id", "created"})
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_batch_base64_pooling(server: RemoteOpenAIServer, model_name: str):
    input_texts = [
        "Hello my name is",
        "The best thing about vLLM is that it supports many different models",
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
    float_data = [np.array(d.data).squeeze(-1).tolist() for d in responses_float.data]

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
            np.frombuffer(base64.b64decode(data.data), dtype="float32").tolist()
        )

    check_embeddings_close(
        embeddings_0_lst=float_data,
        embeddings_1_lst=decoded_responses_base64_data,
        name_0="float32",
        name_1="base64",
    )

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
    default_data = [
        np.array(d.data).squeeze(-1).tolist() for d in responses_default.data
    ]

    check_embeddings_close(
        embeddings_0_lst=float_data,
        embeddings_1_lst=default_data,
        name_0="float32",
        name_1="default",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_base64_embed_dtype_and_endianness(
    server: RemoteOpenAIServer, model_name: str
):
    input_texts = [
        "The best thing about vLLM is that it supports many different models",
    ]

    url = server.url_for("pooling")
    float_response = requests.post(
        url,
        json={
            "model": model_name,
            "input": input_texts,
            "encoding_format": "float",
        },
    )
    responses_float = PoolingResponse.model_validate(float_response.json())
    float_data = [np.array(d.data).squeeze(-1).tolist() for d in responses_float.data]

    for embed_dtype in EMBED_DTYPE_TO_TORCH_DTYPE:
        for endianness in ENDIANNESS:
            responses_base64 = requests.post(
                url,
                json={
                    "model": model_name,
                    "input": input_texts,
                    "encoding_format": "base64",
                    "embed_dtype": embed_dtype,
                    "endianness": endianness,
                },
            )

            base64_data = []
            for data in responses_base64.json()["data"]:
                binary = base64.b64decode(data["data"])
                tensor = binary2tensor(binary, (-1,), embed_dtype, endianness)
                base64_data.append(tensor.to(torch.float32).tolist())

            check_embeddings_close(
                embeddings_0_lst=float_data,
                embeddings_1_lst=base64_data,
                name_0="float_data",
                name_1="base64_data",
                tol=1e-2,
            )


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_bytes_embed_dtype_and_endianness(
    server: RemoteOpenAIServer, model_name: str
):
    input_texts = [
        "The best thing about vLLM is that it supports many different models",
    ]

    url = server.url_for("pooling")
    float_response = requests.post(
        url,
        json={
            "model": model_name,
            "input": input_texts,
            "encoding_format": "float",
        },
    )
    responses_float = PoolingResponse.model_validate(float_response.json())
    float_data = [np.array(d.data).squeeze(-1).tolist() for d in responses_float.data]

    for embed_dtype in list(EMBED_DTYPE_TO_TORCH_DTYPE.keys()):
        for endianness in ENDIANNESS:
            responses_bytes = requests.post(
                url,
                json={
                    "model": model_name,
                    "input": input_texts,
                    "encoding_format": "bytes",
                    "embed_dtype": embed_dtype,
                    "endianness": endianness,
                },
            )

            metadata = json.loads(responses_bytes.headers["metadata"])
            body = responses_bytes.content
            items = [MetadataItem(**x) for x in metadata["data"]]

            bytes_data = decode_pooling_output(items=items, body=body)
            bytes_data = [x.to(torch.float32).view(-1).tolist() for x in bytes_data]

            check_embeddings_close(
                embeddings_0_lst=float_data,
                embeddings_1_lst=bytes_data,
                name_0="float_data",
                name_1="bytes_data",
                tol=1e-2,
            )


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
@pytest.mark.parametrize("param_name", ["encoding_format", "embed_dtype", "endianness"])
async def test_params_not_supported(
    server: RemoteOpenAIServer, model_name: str, param_name: str
):
    input_texts = [
        "The best thing about vLLM is that it supports many different models",
    ]

    responses_base64 = requests.post(
        server.url_for("pooling"),
        json={
            "model": model_name,
            "input": input_texts,
            "encoding_format": "base64",
            param_name: f"bad_{param_name}",
        },
    )

    assert responses_base64.status_code == 400
    assert "literal_error" in responses_base64.json()["error"]["message"]
    assert f"bad_{param_name}" in responses_base64.json()["error"]["message"]


@pytest.mark.asyncio
async def test_invocations(server: RemoteOpenAIServer):
    input_texts = [
        "The chef prepared a delicious meal.",
    ]

    request_args = {
        "model": MODEL_NAME,
        "input": input_texts,
        "encoding_format": "float",
    }

    completion_response = requests.post(server.url_for("pooling"), json=request_args)
    completion_response.raise_for_status()

    invocation_response = requests.post(
        server.url_for("invocations"), json=request_args
    )
    invocation_response.raise_for_status()

    completion_output = completion_response.json()
    invocation_output = invocation_response.json()

    assert completion_output.keys() == invocation_output.keys()
    for completion_data, invocation_data in zip(
        completion_output["data"], invocation_output["data"]
    ):
        assert completion_data.keys() == invocation_data.keys()
        check_embeddings_close(
            embeddings_0_lst=completion_data["data"],
            embeddings_1_lst=invocation_data["data"],
            name_0="completion",
            name_1="invocation",
        )


@pytest.mark.asyncio
async def test_invocations_conversation(server: RemoteOpenAIServer):
    messages = [
        {
            "role": "user",
            "content": "The cat sat on the mat.",
        },
        {
            "role": "assistant",
            "content": "A feline was resting on a rug.",
        },
        {
            "role": "user",
            "content": "Stars twinkle brightly in the night sky.",
        },
    ]

    request_args = {
        "model": MODEL_NAME,
        "messages": messages,
        "encoding_format": "float",
    }

    chat_response = requests.post(server.url_for("pooling"), json=request_args)
    chat_response.raise_for_status()

    invocation_response = requests.post(
        server.url_for("invocations"), json=request_args
    )
    invocation_response.raise_for_status()

    chat_output = chat_response.json()
    invocation_output = invocation_response.json()

    assert chat_output.keys() == invocation_output.keys()
    for chat_data, invocation_data in zip(
        chat_output["data"], invocation_output["data"]
    ):
        assert chat_data.keys() == invocation_data.keys()
        check_embeddings_close(
            embeddings_0_lst=chat_data["data"],
            embeddings_1_lst=invocation_data["data"],
            name_0="chat",
            name_1="invocation",
        )
