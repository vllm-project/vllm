# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import base64

import numpy as np
import openai
import pytest
import pytest_asyncio
import requests
import torch
import torch.nn.functional as F

from vllm.entrypoints.openai.protocol import EmbeddingResponse
from vllm.transformers_utils.tokenizer import get_tokenizer

from ...models.language.pooling.embed_utils import (
    run_embedding_correctness_test)
from ...models.utils import check_embeddings_close
from ...utils import RemoteOpenAIServer

MODEL_NAME = "intfloat/multilingual-e5-small"
DUMMY_CHAT_TEMPLATE = """{% for message in messages %}{{message['role'] + ': ' + message['content'] + '\\n'}}{% endfor %}"""  # noqa: E501
DTYPE = "bfloat16"


@pytest.fixture(autouse=True)
def v1(run_with_both_engines):
    # Simple autouse wrapper to run both engines for each test
    # This can be promoted up to conftest.py to run for every
    # test in a package
    pass


@pytest.fixture(scope="module")
def server():
    args = [
        "--runner",
        "pooling",
        # use half precision for speed and memory savings in CI environment
        "--dtype",
        DTYPE,
        "--enforce-eager",
        "--max-model-len",
        "512",
        "--chat-template",
        DUMMY_CHAT_TEMPLATE,
    ]

    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client() as async_client:
        yield async_client


@pytest.fixture(scope="module")
def hf_model(hf_runner):
    with hf_runner(MODEL_NAME, dtype=DTYPE,
                   is_sentence_transformer=True) as hf_model:
        yield hf_model


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_single_embedding(hf_model, client: openai.AsyncOpenAI,
                                model_name: str):
    input_texts = [
        "The chef prepared a delicious meal.",
    ]

    # test single embedding
    embedding_response = await client.embeddings.create(
        model=model_name,
        input=input_texts,
        encoding_format="float",
    )
    embeddings = EmbeddingResponse.model_validate(
        embedding_response.model_dump(mode="json"))

    assert embeddings.id is not None
    assert len(embeddings.data) == 1
    assert len(embeddings.data[0].embedding) == 384
    assert embeddings.usage.completion_tokens == 0
    assert embeddings.usage.prompt_tokens == 11
    assert embeddings.usage.total_tokens == 11

    vllm_outputs = [d.embedding for d in embeddings.data]
    run_embedding_correctness_test(hf_model, input_texts, vllm_outputs)

    # test using token IDs
    input_tokens = [1, 1, 1, 1, 1]
    embedding_response = await client.embeddings.create(
        model=model_name,
        input=input_tokens,
        encoding_format="float",
    )
    embeddings = EmbeddingResponse.model_validate(
        embedding_response.model_dump(mode="json"))

    assert embeddings.id is not None
    assert len(embeddings.data) == 1
    assert len(embeddings.data[0].embedding) == 384
    assert embeddings.usage.completion_tokens == 0
    assert embeddings.usage.prompt_tokens == 5
    assert embeddings.usage.total_tokens == 5


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_batch_embedding(hf_model, client: openai.AsyncOpenAI,
                               model_name: str):
    # test list[str]
    input_texts = [
        "The cat sat on the mat.", "A feline was resting on a rug.",
        "Stars twinkle brightly in the night sky."
    ]
    embedding_response = await client.embeddings.create(
        model=model_name,
        input=input_texts,
        encoding_format="float",
    )
    embeddings = EmbeddingResponse.model_validate(
        embedding_response.model_dump(mode="json"))

    assert embeddings.id is not None
    assert len(embeddings.data) == 3
    assert len(embeddings.data[0].embedding) == 384
    assert embeddings.usage.completion_tokens == 0
    assert embeddings.usage.prompt_tokens == 33
    assert embeddings.usage.total_tokens == 33

    vllm_outputs = [d.embedding for d in embeddings.data]
    run_embedding_correctness_test(hf_model, input_texts, vllm_outputs)

    # test list[list[int]]
    input_tokens = [[4, 5, 7, 9, 20], [15, 29, 499], [24, 24, 24, 24, 24],
                    [25, 32, 64, 77]]
    embedding_response = await client.embeddings.create(
        model=model_name,
        input=input_tokens,
        encoding_format="float",
    )
    embeddings = EmbeddingResponse.model_validate(
        embedding_response.model_dump(mode="json"))

    assert embeddings.id is not None
    assert len(embeddings.data) == 4
    assert len(embeddings.data[0].embedding) == 384
    assert embeddings.usage.completion_tokens == 0
    assert embeddings.usage.prompt_tokens == 17
    assert embeddings.usage.total_tokens == 17


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_conversation_embedding(server: RemoteOpenAIServer,
                                      client: openai.AsyncOpenAI,
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
        server.url_for("v1/embeddings"),
        json={
            "model": model_name,
            "messages": messages,
            "encoding_format": "float",
        },
    )
    chat_response.raise_for_status()
    chat_embeddings = EmbeddingResponse.model_validate(chat_response.json())

    tokenizer = get_tokenizer(tokenizer_name=model_name, tokenizer_mode="fast")
    prompt = tokenizer.apply_chat_template(
        messages,
        chat_template=DUMMY_CHAT_TEMPLATE,
        add_generation_prompt=True,
        continue_final_message=False,
        tokenize=False,
    )
    completion_response = await client.embeddings.create(
        model=model_name,
        input=prompt,
        encoding_format="float",
        # To be consistent with chat
        extra_body={"add_special_tokens": False},
    )
    completion_embeddings = EmbeddingResponse.model_validate(
        completion_response.model_dump(mode="json"))

    assert chat_embeddings.id is not None
    assert completion_embeddings.id is not None
    assert chat_embeddings.created <= completion_embeddings.created
    assert chat_embeddings.model_dump(
        exclude={"id", "created"}) == (completion_embeddings.model_dump(
            exclude={"id", "created"}))


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_batch_base64_embedding(hf_model, client: openai.AsyncOpenAI,
                                      model_name: str):
    input_texts = [
        "Hello my name is",
        "The best thing about vLLM is that it supports many different models"
    ]

    responses_float = await client.embeddings.create(input=input_texts,
                                                     model=model_name,
                                                     encoding_format="float")
    float_data = [d.embedding for d in responses_float.data]
    run_embedding_correctness_test(hf_model, input_texts, float_data)

    responses_base64 = await client.embeddings.create(input=input_texts,
                                                      model=model_name,
                                                      encoding_format="base64")
    base64_data = []
    for data in responses_base64.data:
        base64_data.append(
            np.frombuffer(base64.b64decode(data.embedding),
                          dtype="float32").tolist())

    run_embedding_correctness_test(hf_model, input_texts, base64_data)

    # Default response is float32 decoded from base64 by OpenAI Client
    responses_default = await client.embeddings.create(input=input_texts,
                                                       model=model_name)
    default_data = [d.embedding for d in responses_default.data]
    run_embedding_correctness_test(hf_model, input_texts, default_data)


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_single_embedding_truncation(client: openai.AsyncOpenAI,
                                           model_name: str):
    input_texts = [
        "Como o Brasil pode fomentar o desenvolvimento de modelos de IA?",
    ]

    # test single embedding
    embedding_response = await client.embeddings.create(
        model=model_name,
        input=input_texts,
        extra_body={"truncate_prompt_tokens": 10})
    embeddings = EmbeddingResponse.model_validate(
        embedding_response.model_dump(mode="json"))

    assert embeddings.id is not None
    assert len(embeddings.data) == 1
    assert len(embeddings.data[0].embedding) == 384
    assert embeddings.usage.completion_tokens == 0
    assert embeddings.usage.prompt_tokens == 10
    assert embeddings.usage.total_tokens == 10

    input_tokens = [
        1, 24428, 289, 18341, 26165, 285, 19323, 283, 289, 26789, 3871, 28728,
        9901, 340, 2229, 385, 340, 315, 28741, 28804, 2
    ]
    embedding_response = await client.embeddings.create(
        model=model_name,
        input=input_tokens,
        extra_body={"truncate_prompt_tokens": 10})
    embeddings = EmbeddingResponse.model_validate(
        embedding_response.model_dump(mode="json"))

    assert embeddings.id is not None
    assert len(embeddings.data) == 1
    assert len(embeddings.data[0].embedding) == 384
    assert embeddings.usage.completion_tokens == 0
    assert embeddings.usage.prompt_tokens == 10
    assert embeddings.usage.total_tokens == 10


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_single_embedding_truncation_invalid(client: openai.AsyncOpenAI,
                                                   model_name: str):
    input_texts = [
        "Como o Brasil pode fomentar o desenvolvimento de modelos de IA?",
    ]

    with pytest.raises(openai.BadRequestError):
        response = await client.embeddings.create(
            model=model_name,
            input=input_texts,
            extra_body={"truncate_prompt_tokens": 8193})
        assert "error" in response.object
        assert "truncate_prompt_tokens value is greater than max_model_len. "\
               "Please, select a smaller truncation size." in response.message


@pytest.mark.asyncio
async def test_invocations(server: RemoteOpenAIServer,
                           client: openai.AsyncOpenAI):
    input_texts = [
        "The chef prepared a delicious meal.",
    ]

    request_args = {
        "model": MODEL_NAME,
        "input": input_texts,
        "encoding_format": "float",
    }

    completion_response = await client.embeddings.create(**request_args)

    invocation_response = requests.post(server.url_for("invocations"),
                                        json=request_args)
    invocation_response.raise_for_status()

    completion_output = completion_response.model_dump()
    invocation_output = invocation_response.json()

    assert completion_output.keys() == invocation_output.keys()
    for completion_data, invocation_data in zip(completion_output["data"],
                                                invocation_output["data"]):
        assert completion_data.keys() == invocation_data.keys()
        check_embeddings_close(embeddings_0_lst=[completion_data["embedding"]],
                               embeddings_1_lst=[invocation_data["embedding"]],
                               name_0="completion",
                               name_1="invocation")


@pytest.mark.asyncio
async def test_invocations_conversation(server: RemoteOpenAIServer):
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

    request_args = {
        "model": MODEL_NAME,
        "messages": messages,
        "encoding_format": "float",
    }

    chat_response = requests.post(server.url_for("v1/embeddings"),
                                  json=request_args)
    chat_response.raise_for_status()

    invocation_response = requests.post(server.url_for("invocations"),
                                        json=request_args)
    invocation_response.raise_for_status()

    chat_output = chat_response.json()
    invocation_output = invocation_response.json()

    assert chat_output.keys() == invocation_output.keys()
    for chat_data, invocation_data in zip(chat_output["data"],
                                          invocation_output["data"]):
        assert chat_data.keys() == invocation_data.keys()
        check_embeddings_close(embeddings_0_lst=[chat_data["embedding"]],
                               embeddings_1_lst=[invocation_data["embedding"]],
                               name_0="chat",
                               name_1="invocation")


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_normalize(server: RemoteOpenAIServer, model_name: str):
    input_text = ["The chef prepared a delicious meal."]

    async def get_outputs(normalize):
        request_args = {
            "model": MODEL_NAME,
            "input": input_text,
            "encoding_format": "float",
            "normalize": normalize
        }

        response = requests.post(server.url_for("v1/embeddings"),
                                 json=request_args)
        outputs = response.json()

        return torch.tensor([x['embedding'] for x in outputs["data"]])

    default = await get_outputs(normalize=None)
    w_normal = await get_outputs(normalize=True)
    wo_normal = await get_outputs(normalize=False)

    assert torch.allclose(default, w_normal,
                          atol=1e-2), "Default should use normal."
    assert not torch.allclose(w_normal, wo_normal,
                              atol=1e-2), "wo_normal should not use normal."
    assert torch.allclose(
        w_normal, F.normalize(wo_normal, p=2, dim=-1),
        atol=1e-2), "w_normal should be close to normal(wo_normal)."
