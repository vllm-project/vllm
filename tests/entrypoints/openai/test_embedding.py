# SPDX-License-Identifier: Apache-2.0

import base64
from collections.abc import Sequence
from http import HTTPStatus
from typing import Optional

import numpy as np
import openai
import pytest
import requests

from vllm.entrypoints.openai.protocol import EmbeddingResponse
from vllm.transformers_utils.tokenizer import get_tokenizer

from ...conftest import HfRunner
from ...models.embedding.utils import (EmbedModelInfo, check_embeddings_close,
                                       matryoshka_fy)
from ...utils import RemoteOpenAIServer

MODELS = [
    EmbedModelInfo("intfloat/multilingual-e5-small", is_matryoshka=False),
    EmbedModelInfo("Snowflake/snowflake-arctic-embed-m-v1.5",
                   is_matryoshka=True,
                   matryoshka_dimensions=[256]),
]

DUMMY_CHAT_TEMPLATE = """{% for message in messages %}{{message['role'] + ': ' + message['content'] + '\\n'}}{% endfor %}"""  # noqa: E501

input_texts = [
    "The chef prepared a delicious meal.",
]


@pytest.fixture(scope="module", params=MODELS)
def model_info(request):
    return request.param


@pytest.fixture(scope="module", params=["bfloat16"])
def dtype(request):
    return request.param


@pytest.fixture(scope="module")
def server(model_info, dtype: str):
    args = [
        "--task",
        "embed",
        # use half precision for speed and memory savings in CI environment
        "--dtype",
        dtype,
        "--enforce-eager",
        "--max-model-len",
        "512",
        "--chat-template",
        DUMMY_CHAT_TEMPLATE
    ]

    if model_info.name == "Snowflake/snowflake-arctic-embed-m-v1.5":
        # Manually enable Matryoshka Embeddings
        args.extend([
            "--trust_remote_code", "--hf_overrides",
            '{"matryoshka_dimensions":[256]}'
        ])

    with RemoteOpenAIServer(model_info.name, args) as remote_server:
        yield remote_server


@pytest.fixture(scope="module")
def hf_model(hf_runner, model_info, dtype: str):
    with hf_runner(model_info.name, dtype=dtype,
                   is_sentence_transformer=True) as hf_model:
        yield hf_model


@pytest.mark.asyncio
async def test_basic(model_info: EmbedModelInfo, server: RemoteOpenAIServer):
    response = requests.get(server.url_for("health"))
    assert response.status_code == HTTPStatus.OK

    client = server.get_async_client()

    models = await client.models.list()
    models = models.data

    assert len(models) == 1
    served_model = models[0]
    assert served_model.id == model_info.name
    assert served_model.root == model_info.name


@pytest.mark.asyncio
async def test_single_embedding(model_info: EmbedModelInfo,
                                server: RemoteOpenAIServer,
                                hf_model: HfRunner):
    client = server.get_async_client()

    # test single embedding
    prompts = input_texts
    embedding_response = await client.embeddings.create(
        model=model_info.name,
        input=prompts,
        encoding_format="float",
    )
    embeddings = EmbeddingResponse.model_validate(
        embedding_response.model_dump(mode="json"))

    assert embeddings.id is not None
    assert len(embeddings.data) == 1
    assert len(embeddings.data[0].embedding) > 0
    assert embeddings.usage.completion_tokens == 0
    assert embeddings.usage.prompt_tokens > 0
    assert embeddings.usage.total_tokens > 0

    vllm_outputs = [d.embedding for d in embeddings.data]
    _correctness_test(hf_model, prompts, vllm_outputs)

    # test using token IDs
    input_tokens = [1, 1, 1, 1, 1]
    embedding_response = await client.embeddings.create(
        model=model_info.name,
        input=input_tokens,
        encoding_format="float",
    )
    embeddings = EmbeddingResponse.model_validate(
        embedding_response.model_dump(mode="json"))

    assert embeddings.id is not None
    assert len(embeddings.data) == 1
    assert len(embeddings.data[0].embedding) > 0
    assert embeddings.usage.completion_tokens == 0
    assert embeddings.usage.prompt_tokens == 5
    assert embeddings.usage.total_tokens == 5


@pytest.mark.asyncio
async def test_batch_embedding(model_info: EmbedModelInfo,
                               server: RemoteOpenAIServer, hf_model: HfRunner):
    client = server.get_async_client()

    # test list[str]
    prompts = [
        "The cat sat on the mat.", "A feline was resting on a rug.",
        "Stars twinkle brightly in the night sky."
    ]
    embedding_response = await client.embeddings.create(
        model=model_info.name,
        input=prompts,
        encoding_format="float",
    )
    embeddings = EmbeddingResponse.model_validate(
        embedding_response.model_dump(mode="json"))

    assert embeddings.id is not None
    assert len(embeddings.data) == 3
    assert len(embeddings.data[0].embedding) > 0
    assert embeddings.usage.completion_tokens == 0
    assert embeddings.usage.prompt_tokens > 0
    assert embeddings.usage.total_tokens > 0

    vllm_outputs = [d.embedding for d in embeddings.data]
    _correctness_test(hf_model, prompts, vllm_outputs)

    # test list[list[int]]
    input_tokens = [[4, 5, 7, 9, 20], [15, 29, 499], [24, 24, 24, 24, 24],
                    [25, 32, 64, 77]]
    embedding_response = await client.embeddings.create(
        model=model_info.name,
        input=input_tokens,
        encoding_format="float",
    )
    embeddings = EmbeddingResponse.model_validate(
        embedding_response.model_dump(mode="json"))

    assert embeddings.id is not None
    assert len(embeddings.data) == 4
    assert len(embeddings.data[0].embedding) > 0
    assert embeddings.usage.completion_tokens == 0
    assert embeddings.usage.prompt_tokens == 17
    assert embeddings.usage.total_tokens == 17


@pytest.mark.asyncio
async def test_conversation_embedding(model_info: EmbedModelInfo,
                                      server: RemoteOpenAIServer):
    client = server.get_async_client()

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
            "model": model_info.name,
            "messages": messages,
            "encoding_format": "float",
        },
    )
    chat_response.raise_for_status()
    chat_embeddings = EmbeddingResponse.model_validate(chat_response.json())

    tokenizer = get_tokenizer(tokenizer_name=model_info.name,
                              tokenizer_mode="fast")
    prompt = tokenizer.apply_chat_template(
        messages,
        chat_template=DUMMY_CHAT_TEMPLATE,
        add_generation_prompt=True,
        continue_final_message=False,
        tokenize=False,
    )
    completion_response = await client.embeddings.create(
        model=model_info.name,
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
async def test_batch_base64_embedding(model_info: EmbedModelInfo,
                                      server: RemoteOpenAIServer,
                                      hf_model: HfRunner):
    client = server.get_async_client()

    prompts = [
        "Hello my name is",
        "The best thing about vLLM is that it supports many different models"
    ]

    # test float responses
    responses_float = await client.embeddings.create(input=prompts,
                                                     model=model_info.name,
                                                     encoding_format="float")
    float_data = [d.embedding for d in responses_float.data]
    _correctness_test(hf_model, prompts, float_data)

    # test base64 responses
    responses_base64 = await client.embeddings.create(input=prompts,
                                                      model=model_info.name,
                                                      encoding_format="base64")
    base64_data = []
    for data in responses_base64.data:
        base64_data.append(
            np.frombuffer(base64.b64decode(data.embedding),
                          dtype="float32").tolist())
    _correctness_test(hf_model, prompts, base64_data)

    # Default response is float32 decoded from base64 by OpenAI Client
    responses_default = await client.embeddings.create(input=prompts,
                                                       model=model_info.name)
    default_data = [d.embedding for d in responses_default.data]
    _correctness_test(hf_model, prompts, default_data)


@pytest.mark.asyncio
async def test_embedding_truncation(
    model_info: EmbedModelInfo,
    server: RemoteOpenAIServer,
):
    client = server.get_async_client()

    input_texts = [
        "Como o Brasil pode fomentar o desenvolvimento de modelos de IA?",
    ]

    # test single embedding
    embedding_response = await client.embeddings.create(
        model=model_info.name,
        input=input_texts,
        extra_body={"truncate_prompt_tokens": 10})
    embeddings = EmbeddingResponse.model_validate(
        embedding_response.model_dump(mode="json"))

    assert embeddings.id is not None
    assert len(embeddings.data) == 1
    assert len(embeddings.data[0].embedding) > 0
    assert embeddings.usage.completion_tokens == 0
    assert embeddings.usage.prompt_tokens > 0
    assert embeddings.usage.total_tokens > 0

    input_tokens = [
        1, 24428, 289, 18341, 26165, 285, 19323, 283, 289, 26789, 3871, 28728,
        9901, 340, 2229, 385, 340, 315, 28741, 28804, 2
    ]
    embedding_response = await client.embeddings.create(
        model=model_info.name,
        input=input_tokens,
        extra_body={"truncate_prompt_tokens": 10})
    embeddings = EmbeddingResponse.model_validate(
        embedding_response.model_dump(mode="json"))

    assert embeddings.id is not None
    assert len(embeddings.data) == 1
    assert len(embeddings.data[0].embedding) > 0
    assert embeddings.usage.completion_tokens == 0
    assert embeddings.usage.prompt_tokens == 10
    assert embeddings.usage.total_tokens == 10

    # test_single_embedding_truncation_invalid
    with pytest.raises(openai.BadRequestError):
        response = await client.embeddings.create(
            model=model_info.name,
            input=input_texts,
            extra_body={"truncate_prompt_tokens": 100000})
        assert "error" in response.object
        assert "truncate_prompt_tokens value is greater than max_model_len. "\
               "Please, select a smaller truncation size." in response.message


@pytest.mark.asyncio
async def test_matryoshka(model_info: EmbedModelInfo,
                          server: RemoteOpenAIServer, hf_model: HfRunner):
    client = server.get_async_client()

    async def make_request_and_correctness_test(dimensions):
        prompts = input_texts * 3

        embedding_response = await client.embeddings.create(
            model=model_info.name,
            input=prompts,
            dimensions=dimensions,
            encoding_format="float",
        )
        embeddings = EmbeddingResponse.model_validate(
            embedding_response.model_dump(mode="json"))

        assert embeddings.id is not None
        assert len(embeddings.data) == 3
        assert len(embeddings.data[0].embedding) > 0
        assert embeddings.usage.completion_tokens == 0
        assert embeddings.usage.prompt_tokens > 0
        assert embeddings.usage.total_tokens > 0

        if dimensions is not None:
            assert len(embeddings.data[0].embedding) == dimensions

        vllm_outputs = [d.embedding for d in embeddings.data]
        _correctness_test(hf_model, prompts, vllm_outputs, dimensions)

    if model_info.is_matryoshka:
        valid_dimensions = [None]
        if model_info.matryoshka_dimensions is not None:
            valid_dimensions += model_info.matryoshka_dimensions[:2]

        for dimensions in valid_dimensions:
            await make_request_and_correctness_test(dimensions)

        invalid_dimensions = [-1]
        if model_info.matryoshka_dimensions is not None:
            assert 5 not in model_info.matryoshka_dimensions
            invalid_dimensions.append(5)

        for dimensions in invalid_dimensions:
            with pytest.raises(openai.BadRequestError):
                await make_request_and_correctness_test(dimensions)

    else:
        for dimensions in [None]:
            await make_request_and_correctness_test(dimensions)

        for dimensions in [-1, 16]:
            with pytest.raises(openai.BadRequestError):
                await make_request_and_correctness_test(dimensions)


def _correctness_test(hf_model: HfRunner,
                      inputs,
                      vllm_outputs: Sequence[list[float]],
                      dimensions: Optional[int] = None):

    hf_outputs = hf_model.encode(inputs)
    if dimensions:
        hf_outputs = matryoshka_fy(hf_outputs, dimensions)

    check_embeddings_close(
        embeddings_0_lst=hf_outputs,
        embeddings_1_lst=vllm_outputs,
        name_0="hf",
        name_1="vllm",
        tol=1e-2,
    )
