# SPDX-License-Identifier: Apache-2.0
"""
Run `pytest tests/entrypoints/openai/test_embedding_dimensions.py`.
"""

from typing import Optional

import openai
import pytest

from vllm.entrypoints.openai.protocol import EmbeddingResponse

from ...conftest import HfRunner
from ...models.embedding.utils import EmbedModelInfo, correctness_test
from ...utils import RemoteOpenAIServer

MODELS = [
    EmbedModelInfo("intfloat/multilingual-e5-small", is_matryoshka=False),
    EmbedModelInfo("Snowflake/snowflake-arctic-embed-m-v1.5",
                   is_matryoshka=True,
                   matryoshka_dimensions=[256]),
]

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
        "512"
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
        correctness_test(hf_model, prompts, vllm_outputs, dimensions)

    if model_info.is_matryoshka:
        valid_dimensions: list[Optional[int]] = [None]
        if model_info.matryoshka_dimensions is not None:
            valid_dimensions += model_info.matryoshka_dimensions[:2]

        for dimensions in valid_dimensions:
            await make_request_and_correctness_test(dimensions)

        invalid_dimensions: list[Optional[int]] = [-1]
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
