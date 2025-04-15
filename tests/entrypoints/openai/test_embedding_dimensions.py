# SPDX-License-Identifier: Apache-2.0
"""
Run `pytest tests/entrypoints/openai/test_embedding_dimensions.py`.
"""

import openai
import pytest

from vllm.entrypoints.openai.protocol import EmbeddingResponse

from ...models.embedding.utils import EmbedModelInfo
from ...utils import RemoteOpenAIServer

MODELS = [
    EmbedModelInfo(name="BAAI/bge-m3", is_matryoshka=False),
    EmbedModelInfo(name="jinaai/jina-embeddings-v3", is_matryoshka=True),
]

input_texts = [
    "The chef prepared a delicious meal.",
] * 3


@pytest.mark.asyncio
@pytest.mark.parametrize("model", MODELS)
async def test_validating_dimensions(model: EmbedModelInfo):
    args = [
        "--task",
        "embed",
        # use half precision for speed and memory savings in CI environment
        "--dtype",
        "bfloat16",
        "--enforce-eager",
        "--max-model-len",
        "512",
        "--trust_remote_code"
    ]
    with RemoteOpenAIServer(model.name, args) as remote_server:
        client = remote_server.get_async_client()

        async def make_request(dimensions):
            embedding_response = await client.embeddings.create(
                model=model.name,
                input=input_texts,
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

        if model.is_matryoshka:
            for dimensions in [None, 16]:
                await make_request(dimensions)

            with pytest.raises(openai.BadRequestError):
                for dimensions in [-1]:
                    await make_request(dimensions)

        else:
            for dimensions in [None]:
                await make_request(dimensions)

            with pytest.raises(openai.BadRequestError):
                for dimensions in [-1, 16]:
                    await make_request(dimensions)
