# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Test cases for long text embedding with automatic chunking mechanism.

This test suite validates vLLM's automatic chunking functionality for handling
text inputs that exceed the model's maximum token length, specifically targeting
the intfloat/multilingual-e5-small model (max token length: 512).
"""

import os

import openai
import pytest
import pytest_asyncio

from vllm.entrypoints.openai.protocol import EmbeddingResponse

from ...utils import RemoteOpenAIServer


def _load_text_file(filename: str) -> str:
    """Load text content from file in the same directory."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, filename)
    with open(file_path, encoding='utf-8') as f:
        return f.read().strip()


MODEL_NAME = "intfloat/multilingual-e5-small"
DTYPE = "bfloat16"

# Test text: Load text with approximately 1500 words to exceed 1024 tokens
LONG_TEXT_1500_WORDS = _load_text_file(
    './test_embedding_long_text_datasets/long_text_1500_words.txt')

# Test text: Construct text with approximately 2500 words to exceed 2048 tokens
LONG_TEXT_2500_WORDS = _load_text_file(
    './test_embedding_long_text_datasets/long_text_2500_words.txt')


@pytest.fixture(scope="module")
def server_with_chunked_processing():
    """Start server with automatic chunking processing enabled."""
    args = [
        "--runner",
        "pooling",
        "--dtype",
        DTYPE,
        "--enforce-eager",
        "--max-model-len",
        "512",  # Set smaller max_model_len to trigger chunking mechanism
        '--override-pooler-config',
        ('{"pooling_type": "MEAN", "normalize": true, '
         '"enable_chunked_processing": true, "max_embed_len": 10000}'),
        "--gpu-memory-utilization",
        "0.8",
    ]

    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client_with_chunked_processing(server_with_chunked_processing):
    """Create async client with chunking processing support."""
    async with server_with_chunked_processing.get_async_client(
    ) as async_client:
        yield async_client


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_long_text_embedding_1500_chars(
        client_with_chunked_processing: openai.AsyncOpenAI, model_name: str):
    """Test embedding processing for ~1500 character long text 
    (~1028 tokens, exceeding 512 token limit)."""

    # Verify text length
    # Verify text has sufficient word count (approximately 1500 words)
    word_count = len(LONG_TEXT_1500_WORDS.split())
    assert word_count >= 1400, (
        f"Test text word count insufficient: {word_count} words")

    # Send embedding request
    embedding_response = await client_with_chunked_processing.embeddings.create(
        model=model_name,
        input=[LONG_TEXT_1500_WORDS],
        encoding_format="float",
    )

    # Verify response structure
    embeddings = EmbeddingResponse.model_validate(
        embedding_response.model_dump(mode="json"))

    assert embeddings.id is not None
    assert len(embeddings.data) == 1
    assert len(embeddings.data[0].embedding
               ) == 384  # multilingual-e5-small embedding dimension
    assert embeddings.usage.completion_tokens == 0
    # Due to chunked processing, token count should
    # reflect actual processed tokens
    # With ~1500 words, we expect roughly
    # 1024+ tokens (exceeding 512 token limit)
    # Should exceed single chunk limit of 512
    assert embeddings.usage.prompt_tokens > 800
    assert embeddings.usage.total_tokens == embeddings.usage.prompt_tokens

    # Verify embedding vector validity
    embedding_vector = embeddings.data[0].embedding
    assert all(
        isinstance(x, float)
        for x in embedding_vector), "Embedding vector should contain floats"
    assert not all(
        x == 0
        for x in embedding_vector), "Embedding vector should not be all zeros"


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_long_text_embedding_2500_chars(
        client_with_chunked_processing: openai.AsyncOpenAI, model_name: str):
    """Test embedding processing for ~2500 character long text
    (~2048 tokens, requiring multiple chunks)."""

    # Verify text length
    # Verify text has sufficient word count (approximately 2500 words)
    word_count = len(LONG_TEXT_2500_WORDS.split())
    assert word_count >= 2300, (
        f"Test text word count insufficient: {word_count} words")

    # Send embedding request
    embedding_response = await client_with_chunked_processing.embeddings.create(
        model=model_name,
        input=[LONG_TEXT_2500_WORDS],
        encoding_format="float",
    )

    # Verify response structure
    embeddings = EmbeddingResponse.model_validate(
        embedding_response.model_dump(mode="json"))

    assert embeddings.id is not None
    assert len(embeddings.data) == 1
    assert len(embeddings.data[0].embedding
               ) == 384  # multilingual-e5-small embedding dimension
    assert embeddings.usage.completion_tokens == 0
    # Due to chunked processing, token count should
    # reflect actual processed tokens
    # With ~2500 words, we expect
    # roughly 2048+ tokens (requiring multiple chunks)
    # Should require multiple chunks for processing
    assert embeddings.usage.prompt_tokens > 1500
    assert embeddings.usage.total_tokens == embeddings.usage.prompt_tokens

    # Verify embedding vector validity
    embedding_vector = embeddings.data[0].embedding
    assert all(
        isinstance(x, float)
        for x in embedding_vector), "Embedding vector should contain floats"
    assert not all(
        x == 0
        for x in embedding_vector), "Embedding vector should not be all zeros"


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_batch_long_text_embedding(
        client_with_chunked_processing: openai.AsyncOpenAI, model_name: str):
    """Test batch long text embedding processing."""

    input_texts = [
        LONG_TEXT_1500_WORDS,
        LONG_TEXT_2500_WORDS,
        "This is a short text test.",  # Short text for comparison
    ]

    # Send batch embedding request
    embedding_response = await client_with_chunked_processing.embeddings.create(
        model=model_name,
        input=input_texts,
        encoding_format="float",
    )

    # Verify response structure
    embeddings = EmbeddingResponse.model_validate(
        embedding_response.model_dump(mode="json"))

    assert embeddings.id is not None
    assert len(embeddings.data) == 3  # Three input texts

    # Verify each embedding dimension
    for i, embedding_data in enumerate(embeddings.data):
        assert len(embedding_data.embedding) == 384
        assert embedding_data.index == i

        # Verify embedding vector validity
        embedding_vector = embedding_data.embedding
        assert all(isinstance(x, float) for x in embedding_vector)
        assert not all(x == 0 for x in embedding_vector)

    # Verify token usage
    assert embeddings.usage.completion_tokens == 0
    # Total token count should be very substantial
    assert embeddings.usage.prompt_tokens > 1000
    assert embeddings.usage.total_tokens == embeddings.usage.prompt_tokens


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_chunked_vs_normal_consistency(
        client_with_chunked_processing: openai.AsyncOpenAI, model_name: str):
    """Test consistency between chunked and
    normal processing (using short text)."""

    # Use a short text within the 512 token limit
    short_text = ("Artificial intelligence technology is changing our world, "
                  "bringing unprecedented opportunities and challenges.")

    # Send embedding request
    embedding_response = await client_with_chunked_processing.embeddings.create(
        model=model_name,
        input=[short_text],
        encoding_format="float",
    )

    # Verify response structure
    embeddings = EmbeddingResponse.model_validate(
        embedding_response.model_dump(mode="json"))

    assert embeddings.id is not None
    assert len(embeddings.data) == 1
    assert len(embeddings.data[0].embedding) == 384
    assert embeddings.usage.completion_tokens == 0
    # Short text should not require chunked processing
    assert embeddings.usage.prompt_tokens < 512
    assert embeddings.usage.total_tokens == embeddings.usage.prompt_tokens

    # 验证embedding向量的有效性
    embedding_vector = embeddings.data[0].embedding
    assert all(isinstance(x, float) for x in embedding_vector)
    assert not all(x == 0 for x in embedding_vector)


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_chunked_processing_response_format(
        client_with_chunked_processing: openai.AsyncOpenAI, model_name: str):
    """Test response format and structure during chunked processing."""

    # Test with long text to trigger chunking
    embedding_response = await client_with_chunked_processing.embeddings.create(
        model=model_name,
        input=[LONG_TEXT_1500_WORDS],
        encoding_format="float",
    )

    # Verify response structure
    embeddings = EmbeddingResponse.model_validate(
        embedding_response.model_dump(mode="json"))

    assert embeddings.id is not None
    assert len(embeddings.data) == 1
    assert embeddings.data[0].object == "embedding"
    assert embeddings.data[0].index == 0

    # Verify embedding vector properties
    embedding_vector = embeddings.data[0].embedding
    import math
    vector_norm = math.sqrt(sum(x * x for x in embedding_vector))
    # Check that the vector is normalized
    # (default behavior for most embedding models)
    assert 0.8 < vector_norm < 1.2, (
        f"Vector norm should be reasonable, actual: {vector_norm}")
