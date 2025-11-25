# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Test cases for long text embedding with automatic chunking mechanism.

This test suite validates vLLM's automatic chunking functionality for handling
text inputs that exceed the model's maximum token length, specifically targeting
the intfloat/multilingual-e5-small model (max token length: 512).
"""

import random

import openai
import pytest
import pytest_asyncio

from tests.utils import RemoteOpenAIServer
from vllm.entrypoints.openai.protocol import EmbeddingResponse
from vllm.platforms import current_platform

if current_platform.is_rocm():
    pytest.skip(
        "Encoder self-attention is not implemented on ROCm.", allow_module_level=True
    )


def _generate_random_text(word_count: int) -> str:
    """Generate random text with approximately the specified word count."""
    # Common English words with focus on verbs and nouns for realistic text
    common_words = [
        # Essential articles and pronouns (minimal)
        "the",
        "and",
        "you",
        "they",
        "this",
        "that",
        "these",
        "those",
        # Action verbs
        "create",
        "build",
        "develop",
        "design",
        "implement",
        "execute",
        "analyze",
        "process",
        "generate",
        "calculate",
        "evaluate",
        "optimize",
        "transform",
        "integrate",
        "configure",
        "deploy",
        "monitor",
        "manage",
        "discover",
        "explore",
        "investigate",
        "research",
        "study",
        "examine",
        "improve",
        "enhance",
        "upgrade",
        "modify",
        "update",
        "maintain",
        "solve",
        "resolve",
        "handle",
        "address",
        "tackle",
        "overcome",
        "communicate",
        "collaborate",
        "coordinate",
        "organize",
        "plan",
        "achieve",
        "accomplish",
        "complete",
        "finish",
        "deliver",
        "provide",
        # Technology and science nouns
        "system",
        "application",
        "software",
        "hardware",
        "network",
        "database",
        "algorithm",
        "model",
        "framework",
        "platform",
        "interface",
        "protocol",
        "architecture",
        "infrastructure",
        "component",
        "module",
        "service",
        "technology",
        "innovation",
        "solution",
        "methodology",
        "approach",
        "artificial",
        "intelligence",
        "machine",
        "learning",
        "neural",
        "network",
        "computer",
        "processor",
        "memory",
        "storage",
        "computation",
        "data",
        "information",
        "knowledge",
        "insight",
        "pattern",
        "trend",
        "analysis",
        "research",
        "development",
        "engineering",
        "science",
        "mathematics",
        "statistics",
        "probability",
        "optimization",
        "performance",
        "efficiency",
        # General nouns
        "project",
        "team",
        "organization",
        "company",
        "business",
        "industry",
        "market",
        "customer",
        "user",
        "client",
        "product",
        "feature",
        "function",
        "requirement",
        "specification",
        "documentation",
        "report",
        "result",
        "outcome",
        "impact",
        "benefit",
        "advantage",
        "challenge",
        "problem",
        "opportunity",
        "strategy",
        "goal",
        "objective",
        "target",
        "milestone",
        "process",
        "procedure",
        "workflow",
        "pipeline",
        "operation",
        "task",
        "activity",
        "event",
        "session",
        "meeting",
        "discussion",
        "decision",
    ]

    words = []
    for _ in range(word_count):
        words.append(random.choice(common_words))

    # Add some punctuation for more realistic text
    text = " ".join(words)
    # Add periods every 10-20 words
    words_list = text.split()
    result = []
    for i, word in enumerate(words_list):
        result.append(word)
        if (i + 1) % random.randint(10, 20) == 0 and i < len(words_list) - 1:
            result[-1] += "."

    return " ".join(result)


MODEL_NAME = "intfloat/multilingual-e5-small"
DTYPE = "bfloat16"

# Test text: Generate text with approximately 1500 words to exceed 1024 tokens
LONG_TEXT_1500_WORDS = _generate_random_text(1500)

# Test text: Generate text with approximately 2500 words to exceed 2048 tokens
LONG_TEXT_2500_WORDS = _generate_random_text(2500)


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
        "--pooler-config",
        (
            '{"pooling_type": "MEAN", "normalize": true, '
            '"enable_chunked_processing": true, "max_embed_len": 10000}'
        ),
        "--gpu-memory-utilization",
        "0.8",
    ]

    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client_with_chunked_processing(server_with_chunked_processing):
    """Create async client with chunking processing support."""
    async with server_with_chunked_processing.get_async_client() as async_client:
        yield async_client


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_long_text_embedding_1500_chars(
    client_with_chunked_processing: openai.AsyncOpenAI, model_name: str
):
    """Test embedding processing for ~1500 character long text
    (~1028 tokens, exceeding 512 token limit)."""

    # Verify text length
    # Verify text has sufficient word count (approximately 1500 words)
    word_count = len(LONG_TEXT_1500_WORDS.split())
    assert word_count >= 1400, f"Test text word count insufficient: {word_count} words"

    # Send embedding request
    embedding_response = await client_with_chunked_processing.embeddings.create(
        model=model_name,
        input=[LONG_TEXT_1500_WORDS],
        encoding_format="float",
    )

    # Verify response structure
    embeddings = EmbeddingResponse.model_validate(
        embedding_response.model_dump(mode="json")
    )

    assert embeddings.id is not None
    assert len(embeddings.data) == 1
    assert (
        len(embeddings.data[0].embedding) == 384
    )  # multilingual-e5-small embedding dimension
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
    assert all(isinstance(x, float) for x in embedding_vector), (
        "Embedding vector should contain floats"
    )
    assert not all(x == 0 for x in embedding_vector), (
        "Embedding vector should not be all zeros"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_long_text_embedding_2500_chars(
    client_with_chunked_processing: openai.AsyncOpenAI, model_name: str
):
    """Test embedding processing for ~2500 character long text
    (~2048 tokens, requiring multiple chunks)."""

    # Verify text length
    # Verify text has sufficient word count (approximately 2500 words)
    word_count = len(LONG_TEXT_2500_WORDS.split())
    assert word_count >= 2300, f"Test text word count insufficient: {word_count} words"

    # Send embedding request
    embedding_response = await client_with_chunked_processing.embeddings.create(
        model=model_name,
        input=[LONG_TEXT_2500_WORDS],
        encoding_format="float",
    )

    # Verify response structure
    embeddings = EmbeddingResponse.model_validate(
        embedding_response.model_dump(mode="json")
    )

    assert embeddings.id is not None
    assert len(embeddings.data) == 1
    assert (
        len(embeddings.data[0].embedding) == 384
    )  # multilingual-e5-small embedding dimension
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
    assert all(isinstance(x, float) for x in embedding_vector), (
        "Embedding vector should contain floats"
    )
    assert not all(x == 0 for x in embedding_vector), (
        "Embedding vector should not be all zeros"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_batch_long_text_embedding(
    client_with_chunked_processing: openai.AsyncOpenAI, model_name: str
):
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
        embedding_response.model_dump(mode="json")
    )

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
    client_with_chunked_processing: openai.AsyncOpenAI, model_name: str
):
    """Test consistency between chunked and
    normal processing (using short text)."""

    # Use a short text within the 512 token limit
    short_text = (
        "Artificial intelligence technology is changing our world, "
        "bringing unprecedented opportunities and challenges."
    )

    # Send embedding request
    embedding_response = await client_with_chunked_processing.embeddings.create(
        model=model_name,
        input=[short_text],
        encoding_format="float",
    )

    # Verify response structure
    embeddings = EmbeddingResponse.model_validate(
        embedding_response.model_dump(mode="json")
    )

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
    client_with_chunked_processing: openai.AsyncOpenAI, model_name: str
):
    """Test response format and structure during chunked processing."""

    # Test with long text to trigger chunking
    embedding_response = await client_with_chunked_processing.embeddings.create(
        model=model_name,
        input=[LONG_TEXT_1500_WORDS],
        encoding_format="float",
    )

    # Verify response structure
    embeddings = EmbeddingResponse.model_validate(
        embedding_response.model_dump(mode="json")
    )

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
        f"Vector norm should be reasonable, actual: {vector_norm}"
    )
