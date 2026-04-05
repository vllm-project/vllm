# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Integration tests for usage policy configuration across all API endpoints.
"""

import anthropic
import openai
import pytest
import pytest_asyncio

from tests.utils import RemoteOpenAIServer

# =============================================================================
# Chat APIs (OpenAI, Anthropic, etc.)
# =============================================================================

CHAT_MODEL_NAME = "Qwen/Qwen3-0.6B"
CHAT_BASE_ARGS = [
    "--max-model-len",
    "4096",
    "--gpu-memory-utilization",
    "0.8",
    "--enforce-eager",
]


@pytest.fixture(scope="class")
def chat_server(request):
    """Class-level fixture for Chat model tests."""
    with RemoteOpenAIServer(CHAT_MODEL_NAME, CHAT_BASE_ARGS) as remote_server:
        request.cls.server = remote_server
        yield remote_server


@pytest_asyncio.fixture
async def chat_client(chat_server):
    async with chat_server.get_async_client() as async_client:
        yield async_client


@pytest_asyncio.fixture
async def chat_anthropic_client(chat_server):
    async with chat_server.get_async_client_anthropic() as async_client:
        yield async_client


@pytest.mark.usefixtures("chat_server")
class TestDefaultUsagePolicy:
    """Usage policy tests for all APIs using Chat model."""

    # -------------------------------------------------------------------------
    # Chat Completion tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_chat_completion_non_streaming(self, chat_client: openai.AsyncOpenAI):
        response = await chat_client.chat.completions.create(
            model=CHAT_MODEL_NAME,
            messages=[{"role": "user", "content": "Hello"}],
            max_completion_tokens=5,
            temperature=0.0,
            stream=False,
        )

        assert response.usage is not None
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0

    @pytest.mark.asyncio
    async def test_chat_completion_streaming_no_usage(
        self, chat_client: openai.AsyncOpenAI
    ):
        stream = await chat_client.chat.completions.create(
            model=CHAT_MODEL_NAME,
            messages=[{"role": "user", "content": "Hello"}],
            max_completion_tokens=5,
            temperature=0.0,
            stream=True,
        )

        chunk_count = 0
        usage_in_chunks = False

        async for chunk in stream:
            chunk_count += 1
            if chunk.usage is not None and not chunk.choices:
                usage_in_chunks = True

        assert not usage_in_chunks
        assert chunk_count > 0

    @pytest.mark.asyncio
    async def test_chat_completion_streaming_with_usage_option(
        self, chat_client: openai.AsyncOpenAI
    ):
        stream = await chat_client.chat.completions.create(
            model=CHAT_MODEL_NAME,
            messages=[{"role": "user", "content": "Hello"}],
            max_completion_tokens=5,
            temperature=0.0,
            stream=True,
            stream_options={"include_usage": True},
        )

        chunk_count = 0
        final_chunk_with_usage = False

        async for chunk in stream:
            chunk_count += 1
            if chunk.usage is not None and not chunk.choices:
                final_chunk_with_usage = True
                assert chunk.usage.prompt_tokens > 0

        assert final_chunk_with_usage

    # -------------------------------------------------------------------------
    # Completion tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_completion_non_streaming(self, chat_client: openai.AsyncOpenAI):
        response = await chat_client.completions.create(
            model=CHAT_MODEL_NAME,
            prompt="Hello",
            max_tokens=5,
            temperature=0.0,
            stream=False,
        )

        assert response.usage is not None
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0

    @pytest.mark.asyncio
    async def test_completion_streaming_no_usage(self, chat_client: openai.AsyncOpenAI):
        stream = await chat_client.completions.create(
            model=CHAT_MODEL_NAME,
            prompt="Hello",
            max_tokens=5,
            temperature=0.0,
            stream=True,
        )

        chunk_count = 0
        usage_in_chunks = False

        async for chunk in stream:
            chunk_count += 1
            if chunk.usage is not None and not chunk.choices:
                usage_in_chunks = True

        assert not usage_in_chunks
        assert chunk_count > 0

    @pytest.mark.asyncio
    async def test_completion_streaming_with_usage_option(
        self, chat_client: openai.AsyncOpenAI
    ):
        stream = await chat_client.completions.create(
            model=CHAT_MODEL_NAME,
            prompt="Hello",
            max_tokens=5,
            temperature=0.0,
            stream=True,
            stream_options={"include_usage": True},
        )

        chunk_count = 0
        final_chunk_with_usage = False

        async for chunk in stream:
            chunk_count += 1
            if chunk.usage is not None and not chunk.choices:
                final_chunk_with_usage = True
                assert chunk.usage.prompt_tokens > 0

        assert final_chunk_with_usage

    # -------------------------------------------------------------------------
    # Responses tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_responses_non_streaming(self, chat_client: openai.AsyncOpenAI):
        response = await chat_client.responses.create(
            model=CHAT_MODEL_NAME,
            input="Hello",
        )

        assert response.usage is not None
        assert response.usage.input_tokens > 0
        assert response.usage.output_tokens > 0
        assert response.usage.total_tokens > 0

    @pytest.mark.asyncio
    async def test_responses_streaming(self, chat_client: openai.AsyncOpenAI):
        stream = await chat_client.responses.create(
            model=CHAT_MODEL_NAME,
            input="Hello",
            stream=True,
        )

        events_with_usage_before_completed = []
        final_event = None

        async for event in stream:
            if event.type == "response.completed":
                final_event = event
                break
            if (
                hasattr(event, "response")
                and event.response
                and event.response.usage is not None
            ):
                events_with_usage_before_completed.append(event.type)

        assert final_event is not None
        assert len(events_with_usage_before_completed) == 0

        assert final_event.response.usage is not None
        assert final_event.response.usage.input_tokens > 0
        assert final_event.response.usage.output_tokens > 0

    # -------------------------------------------------------------------------
    # Anthropic tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_anthropic_non_streaming(
        self, chat_anthropic_client: anthropic.AsyncAnthropic
    ):
        response = await chat_anthropic_client.messages.create(
            model=CHAT_MODEL_NAME,
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello"}],
        )

        assert response.usage is not None
        assert response.usage.input_tokens > 0
        assert response.usage.output_tokens > 0

    @pytest.mark.asyncio
    async def test_anthropic_streaming(
        self, chat_anthropic_client: anthropic.AsyncAnthropic
    ):
        stream = await chat_anthropic_client.messages.create(
            model=CHAT_MODEL_NAME,
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello"}],
            stream=True,
        )

        message_start_usage = None
        message_delta_usage = None

        async for event in stream:
            if event.type == "message_start":
                message_start_usage = event.message.usage
            elif event.type == "message_delta":
                message_delta_usage = event.usage

        assert message_start_usage is not None
        assert message_start_usage.input_tokens > 0

        assert message_delta_usage is not None
        assert message_delta_usage.output_tokens > 0


@pytest.fixture(scope="class")
def chat_server_always(request):
    """Class-level fixture for Chat model tests."""
    with RemoteOpenAIServer(
        CHAT_MODEL_NAME,
        CHAT_BASE_ARGS
        + [
            "--include-usage-policy",
            "always",
            "--continuous-usage-policy",
            "always",
        ],
    ) as remote_server:
        request.cls.server = remote_server
        yield remote_server


@pytest_asyncio.fixture
async def chat_client_always(chat_server_always):
    async with chat_server_always.get_async_client() as async_client:
        yield async_client


@pytest_asyncio.fixture
async def chat_anthropic_client_always(chat_server_always):
    async with chat_server_always.get_async_client_anthropic() as async_client:
        yield async_client


@pytest.mark.usefixtures("chat_server_always")
class TestAlwaysUsagePolicy:
    """Usage policy tests for all APIs using Chat model with always policy enabled."""

    # -------------------------------------------------------------------------
    # Chat Completion tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_chat_completion_non_streaming(
        self, chat_client_always: openai.AsyncOpenAI
    ):
        response = await chat_client_always.chat.completions.create(
            model=CHAT_MODEL_NAME,
            messages=[{"role": "user", "content": "Hello"}],
            max_completion_tokens=5,
            temperature=0.0,
            stream=False,
        )

        assert response.usage is not None
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0

    @pytest.mark.asyncio
    async def test_chat_completion_streaming(
        self, chat_client_always: openai.AsyncOpenAI
    ):
        stream = await chat_client_always.chat.completions.create(
            model=CHAT_MODEL_NAME,
            messages=[{"role": "user", "content": "Hello"}],
            max_completion_tokens=5,
            temperature=0.0,
            stream=True,
        )

        async for chunk in stream:
            assert chunk.usage is not None
            assert chunk.usage.prompt_tokens > 0

    @pytest.mark.asyncio
    async def test_chat_completion_streaming_with_stream_options(
        self, chat_client_always: openai.AsyncOpenAI
    ):
        stream = await chat_client_always.chat.completions.create(
            model=CHAT_MODEL_NAME,
            messages=[{"role": "user", "content": "Hello"}],
            max_completion_tokens=5,
            temperature=0.0,
            stream=True,
            stream_options={"include_usage": False, "continuous_usage": False},
        )

        # should ignore stream_options
        async for chunk in stream:
            assert chunk.usage is not None
            assert chunk.usage.prompt_tokens > 0

    # -------------------------------------------------------------------------
    # Completion tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_completion_non_streaming(
        self, chat_client_always: openai.AsyncOpenAI
    ):
        response = await chat_client_always.completions.create(
            model=CHAT_MODEL_NAME,
            prompt="Hello",
            max_tokens=5,
            temperature=0.0,
            stream=False,
        )

        assert response.usage is not None
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0

    @pytest.mark.asyncio
    async def test_completion_streaming(self, chat_client_always: openai.AsyncOpenAI):
        stream = await chat_client_always.completions.create(
            model=CHAT_MODEL_NAME,
            prompt="Hello",
            max_tokens=5,
            temperature=0.0,
            stream=True,
        )

        async for chunk in stream:
            assert chunk.usage is not None
            assert chunk.usage.prompt_tokens > 0

    @pytest.mark.asyncio
    async def test_completion_streaming_with_usage_option(
        self, chat_client_always: openai.AsyncOpenAI
    ):
        stream = await chat_client_always.completions.create(
            model=CHAT_MODEL_NAME,
            prompt="Hello",
            max_tokens=5,
            temperature=0.0,
            stream=True,
            stream_options={"include_usage": False, "continuous_usage": False},
        )

        # should ignore stream_options
        async for chunk in stream:
            assert chunk.usage is not None
            assert chunk.usage.prompt_tokens > 0

    # -------------------------------------------------------------------------
    # Responses tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_responses_non_streaming(
        self, chat_client_always: openai.AsyncOpenAI
    ):
        response = await chat_client_always.responses.create(
            model=CHAT_MODEL_NAME,
            input="Hello",
        )

        assert response.usage is not None
        assert response.usage.input_tokens > 0
        assert response.usage.output_tokens > 0
        assert response.usage.total_tokens > 0

    # TODO responses_streaming

    # -------------------------------------------------------------------------
    # Anthropic tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_anthropic_non_streaming(
        self, chat_anthropic_client_always: anthropic.AsyncAnthropic
    ):
        response = await chat_anthropic_client_always.messages.create(
            model=CHAT_MODEL_NAME,
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello"}],
        )

        assert response.usage is not None
        assert response.usage.input_tokens > 0
        assert response.usage.output_tokens > 0

    @pytest.mark.asyncio
    async def test_anthropic_streaming(
        self, chat_anthropic_client_always: anthropic.AsyncAnthropic
    ):
        stream = await chat_anthropic_client_always.messages.create(
            model=CHAT_MODEL_NAME,
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello"}],
            stream=True,
        )

        message_start_usage = None
        message_delta_usage = None

        async for event in stream:
            if event.type == "message_start":
                message_start_usage = event.message.usage
            elif event.type == "message_delta":
                message_delta_usage = event.usage

        assert message_start_usage is not None
        assert message_start_usage.input_tokens > 0

        assert message_delta_usage is not None
        assert message_delta_usage.output_tokens > 0


# =============================================================================
# Group 2: Audio
# =============================================================================

AUDIO_MODEL_NAME = "openai/whisper-large-v3-turbo"
AUDIO_BASE_ARGS = [
    "--gpu-memory-utilization",
    "0.8",
    "--enforce-eager",
]


@pytest.fixture(scope="class")
def audio_server(request):
    """Class-level fixture for Audio model tests."""
    with RemoteOpenAIServer(AUDIO_MODEL_NAME, AUDIO_BASE_ARGS) as remote_server:
        request.cls.server = remote_server
        yield remote_server


@pytest_asyncio.fixture
async def audio_client(audio_server):
    async with audio_server.get_async_client() as async_client:
        yield async_client


@pytest.fixture
def winning_call():
    """Audio fixture for speech-to-text tests."""
    from vllm.assets.audio import AudioAsset

    path = AudioAsset("winning_call").get_local_path()
    with open(str(path), "rb") as f:
        yield f


@pytest.mark.usefixtures("audio_server")
class TestSpeechToTextUsagePolicy:
    """Usage policy tests for Speech-to-Text API."""

    @pytest.mark.asyncio
    async def test_transcription_non_streaming(
        self, audio_client: openai.AsyncOpenAI, winning_call
    ):
        response = await audio_client.audio.transcriptions.create(
            model=AUDIO_MODEL_NAME,
            file=winning_call,
            language="en",
            temperature=0.0,
            response_format="json",
        )

        assert response.usage is not None
        assert response.usage.seconds > 0

    @pytest.mark.asyncio
    async def test_transcription_streaming(
        self, audio_client: openai.AsyncOpenAI, winning_call
    ):
        stream = await audio_client.audio.transcriptions.create(
            model=AUDIO_MODEL_NAME,
            file=winning_call,
            language="en",
            temperature=0.0,
            response_format="json",
            stream=True,
        )

        chunk_count = 0
        usage_in_chunks = False

        async for chunk in stream:
            chunk_count += 1
            has_usage = hasattr(chunk, "usage") and chunk.usage is not None
            if not chunk.choices and has_usage:
                usage_in_chunks = True

        assert not usage_in_chunks
        assert chunk_count > 0


# =============================================================================
# Embeddings
# =============================================================================

EMBED_MODEL_NAME = "BAAI/bge-base-en-v1.5"
EMBED_BASE_ARGS = [
    "--runner",
    "pooling",
    "--dtype",
    "bfloat16",
    "--enforce-eager",
    "--max-model-len",
    "512",
]


@pytest.fixture(scope="class")
def embed_server(request):
    """Class-level fixture for BGE embedding model tests."""
    with RemoteOpenAIServer(EMBED_MODEL_NAME, EMBED_BASE_ARGS) as remote_server:
        request.cls.server = remote_server
        yield remote_server


@pytest_asyncio.fixture
async def embed_client(embed_server):
    async with embed_server.get_async_client() as async_client:
        yield async_client


@pytest.mark.usefixtures("embed_server")
class TestEmbeddingsUsagePolicy:
    """Usage policy tests for Embeddings API."""

    @pytest.mark.asyncio
    async def test_non_streaming(self, embed_client: openai.AsyncOpenAI):
        response = await embed_client.embeddings.create(
            model=EMBED_MODEL_NAME,
            input="Hello world",
            encoding_format="float",
        )

        assert response.usage is not None
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens == 0
        assert response.usage.total_tokens == response.usage.prompt_tokens

    @pytest.mark.asyncio
    async def test_non_streaming_batch(self, embed_client: openai.AsyncOpenAI):
        response = await embed_client.embeddings.create(
            model=EMBED_MODEL_NAME,
            input=["Hello", "World"],
            encoding_format="float",
        )

        assert response.usage is not None
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens == 0
        assert response.usage.total_tokens > 0
