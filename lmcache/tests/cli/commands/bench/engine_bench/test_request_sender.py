# SPDX-License-Identifier: Apache-2.0
"""Tests for bench engine request sender."""

# Standard
from unittest.mock import AsyncMock, MagicMock, patch
import os

# Third Party
from openai.types import Completion, CompletionUsage
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta
from openai.types.completion_choice import CompletionChoice
import pytest

# First Party
from lmcache.cli.commands.bench.engine_bench.request_sender import (
    RequestSender,
    _extract_content,
    _normalize_url,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chat_chunk(
    content: str = "",
    usage: CompletionUsage = None,
) -> ChatCompletionChunk:
    """Build a minimal ``ChatCompletionChunk``."""
    choices = []
    if content:
        choices.append(
            Choice(
                delta=ChoiceDelta(content=content),
                index=0,
            )
        )
    return ChatCompletionChunk(
        id="chunk-1",
        choices=choices,
        created=0,
        model="test-model",
        object="chat.completion.chunk",
        usage=usage,
    )


def _make_completions_chunk(
    text: str = "",
    usage: CompletionUsage = None,
) -> Completion:
    """Build a minimal ``Completion`` chunk."""
    choices = []
    if text:
        choices.append(
            CompletionChoice(
                text=text,
                index=0,
                finish_reason="stop",
            )
        )
    return Completion(
        id="cmpl-1",
        choices=choices,
        created=0,
        model="test-model",
        object="text_completion",
        usage=usage,
    )


async def _fake_stream(chunks):
    """Async generator yielding chunks."""
    for chunk in chunks:
        yield chunk


async def _error_stream(chunks, error_after: int = 1):
    """Async generator that raises after yielding some chunks."""
    for i, chunk in enumerate(chunks):
        if i >= error_after:
            raise RuntimeError("stream interrupted")
        yield chunk


def _usage(prompt: int = 100, completion: int = 2) -> CompletionUsage:
    return CompletionUsage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion,
    )


# ---------------------------------------------------------------------------
# _extract_content
# ---------------------------------------------------------------------------


class TestExtractContent:
    def test_chat_content(self) -> None:
        chunk = _make_chat_chunk(content="hello")
        assert _extract_content(chunk, completions_mode=False) == "hello"

    def test_chat_no_choices(self) -> None:
        chunk = _make_chat_chunk()  # no content → empty choices
        assert _extract_content(chunk, completions_mode=False) == ""

    def test_chat_none_content(self) -> None:
        chunk = ChatCompletionChunk(
            id="c1",
            choices=[Choice(delta=ChoiceDelta(content=None), index=0)],
            created=0,
            model="m",
            object="chat.completion.chunk",
        )
        assert _extract_content(chunk, completions_mode=False) == ""

    def test_completions_text(self) -> None:
        chunk = _make_completions_chunk(text="world")
        assert _extract_content(chunk, completions_mode=True) == "world"

    def test_completions_no_choices(self) -> None:
        chunk = _make_completions_chunk()  # no text → empty choices
        assert _extract_content(chunk, completions_mode=True) == ""


# ---------------------------------------------------------------------------
# _normalize_url
# ---------------------------------------------------------------------------


class TestNormalizeUrl:
    def test_appends_v1(self) -> None:
        assert _normalize_url("http://localhost:8000") == ("http://localhost:8000/v1")

    def test_keeps_existing_v1(self) -> None:
        assert _normalize_url("http://localhost:8000/v1") == (
            "http://localhost:8000/v1"
        )

    def test_strips_trailing_slash(self) -> None:
        assert _normalize_url("http://localhost:8000/") == ("http://localhost:8000/v1")


# ---------------------------------------------------------------------------
# RequestSender — construction
# ---------------------------------------------------------------------------


class TestRequestSenderInit:
    @patch(
        "lmcache.cli.commands.bench.engine_bench.request_sender.AsyncOpenAI",
    )
    def test_default_api_key(self, mock_openai_cls) -> None:
        env = os.environ.copy()
        env.pop("OPENAI_API_KEY", None)
        with patch.dict(os.environ, env, clear=True):
            RequestSender("http://localhost:8000", "test-model")
        _, kwargs = mock_openai_cls.call_args
        assert kwargs["api_key"] == "sk-dummy"

    @patch(
        "lmcache.cli.commands.bench.engine_bench.request_sender.AsyncOpenAI",
    )
    def test_env_api_key(self, mock_openai_cls) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
            RequestSender("http://localhost:8000", "test-model")
        _, kwargs = mock_openai_cls.call_args
        assert kwargs["api_key"] == "sk-test"

    @patch(
        "lmcache.cli.commands.bench.engine_bench.request_sender.AsyncOpenAI",
    )
    def test_url_normalization(self, mock_openai_cls) -> None:
        RequestSender("http://localhost:8000", "test-model")
        _, kwargs = mock_openai_cls.call_args
        assert kwargs["base_url"] == "http://localhost:8000/v1"


# ---------------------------------------------------------------------------
# RequestSender — send_request (chat mode)
# ---------------------------------------------------------------------------


class TestRequestSenderSendRequest:
    @pytest.mark.asyncio
    @patch(
        "lmcache.cli.commands.bench.engine_bench.request_sender.AsyncOpenAI",
    )
    async def test_successful_chat_request(self, mock_openai_cls) -> None:
        chunks = [
            _make_chat_chunk(content="Hello"),
            _make_chat_chunk(content=" world"),
            _make_chat_chunk(usage=_usage(prompt=100, completion=2)),
        ]
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create = AsyncMock(
            return_value=_fake_stream(chunks)
        )

        sender = RequestSender("http://localhost:8000", "test-model")
        result = await sender.send_request("req_0", [{"role": "user", "content": "Hi"}])

        assert result.successful is True
        assert result.error == ""
        assert result.ttft > 0
        assert result.request_latency > 0
        assert result.num_input_tokens == 100
        assert result.num_output_tokens == 2
        assert result.decode_speed > 0
        assert result.submit_time < result.first_token_time < result.finish_time

    @pytest.mark.asyncio
    @patch(
        "lmcache.cli.commands.bench.engine_bench.request_sender.AsyncOpenAI",
    )
    async def test_ignore_eos_adds_extra_body(self, mock_openai_cls) -> None:
        chunks = [_make_chat_chunk(usage=_usage(prompt=10, completion=1))]
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create = AsyncMock(
            return_value=_fake_stream(chunks)
        )

        sender = RequestSender("http://localhost:8000", "test-model", ignore_eos=True)
        await sender.send_request("req_0", [{"role": "user", "content": "Hi"}])

        _, kwargs = mock_client.chat.completions.create.call_args
        assert kwargs["extra_body"] == {"ignore_eos": True}

    @pytest.mark.asyncio
    @patch(
        "lmcache.cli.commands.bench.engine_bench.request_sender.AsyncOpenAI",
    )
    async def test_default_omits_extra_body(self, mock_openai_cls) -> None:
        chunks = [_make_chat_chunk(usage=_usage(prompt=10, completion=1))]
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create = AsyncMock(
            return_value=_fake_stream(chunks)
        )

        sender = RequestSender("http://localhost:8000", "test-model")
        await sender.send_request("req_0", [{"role": "user", "content": "Hi"}])

        _, kwargs = mock_client.chat.completions.create.call_args
        assert "extra_body" not in kwargs

    @pytest.mark.asyncio
    @patch(
        "lmcache.cli.commands.bench.engine_bench.request_sender.AsyncOpenAI",
    )
    async def test_usage_extraction(self, mock_openai_cls) -> None:
        chunks = [
            _make_chat_chunk(content="tok"),
            _make_chat_chunk(usage=_usage(prompt=500, completion=20)),
        ]
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create = AsyncMock(
            return_value=_fake_stream(chunks)
        )

        sender = RequestSender("http://localhost:8000", "test-model")
        result = await sender.send_request("req_0", [{"role": "user", "content": "Hi"}])

        assert result.num_input_tokens == 500
        assert result.num_output_tokens == 20

    @pytest.mark.asyncio
    @patch(
        "lmcache.cli.commands.bench.engine_bench.request_sender.AsyncOpenAI",
    )
    async def test_failed_request_on_exception(self, mock_openai_cls) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create = AsyncMock(
            side_effect=ConnectionError("refused")
        )

        sender = RequestSender("http://localhost:8000", "test-model")
        result = await sender.send_request("req_0", [{"role": "user", "content": "Hi"}])

        assert result.successful is False
        assert result.ttft == -1.0
        assert "refused" in result.error
        assert result.num_input_tokens == 0

    @pytest.mark.asyncio
    @patch(
        "lmcache.cli.commands.bench.engine_bench.request_sender.AsyncOpenAI",
    )
    async def test_exception_during_streaming(self, mock_openai_cls) -> None:
        chunks = [
            _make_chat_chunk(content="Hello"),
            _make_chat_chunk(content=" world"),  # won't be reached
        ]
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create = AsyncMock(
            return_value=_error_stream(chunks, error_after=1)
        )

        sender = RequestSender("http://localhost:8000", "test-model")
        result = await sender.send_request("req_0", [{"role": "user", "content": "Hi"}])

        assert result.successful is False
        assert "stream interrupted" in result.error

    @pytest.mark.asyncio
    @patch(
        "lmcache.cli.commands.bench.engine_bench.request_sender.AsyncOpenAI",
    )
    async def test_no_content_chunks(self, mock_openai_cls) -> None:
        # Only usage chunk, no content
        chunks = [
            _make_chat_chunk(usage=_usage(prompt=100, completion=0)),
        ]
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create = AsyncMock(
            return_value=_fake_stream(chunks)
        )

        sender = RequestSender("http://localhost:8000", "test-model")
        result = await sender.send_request("req_0", [{"role": "user", "content": "Hi"}])

        assert result.successful is False

    @pytest.mark.asyncio
    @patch(
        "lmcache.cli.commands.bench.engine_bench.request_sender.AsyncOpenAI",
    )
    async def test_callbacks_called_on_success(self, mock_openai_cls) -> None:
        chunks = [
            _make_chat_chunk(content="Hello"),
            _make_chat_chunk(content=" world"),
            _make_chat_chunk(usage=_usage(prompt=100, completion=2)),
        ]
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create = AsyncMock(
            return_value=_fake_stream(chunks)
        )

        callback_args: list[tuple] = []

        def on_finished(result, text):
            callback_args.append((result, text))

        sender = RequestSender(
            "http://localhost:8000",
            "test-model",
            on_finished=[on_finished],
        )
        result = await sender.send_request("req_0", [{"role": "user", "content": "Hi"}])

        assert len(callback_args) == 1
        cb_result, cb_text = callback_args[0]
        assert cb_result is result
        assert cb_result.successful is True
        assert cb_text == "Hello world"

    @pytest.mark.asyncio
    @patch(
        "lmcache.cli.commands.bench.engine_bench.request_sender.AsyncOpenAI",
    )
    async def test_callbacks_called_on_failure(self, mock_openai_cls) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create = AsyncMock(
            side_effect=ConnectionError("refused")
        )

        callback_args: list[tuple] = []

        def on_finished(result, text):
            callback_args.append((result, text))

        sender = RequestSender(
            "http://localhost:8000",
            "test-model",
            on_finished=[on_finished],
        )
        await sender.send_request("req_0", [{"role": "user", "content": "Hi"}])

        assert len(callback_args) == 1
        cb_result, cb_text = callback_args[0]
        assert cb_result.successful is False
        assert cb_text == ""


# ---------------------------------------------------------------------------
# RequestSender — completions mode
# ---------------------------------------------------------------------------


class TestRequestSenderCompletionsMode:
    @pytest.mark.asyncio
    @patch(
        "lmcache.cli.commands.bench.engine_bench.request_sender.AsyncOpenAI",
    )
    async def test_uses_completions_api(self, mock_openai_cls) -> None:
        chunks = [
            _make_completions_chunk(text="Hello"),
            _make_completions_chunk(usage=_usage(prompt=50, completion=1)),
        ]
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.completions.create = AsyncMock(return_value=_fake_stream(chunks))

        sender = RequestSender(
            "http://localhost:8000", "test-model", completions_mode=True
        )
        result = await sender.send_request(
            "req_0", [{"role": "user", "content": "Test prompt"}]
        )

        # Verify completions API was called
        mock_client.completions.create.assert_called_once()
        call_kwargs = mock_client.completions.create.call_args[1]
        assert call_kwargs["prompt"] == "Test prompt"

        # Chat API should NOT have been called
        mock_client.chat.completions.create.assert_not_called()

        assert result.successful is True

    @pytest.mark.asyncio
    @patch(
        "lmcache.cli.commands.bench.engine_bench.request_sender.AsyncOpenAI",
    )
    async def test_completions_content_extraction(self, mock_openai_cls) -> None:
        chunks = [
            _make_completions_chunk(text="Hello"),
            _make_completions_chunk(text=" there"),
            _make_completions_chunk(usage=_usage(prompt=50, completion=2)),
        ]
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.completions.create = AsyncMock(return_value=_fake_stream(chunks))

        callback_args: list[tuple] = []

        sender = RequestSender(
            "http://localhost:8000",
            "test-model",
            completions_mode=True,
            on_finished=[lambda r, t: callback_args.append((r, t))],
        )
        result = await sender.send_request(
            "req_0", [{"role": "user", "content": "Test"}]
        )

        assert result.successful is True
        assert result.num_input_tokens == 50
        assert result.num_output_tokens == 2
        # Verify response text via callback
        assert callback_args[0][1] == "Hello there"


# ---------------------------------------------------------------------------
# RequestSender — warmup
# ---------------------------------------------------------------------------


class TestRequestSenderWarmup:
    @pytest.mark.asyncio
    @patch(
        "lmcache.cli.commands.bench.engine_bench.request_sender.AsyncOpenAI",
    )
    async def test_warmup_defaults_max_tokens_1(self, mock_openai_cls) -> None:
        chunks = [
            _make_chat_chunk(content="X"),
            _make_chat_chunk(usage=_usage(prompt=100, completion=1)),
        ]
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create = AsyncMock(
            return_value=_fake_stream(chunks)
        )

        sender = RequestSender("http://localhost:8000", "test-model")
        await sender.send_warmup_request(
            "warmup_0", [{"role": "user", "content": "Hi"}]
        )

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["max_tokens"] == 1


# ---------------------------------------------------------------------------
# RequestSender — close
# ---------------------------------------------------------------------------


class TestRequestSenderClose:
    @pytest.mark.asyncio
    @patch(
        "lmcache.cli.commands.bench.engine_bench.request_sender.AsyncOpenAI",
    )
    async def test_close_calls_client_close(self, mock_openai_cls) -> None:
        mock_client = MagicMock()
        mock_client.close = AsyncMock()
        mock_openai_cls.return_value = mock_client

        sender = RequestSender("http://localhost:8000", "test-model")
        await sender.close()

        mock_client.close.assert_called_once()
