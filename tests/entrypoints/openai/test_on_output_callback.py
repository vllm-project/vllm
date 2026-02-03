# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the on_output callback functionality in OpenAI serving endpoints."""

import asyncio
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from vllm.config.multimodal import MultiModalConfig
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.completion.protocol import (
    CompletionRequest,
    CompletionResponse,
)
from vllm.entrypoints.openai.completion.serving import OpenAIServingCompletion
from vllm.entrypoints.openai.models.protocol import BaseModelPath
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.inputs.data import TokensPrompt
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.v1.engine.async_llm import AsyncLLM

MODEL_NAME = "test-model"
BASE_MODEL_PATHS = [
    BaseModelPath(name=MODEL_NAME, model_path=MODEL_NAME),
]

CHAT_TEMPLATE = "{% for message in messages %}{{message['role'] + ': ' + message['content'] + '\\n'}}{% endfor %}"  # noqa: E501


@dataclass
class MockHFConfig:
    model_type: str = "any"


@dataclass
class MockModelConfig:
    task = "generate"
    runner_type = "generate"
    tokenizer = MODEL_NAME
    trust_remote_code = False
    tokenizer_mode = "auto"
    max_model_len = 100
    tokenizer_revision = None
    multimodal_config = MultiModalConfig()
    hf_config = MockHFConfig()
    hf_text_config = MockHFConfig()
    logits_processor_pattern = None
    logits_processors: list[str] | None = None
    diff_sampling_param: dict | None = None
    allowed_local_media_path: str = ""
    allowed_media_domains: list[str] | None = None
    encoder_config = None
    generation_config: str = "auto"
    media_io_kwargs: dict[str, dict[str, Any]] = field(default_factory=dict)
    skip_tokenizer_init = False

    def get_diff_sampling_param(self):
        return self.diff_sampling_param or {}


def _create_mock_renderer():
    """Create a mock renderer that doesn't require downloading a tokenizer."""
    mock_renderer = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.return_value = [1, 2, 3]
    mock_tokenizer.decode.return_value = "decoded"
    mock_renderer.tokenizer = mock_tokenizer

    # Mock async methods
    engine_prompt = TokensPrompt(prompt_token_ids=[1, 2, 3])
    mock_renderer.render_completions_async = AsyncMock(return_value=["Test prompt"])
    mock_renderer.tokenize_prompts_async = AsyncMock(return_value=[engine_prompt])
    mock_renderer.render_chat_template_async = AsyncMock(
        return_value=([], [engine_prompt])
    )
    # For chat completion
    mock_renderer.render_messages_async = AsyncMock(
        return_value=([{"role": "user", "content": "Hello"}], "user: Hello\n")
    )
    mock_renderer.tokenize_chat_prompt_async = AsyncMock(return_value=engine_prompt)
    mock_renderer.tokenize_prompt_async = AsyncMock(return_value=engine_prompt)
    return mock_renderer


def _build_serving_completion(engine: AsyncLLM) -> OpenAIServingCompletion:
    models = OpenAIServingModels(
        engine_client=engine,
        base_model_paths=BASE_MODEL_PATHS,
    )
    return OpenAIServingCompletion(
        engine,
        models,
        request_logger=None,
    )


def _build_serving_chat(engine: AsyncLLM) -> OpenAIServingChat:
    models = OpenAIServingModels(
        engine_client=engine,
        base_model_paths=BASE_MODEL_PATHS,
    )
    return OpenAIServingChat(
        engine,
        models,
        response_role="assistant",
        chat_template=CHAT_TEMPLATE,
        chat_template_content_format="auto",
        request_logger=None,
    )


def _create_mock_engine() -> MagicMock:
    """Create a mock engine with all required attributes."""
    mock_engine = MagicMock(spec=AsyncLLM)
    mock_engine.errored = False
    mock_engine.model_config = MockModelConfig()
    mock_engine.input_processor = MagicMock()
    mock_engine.io_processor = MagicMock()
    mock_engine.renderer = _create_mock_renderer()
    return mock_engine


def _create_request_output(
    request_id: str = "test-id",
    text: str = "Hello",
    token_ids: list[int] | None = None,
    finished: bool = True,
    finish_reason: str | None = "stop",
) -> RequestOutput:
    """Create a RequestOutput for testing."""
    if token_ids is None:
        token_ids = [100, 101, 102]

    completion_output = CompletionOutput(
        index=0,
        text=text,
        token_ids=token_ids,
        cumulative_logprob=None,
        logprobs=None,
        finish_reason=finish_reason,
    )

    return RequestOutput(
        request_id=request_id,
        prompt="Test prompt",
        prompt_token_ids=[1, 2, 3],
        prompt_logprobs=None,
        outputs=[completion_output],
        finished=finished,
        metrics=None,
        lora_request=None,
        encoder_prompt=None,
        encoder_prompt_token_ids=None,
    )


# =============================================================================
# Completion endpoint tests
# =============================================================================


def test_completion_on_output_callback_non_streaming():
    """Test on_output callback is invoked for non-streaming completion."""

    async def run_test():
        mock_engine = _create_mock_engine()
        serving_completion = _build_serving_completion(mock_engine)

        request_output_1 = _create_request_output(
            text="Hello", token_ids=[100], finished=False, finish_reason=None
        )
        request_output_2 = _create_request_output(
            text="Hello world",
            token_ids=[100, 101],
            finished=True,
            finish_reason="stop",
        )

        async def mock_generate(*args, **kwargs):
            yield request_output_1
            yield request_output_2

        mock_engine.generate = MagicMock(side_effect=mock_generate)

        request = CompletionRequest(
            model=MODEL_NAME,
            prompt="Test prompt",
            max_tokens=10,
            stream=False,
        )

        # Track callback invocations
        captured_outputs: list[RequestOutput] = []

        def on_output_callback(output: RequestOutput) -> None:
            captured_outputs.append(output)

        response = await serving_completion.create_completion(
            request, on_output=on_output_callback
        )

        # Verify callback was invoked for each RequestOutput
        assert len(captured_outputs) == 2
        assert captured_outputs[0] is request_output_1
        assert captured_outputs[1] is request_output_2

        # Verify the response is still correct
        assert isinstance(response, CompletionResponse)

    asyncio.run(run_test())


def test_completion_on_output_callback_streaming():
    """Test on_output callback is invoked for streaming completion."""

    async def run_test():
        mock_engine = _create_mock_engine()
        serving_completion = _build_serving_completion(mock_engine)

        request_output_1 = _create_request_output(
            text="Hello", token_ids=[100], finished=False, finish_reason=None
        )
        request_output_2 = _create_request_output(
            text="Hello world",
            token_ids=[100, 101],
            finished=True,
            finish_reason="stop",
        )

        async def mock_generate(*args, **kwargs):
            yield request_output_1
            yield request_output_2

        mock_engine.generate = MagicMock(side_effect=mock_generate)

        request = CompletionRequest(
            model=MODEL_NAME,
            prompt="Test prompt",
            max_tokens=10,
            stream=True,
        )

        captured_outputs: list[RequestOutput] = []

        def on_output_callback(output: RequestOutput) -> None:
            captured_outputs.append(output)

        response = await serving_completion.create_completion(
            request, on_output=on_output_callback
        )

        # Consume the streaming response
        chunks = []
        async for chunk in response:
            chunks.append(chunk)

        # Verify callback was invoked for each RequestOutput
        assert len(captured_outputs) == 2
        assert captured_outputs[0] is request_output_1
        assert captured_outputs[1] is request_output_2

        # Verify streaming response was generated
        assert len(chunks) > 0
        assert chunks[-1] == "data: [DONE]\n\n"

    asyncio.run(run_test())


def test_completion_on_output_callback_none():
    """Test that None callback works without error."""

    async def run_test():
        mock_engine = _create_mock_engine()
        serving_completion = _build_serving_completion(mock_engine)

        request_output = _create_request_output()

        async def mock_generate(*args, **kwargs):
            yield request_output

        mock_engine.generate = MagicMock(side_effect=mock_generate)

        request = CompletionRequest(
            model=MODEL_NAME,
            prompt="Test prompt",
            max_tokens=10,
            stream=False,
        )

        # Should work without error when on_output is None (default)
        response = await serving_completion.create_completion(request, on_output=None)
        assert isinstance(response, CompletionResponse)

    asyncio.run(run_test())


# =============================================================================
# Chat completion endpoint tests
# =============================================================================


def test_chat_completion_on_output_callback_non_streaming():
    """Test on_output callback is invoked for non-streaming chat completion."""

    async def run_test():
        mock_engine = _create_mock_engine()
        serving_chat = _build_serving_chat(mock_engine)

        request_output_1 = _create_request_output(
            text="Hello", token_ids=[100], finished=False, finish_reason=None
        )
        request_output_2 = _create_request_output(
            text="Hello world",
            token_ids=[100, 101],
            finished=True,
            finish_reason="stop",
        )

        async def mock_generate(*args, **kwargs):
            yield request_output_1
            yield request_output_2

        mock_engine.generate = MagicMock(side_effect=mock_generate)

        request = ChatCompletionRequest(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10,
            stream=False,
        )

        captured_outputs: list[RequestOutput] = []

        def on_output_callback(output: RequestOutput) -> None:
            captured_outputs.append(output)

        response = await serving_chat.create_chat_completion(
            request, on_output=on_output_callback
        )

        # Verify callback was invoked for each RequestOutput
        assert len(captured_outputs) == 2
        assert captured_outputs[0] is request_output_1
        assert captured_outputs[1] is request_output_2

        # Verify the response is still correct
        assert isinstance(response, ChatCompletionResponse)

    asyncio.run(run_test())


def test_chat_completion_on_output_callback_streaming():
    """Test on_output callback is invoked for streaming chat completion."""

    async def run_test():
        mock_engine = _create_mock_engine()
        serving_chat = _build_serving_chat(mock_engine)

        request_output_1 = _create_request_output(
            text="Hello", token_ids=[100], finished=False, finish_reason=None
        )
        request_output_2 = _create_request_output(
            text="Hello world",
            token_ids=[100, 101],
            finished=True,
            finish_reason="stop",
        )

        async def mock_generate(*args, **kwargs):
            yield request_output_1
            yield request_output_2

        mock_engine.generate = MagicMock(side_effect=mock_generate)

        request = ChatCompletionRequest(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10,
            stream=True,
        )

        captured_outputs: list[RequestOutput] = []

        def on_output_callback(output: RequestOutput) -> None:
            captured_outputs.append(output)

        response = await serving_chat.create_chat_completion(
            request, on_output=on_output_callback
        )

        # Consume the streaming response
        chunks = []
        async for chunk in response:
            chunks.append(chunk)

        # Verify callback was invoked for each RequestOutput
        assert len(captured_outputs) == 2
        assert captured_outputs[0] is request_output_1
        assert captured_outputs[1] is request_output_2

        # Verify streaming response was generated
        assert len(chunks) > 0
        assert chunks[-1] == "data: [DONE]\n\n"

    asyncio.run(run_test())


def test_chat_completion_on_output_callback_none():
    """Test that None callback works without error for chat completion."""

    async def run_test():
        mock_engine = _create_mock_engine()
        serving_chat = _build_serving_chat(mock_engine)

        request_output = _create_request_output()

        async def mock_generate(*args, **kwargs):
            yield request_output

        mock_engine.generate = MagicMock(side_effect=mock_generate)

        request = ChatCompletionRequest(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10,
            stream=False,
        )

        # Should work without error when on_output is None (default)
        response = await serving_chat.create_chat_completion(request, on_output=None)
        assert isinstance(response, ChatCompletionResponse)

    asyncio.run(run_test())


def test_on_output_callback_receives_all_intermediate_outputs():
    """Test that callback receives all intermediate outputs during generation."""

    async def run_test():
        mock_engine = _create_mock_engine()
        serving_completion = _build_serving_completion(mock_engine)

        # Create multiple intermediate outputs to simulate token-by-token generation
        outputs = [
            _create_request_output(
                text="H", token_ids=[1], finished=False, finish_reason=None
            ),
            _create_request_output(
                text="He", token_ids=[1, 2], finished=False, finish_reason=None
            ),
            _create_request_output(
                text="Hel", token_ids=[1, 2, 3], finished=False, finish_reason=None
            ),
            _create_request_output(
                text="Hell", token_ids=[1, 2, 3, 4], finished=False, finish_reason=None
            ),
            _create_request_output(
                text="Hello",
                token_ids=[1, 2, 3, 4, 5],
                finished=True,
                finish_reason="stop",
            ),
        ]

        async def mock_generate(*args, **kwargs):
            for output in outputs:
                yield output

        mock_engine.generate = MagicMock(side_effect=mock_generate)

        request = CompletionRequest(
            model=MODEL_NAME,
            prompt="Test prompt",
            max_tokens=10,
            stream=False,
        )

        captured_outputs: list[RequestOutput] = []

        def on_output_callback(output: RequestOutput) -> None:
            captured_outputs.append(output)

        await serving_completion.create_completion(
            request, on_output=on_output_callback
        )

        # Verify ALL intermediate outputs were captured
        assert len(captured_outputs) == 5
        for i, captured in enumerate(captured_outputs):
            assert captured is outputs[i]

    asyncio.run(run_test())
