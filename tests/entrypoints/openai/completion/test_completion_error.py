# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from vllm.config.multimodal import MultiModalConfig
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.openai.completion.serving import OpenAIServingCompletion
from vllm.entrypoints.openai.engine.protocol import (
    GenerationError,
    RequestResponseMetadata,
    StreamOptions,
)
from vllm.entrypoints.openai.models.protocol import BaseModelPath
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.serve.render.serving import OpenAIServingRender
from vllm.logprobs import Logprob
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.renderers.hf import HfRenderer
from vllm.tokenizers.registry import cached_tokenizer_from_config
from vllm.v1.engine.async_llm import AsyncLLM

MODEL_NAME = "openai-community/gpt2"
MODEL_NAME_SHORT = "gpt2"
BASE_MODEL_PATHS = [
    BaseModelPath(name=MODEL_NAME, model_path=MODEL_NAME),
    BaseModelPath(name=MODEL_NAME_SHORT, model_path=MODEL_NAME_SHORT),
]


@dataclass
class MockHFConfig:
    model_type: str = "any"


@dataclass
class MockModelConfig:
    task = "generate"
    runner_type = "generate"
    model = MODEL_NAME
    tokenizer = MODEL_NAME
    trust_remote_code = False
    tokenizer_mode = "auto"
    max_model_len = 100
    tokenizer_revision = None
    multimodal_config = MultiModalConfig()
    hf_config = MockHFConfig()
    logits_processors: list[str] | None = None
    diff_sampling_param: dict | None = None
    allowed_local_media_path: str = ""
    allowed_media_domains: list[str] | None = None
    encoder_config = None
    generation_config: str = "auto"
    media_io_kwargs: dict[str, dict[str, Any]] = field(default_factory=dict)
    skip_tokenizer_init = False
    is_encoder_decoder: bool = False
    is_multimodal_model: bool = False
    renderer_num_workers: int = 1

    def get_diff_sampling_param(self):
        return self.diff_sampling_param or {}


@dataclass
class MockParallelConfig:
    _api_process_rank: int = 0


@dataclass
class MockVllmConfig:
    model_config: MockModelConfig
    parallel_config: MockParallelConfig


def _build_serving_completion(engine: AsyncLLM) -> OpenAIServingCompletion:
    models = OpenAIServingModels(
        engine_client=engine,
        base_model_paths=BASE_MODEL_PATHS,
    )
    serving_render = OpenAIServingRender(
        model_config=engine.model_config,
        renderer=engine.renderer,
        model_registry=models.registry,
        request_logger=None,
        chat_template=None,
        chat_template_content_format="auto",
    )
    return OpenAIServingCompletion(
        engine,
        models,
        openai_serving_render=serving_render,
        request_logger=None,
    )


def _build_renderer(model_config: MockModelConfig):
    return HfRenderer(
        MockVllmConfig(model_config, parallel_config=MockParallelConfig()),
        cached_tokenizer_from_config(model_config),
    )


def _build_mock_engine() -> MagicMock:
    mock_engine = MagicMock(spec=AsyncLLM)
    mock_engine.errored = False
    mock_engine.model_config = MockModelConfig()
    mock_engine.input_processor = MagicMock()
    mock_engine.io_processor = MagicMock()
    mock_engine.renderer = _build_renderer(mock_engine.model_config)
    return mock_engine


def _build_request_output(
    *,
    prompt: str = "Test prompt",
    prompt_token_ids: list[int] | None = None,
    text: str = "Hello",
    token_ids: list[int] | None = None,
    logprobs: list[dict[int, Logprob] | None] | None = None,
    finish_reason: str | None = "length",
    finished: bool = True,
) -> RequestOutput:
    return RequestOutput(
        request_id="test-id",
        prompt=prompt,
        prompt_token_ids=[1, 2, 3] if prompt_token_ids is None else prompt_token_ids,
        prompt_logprobs=None,
        outputs=[
            CompletionOutput(
                index=0,
                text=text,
                token_ids=[100] if token_ids is None else token_ids,
                cumulative_logprob=None,
                logprobs=logprobs,
                finish_reason=finish_reason,
            )
        ],
        finished=finished,
        metrics=None,
        lora_request=None,
        encoder_prompt=None,
        encoder_prompt_token_ids=None,
    )


def _build_engine_input(
    *,
    prompt: str = "Test prompt",
    prompt_token_ids: list[int] | None = None,
) -> dict[str, str | list[int]]:
    return {
        "prompt": prompt,
        "prompt_token_ids": [1, 2, 3] if prompt_token_ids is None else prompt_token_ids,
    }


def _build_output_logprobs(
    token_id: int,
    decoded_token: str,
    *,
    logprob: float = -0.5,
) -> list[dict[int, Logprob] | None]:
    return [
        {
            token_id: Logprob(
                logprob=logprob,
                rank=1,
                decoded_token=decoded_token,
            )
        }
    ]


async def _stream_results(results: list[tuple[int, RequestOutput]]):
    for result in results:
        yield result


async def _collect_stream_payloads(
    serving_completion: OpenAIServingCompletion,
    request: CompletionRequest,
    results: list[tuple[int, RequestOutput]],
    *,
    max_tokens_per_prompt: list[int],
):
    response = serving_completion.completion_stream_generator(
        request=request,
        engine_inputs=[_build_engine_input()],
        result_generator=_stream_results(results),
        request_id="test-id",
        created_time=0,
        model_name=MODEL_NAME,
        num_prompts=1,
        max_tokens_per_prompt=max_tokens_per_prompt,
        tokenizer=serving_completion.renderer.tokenizer,
        request_metadata=RequestResponseMetadata(request_id="test-id"),
    )

    chunks = []
    async for chunk in response:
        chunks.append(chunk)

    payloads = [
        json.loads(chunk.removeprefix("data: ").removesuffix("\n\n"))
        for chunk in chunks
        if chunk.startswith("data: {")
    ]
    return (
        chunks,
        payloads,
        [payload for payload in payloads if payload.get("choices")],
        [payload for payload in payloads if payload.get("usage")],
    )


@pytest.mark.asyncio
async def test_completion_error_non_stream():
    """test finish_reason='error' returns 500 InternalServerError (non-streaming)"""
    mock_engine = MagicMock(spec=AsyncLLM)
    mock_engine.errored = False
    mock_engine.model_config = MockModelConfig()
    mock_engine.input_processor = MagicMock()
    mock_engine.renderer = _build_renderer(mock_engine.model_config)

    serving_completion = _build_serving_completion(mock_engine)

    completion_output = CompletionOutput(
        index=0,
        text="",
        token_ids=[],
        cumulative_logprob=None,
        logprobs=None,
        finish_reason="error",
    )

    request_output = RequestOutput(
        request_id="test-id",
        prompt="Test prompt",
        prompt_token_ids=[1, 2, 3],
        prompt_logprobs=None,
        outputs=[completion_output],
        finished=True,
        metrics=None,
        lora_request=None,
        encoder_prompt=None,
        encoder_prompt_token_ids=None,
    )

    async def mock_generate(*args, **kwargs):
        yield request_output

    mock_engine.generate = MagicMock(side_effect=mock_generate)

    request = CompletionRequest(
        model=MODEL_NAME,
        prompt="Test prompt",
        max_tokens=10,
        stream=False,
    )

    with pytest.raises(GenerationError):
        await serving_completion.create_completion(request)


@pytest.mark.asyncio
async def test_completion_error_stream():
    """test finish_reason='error' returns 500 InternalServerError (streaming)"""
    mock_engine = MagicMock(spec=AsyncLLM)
    mock_engine.errored = False
    mock_engine.model_config = MockModelConfig()
    mock_engine.input_processor = MagicMock()
    mock_engine.renderer = _build_renderer(mock_engine.model_config)

    serving_completion = _build_serving_completion(mock_engine)

    completion_output_1 = CompletionOutput(
        index=0,
        text="Hello",
        token_ids=[100],
        cumulative_logprob=None,
        logprobs=None,
        finish_reason=None,
    )

    request_output_1 = RequestOutput(
        request_id="test-id",
        prompt="Test prompt",
        prompt_token_ids=[1, 2, 3],
        prompt_logprobs=None,
        outputs=[completion_output_1],
        finished=False,
        metrics=None,
        lora_request=None,
        encoder_prompt=None,
        encoder_prompt_token_ids=None,
    )

    completion_output_2 = CompletionOutput(
        index=0,
        text="Hello",
        token_ids=[100],
        cumulative_logprob=None,
        logprobs=None,
        finish_reason="error",
    )

    request_output_2 = RequestOutput(
        request_id="test-id",
        prompt="Test prompt",
        prompt_token_ids=[1, 2, 3],
        prompt_logprobs=None,
        outputs=[completion_output_2],
        finished=True,
        metrics=None,
        lora_request=None,
        encoder_prompt=None,
        encoder_prompt_token_ids=None,
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

    response = await serving_completion.create_completion(request)

    chunks = []
    async for chunk in response:
        chunks.append(chunk)

    assert len(chunks) >= 2
    assert any("Internal server error" in chunk for chunk in chunks), (
        f"Expected error message in chunks: {chunks}"
    )
    assert chunks[-1] == "data: [DONE]\n\n"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("echo", "max_tokens_per_prompt", "expected_text"),
    [
        (False, [16], "Hello"),
        (True, [16], "Test promptHello"),
        (True, [0], "Test prompt"),
    ],
)
async def test_request_output_to_completion_response_uses_normalized_max_tokens(
    echo: bool,
    max_tokens_per_prompt: list[int],
    expected_text: str,
):
    mock_engine = _build_mock_engine()
    serving_completion = _build_serving_completion(mock_engine)
    request = CompletionRequest(
        model=MODEL_NAME,
        prompt="Test prompt",
        max_tokens=None,
        stream=False,
        echo=echo,
    )
    request_output = _build_request_output()

    response = serving_completion.request_output_to_completion_response(
        [request_output],
        request,
        request_id="test-id",
        created_time=0,
        model_name=MODEL_NAME,
        max_tokens_per_prompt=max_tokens_per_prompt,
        tokenizer=serving_completion.renderer.tokenizer,
        request_metadata=RequestResponseMetadata(request_id="test-id"),
    )

    assert response.choices[0].text == expected_text


@pytest.mark.asyncio
async def test_request_output_to_completion_response_uses_per_prompt_max_tokens():
    mock_engine = _build_mock_engine()
    serving_completion = _build_serving_completion(mock_engine)
    request = CompletionRequest(
        model=MODEL_NAME,
        prompt=["First prompt", "Second prompt"],
        max_tokens=None,
        stream=False,
        echo=True,
    )

    response = serving_completion.request_output_to_completion_response(
        [
            _build_request_output(
                prompt="First prompt",
                prompt_token_ids=[1, 2],
                text="Alpha",
                token_ids=[101],
            ),
            _build_request_output(
                prompt="Second prompt",
                prompt_token_ids=[3, 4],
                text="Beta",
                token_ids=[202],
            ),
        ],
        request,
        request_id="test-id",
        created_time=0,
        model_name=MODEL_NAME,
        max_tokens_per_prompt=[0, 16],
        tokenizer=serving_completion.renderer.tokenizer,
        request_metadata=RequestResponseMetadata(request_id="test-id"),
    )

    assert [choice.text for choice in response.choices] == [
        "First prompt",
        "Second promptBeta",
    ]


@pytest.mark.asyncio
async def test_request_output_to_completion_response_allows_empty_logprobs():
    mock_engine = _build_mock_engine()
    serving_completion = _build_serving_completion(mock_engine)
    request = CompletionRequest(
        model=MODEL_NAME,
        prompt="Test prompt",
        max_tokens=None,
        stream=False,
        echo=False,
        logprobs=1,
    )

    response = serving_completion.request_output_to_completion_response(
        [
            _build_request_output(
                text="",
                token_ids=[],
                logprobs=None,
                finish_reason="length",
            )
        ],
        request,
        request_id="test-id",
        created_time=0,
        model_name=MODEL_NAME,
        max_tokens_per_prompt=[0],
        tokenizer=serving_completion.renderer.tokenizer,
        request_metadata=RequestResponseMetadata(request_id="test-id"),
    )

    assert response.choices[0].text == ""
    assert response.choices[0].logprobs is None
    assert response.usage.completion_tokens == 0


@pytest.mark.asyncio
async def test_request_output_to_completion_response_allows_empty_echo_logprobs():
    mock_engine = _build_mock_engine()
    serving_completion = _build_serving_completion(mock_engine)
    request = CompletionRequest(
        model=MODEL_NAME,
        prompt="Test prompt",
        max_tokens=None,
        stream=False,
        echo=True,
        logprobs=1,
    )
    request_output = _build_request_output(
        text="",
        token_ids=[],
        logprobs=None,
        finish_reason="length",
    )
    request_output.prompt_logprobs = [None, None, None]

    response = serving_completion.request_output_to_completion_response(
        [request_output],
        request,
        request_id="test-id",
        created_time=0,
        model_name=MODEL_NAME,
        max_tokens_per_prompt=[16],
        tokenizer=serving_completion.renderer.tokenizer,
        request_metadata=RequestResponseMetadata(request_id="test-id"),
    )

    assert response.choices[0].text == "Test prompt"
    assert response.choices[0].logprobs is not None
    assert response.choices[0].token_ids is None
    assert response.usage.completion_tokens == 0


@pytest.mark.asyncio
async def test_completion_stream_generator_uses_normalized_max_tokens():
    mock_engine = _build_mock_engine()
    serving_completion = _build_serving_completion(mock_engine)
    request = CompletionRequest(
        model=MODEL_NAME,
        prompt="Test prompt",
        max_tokens=None,
        stream=True,
        echo=True,
    )

    chunks, payloads, _, _ = await _collect_stream_payloads(
        serving_completion,
        request,
        [(0, _build_request_output())],
        max_tokens_per_prompt=[16],
    )
    payload = next(payload for payload in payloads if payload.get("choices"))

    assert payload["choices"][0]["text"] == "Test promptHello"
    assert all("error" not in payload for payload in payloads)
    assert chunks[-1] == "data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_stream_generator_hides_prompt_token_ids_without_return_token_ids():
    mock_engine = _build_mock_engine()
    serving_completion = _build_serving_completion(mock_engine)
    request = CompletionRequest(
        model=MODEL_NAME,
        prompt="Test prompt",
        max_tokens=None,
        stream=True,
        echo=True,
        return_token_ids=False,
    )

    chunks, payloads, choice_payloads, _ = await _collect_stream_payloads(
        serving_completion,
        request,
        [(0, _build_request_output())],
        max_tokens_per_prompt=[16],
    )

    assert all("error" not in payload for payload in payloads)
    assert choice_payloads[0]["choices"][0]["text"] == "Test promptHello"
    assert choice_payloads[0]["choices"][0]["prompt_token_ids"] is None
    assert choice_payloads[0]["choices"][0]["token_ids"] is None
    assert chunks[-1] == "data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_completion_stream_generator_allows_empty_prefill_logprobs():
    mock_engine = _build_mock_engine()
    serving_completion = _build_serving_completion(mock_engine)
    request = CompletionRequest(
        model=MODEL_NAME,
        prompt="Test prompt",
        max_tokens=16,
        stream=True,
        logprobs=1,
        return_token_ids=True,
    )

    chunks, payloads, choice_payloads, _ = await _collect_stream_payloads(
        serving_completion,
        request,
        [
            (
                0,
                _build_request_output(
                    text="",
                    token_ids=[],
                    logprobs=None,
                    finish_reason=None,
                    finished=False,
                ),
            ),
            (
                0,
                _build_request_output(
                    text="Hello",
                    token_ids=[100],
                    logprobs=_build_output_logprobs(100, "Hello"),
                ),
            ),
        ],
        max_tokens_per_prompt=[16],
    )

    assert all("error" not in payload for payload in payloads)
    assert choice_payloads[0]["choices"][0]["prompt_token_ids"] == [1, 2, 3]
    assert choice_payloads[0]["choices"][0]["token_ids"] == []
    assert choice_payloads[0]["choices"][0]["logprobs"] is None
    assert choice_payloads[1]["choices"][0]["token_ids"] == [100]
    assert choice_payloads[1]["choices"][0]["logprobs"] is not None
    assert chunks[-1] == "data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_zero_budget_stream_hides_prompt_token_ids():
    mock_engine = _build_mock_engine()
    serving_completion = _build_serving_completion(mock_engine)
    request = CompletionRequest(
        model=MODEL_NAME,
        prompt="Test prompt",
        max_tokens=None,
        stream=True,
        echo=True,
        return_token_ids=False,
    )

    chunks, payloads, choice_payloads, _ = await _collect_stream_payloads(
        serving_completion,
        request,
        [
            (
                0,
                _build_request_output(
                    text="H",
                    token_ids=[999],
                    finish_reason="length",
                ),
            ),
        ],
        max_tokens_per_prompt=[0],
    )

    assert all("error" not in payload for payload in payloads)
    assert choice_payloads[0]["choices"][0]["text"] == "Test prompt"
    assert choice_payloads[0]["choices"][0]["prompt_token_ids"] is None
    assert choice_payloads[0]["choices"][0]["token_ids"] is None
    assert chunks[-1] == "data: [DONE]\n\n"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "results",
    [
        [
            (
                0,
                _build_request_output(
                    text="H",
                    token_ids=[999],
                    finish_reason="length",
                ),
            ),
        ],
        [
            (
                0,
                _build_request_output(
                    text="",
                    token_ids=[],
                    finish_reason=None,
                    finished=False,
                ),
            ),
            (
                0,
                _build_request_output(
                    text="H",
                    token_ids=[999],
                    finish_reason="length",
                ),
            ),
        ],
    ],
)
async def test_stream_generator_suppresses_helper_token(
    results: list[tuple[int, RequestOutput]],
):
    mock_engine = _build_mock_engine()
    serving_completion = _build_serving_completion(mock_engine)
    request = CompletionRequest(
        model=MODEL_NAME,
        prompt="Test prompt",
        max_tokens=None,
        stream=True,
        echo=True,
        return_token_ids=True,
        stream_options=StreamOptions(
            include_usage=True,
            continuous_usage_stats=True,
        ),
    )

    chunks, payloads, choice_payloads, usage_payloads = await _collect_stream_payloads(
        serving_completion,
        request,
        results,
        max_tokens_per_prompt=[0],
    )

    assert all("error" not in payload for payload in payloads)
    assert choice_payloads[0]["choices"][0]["prompt_token_ids"] == [1, 2, 3]
    assert all(payload["choices"][0]["text"] == "" for payload in choice_payloads)
    assert choice_payloads[0]["choices"][0]["token_ids"] == []
    assert choice_payloads[-1]["choices"][0]["finish_reason"] == "length"
    assert all(payload["usage"]["completion_tokens"] == 0 for payload in usage_payloads)
    assert chunks[-1] == "data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_create_completion_threads_normalized_max_tokens():
    mock_engine = _build_mock_engine()
    serving_completion = _build_serving_completion(mock_engine)
    serving_completion.render_completion_request = AsyncMock(
        return_value=[_build_engine_input()]
    )
    request_output = _build_request_output()

    async def mock_generate(*args, **kwargs):
        yield request_output

    mock_engine.generate = MagicMock(side_effect=mock_generate)

    request = CompletionRequest(
        model=MODEL_NAME,
        prompt="Test prompt",
        max_tokens=None,
        stream=False,
    )

    response = await serving_completion.create_completion(request)

    assert response.choices[0].text == "Hello"
    assert mock_engine.generate.call_args.args[1].max_tokens > 0


def test_json_schema_response_format_missing_schema():
    """When response_format type is 'json_schema' but the json_schema field
    is not provided, request construction should raise a validation error
    so the API returns 400 instead of 500."""
    with pytest.raises(Exception, match="json_schema.*must be provided"):
        CompletionRequest(
            model=MODEL_NAME,
            prompt="Test prompt",
            max_tokens=10,
            response_format={"type": "json_schema"},
        )


def test_negative_prompt_token_ids_nested():
    """Negative token IDs in prompt (nested list) should raise validation error."""
    with pytest.raises(Exception, match="greater than or equal to 0"):
        CompletionRequest(
            model=MODEL_NAME,
            prompt=[[-1]],
            max_tokens=10,
        )


def test_negative_prompt_token_ids_flat():
    """Negative token IDs in prompt (flat list) should raise validation error."""
    with pytest.raises(Exception, match="greater than or equal to 0"):
        CompletionRequest(
            model=MODEL_NAME,
            prompt=[-1],
            max_tokens=10,
        )
