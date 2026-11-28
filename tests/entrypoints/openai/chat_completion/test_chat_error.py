# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vllm.config.multimodal import MultiModalConfig
from vllm.entrypoints.openai.chat_completion.protocol import (
    BatchChatCompletionRequest,
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.models.protocol import BaseModelPath
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.scale_out.render.serving import ServingRender
from vllm.entrypoints.serve.engine.protocol import ErrorResponse
from vllm.exceptions import GenerationError, VLLMValidationError
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.renderers.hf import HfRenderer
from vllm.renderers.online_renderer import OnlineRenderer
from vllm.tokenizers.registry import cached_tokenizer_from_config
from vllm.v1.engine.async_llm import AsyncLLM

MODEL_NAME = "openai-community/gpt2"
MODEL_NAME_SHORT = "gpt2"
BASE_MODEL_PATHS = [
    BaseModelPath(name=MODEL_NAME, model_path=MODEL_NAME),
    BaseModelPath(name=MODEL_NAME_SHORT, model_path=MODEL_NAME_SHORT),
]


@dataclass(slots=True)
class MockHFConfig:
    model_type: str = "any"


@dataclass
class MockModelConfig:
    task = "generate"
    runner_type = "generate"
    model = MODEL_NAME
    served_model_name = MODEL_NAME
    tokenizer = MODEL_NAME
    trust_remote_code = False
    tokenizer_mode = "auto"
    max_model_len = 100
    revision = None
    code_revision = None
    tokenizer_revision = None
    multimodal_config = MultiModalConfig()
    hf_config = MockHFConfig()
    hf_text_config = MockHFConfig()
    logits_processors: list[str] | None = None
    diff_sampling_param: dict | None = None
    allowed_local_media_path: str = ""
    allowed_media_domains: list[str] | None = None
    encoder_config = None
    generation_config: str = "auto"
    media_io_kwargs: dict[str, dict[str, Any]] = field(default_factory=dict)
    skip_tokenizer_init: bool = False
    is_encoder_decoder: bool = False
    is_multimodal_model: bool = False
    supports_multimodal_inputs: bool = False
    renderer_num_workers: int = 1

    def get_diff_sampling_param(self):
        return self.diff_sampling_param or {}


@dataclass(slots=True)
class MockParallelConfig:
    _api_process_rank: int = 0


@dataclass(slots=True)
class MockVllmConfig:
    model_config: MockModelConfig
    parallel_config: MockParallelConfig


def _build_renderer(model_config: MockModelConfig):
    return HfRenderer(
        MockVllmConfig(model_config, parallel_config=MockParallelConfig()),
        cached_tokenizer_from_config(model_config),
    )


def _build_serving_chat(
    engine: AsyncLLM, monkeypatch: pytest.MonkeyPatch
) -> OpenAIServingChat:
    models = OpenAIServingModels(
        engine_client=engine,
        base_model_paths=BASE_MODEL_PATHS,
    )
    online_renderer = OnlineRenderer(
        model_config=engine.model_config,
        renderer=engine.renderer,
        request_logger=None,
        chat_template=None,
        chat_template_content_format="auto",
    )

    serving_chat = OpenAIServingChat(
        engine,
        models,
        response_role="assistant",
        online_renderer=online_renderer,
        request_logger=None,
        chat_template=None,
        chat_template_content_format="auto",
    )

    async def _fake_preprocess_chat(*args, **kwargs):
        # return conversation, engine_inputs
        return (
            [{"role": "user", "content": "Test"}],
            [{"prompt_token_ids": [1, 2, 3]}],
        )

    monkeypatch.setattr(
        serving_chat.online_renderer,
        "preprocess_chat",
        AsyncMock(side_effect=_fake_preprocess_chat),
    )
    return serving_chat


@pytest.mark.asyncio
async def test_chat_error_non_stream(monkeypatch: pytest.MonkeyPatch):
    """Test finish_reason='error' returns 500 InternalServerError (non-streaming)."""
    mock_engine = MagicMock(spec=AsyncLLM)
    mock_engine.errored = False
    mock_engine.model_config = MockModelConfig()
    mock_engine.input_processor = MagicMock()
    mock_engine.renderer = _build_renderer(mock_engine.model_config)

    serving_chat = _build_serving_chat(mock_engine, monkeypatch)

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

    request = ChatCompletionRequest(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "Test prompt"}],
        max_tokens=10,
        stream=False,
    )

    with pytest.raises(GenerationError):
        await serving_chat.create_chat_completion(request)


@pytest.mark.asyncio
async def test_openai_chat_keeps_mm_cache_for_engine_execution(
    monkeypatch: pytest.MonkeyPatch,
):
    mock_engine = MagicMock(spec=AsyncLLM)
    mock_engine.errored = False
    mock_engine.model_config = MockModelConfig()
    mock_engine.input_processor = MagicMock()
    mock_engine.renderer = _build_renderer(mock_engine.model_config)

    serving_chat = _build_serving_chat(mock_engine, monkeypatch)

    request = ChatCompletionRequest(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "Test prompt"}],
    )

    result = await serving_chat.render_chat_request(request)

    assert isinstance(result, tuple)
    preprocess_chat = serving_chat.online_renderer.preprocess_chat
    assert isinstance(preprocess_chat, AsyncMock)
    assert preprocess_chat.call_args.kwargs["skip_mm_cache"] is False


def _build_serving_render(
    engine: AsyncLLM, monkeypatch: pytest.MonkeyPatch
) -> ServingRender:
    models = OpenAIServingModels(
        engine_client=engine,
        base_model_paths=BASE_MODEL_PATHS,
    )
    online_renderer = OnlineRenderer(
        model_config=engine.model_config,
        renderer=engine.renderer,
        request_logger=None,
        chat_template=None,
        chat_template_content_format="auto",
    )

    serving_render = ServingRender(models, online_renderer)

    async def _fake_preprocess_chat(*args, **kwargs):
        # return conversation, engine_inputs
        return (
            [{"role": "user", "content": "Test"}],
            [{"prompt_token_ids": [1, 2, 3]}],
        )

    monkeypatch.setattr(
        serving_render.online_renderer,
        "preprocess_chat",
        AsyncMock(side_effect=_fake_preprocess_chat),
    )
    return serving_render


@pytest.mark.asyncio
async def test_renderer_only_chat_request_skips_mm_cache(
    monkeypatch: pytest.MonkeyPatch,
):
    mock_engine = MagicMock(spec=AsyncLLM)
    mock_engine.errored = False
    mock_engine.model_config = MockModelConfig()
    mock_engine.input_processor = MagicMock()
    mock_engine.renderer = _build_renderer(mock_engine.model_config)

    serving_render = _build_serving_render(mock_engine, monkeypatch)

    request = ChatCompletionRequest(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "Test prompt"}],
    )

    result = await serving_render.render_chat_request(request)

    assert not isinstance(result, ErrorResponse)
    assert result.token_ids == [1, 2, 3]
    preprocess_chat = serving_render.online_renderer.preprocess_chat
    assert isinstance(preprocess_chat, AsyncMock)
    assert preprocess_chat.call_args.kwargs["skip_mm_cache"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("include_usage", [False, True])
async def test_chat_error_stream(include_usage: bool, monkeypatch: pytest.MonkeyPatch):
    """Test finish_reason='error' returns 500 InternalServerError (streaming)."""
    mock_engine = MagicMock(spec=AsyncLLM)
    mock_engine.errored = False
    mock_engine.model_config = MockModelConfig()
    mock_engine.input_processor = MagicMock()
    mock_engine.renderer = _build_renderer(mock_engine.model_config)

    serving_chat = _build_serving_chat(mock_engine, monkeypatch)

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

    request = ChatCompletionRequest(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "Test prompt"}],
        max_tokens=10,
        stream=True,
        stream_options={"include_usage": include_usage},
    )

    response = await serving_chat.create_chat_completion(request)

    assert isinstance(response, AsyncGenerator)
    chunks = []
    async for chunk in response:
        chunks.append(chunk)

    assert len(chunks) >= 2
    assert any("Internal server error" in chunk for chunk in chunks), (
        f"Expected error message in chunks: {chunks}"
    )
    assert chunks[-1] == "data: [DONE]\n\n"


@pytest.mark.asyncio
@pytest.mark.parametrize("include_usage", [False, True])
async def test_chat_error_stream_first_chunk(
    include_usage: bool, monkeypatch: pytest.MonkeyPatch
):
    """An error before the first token must reach the streaming client."""
    mock_engine = MagicMock(spec=AsyncLLM)
    mock_engine.errored = False
    mock_engine.model_config = MockModelConfig()
    mock_engine.input_processor = MagicMock()
    mock_engine.renderer = _build_renderer(mock_engine.model_config)

    serving_chat = _build_serving_chat(mock_engine, monkeypatch)

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

    request = ChatCompletionRequest(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "Test prompt"}],
        max_tokens=10,
        stream=True,
        stream_options={"include_usage": include_usage},
    )

    response = await serving_chat.create_chat_completion(request)

    assert isinstance(response, AsyncGenerator)
    chunks = []
    async for chunk in response:
        chunks.append(chunk)

    assert len(chunks) >= 2
    assert any("Internal server error" in chunk for chunk in chunks), (
        f"Expected error message in chunks: {chunks}"
    )
    assert chunks[-1] == "data: [DONE]\n\n"


@pytest.mark.parametrize(
    "image_content",
    [
        [{"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}],
        [{"image_url": {"url": "https://example.com/image.jpg"}}],
    ],
)
def test_system_message_warns_on_image(image_content):
    """Test that system messages with image content trigger a warning."""
    with patch(
        "vllm.entrypoints.openai.chat_completion.protocol.logger"
    ) as mock_logger:
        ChatCompletionRequest(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": image_content,
                }
            ],
        )

    mock_logger.warning_once.assert_called()
    call_args = str(mock_logger.warning_once.call_args)
    assert "System messages should only contain text" in call_args
    assert "image_url" in call_args


def test_system_message_accepts_text():
    """Test that system messages can contain text content."""
    # Should not raise an exception
    request = ChatCompletionRequest(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
        ],
    )
    assert request.messages[0]["role"] == "system"


def test_system_message_accepts_text_array():
    """Test that system messages can contain an array with text content."""
    # Should not raise an exception
    request = ChatCompletionRequest(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}],
            },
        ],
    )
    assert request.messages[0]["role"] == "system"


def test_user_message_accepts_image():
    """Test that user messages can still contain image content."""
    # Should not raise an exception
    request = ChatCompletionRequest(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/image.jpg"},
                    },
                ],
            },
        ],
    )
    assert request.messages[0]["role"] == "user"


@pytest.mark.parametrize(
    "audio_content",
    [
        [
            {
                "type": "input_audio",
                "input_audio": {"data": "base64data", "format": "wav"},
            }
        ],
        [{"input_audio": {"data": "base64data", "format": "wav"}}],
    ],
)
def test_system_message_warns_on_audio(audio_content):
    """Test that system messages with audio content trigger a warning."""
    with patch(
        "vllm.entrypoints.openai.chat_completion.protocol.logger"
    ) as mock_logger:
        ChatCompletionRequest(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": audio_content,
                }
            ],
        )

    mock_logger.warning_once.assert_called()
    call_args = str(mock_logger.warning_once.call_args)
    assert "System messages should only contain text" in call_args
    assert "input_audio" in call_args


@pytest.mark.parametrize(
    "video_content",
    [
        [{"type": "video_url", "video_url": {"url": "https://example.com/video.mp4"}}],
        [{"video_url": {"url": "https://example.com/video.mp4"}}],
    ],
)
def test_system_message_warns_on_video(video_content):
    """Test that system messages with video content trigger a warning."""
    with patch(
        "vllm.entrypoints.openai.chat_completion.protocol.logger"
    ) as mock_logger:
        ChatCompletionRequest(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": video_content,
                }
            ],
        )

    mock_logger.warning_once.assert_called()
    call_args = str(mock_logger.warning_once.call_args)
    assert "System messages should only contain text" in call_args
    assert "video_url" in call_args


def test_json_schema_response_format_missing_schema():
    """When response_format type is 'json_schema' but the json_schema field
    is not provided, request construction should raise a validation error
    so the API returns 400 instead of 500."""
    with pytest.raises(Exception, match="json_schema.*must be provided"):
        ChatCompletionRequest(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "hello"}],
            response_format={"type": "json_schema"},
        )


@pytest.mark.asyncio
async def test_online_renderer_rejects_mm_processor_kwargs_by_default():
    model_config = MockModelConfig()
    model_config.multimodal_config = MultiModalConfig()
    online_renderer = OnlineRenderer(
        model_config=model_config,
        renderer=MagicMock(),
        request_logger=None,
        chat_template=None,
        chat_template_content_format="auto",
    )

    request = ChatCompletionRequest(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "hello"}],
        mm_processor_kwargs={
            "patch_size": 1,
            "vision_min_num_patches": 3_000_000_000,
        },
    )

    with pytest.raises(
        VLLMValidationError,
        match="Per-request mm_processor_kwargs are disabled",
    ):
        await online_renderer.preprocess_chat(
            request,
            request.messages,
            default_template=None,
            default_template_content_format="auto",
            default_template_kwargs=None,
        )


@pytest.mark.asyncio
async def test_online_renderer_rejects_media_io_kwargs_by_default():
    model_config = MockModelConfig()
    model_config.multimodal_config = MultiModalConfig()
    online_renderer = OnlineRenderer(
        model_config=model_config,
        renderer=MagicMock(),
        request_logger=None,
        chat_template=None,
        chat_template_content_format="auto",
    )

    request = ChatCompletionRequest(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "hello"}],
        media_io_kwargs={"video": {"num_frames": 1_000_000}},
    )

    with pytest.raises(
        VLLMValidationError,
        match="Per-request media_io_kwargs are disabled",
    ):
        await online_renderer.preprocess_chat(
            request,
            request.messages,
            default_template=None,
            default_template_content_format="auto",
            default_template_kwargs=None,
        )


def test_trust_request_mm_kwargs_opt_in_allows_overrides():
    from vllm.entrypoints.generate.base.protocol import validate_request_mm_kwargs

    validate_request_mm_kwargs(
        mm_processor_kwargs={"use_audio_in_video": True},
        media_io_kwargs={"video": {"num_frames": 4}},
        trust_request_mm_kwargs=True,
    )


@pytest.mark.parametrize("format_value", [None, {}])
def test_structural_tag_response_format_invalid(format_value):
    """Malformed structural tags should be rejected during request validation."""
    with pytest.raises(
        VLLMValidationError,
        match="Invalid response_format structural_tag",
    ):
        ChatCompletionRequest(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "hello"}],
            response_format={"type": "structural_tag", "format": format_value},
        )


@pytest.mark.parametrize("format_value", [None, {}])
def test_batch_structural_tag_response_format_invalid(format_value):
    """Batch chat should reject malformed structural tags at request parsing."""
    with pytest.raises(
        VLLMValidationError,
        match="Invalid response_format structural_tag",
    ):
        BatchChatCompletionRequest(
            model=MODEL_NAME,
            messages=[[{"role": "user", "content": "hello"}]],
            response_format={"type": "structural_tag", "format": format_value},
        )


@pytest.mark.parametrize(
    "structural_tag",
    [
        "not json",
        "",
        # json.loads raises a plain ValueError, not JSONDecodeError
        '{"type": "structural_tag", "x": ' + "1" * 5000 + "}",
    ],
)
def test_structured_outputs_structural_tag_invalid(structural_tag):
    """Malformed direct structured_outputs structural tags should be rejected."""
    with pytest.raises(
        VLLMValidationError,
        match="Invalid structured_outputs structural_tag",
    ):
        ChatCompletionRequest(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "hello"}],
            structured_outputs={"structural_tag": structural_tag},
        )


@pytest.mark.parametrize("field", ["response_format", "structured_outputs"])
def test_deeply_nested_structural_tag_rejected(field):
    """Structural tags are compiled during request parsing, so a deeply nested
    one must be rejected there instead of blocking the server while converting,
    and reported as too deep rather than malformed."""
    schema = '{"type": "array", "items": ' * 200 + "{}" + "}" * 200
    tag = (
        '{"type": "structural_tag", "format": {"type": "json_schema", '
        f'"json_schema": {schema}}}}}'
    )
    value = json.loads(tag) if field == "response_format" else {"structural_tag": tag}
    with pytest.raises(VLLMValidationError, match="nested too deeply"):
        ChatCompletionRequest(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "hello"}],
            **{field: value},
        )


@pytest.mark.parametrize("field_name", ["prompt_logprobs", "top_logprobs"])
def test_non_numeric_logprobs_rejected(field_name):
    """A non-numeric logprobs value must be a clean 400 validation error, not a
    TypeError from the mode='before' comparison (which surfaces as HTTP 500)."""
    with pytest.raises(VLLMValidationError, match=f"`{field_name}` must be an integer"):
        ChatCompletionRequest(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "hello"}],
            **{field_name: "2"},
        )


def _build_local_chat_and_tokenize_client(
    tmp_path,
    tokenizer_mode: str,
    chat_template: str | None = None,
):
    from types import SimpleNamespace

    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import LlamaConfig, TokenizersBackend

    from vllm.config import (
        CacheConfig,
        DeviceConfig,
        LoadConfig,
        ModelConfig,
        SchedulerConfig,
        VllmConfig,
    )
    from vllm.entrypoints.openai.chat_completion.api_router import (
        router as chat_router,
    )
    from vllm.entrypoints.serve.exception_handling.register import (
        init_exception_handler,
    )
    from vllm.entrypoints.serve.tokenize.api_router import (
        router as tokenize_router,
    )
    from vllm.entrypoints.serve.tokenize.serving import ServingTokenization
    from vllm.renderers import renderer_from_config

    vocab = {"[UNK]": 0, "<s>": 1, "</s>": 2}
    for tok in ("hi", "real_fn", "evil_fn", "user", "assistant", "TOOLS"):
        vocab[tok] = len(vocab)
    extra: list[str] = []
    if tokenizer_mode != "hf":
        extra = [
            "<｜begin▁of▁sentence｜>",
            "<｜end▁of▁sentence｜>",
            "<｜User｜>",
            "<｜Assistant｜>",
            "<think>",
            "</think>",
            "<｜DSML｜>",
            "<｜System｜>",
            "<|image|>",
        ]
        for tok in extra:
            vocab[tok] = len(vocab)

    LlamaConfig(
        max_position_embeddings=4096,
        vocab_size=len(vocab),
    ).save_pretrained(tmp_path)
    tok_obj = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tok_obj.pre_tokenizer = Whitespace()
    tok = TokenizersBackend(
        tokenizer_object=tok_obj,
        unk_token="[UNK]",
        bos_token="<s>",
        eos_token="</s>",
        chat_template=chat_template,
    )
    if extra:
        tok.add_special_tokens({"additional_special_tokens": extra})
    tok.save_pretrained(tmp_path)

    cfg = VllmConfig(
        model_config=ModelConfig(
            model=str(tmp_path),
            tokenizer=str(tmp_path),
            tokenizer_mode=tokenizer_mode,
            dtype="float32",
            max_model_len=4096,
            skip_tokenizer_init=False,
        ),
        cache_config=CacheConfig(),
        scheduler_config=SchedulerConfig(
            max_model_len=4096,
            is_encoder_decoder=False,
            runner_type="generate",
        ),
        device_config=DeviceConfig(device="cpu"),
        load_config=LoadConfig(),
    )
    renderer = renderer_from_config(cfg)
    captured_prompts: list[list[int]] = []

    engine_client = MagicMock()
    engine_client.vllm_config = cfg
    engine_client.model_config = cfg.model_config
    engine_client.renderer = renderer
    engine_client.errored = False

    async def fake_generate(prompt, sampling_params, request_id, **kwargs):
        captured_prompts.append(prompt["prompt_token_ids"])
        yield RequestOutput(
            request_id=request_id,
            prompt=None,
            prompt_token_ids=prompt["prompt_token_ids"],
            prompt_logprobs=None,
            outputs=[
                CompletionOutput(
                    index=0,
                    text="ok",
                    token_ids=[3],
                    cumulative_logprob=0.0,
                    logprobs=None,
                    finish_reason="stop",
                )
            ],
            finished=True,
        )

    engine_client.generate = fake_generate
    engine_client.check_health = AsyncMock()
    online_renderer = OnlineRenderer(
        cfg.model_config,
        renderer,
        request_logger=None,
        chat_template=chat_template,
        chat_template_content_format="auto",
        enable_auto_tools=True,
        exclude_tools_when_tool_choice_none=True,
    )
    models = OpenAIServingModels(
        engine_client,
        [BaseModelPath(name="stub", model_path=str(tmp_path))],
    )
    chat = OpenAIServingChat(
        engine_client,
        models,
        "assistant",
        online_renderer=online_renderer,
        request_logger=None,
        chat_template=chat_template,
        chat_template_content_format="auto",
        enable_auto_tools=True,
        exclude_tools_when_tool_choice_none=True,
    )
    tok_srv = ServingTokenization(
        models,
        online_renderer,
        chat_template=chat_template,
        chat_template_content_format="auto",
        request_logger=None,
    )
    app = FastAPI()
    app.state.args = SimpleNamespace(log_error_stack=False)
    init_exception_handler(app)
    app.state.openai_serving_chat = chat
    app.state.serving_tokenization = tok_srv
    app.include_router(chat_router)
    app.include_router(tokenize_router)
    return TestClient(app), renderer, captured_prompts


@pytest.mark.parametrize(
    "tokenizer_mode",
    ["hf", "deepseek_v32", "deepseek_v4", "deepseek_v41"],
)
def test_chat_template_kwargs_rejects_tools_override_and_malformed_tools(
    tmp_path, tokenizer_mode
):
    """Requests must not inject tools via chat_template_kwargs (HTTP 400)."""
    template = (
        "{% if tools %}TOOLS {% for t in tools %}{{ t.function.name }} "
        "{% endfor %}{% endif %}"
        "{% for m in messages %}{{ m.role }} {{ m.content }} {% endfor %}"
        if tokenizer_mode == "hf"
        else None
    )
    client, renderer, captured = _build_local_chat_and_tokenize_client(
        tmp_path, tokenizer_mode, chat_template=template
    )
    real_tool = {
        "type": "function",
        "function": {
            "name": "real_fn",
            "description": "r",
            "parameters": {"type": "object"},
        },
    }
    evil_tool = {
        "type": "function",
        "function": {
            "name": "evil_fn",
            "description": "e",
            "parameters": {"type": "object"},
        },
    }

    def post(endpoint: str, payload: dict[str, Any]) -> tuple[Any, str, int]:
        before = len(captured)
        resp = client.post(
            endpoint,
            json={
                "model": "stub",
                "messages": [{"role": "user", "content": "hi"}],
                **payload,
            },
        )
        calls = len(captured) - before
        ids = captured[-1] if calls else resp.json().get("tokens", [])
        prompt = (
            renderer.tokenizer.decode(ids, skip_special_tokens=False) if ids else ""
        )
        return resp, prompt, calls

    reserved_cases: list[tuple[str, Any]] = [
        ("tools", None),
        ("tools", []),
        ("tools", [evil_tool]),
        ("tools", [{}]),
        ("conversation", None),
        ("conversation", []),
        ("messages", None),
        ("messages", []),
    ]
    if tokenizer_mode == "deepseek_v41":
        reserved_cases.extend([("image_order", None), ("image_order", [])])
    for endpoint in ("/v1/chat/completions", "/tokenize"):
        for tool_choice in ("none", "auto"):
            resp, prompt, calls = post(
                endpoint,
                {"tools": [real_tool], "tool_choice": tool_choice},
            )
            assert resp.status_code == 200, resp.text
            assert ("real_fn" in prompt) == (
                endpoint == "/tokenize" or tool_choice == "auto"
            ), prompt
            if (
                tokenizer_mode == "hf"
                and endpoint == "/v1/chat/completions"
                and tool_choice == "none"
            ):
                assert prompt == "user hi"
            assert "evil_fn" not in prompt
            assert calls == (1 if endpoint == "/v1/chat/completions" else 0)

        resp, _, calls = post(
            endpoint,
            {"chat_template_kwargs": {"enable_thinking": False, "custom_x": "value"}},
        )
        assert resp.status_code == 200, resp.text
        assert calls == (1 if endpoint == "/v1/chat/completions" else 0)

        resp, _, calls = post(endpoint, {"tools": [{}]})
        assert 400 <= resp.status_code < 500, resp.text
        assert calls == 0

        for key, value in reserved_cases:
            case_payload: dict[str, Any] = {"chat_template_kwargs": {key: value}}
            if key == "tools":
                case_payload.update({"tools": [real_tool], "tool_choice": "none"})
            resp, _, calls = post(endpoint, case_payload)
            assert resp.status_code == 400, resp.text
            error = resp.json()["error"]
            assert error["param"] == "chat_template_kwargs"
            assert key in error["message"]
            assert calls == 0


def test_online_renderer_pins_server_tools_above_chat_template_kwargs(tmp_path):
    """Even for renderers without _server_owned_chat_template_kwargs,
    OnlineRenderer keeps server-resolved tool_dicts authoritative."""
    import asyncio

    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import LlamaConfig, TokenizersBackend

    from vllm.config import (
        CacheConfig,
        DeviceConfig,
        LoadConfig,
        ModelConfig,
        SchedulerConfig,
        VllmConfig,
    )
    from vllm.renderers.base import BaseRenderer
    from vllm.renderers.inputs.preprocess import parse_dec_only_prompt

    vocab = {"[UNK]": 0, "<s>": 1, "</s>": 2, "hi": 3}
    LlamaConfig(max_position_embeddings=64, vocab_size=len(vocab)).save_pretrained(
        tmp_path
    )
    tok = TokenizersBackend(
        tokenizer_object=Tokenizer(WordLevel(vocab, unk_token="[UNK]")),
        unk_token="[UNK]",
        bos_token="<s>",
        eos_token="</s>",
    )
    tok.save_pretrained(tmp_path)

    seen_tools: list[Any] = []

    class CustomRenderer(BaseRenderer):
        def render_messages(self, messages, params):
            seen_tools.append(params.chat_template_kwargs.get("tools"))
            return [{"role": "user", "content": "hi"}], parse_dec_only_prompt([3])

        async def render_messages_async(self, messages, params):
            return self.render_messages(messages, params)

    cfg = VllmConfig(
        model_config=ModelConfig(
            model=str(tmp_path),
            tokenizer=str(tmp_path),
            dtype="float32",
            max_model_len=64,
            skip_tokenizer_init=False,
        ),
        cache_config=CacheConfig(),
        scheduler_config=SchedulerConfig(
            max_model_len=64,
            is_encoder_decoder=False,
            runner_type="generate",
        ),
        device_config=DeviceConfig(device="cpu"),
        load_config=LoadConfig(),
    )
    renderer = CustomRenderer(cfg, tok)
    online_renderer = OnlineRenderer(
        cfg.model_config,
        renderer,
        request_logger=None,
        chat_template=None,
        chat_template_content_format="auto",
        enable_auto_tools=True,
        exclude_tools_when_tool_choice_none=True,
    )

    real_tool = {
        "type": "function",
        "function": {
            "name": "real_fn",
            "description": "r",
            "parameters": {"type": "object"},
        },
    }
    evil_tool = {
        "type": "function",
        "function": {
            "name": "evil_fn",
            "description": "e",
            "parameters": {"type": "object"},
        },
    }

    req_none = ChatCompletionRequest(
        model="stub",
        messages=[{"role": "user", "content": "hi"}],
        tools=[real_tool],
        tool_choice="none",
        chat_template_kwargs={"tools": [evil_tool]},
    )
    asyncio.run(online_renderer.render_chat(req_none))
    assert seen_tools[-1] is None

    req_auto = ChatCompletionRequest(
        model="stub",
        messages=[{"role": "user", "content": "hi"}],
        tools=[real_tool],
        tool_choice="auto",
        chat_template_kwargs={"tools": [evil_tool]},
    )
    asyncio.run(online_renderer.render_chat(req_auto))
    assert seen_tools[-1] is not None
    assert [t["function"]["name"] for t in seen_tools[-1]] == ["real_fn"]
