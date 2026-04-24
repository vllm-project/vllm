# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from vllm.config.multimodal import MultiModalConfig
from vllm.entrypoints.openai.engine.protocol import StreamOptions
from vllm.entrypoints.openai.models.protocol import BaseModelPath
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.serve.disagg.protocol import GenerateRequest
from vllm.entrypoints.serve.disagg.serving import ServingTokens
from vllm.entrypoints.serve.render.serving import OpenAIServingRender
from vllm.logprobs import Logprob
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.renderers import renderer_from_config
from vllm.sampling_params import SamplingParams
from vllm.v1.engine.async_llm import AsyncLLM

MODEL_NAME = "openai-community/gpt2"
BASE_MODEL_PATHS = [
    BaseModelPath(name=MODEL_NAME, model_path=MODEL_NAME),
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
    hf_text_config = MockHFConfig()
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


def _build_renderer(model_config: MockModelConfig):
    return renderer_from_config(
        MockVllmConfig(model_config, parallel_config=MockParallelConfig()),
    )


def _build_serving_tokens(engine: AsyncLLM, **kwargs) -> ServingTokens:
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
    serving = ServingTokens(
        engine,
        models,
        openai_serving_render=serving_render,
        request_logger=None,
        **kwargs,
    )

    async def _fake_preprocess(*args, **kwargs):
        return [{"prompt_token_ids": [1, 2, 3]}]

    serving.openai_serving_render.preprocess_completion = AsyncMock(
        side_effect=_fake_preprocess
    )
    return serving


def _make_request_output(
    request_id: str,
    token_ids: list[int],
    finish_reason: str | None = None,
    finished: bool = False,
    prompt_token_ids: list[int] | None = None,
    logprobs: list[dict[int, Any] | None] | None = None,
    num_cached_tokens: int | None = None,
    index: int = 0,
) -> RequestOutput:
    return RequestOutput(
        request_id=request_id,
        prompt=None,
        prompt_token_ids=prompt_token_ids or [1, 2, 3],
        prompt_logprobs=None,
        outputs=[
            CompletionOutput(
                index=index,
                text="",
                token_ids=token_ids,
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
        num_cached_tokens=num_cached_tokens,
    )


def _mock_engine() -> MagicMock:
    engine = MagicMock(spec=AsyncLLM)
    engine.errored = False
    engine.model_config = MockModelConfig()
    engine.input_processor = MagicMock()
    engine.renderer = _build_renderer(engine.model_config)
    return engine


def _parse_sse_chunks(chunks: list[str]) -> list[Any]:
    """Parse SSE chunks into dicts (JSON) or raw strings ([DONE])."""
    parsed: list[Any] = []
    for chunk in chunks:
        assert chunk.startswith("data: ") and chunk.endswith("\n\n")
        payload = chunk[len("data: ") : -len("\n\n")]
        if payload == "[DONE]":
            parsed.append("[DONE]")
        else:
            parsed.append(json.loads(payload))
    return parsed


@pytest.mark.asyncio
async def test_stream_basic():
    """Streaming returns SSE chunks with correct token_ids and ends with [DONE]."""
    engine = _mock_engine()

    async def mock_generate(*args, **kwargs):
        yield _make_request_output("req-1", token_ids=[10])
        yield _make_request_output("req-1", token_ids=[20, 30])
        yield _make_request_output(
            "req-1", token_ids=[40], finish_reason="stop", finished=True
        )

    engine.generate = MagicMock(side_effect=mock_generate)
    serving = _build_serving_tokens(engine)

    request = GenerateRequest(
        token_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=10),
        model=MODEL_NAME,
        stream=True,
    )

    response = await serving.serve_tokens(request)
    chunks = []
    async for chunk in response:
        chunks.append(chunk)

    parsed = _parse_sse_chunks(chunks)

    # 3 data chunks + [DONE]
    assert parsed[-1] == "[DONE]"
    data_chunks = [c for c in parsed if c != "[DONE]"]
    assert len(data_chunks) == 3

    assert data_chunks[0]["choices"][0]["token_ids"] == [10]
    assert data_chunks[1]["choices"][0]["token_ids"] == [20, 30]
    assert data_chunks[2]["choices"][0]["token_ids"] == [40]
    assert data_chunks[2]["choices"][0]["finish_reason"] == "stop"


@pytest.mark.asyncio
async def test_stream_error_mid_generation():
    """finish_reason='error' mid-stream yields error chunk then [DONE]."""
    engine = _mock_engine()

    async def mock_generate(*args, **kwargs):
        yield _make_request_output("req-1", token_ids=[10])
        yield _make_request_output(
            "req-1", token_ids=[20], finish_reason="error", finished=True
        )

    engine.generate = MagicMock(side_effect=mock_generate)
    serving = _build_serving_tokens(engine)

    request = GenerateRequest(
        token_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=10),
        model=MODEL_NAME,
        stream=True,
    )

    response = await serving.serve_tokens(request)
    chunks = []
    async for chunk in response:
        chunks.append(chunk)

    assert len(chunks) >= 2
    assert any("Internal server error" in chunk for chunk in chunks), (
        f"Expected error message in chunks: {chunks}"
    )
    assert chunks[-1] == "data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_stream_error_with_empty_delta():
    """finish_reason='error' with empty delta_token_ids still raises."""
    engine = _mock_engine()

    async def mock_generate(*args, **kwargs):
        yield _make_request_output("req-1", token_ids=[10])
        yield _make_request_output(
            "req-1", token_ids=[], finish_reason="error", finished=True
        )

    engine.generate = MagicMock(side_effect=mock_generate)
    serving = _build_serving_tokens(engine)

    request = GenerateRequest(
        token_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=10),
        model=MODEL_NAME,
        stream=True,
    )

    response = await serving.serve_tokens(request)
    chunks = []
    async for chunk in response:
        chunks.append(chunk)

    assert any("Internal server error" in chunk for chunk in chunks), (
        f"Expected error message in chunks: {chunks}"
    )
    assert chunks[-1] == "data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_stream_skips_empty_token_output():
    """Outputs with empty token_ids are skipped (no chunk emitted)."""
    engine = _mock_engine()

    async def mock_generate(*args, **kwargs):
        yield _make_request_output("req-1", token_ids=[10])
        yield _make_request_output("req-1", token_ids=[])
        yield _make_request_output(
            "req-1", token_ids=[20], finish_reason="stop", finished=True
        )

    engine.generate = MagicMock(side_effect=mock_generate)
    serving = _build_serving_tokens(engine)

    request = GenerateRequest(
        token_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=10),
        model=MODEL_NAME,
        stream=True,
    )

    response = await serving.serve_tokens(request)
    chunks = []
    async for chunk in response:
        chunks.append(chunk)

    parsed = _parse_sse_chunks(chunks)
    assert parsed[-1] == "[DONE]"
    data_chunks = [c for c in parsed if c != "[DONE]"]

    # Only 2 data chunks — the empty one is skipped
    assert len(data_chunks) == 2
    assert data_chunks[0]["choices"][0]["token_ids"] == [10]
    assert data_chunks[1]["choices"][0]["token_ids"] == [20]


@pytest.mark.asyncio
async def test_stream_include_usage():
    """stream_options.include_usage emits a final usage-only chunk."""
    engine = _mock_engine()

    async def mock_generate(*args, **kwargs):
        yield _make_request_output("req-1", token_ids=[10])
        yield _make_request_output(
            "req-1", token_ids=[20], finish_reason="stop", finished=True
        )

    engine.generate = MagicMock(side_effect=mock_generate)
    serving = _build_serving_tokens(engine)

    request = GenerateRequest(
        token_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=10),
        model=MODEL_NAME,
        stream=True,
        stream_options=StreamOptions(include_usage=True),
    )

    response = await serving.serve_tokens(request)
    chunks = []
    async for chunk in response:
        chunks.append(chunk)

    parsed = _parse_sse_chunks(chunks)
    assert parsed[-1] == "[DONE]"

    # The chunk before [DONE] should be the usage-only chunk
    usage_chunk = parsed[-2]
    assert usage_chunk["choices"] == []
    assert usage_chunk["usage"]["prompt_tokens"] == 3
    assert usage_chunk["usage"]["completion_tokens"] == 2
    assert usage_chunk["usage"]["total_tokens"] == 5


@pytest.mark.asyncio
async def test_stream_continuous_usage():
    """continuous_usage_stats adds usage to every data chunk."""
    engine = _mock_engine()

    async def mock_generate(*args, **kwargs):
        yield _make_request_output("req-1", token_ids=[10])
        yield _make_request_output(
            "req-1", token_ids=[20], finish_reason="stop", finished=True
        )

    engine.generate = MagicMock(side_effect=mock_generate)
    serving = _build_serving_tokens(engine)

    request = GenerateRequest(
        token_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=10),
        model=MODEL_NAME,
        stream=True,
        stream_options=StreamOptions(
            include_usage=True,
            continuous_usage_stats=True,
        ),
    )

    response = await serving.serve_tokens(request)
    chunks = []
    async for chunk in response:
        chunks.append(chunk)

    parsed = _parse_sse_chunks(chunks)
    data_chunks = [c for c in parsed if isinstance(c, dict) and c.get("choices")]

    # Every data chunk should have usage
    for i, dc in enumerate(data_chunks):
        assert dc["usage"] is not None, f"chunk {i} missing usage"
        assert dc["usage"]["prompt_tokens"] == 3

    # First chunk: 1 completion token
    assert data_chunks[0]["usage"]["completion_tokens"] == 1
    assert data_chunks[0]["usage"]["total_tokens"] == 4

    # Second chunk: 2 completion tokens (cumulative)
    assert data_chunks[1]["usage"]["completion_tokens"] == 2
    assert data_chunks[1]["usage"]["total_tokens"] == 5


@pytest.mark.asyncio
async def test_stream_with_logprobs():
    """Streaming with logprobs includes logprob data in each chunk."""
    engine = _mock_engine()

    async def mock_generate(*args, **kwargs):
        yield _make_request_output(
            "req-1",
            token_ids=[10],
            logprobs=[{10: Logprob(logprob=-0.5)}],
        )
        yield _make_request_output(
            "req-1",
            token_ids=[20],
            logprobs=[{20: Logprob(logprob=-1.0)}],
            finish_reason="stop",
            finished=True,
        )

    engine.generate = MagicMock(side_effect=mock_generate)
    serving = _build_serving_tokens(engine)

    request = GenerateRequest(
        token_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=10, logprobs=1),
        model=MODEL_NAME,
        stream=True,
    )

    response = await serving.serve_tokens(request)
    chunks = []
    async for chunk in response:
        chunks.append(chunk)

    parsed = _parse_sse_chunks(chunks)
    data_chunks = [c for c in parsed if isinstance(c, dict) and c.get("choices")]

    for dc in data_chunks:
        lp = dc["choices"][0]["logprobs"]
        assert lp is not None
        assert len(lp["content"]) == 1
        assert lp["content"][0]["token"].startswith("token_id:")


@pytest.mark.asyncio
async def test_stream_prompt_tokens_details():
    """enable_prompt_tokens_details includes cached_tokens in final usage."""
    engine = _mock_engine()

    async def mock_generate(*args, **kwargs):
        yield _make_request_output(
            "req-1",
            token_ids=[10],
            finish_reason="stop",
            finished=True,
            num_cached_tokens=2,
        )

    engine.generate = MagicMock(side_effect=mock_generate)
    serving = _build_serving_tokens(engine, enable_prompt_tokens_details=True)

    request = GenerateRequest(
        token_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=10),
        model=MODEL_NAME,
        stream=True,
        stream_options=StreamOptions(include_usage=True),
    )

    response = await serving.serve_tokens(request)
    chunks = []
    async for chunk in response:
        chunks.append(chunk)

    parsed = _parse_sse_chunks(chunks)
    # Usage-only chunk (before [DONE])
    usage_chunk = parsed[-2]
    assert usage_chunk["choices"] == []
    assert usage_chunk["usage"]["prompt_tokens_details"]["cached_tokens"] == 2
