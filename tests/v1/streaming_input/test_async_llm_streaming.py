# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from vllm.config.multimodal import MultiModalConfig
from vllm.inputs import TextPrompt
from vllm.outputs import RequestOutput
from vllm.renderers.hf import HfRenderer
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.tokenizers.registry import tokenizer_args_from_config
from vllm.v1.engine.async_llm import AsyncLLM, StreamingInput
from vllm.v1.engine.output_processor import RequestOutputCollector

MODEL_NAME = "openai-community/gpt2"


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
    max_model_len = 2048
    tokenizer_revision = None
    multimodal_config = MultiModalConfig()
    hf_config = MockHFConfig()
    hf_text_config = MockHFConfig()
    logits_processors: list[str] | None = None
    logits_processor_pattern = None
    diff_sampling_param: dict | None = None
    allowed_local_media_path: str = ""
    allowed_media_domains: list[str] | None = None
    encoder_config = None
    generation_config: str = "auto"
    media_io_kwargs: dict[str, dict[str, Any]] = field(default_factory=dict)
    skip_tokenizer_init: bool = False
    is_encoder_decoder: bool = False

    def get_diff_sampling_param(self):
        return self.diff_sampling_param or {}


def _build_renderer(model_config: MockModelConfig):
    _, tokenizer_name, _, kwargs = tokenizer_args_from_config(model_config)

    return HfRenderer(
        model_config,
        tokenizer_kwargs={**kwargs, "tokenizer_name": tokenizer_name},
    )


@pytest.fixture
def mock_async_llm():
    """Create a mock AsyncLLM with mocked dependencies."""
    # Create a minimal mock without initializing the full engine
    llm = MagicMock(spec=AsyncLLM)

    # Mock the essential attributes
    llm.vllm_config = MagicMock()
    llm.vllm_config.cache_config.kv_sharing_fast_prefill = False
    llm.model_config = MockModelConfig()
    llm.input_processor = MagicMock()
    llm.io_processor = MagicMock()
    llm.renderer = _build_renderer(llm.model_config)
    llm.log_requests = False
    llm.errored = False
    llm._pause_cond = asyncio.Condition()
    llm._paused = False

    # Mock methods
    llm._run_output_handler = MagicMock()
    llm.abort = AsyncMock()

    # Use the real generate method from AsyncLLM
    llm.generate = AsyncLLM.generate.__get__(llm, AsyncLLM)

    return llm


def streaming_input_from_text(
    engine: AsyncLLM, text: str, sampling_params: SamplingParams
):
    prompt = TextPrompt(prompt=text)
    (tok_prompt,) = engine.renderer.render_cmpl([prompt])

    return StreamingInput(prompt=tok_prompt, sampling_params=sampling_params)


@pytest.mark.asyncio
async def test_generate_normal_flow(mock_async_llm):
    """Test normal generation flow with streaming requests."""
    request_id = "test_request"
    prompt = "Tell me about Paris"
    sampling_params = SamplingParams(max_tokens=10)

    # Create a mock queue with outputs
    queue = RequestOutputCollector(RequestOutputKind.FINAL_ONLY, request_id)
    output1 = RequestOutput(
        request_id=request_id,
        prompt="Tell me about Paris",
        prompt_token_ids=[1, 2, 3],
        prompt_logprobs=None,
        outputs=[],
        finished=False,
    )
    output2 = RequestOutput(
        request_id=request_id,
        prompt="Tell me about Paris",
        prompt_token_ids=[1, 2, 3],
        prompt_logprobs=None,
        outputs=[],
        finished=True,
    )

    # Feed outputs to queue as they're consumed to avoid aggregation
    async def feed_outputs():
        queue.put(output1)
        await asyncio.sleep(1)  # Let first output be consumed
        queue.put(output2)

    asyncio.create_task(feed_outputs())  # noqa

    # Mock add_request to return the queue
    async def mock_add_request(*args, **kwargs):
        return queue

    mock_async_llm.add_request = mock_add_request

    # Collect outputs from generate
    outputs = []
    async for output in mock_async_llm.generate(
        prompt=prompt,
        sampling_params=sampling_params,
        request_id=request_id,
    ):
        outputs.append(output)

    assert len(outputs) == 2
    assert outputs[0].finished is False
    assert outputs[1].finished is True


def make_output(request_id: str, finished: bool) -> RequestOutput:
    """Helper to create a RequestOutput."""
    return RequestOutput(
        request_id=request_id,
        prompt="test",
        prompt_token_ids=[1, 2, 3],
        prompt_logprobs=None,
        outputs=[],
        finished=finished,
    )


@pytest.mark.asyncio
async def test_generate_with_async_generator():
    """Test generate with an async input generator.

    With the new streaming input API, completion is signaled by finishing
    the input generator (not via a resumable flag). Each input chunk
    produces intermediate outputs, and the final output has finished=True.
    """
    request_id = "test"
    sampling_params = SamplingParams(max_tokens=10)

    llm = MagicMock(spec=AsyncLLM)
    llm.vllm_config = MagicMock()
    llm.vllm_config.cache_config.kv_sharing_fast_prefill = False
    llm.model_config = MockModelConfig()
    llm.input_processor = MagicMock()
    llm.io_processor = MagicMock()
    llm.renderer = _build_renderer(llm.model_config)
    llm.log_requests = False
    llm.errored = False
    llm._pause_cond = asyncio.Condition()
    llm._paused = False
    llm._run_output_handler = MagicMock()
    llm.abort = AsyncMock()

    # Bind the real generate method
    llm.generate = AsyncLLM.generate.__get__(llm, AsyncLLM)

    # Track inputs processed
    inputs_received = []
    queue = RequestOutputCollector(RequestOutputKind.DELTA, request_id)

    async def mock_add_request(req_id, prompt, params, *args, **kwargs):
        # When prompt is an AsyncGenerator, process streaming inputs
        if isinstance(prompt, AsyncGenerator):
            # Process inputs in background, produce outputs
            async def handle_stream():
                async for input_chunk in prompt:
                    inputs_received.append(input_chunk.prompt)
                    # Each input produces an intermediate output
                    queue.put(make_output(req_id, finished=False))
                    await asyncio.sleep(0.01)
                # Final output when stream ends
                queue.put(make_output(req_id, finished=True))

            asyncio.create_task(handle_stream())
            return queue
        return queue

    llm.add_request = mock_add_request

    async def input_generator() -> AsyncGenerator[StreamingInput, None]:
        yield streaming_input_from_text(llm, "Hello", sampling_params=sampling_params)
        yield streaming_input_from_text(llm, " world", sampling_params=sampling_params)

    outputs = []
    async for output in llm.generate(input_generator(), sampling_params, request_id):
        outputs.append(output)

    # Two intermediate outputs + one final output
    assert len(outputs) == 3
    assert outputs[0].finished is False
    assert outputs[1].finished is False
    assert outputs[2].finished is True
    # Both inputs were processed
    assert inputs_received == ["Hello", " world"]
