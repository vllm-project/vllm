# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

import pytest

from vllm.outputs import RequestOutput
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.v1.engine.async_llm import AsyncLLM, StreamingInput
from vllm.v1.engine.output_processor import RequestOutputCollector


@pytest.fixture
def mock_async_llm():
    """Create a mock AsyncLLM with mocked dependencies."""
    # Create a minimal mock without initializing the full engine
    llm = MagicMock(spec=AsyncLLM)

    # Mock the essential attributes
    llm.vllm_config = MagicMock()
    llm.vllm_config.cache_config.kv_sharing_fast_prefill = False
    llm.model_config = MagicMock()
    llm.model_config.max_model_len = 2048
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


@pytest.fixture
def mock_async_llm_streaming():
    """Create a mock AsyncLLM for generate with async generator."""
    llm = MagicMock(spec=AsyncLLM)

    llm.vllm_config = MagicMock()
    llm.vllm_config.cache_config.kv_sharing_fast_prefill = False
    llm.model_config = MagicMock()
    llm.model_config.max_model_len = 2048
    llm.log_requests = False
    llm.errored = False
    llm._pause_cond = asyncio.Condition()
    llm._paused = False

    llm._run_output_handler = MagicMock()
    llm.abort = AsyncMock()

    # Bind the real methods
    llm.generate = AsyncLLM.generate.__get__(llm, AsyncLLM)
    llm._add_streaming_request = AsyncLLM._add_streaming_request.__get__(llm, AsyncLLM)

    return llm


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
async def test_generate_with_async_generator(mock_async_llm_streaming):
    """Test generate with an async input generator."""
    request_id = "test"
    sampling_params = SamplingParams(max_tokens=10)

    segment = 0
    # Use a single shared queue to simulate real behavior where
    # streaming session outputs go to the same queue
    shared_queue = RequestOutputCollector(RequestOutputKind.FINAL_ONLY, request_id)

    async def mock_add_request(*args, **kwargs):
        nonlocal segment
        # Add output to shared queue for each segment
        shared_queue.put(make_output(request_id, finished=True))
        segment += 1
        return shared_queue

    mock_async_llm_streaming.add_request = mock_add_request

    async def input_generator() -> AsyncGenerator[StreamingInput, None]:
        yield StreamingInput(
            prompt="Hello", sampling_params=sampling_params, resumable=True
        )
        yield StreamingInput(
            prompt=" world", sampling_params=sampling_params, resumable=False
        )

    outputs = []
    async for output in mock_async_llm_streaming.generate(
        input_generator(), None, request_id
    ):
        outputs.append(output)

    assert len(outputs) == 2
    assert segment == 2
