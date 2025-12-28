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
        resumable=True,
    ):
        outputs.append(output)

    assert len(outputs) == 2
    assert outputs[0].finished is False
    assert outputs[1].finished is True


@pytest.mark.asyncio
async def test_generate_multiple_streaming_requests(mock_async_llm):
    """Test session continuation across multiple streaming requests."""
    request_id = "session"
    prompt1 = "Tell me about Paris"
    prompt2 = "Tell me more"
    sampling_params = SamplingParams(max_tokens=10)

    # First streaming request (sequence_id=0)
    queue1 = RequestOutputCollector(RequestOutputKind.FINAL_ONLY, request_id)
    output1 = RequestOutput(
        request_id=request_id,
        prompt=prompt1,
        prompt_token_ids=[1, 2, 3],
        prompt_logprobs=None,
        outputs=[],
        finished=True,
    )

    queue1.put(output1)

    async def mock_add_request1(*args, **kwargs):
        return queue1

    mock_async_llm.add_request = mock_add_request1

    # Generate first request
    outputs1 = []
    async for output in mock_async_llm.generate(
        prompt=prompt1,
        sampling_params=sampling_params,
        request_id=request_id,
        resumable=True,
    ):
        outputs1.append(output)

    assert len(outputs1) == 1
    assert outputs1[0].finished is True

    # Second streaming request (sequence_id=1)
    queue2 = RequestOutputCollector(RequestOutputKind.FINAL_ONLY, request_id)
    output2 = RequestOutput(
        request_id=request_id,
        prompt=prompt2,
        prompt_token_ids=[4, 5],
        prompt_logprobs=None,
        outputs=[],
        finished=True,
    )

    queue2.put(output2)

    async def mock_add_request2(*args, **kwargs):
        return queue2

    mock_async_llm.add_request = mock_add_request2

    # Generate second request
    outputs2 = []
    async for output in mock_async_llm.generate(
        prompt=prompt2,
        sampling_params=sampling_params,
        request_id=request_id,
        resumable=True,
    ):
        outputs2.append(output)

    assert len(outputs2) == 1
    assert outputs2[0].finished is True


@pytest.mark.asyncio
async def test_generate_generator_exit(mock_async_llm):
    """Test that GeneratorExit is handled gracefully for streaming sessions."""
    request_id = "test_request"
    prompt = "Test prompt"
    sampling_params = SamplingParams(max_tokens=10)

    # Create a queue with one output
    queue = RequestOutputCollector(RequestOutputKind.FINAL_ONLY, request_id)
    output1 = RequestOutput(
        request_id=request_id,
        prompt="Test prompt",
        prompt_token_ids=[1, 2, 3],
        prompt_logprobs=None,
        outputs=[],
        finished=False,
    )

    queue.put(output1)

    async def mock_add_request(*args, **kwargs):
        return queue

    mock_async_llm.add_request = mock_add_request

    # Create generator and close it early (simulates GeneratorExit)
    gen = mock_async_llm.generate(
        prompt=prompt,
        sampling_params=sampling_params,
        request_id=request_id,
        resumable=True,
    )

    # Get first output then close generator
    output = await gen.__anext__()
    assert output.finished is False

    # Closing the generator should not raise or abort (streaming session continues)
    await gen.aclose()


@pytest.fixture
def mock_async_llm_streaming():
    """Create a mock AsyncLLM for generate_streaming and generate_from_stream."""
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
    llm.generate_streaming = AsyncLLM.generate_streaming.__get__(llm, AsyncLLM)
    llm.generate_from_stream = AsyncLLM.generate_from_stream.__get__(llm, AsyncLLM)

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
async def test_generate_streaming_single_input(mock_async_llm_streaming):
    """Test generate_streaming with a single non-resumable input."""
    request_id = "test"
    sampling_params = SamplingParams(max_tokens=10)

    queue = RequestOutputCollector(RequestOutputKind.FINAL_ONLY, request_id)
    queue.put(make_output(request_id, finished=True))

    async def mock_add_request(*args, **kwargs):
        return queue

    mock_async_llm_streaming.add_request = mock_add_request

    initial_input = StreamingInput(
        prompt="Hello",
        sampling_params=sampling_params,
        resumable=False,
    )

    outputs = []
    async for output in mock_async_llm_streaming.generate_streaming(
        initial_input, request_id
    ):
        outputs.append(output)

    assert len(outputs) == 1
    assert outputs[0].finished is True


@pytest.mark.asyncio
async def test_generate_streaming_with_asend(mock_async_llm_streaming):
    """Test generate_streaming with multiple inputs via asend()."""
    request_id = "test"
    sampling_params = SamplingParams(max_tokens=10)

    # Track which segment we're on
    segment = 0
    queues = []

    async def mock_add_request(*args, **kwargs):
        nonlocal segment
        queue = RequestOutputCollector(RequestOutputKind.FINAL_ONLY, request_id)
        queues.append(queue)
        # Each segment produces one finished output
        queue.put(make_output(request_id, finished=True))
        segment += 1
        return queue

    mock_async_llm_streaming.add_request = mock_add_request

    # First input (resumable)
    input1 = StreamingInput(
        prompt="Hello", sampling_params=sampling_params, resumable=True
    )

    gen = mock_async_llm_streaming.generate_streaming(input1, request_id)

    # Get first output
    out1 = await gen.__anext__()
    assert out1.finished is True

    # Send second input via asend
    input2 = StreamingInput(
        prompt=" world", sampling_params=sampling_params, resumable=False
    )
    out2 = await gen.asend(input2)
    assert out2.finished is True

    # Generator should be done
    with pytest.raises(StopAsyncIteration):
        await gen.__anext__()

    assert segment == 2


@pytest.mark.asyncio
async def test_generate_from_stream_basic(mock_async_llm_streaming):
    """Test generate_from_stream with an async input generator."""
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
    async for output in mock_async_llm_streaming.generate_from_stream(
        input_generator(), request_id
    ):
        outputs.append(output)

    assert len(outputs) == 2
    assert segment == 2


@pytest.mark.asyncio
async def test_generate_streaming_abort_on_early_close(mock_async_llm_streaming):
    """Test that closing generator before session ends triggers abort."""
    request_id = "test"
    sampling_params = SamplingParams(max_tokens=10)

    queue = RequestOutputCollector(RequestOutputKind.FINAL_ONLY, request_id)
    # Use finished=True so generator exits inner loop and waits for next input
    queue.put(make_output(request_id, finished=True))

    async def mock_add_request(*args, **kwargs):
        return queue

    mock_async_llm_streaming.add_request = mock_add_request

    # Resumable input - session stays open after this segment
    input1 = StreamingInput(
        prompt="Hello", sampling_params=sampling_params, resumable=True
    )

    gen = mock_async_llm_streaming.generate_streaming(input1, request_id)
    await gen.__anext__()  # Get finished output, generator now waits for next input

    # Close generator while session is still open (no resumable=False sent)
    # Use wait_for to avoid hanging if aclose() blocks
    await asyncio.wait_for(gen.aclose(), timeout=1.0)

    # Should have called abort since session wasn't closed
    mock_async_llm_streaming.abort.assert_called_once()
