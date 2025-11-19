import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from vllm.outputs import RequestOutput
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.v1.engine.output_processor import RequestOutputCollector
from vllm.v1.streaming.engine.streaming_async_llm import StreamingAsyncLLM


@pytest.fixture
def mock_streaming_llm():
    """Create a mock StreamingAsyncLLM with mocked dependencies."""
    # Create a minimal mock without initializing the full engine
    llm = MagicMock(spec=StreamingAsyncLLM)

    # Mock the essential attributes
    llm.vllm_config = MagicMock()
    llm.vllm_config.cache_config.kv_sharing_fast_prefill = False
    llm.model_config = MagicMock()
    llm.model_config.max_model_len = 2048
    llm.log_requests = False
    llm.errored = False

    # Mock methods
    llm._run_output_handler = MagicMock()
    llm.abort = AsyncMock()

    # Use the real generate method from StreamingAsyncLLM
    llm.generate = StreamingAsyncLLM.generate.__get__(llm, StreamingAsyncLLM)

    return llm


@pytest.mark.asyncio
async def test_generate_normal_flow(mock_streaming_llm):
    """Test normal generation flow with streaming requests."""
    request_id = "test_request"
    prompt = "Tell me about Paris"
    sampling_params = SamplingParams(max_tokens=10)
    streaming_sequence_id = 0
    close_session = False

    # Create a mock queue with outputs
    queue = RequestOutputCollector(output_kind=RequestOutputKind.FINAL_ONLY)
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

    mock_streaming_llm.add_request = mock_add_request

    # Collect outputs from generate
    outputs = []
    async for output in mock_streaming_llm.generate(
        prompt=prompt,
        sampling_params=sampling_params,
        request_id=request_id,
        streaming_sequence_id=streaming_sequence_id,
        close_session=close_session,
    ):
        outputs.append(output)

    assert len(outputs) == 2
    assert outputs[0].finished is False
    assert outputs[1].finished is True


@pytest.mark.asyncio
async def test_generate_multiple_streaming_requests(mock_streaming_llm):
    """Test session continuation across multiple streaming requests."""
    request_id = "session"
    prompt1 = "Tell me about Paris"
    prompt2 = "Tell me more"
    sampling_params = SamplingParams(max_tokens=10)

    # First streaming request (sequence_id=0)
    queue1 = RequestOutputCollector(output_kind=RequestOutputKind.FINAL_ONLY)
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

    mock_streaming_llm.add_request = mock_add_request1

    # Generate first request
    outputs1 = []
    async for output in mock_streaming_llm.generate(
        prompt=prompt1,
        sampling_params=sampling_params,
        request_id=request_id,
        streaming_sequence_id=0,
        close_session=False,
    ):
        outputs1.append(output)

    assert len(outputs1) == 1
    assert outputs1[0].finished is True

    # Second streaming request (sequence_id=1)
    queue2 = RequestOutputCollector(output_kind=RequestOutputKind.FINAL_ONLY)
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

    mock_streaming_llm.add_request = mock_add_request2

    # Generate second request
    outputs2 = []
    async for output in mock_streaming_llm.generate(
        prompt=prompt2,
        sampling_params=sampling_params,
        request_id=request_id,
        streaming_sequence_id=1,
        close_session=False,
    ):
        outputs2.append(output)

    assert len(outputs2) == 1
    assert outputs2[0].finished is True


@pytest.mark.asyncio
async def test_generate_generator_exit(mock_streaming_llm):
    """Test that GeneratorExit is handled gracefully for streaming sessions."""
    request_id = "test_request"
    prompt = "Test prompt"
    sampling_params = SamplingParams(max_tokens=10)

    # Create a queue with one output
    queue = RequestOutputCollector(output_kind=RequestOutputKind.FINAL_ONLY)
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

    mock_streaming_llm.add_request = mock_add_request

    # Create generator and close it early (simulates GeneratorExit)
    gen = mock_streaming_llm.generate(
        prompt=prompt,
        sampling_params=sampling_params,
        request_id=request_id,
        streaming_sequence_id=0,
        close_session=False,
    )

    # Get first output then close generator
    output = await gen.__anext__()
    assert output.finished is False

    # Closing the generator should not raise or abort (streaming session continues)
    await gen.aclose()
