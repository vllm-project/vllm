# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

import pytest

from vllm.inputs.data import StreamingInput
from vllm.outputs import RequestOutput
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.v1.engine.async_llm import AsyncLLM
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
    llm.model_config = MagicMock()
    llm.model_config.max_model_len = 2048
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
        yield StreamingInput(prompt="Hello", sampling_params=sampling_params)
        yield StreamingInput(prompt=" world", sampling_params=sampling_params)

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
