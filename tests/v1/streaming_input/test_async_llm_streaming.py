# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import contextlib
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

import pytest

from vllm.engine.protocol import StreamingInput
from vllm.outputs import RequestOutput
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.v1.engine import (
    EngineCoreOutput,
    EngineCoreOutputs,
    EngineCoreRequest,
    FinishReason,
)
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.core_client import DPLBAsyncMPClient
from vllm.v1.engine.output_processor import (
    OutputProcessorOutput,
    RequestOutputCollector,
)


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


def test_dplb_resumable_updates_reuse_existing_route():
    client = object.__new__(DPLBAsyncMPClient)
    client.client_count = 1
    client.reqs_in_flight = {}
    client.streaming_req_ids = set()
    client.pending_streaming_cleanup_req_ids = set()
    client.core_engines = [b"\x00\x00", b"\x01\x00"]
    client.lb_engines = [[0, 0], [0, 0]]
    client.eng_start_index = 0

    def make_request() -> EngineCoreRequest:
        return EngineCoreRequest(
            request_id="stream",
            prompt_token_ids=[1],
            mm_features=None,
            sampling_params=SamplingParams(max_tokens=1),
            pooling_params=None,
            arrival_time=0.0,
            lora_request=None,
            cache_salt=None,
            data_parallel_rank=None,
            resumable=True,
        )

    first_engine = client.get_core_engine_for_request(make_request())
    second_engine = client.get_core_engine_for_request(make_request())

    assert second_engine == first_engine
    assert client.reqs_in_flight == {"stream": first_engine}
    assert client.lb_engines == [[1, 0], [0, 0]]


def test_dplb_final_streaming_sentinel_reuses_existing_route():
    client = object.__new__(DPLBAsyncMPClient)
    client.client_count = 1
    client.reqs_in_flight = {}
    client.streaming_req_ids = set()
    client.pending_streaming_cleanup_req_ids = set()
    client.core_engines = [b"\x00\x00", b"\x01\x00"]
    client.lb_engines = [[0, 0], [0, 0]]
    client.eng_start_index = 0

    resumable_request = EngineCoreRequest(
        request_id="stream",
        prompt_token_ids=[1],
        mm_features=None,
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        resumable=True,
    )
    final_sentinel = EngineCoreRequest(
        request_id="stream",
        prompt_token_ids=[0],
        mm_features=None,
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )

    first_engine = client.get_core_engine_for_request(resumable_request)
    final_engine = client.get_core_engine_for_request(final_sentinel)

    assert final_engine == first_engine
    assert client.reqs_in_flight == {"stream": first_engine}
    assert client.lb_engines == [[1, 0], [0, 0]]


@pytest.mark.asyncio
async def test_dplb_terminal_streaming_route_survives_until_abort_ack():
    client = object.__new__(DPLBAsyncMPClient)
    engine = b"\x01\x00"
    client.reqs_in_flight = {
        "terminal": engine,
        "ordinary": engine,
    }
    client.streaming_req_ids = {"terminal"}
    client.pending_streaming_cleanup_req_ids = set()
    client.resources = MagicMock()
    client.resources.engine_dead = False

    terminal_outputs = EngineCoreOutputs(
        outputs=[
            EngineCoreOutput("terminal", [], finish_reason=FinishReason.LENGTH),
        ],
        finished_requests={"terminal"},
    )
    await DPLBAsyncMPClient.process_engine_outputs(client, terminal_outputs)
    await DPLBAsyncMPClient.process_engine_outputs(
        client,
        EngineCoreOutputs(
            outputs=[
                EngineCoreOutput("ordinary", [1], finish_reason=FinishReason.LENGTH),
            ],
            finished_requests={"ordinary"},
        ),
    )

    assert client.reqs_in_flight == {"terminal": engine}

    abort_calls: list[tuple[list[str], bytes]] = []
    abort_seen = asyncio.Event()

    async def abort_requests(request_ids: list[str], target_engine: bytes):
        abort_calls.append((request_ids, target_engine))
        abort_seen.set()

    client._abort_requests = abort_requests
    outputs_delivered = False

    async def get_output_async():
        nonlocal outputs_delivered
        if not outputs_delivered:
            outputs_delivered = True
            return terminal_outputs
        await asyncio.Future()

    client.get_output_async = get_output_async
    output_processor = MagicMock()
    output_processor.process_outputs.return_value = OutputProcessorOutput(
        request_outputs=[],
        reqs_to_abort=["terminal"],
    )

    llm = object.__new__(AsyncLLM)
    llm.output_handler = None
    llm.engine_core = client
    llm.output_processor = output_processor
    llm.log_stats = False
    llm.logger_manager = None
    llm.renderer = MagicMock()

    AsyncLLM._run_output_handler(llm)
    assert llm.output_handler is not None
    try:
        await asyncio.wait_for(abort_seen.wait(), timeout=1.0)
        await asyncio.sleep(0)

        assert abort_calls == [(["terminal"], engine)]
        assert "terminal" not in client.reqs_in_flight
        assert not client.streaming_req_ids
    finally:
        llm.output_handler.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await llm.output_handler
        llm.engine_core = None


@pytest.mark.asyncio
async def test_dplb_finished_only_batch_waits_for_terminal_streaming_cleanup_ack():
    client = object.__new__(DPLBAsyncMPClient)
    engine = b"\x01\x00"
    client.reqs_in_flight = {"terminal": engine}
    client.streaming_req_ids = {"terminal"}
    client.pending_streaming_cleanup_req_ids = set()

    await DPLBAsyncMPClient.process_engine_outputs(
        client,
        EngineCoreOutputs(
            outputs=[
                EngineCoreOutput("terminal", [], finish_reason=FinishReason.LENGTH),
            ],
            finished_requests={"terminal"},
        ),
    )
    await DPLBAsyncMPClient.process_engine_outputs(
        client,
        EngineCoreOutputs(finished_requests={"terminal"}),
    )

    assert client.reqs_in_flight == {"terminal": engine}
    assert client.streaming_req_ids == {"terminal"}

    await client.cleanup_finished_requests_async({"terminal"})

    assert not client.reqs_in_flight
    assert not client.streaming_req_ids
    assert not client.pending_streaming_cleanup_req_ids


@pytest.mark.asyncio
async def test_dplb_finished_only_streaming_completion_cleans_without_terminal_output():
    client = object.__new__(DPLBAsyncMPClient)
    engine = b"\x01\x00"
    client.reqs_in_flight = {"finished-only": engine}
    client.streaming_req_ids = {"finished-only"}
    client.pending_streaming_cleanup_req_ids = set()

    await DPLBAsyncMPClient.process_engine_outputs(
        client,
        EngineCoreOutputs(finished_requests={"finished-only"}),
    )

    assert not client.reqs_in_flight
    assert not client.streaming_req_ids
    assert not client.pending_streaming_cleanup_req_ids


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
