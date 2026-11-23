# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from collections.abc import AsyncGenerator
from unittest import mock
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from transformers import Qwen2Config

from vllm.config import (
    CacheConfig,
    DeviceConfig,
    ModelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.protocol import StreamingInput
from vllm.exceptions import VLLMValidationError
from vllm.outputs import STREAM_FINISHED, RequestOutput
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs, FinishReason
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.core_client import EngineCoreClient
from vllm.v1.engine.output_processor import OutputProcessor, RequestOutputCollector
from vllm.v1.executor.uniproc_executor import UniProcExecutor


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


@pytest.mark.parametrize(
    ("is_mm_encoder_only", "is_diffusion"),
    [(True, False), (False, True)],
    ids=["encoder-only", "diffusion"],
)
def test_resumable_requests_rejected_for_non_decoder_models(
    is_mm_encoder_only: bool, is_diffusion: bool
):
    llm = MagicMock()
    llm.vllm_config.is_mm_encoder_only = is_mm_encoder_only
    llm.model_config.is_diffusion = is_diffusion

    with pytest.raises(
        VLLMValidationError,
        match="Resumable requests are not supported",
    ):
        AsyncLLM._validate_resumable_request(llm)


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


@pytest.mark.asyncio
async def test_empty_input_stream_finishes_without_engine_request(monkeypatch):
    llm = AsyncLLM.__new__(AsyncLLM)
    llm.output_processor = OutputProcessor(None, log_stats=False)
    monkeypatch.setattr(llm, "_validate_streaming_input_sampling_params", MagicMock())
    monkeypatch.setattr(
        llm, "get_supported_tasks", AsyncMock(return_value=("generate",))
    )
    monkeypatch.setattr(llm, "_run_output_handler", MagicMock())
    add_request = AsyncMock()
    monkeypatch.setattr(llm, "_add_request", add_request)
    llm.vllm_config = MagicMock(is_mm_encoder_only=False)
    llm.model_config = MagicMock(is_diffusion=False)

    final_req = MagicMock()
    final_req.request_id = "request-int"
    llm.input_processor = MagicMock()
    llm.input_processor.process_inputs.return_value = final_req

    async def empty_stream():
        return
        yield  # pragma: no cover

    queue = await llm._add_streaming_input_request(
        "request", empty_stream(), SamplingParams(max_tokens=4)
    )
    task = queue._input_stream_task
    assert task is not None
    await task

    assert queue.get_nowait() is STREAM_FINISHED
    add_request.assert_not_awaited()


def _make_streaming_llm(
    monkeypatch: pytest.MonkeyPatch, num_chunks: int, has_request: bool = True
):
    llm = AsyncLLM.__new__(AsyncLLM)
    monkeypatch.setattr(llm, "_validate_streaming_input_sampling_params", MagicMock())
    monkeypatch.setattr(llm, "_validate_resumable_request", MagicMock())
    monkeypatch.setattr(
        llm, "get_supported_tasks", AsyncMock(return_value=("generate",))
    )
    monkeypatch.setattr(llm, "_run_output_handler", MagicMock())
    monkeypatch.setattr(llm, "_add_request", AsyncMock())
    llm.model_config = MagicMock()
    llm.output_processor = MagicMock()
    llm.output_processor.has_request.return_value = has_request

    # The first processed request is the final (end-of-stream) request.
    reqs = [MagicMock(prompt_embeds=None) for _ in range(num_chunks + 1)]
    reqs[0].request_id = "r1-int"
    llm.input_processor = MagicMock()
    llm.input_processor.process_inputs.side_effect = reqs
    return llm, reqs


async def _stream_chunks(llm: AsyncLLM, num_chunks: int) -> None:
    async def chunks():
        for i in range(num_chunks):
            yield StreamingInput(prompt=f"frame-{i}")

    with mock.patch(
        "vllm.v1.engine.async_llm.extract_prompt_components",
        return_value=("", None, None),
    ):
        queue = await llm._add_streaming_input_request(
            "r1", chunks(), SamplingParams(max_tokens=4)
        )
        task = queue._input_stream_task
        assert task is not None
        await task


@pytest.mark.asyncio
@pytest.mark.parametrize("session_live", [True, False])
async def test_handle_inputs_stops_after_session_finished(
    session_live: bool, monkeypatch: pytest.MonkeyPatch
):
    """Once the session's request state is gone (aborted or errored), no
    further chunks or final request are sent: they would recreate a request
    state that is never cleaned up."""
    llm, reqs = _make_streaming_llm(monkeypatch, num_chunks=2, has_request=session_live)
    await _stream_chunks(llm, num_chunks=2)

    sent = [call.args[0] for call in llm._add_request.await_args_list]
    # Both chunks then the final request, or only the first chunk.
    assert sent == ([reqs[1], reqs[2], reqs[0]] if session_live else [reqs[1]])


@pytest_asyncio.fixture
async def streaming_frontend(tmp_path, monkeypatch):
    """Real frontend objects with only the remote EngineCore service replaced."""
    Qwen2Config(
        architectures=["Qwen2ForCausalLM"],
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=128,
    ).save_pretrained(tmp_path)
    config = VllmConfig(
        model_config=ModelConfig(
            model=str(tmp_path),
            dtype="float32",
            enforce_eager=True,
            max_model_len=128,
            skip_tokenizer_init=True,
        ),
        device_config=DeviceConfig("cpu"),
        cache_config=CacheConfig(enable_prefix_caching=False),
        scheduler_config=SchedulerConfig(
            max_model_len=128,
            max_num_seqs=2,
            max_num_batched_tokens=128,
            is_encoder_decoder=False,
        ),
    )
    service = MagicMock()
    service.resources.engine_dead = False
    service.get_supported_tasks_async = AsyncMock(return_value=("generate",))
    outputs: asyncio.Queue[EngineCoreOutputs] = asyncio.Queue()

    async def submit(request):
        if request.resumable:
            outputs.put_nowait(
                EngineCoreOutputs(
                    outputs=[
                        EngineCoreOutput(
                            request_id=request.request_id,
                            new_token_ids=[7],
                            finish_reason=FinishReason.LENGTH,
                        )
                    ]
                )
            )

    service.get_output_async = AsyncMock(side_effect=outputs.get)
    service.add_request_async = AsyncMock(side_effect=submit)
    service.abort_requests_async = AsyncMock()
    monkeypatch.setattr(
        EngineCoreClient, "make_async_mp_client", MagicMock(return_value=service)
    )
    engine = AsyncLLM(config, UniProcExecutor, log_stats=False, log_requests=False)
    try:
        yield engine
    finally:
        engine.shutdown()
        if engine.output_handler is not None:
            await asyncio.gather(engine.output_handler, return_exceptions=True)


async def _abort_before_first_chunk(engine: AsyncLLM) -> list[int]:
    """Explicit abort closes both sides of a stream awaiting its first input."""
    entered, release, closed = asyncio.Event(), asyncio.Event(), asyncio.Event()
    yielded = []

    async def inputs():
        entered.set()
        try:
            await release.wait()
            yielded.append(True)
            yield StreamingInput(prompt={"prompt_token_ids": [1, 2, 3]})
        finally:
            closed.set()

    async def consume():
        return [
            output
            async for output in engine.generate(
                inputs(),
                SamplingParams(max_tokens=4, output_kind=RequestOutputKind.DELTA),
                "before-first",
            )
        ]

    consumer = asyncio.create_task(consume())
    try:
        await asyncio.wait_for(entered.wait(), 5)
        await engine.abort("before-first")
        assert await asyncio.wait_for(asyncio.shield(consumer), 5) == []
        await asyncio.wait_for(closed.wait(), 5)
        assert not yielded
        assert engine.get_num_unfinished_requests() == 0

        # Repeated aborts must not poison a new session using the same ID.
        await engine.abort("before-first")

        async def healthy_inputs():
            yield StreamingInput(prompt={"prompt_token_ids": [1, 2, 3]})

        async def reuse_id():
            return [
                token
                async for output in engine.generate(
                    healthy_inputs(),
                    SamplingParams(
                        temperature=0,
                        max_tokens=1,
                        ignore_eos=True,
                        output_kind=RequestOutputKind.DELTA,
                    ),
                    "before-first",
                )
                for completion in output.outputs
                for token in completion.token_ids
            ]

        return await asyncio.wait_for(reuse_id(), 30)
    finally:
        consumer.cancel()
        await asyncio.gather(consumer, return_exceptions=True)
        release.set()


@pytest.mark.cpu_test
@pytest.mark.asyncio
async def test_abort_stream_before_first_chunk(streaming_frontend):
    assert await _abort_before_first_chunk(streaming_frontend) == [7]


@pytest.mark.asyncio
async def test_abort_stream_before_first_chunk_e2e():
    engine = AsyncLLM.from_engine_args(
        AsyncEngineArgs(
            model="Qwen/Qwen2.5-0.5B-Instruct",
            enforce_eager=True,
            max_model_len=256,
            max_num_batched_tokens=128,
            max_num_seqs=2,
            gpu_memory_utilization=0.2,
            enable_prefix_caching=False,
            disable_log_stats=True,
        )
    )
    try:
        assert len(await _abort_before_first_chunk(engine)) == 1
    finally:
        engine.shutdown()


@pytest.mark.cpu_test
@pytest.mark.asyncio
@pytest.mark.parametrize("internal", [False, True])
async def test_abort_stream_before_producer_starts(streaming_frontend, internal):
    """Cancellation before task startup must also finish the output collector."""
    engine = streaming_frontend
    entered = asyncio.Event()

    async def inputs():
        entered.set()
        yield StreamingInput(prompt={"prompt_token_ids": [1, 2, 3]})

    collector = await engine.add_request(
        "not-started",
        inputs(),
        SamplingParams(max_tokens=4, output_kind=RequestOutputKind.DELTA),
    )
    try:
        await engine.abort(
            collector.request_id if internal else "not-started", internal=internal
        )
        assert await asyncio.wait_for(collector.get(), 5) is STREAM_FINISHED
        assert not entered.is_set()
    finally:
        collector.close()
        await engine.abort("not-started")
