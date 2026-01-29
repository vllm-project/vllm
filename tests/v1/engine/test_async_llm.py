# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import time
from contextlib import ExitStack
from unittest.mock import MagicMock

import pytest

from vllm import SamplingParams
from vllm.assets.image import ImageAsset
from vllm.config import VllmConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.engine.protocol import (
    ErrorResponse,
)
from vllm.entrypoints.openai.serving_models import BaseModelPath, OpenAIServingModels
from vllm.inputs import PromptType
from vllm.outputs import RequestOutput
from vllm.platforms import current_platform
from vllm.sampling_params import RequestOutputKind
from vllm.utils.torch_utils import set_default_torch_num_threads
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.metrics.loggers import (
    AggregatedLoggingStatLogger,
    LoggingStatLogger,
    PerEngineStatLoggerAdapter,
    PrometheusStatLogger,
)

if not current_platform.is_cuda():
    pytest.skip(reason="V1 currently only supported on CUDA.", allow_module_level=True)

TEXT_ENGINE_ARGS = AsyncEngineArgs(
    model="meta-llama/Llama-3.2-1B-Instruct",
    enforce_eager=True,
)

VISION_ENGINE_ARGS = AsyncEngineArgs(
    model="Qwen/Qwen2-VL-2B-Instruct", enforce_eager=True
)

TEXT_PROMPT = "Hello my name is Robert and"

VISION_PROMPT_TEMPLATE = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>"
    "\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
    "What is in the image?<|im_end|>\n"
    "<|im_start|>assistant\n"
)
VISION_PROMPT = {
    "prompt": VISION_PROMPT_TEMPLATE,
    "multi_modal_data": {"image": ImageAsset("stop_sign").pil_image},
}


async def generate(
    engine: AsyncLLM,
    request_id: str,
    prompt: PromptType,
    output_kind: RequestOutputKind,
    max_tokens: int,
    n: int = 1,
    prompt_logprobs: int | None = None,
    cancel_after: int | None = None,
) -> tuple[int, str]:
    # Ensure generate doesn't complete too fast for cancellation test.
    await asyncio.sleep(0.2)

    count = 0
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        ignore_eos=True,
        output_kind=output_kind,
        temperature=0.5,
        seed=33,
        n=n,
        prompt_logprobs=prompt_logprobs,
    )
    async for out in engine.generate(
        request_id=request_id, prompt=prompt, sampling_params=sampling_params
    ):
        num_tokens = sum(len(output.token_ids) for output in out.outputs)
        if output_kind == RequestOutputKind.DELTA:
            count += num_tokens
        else:
            count = num_tokens

        if cancel_after is not None and count >= cancel_after:
            return count, request_id

        await asyncio.sleep(0.0)

    return count, request_id


@pytest.mark.parametrize(
    "output_kind", [RequestOutputKind.DELTA, RequestOutputKind.FINAL_ONLY]
)
@pytest.mark.parametrize(
    "engine_args,prompt",
    [(TEXT_ENGINE_ARGS, TEXT_PROMPT), (VISION_ENGINE_ARGS, VISION_PROMPT)],
)
@pytest.mark.asyncio
async def test_load(
    output_kind: RequestOutputKind,
    engine_args: AsyncEngineArgs,
    prompt: PromptType,
):
    with ExitStack() as after:
        with set_default_torch_num_threads(1):
            engine = AsyncLLM.from_engine_args(engine_args)
        after.callback(engine.shutdown)

        NUM_REQUESTS = 100
        NUM_EXPECTED_TOKENS = 10

        request_ids = [f"request-{i}" for i in range(NUM_REQUESTS)]

        # Create concurrent requests.
        tasks = []
        for request_id in request_ids:
            tasks.append(
                asyncio.create_task(
                    generate(
                        engine, request_id, prompt, output_kind, NUM_EXPECTED_TOKENS
                    )
                )
            )

        # Confirm that we got all the EXPECTED tokens from the requests.
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
        for task in pending:
            task.cancel()
        for task in done:
            num_generated_tokens, request_id = await task
            assert num_generated_tokens == NUM_EXPECTED_TOKENS, (
                f"{request_id} generated {num_generated_tokens} but "
                f"expected {NUM_EXPECTED_TOKENS}"
            )

        assert not engine.output_processor.has_unfinished_requests()


@pytest.mark.parametrize(
    "output_kind", [RequestOutputKind.DELTA, RequestOutputKind.FINAL_ONLY]
)
@pytest.mark.parametrize(
    "engine_args,prompt",
    [(TEXT_ENGINE_ARGS, TEXT_PROMPT), (VISION_ENGINE_ARGS, VISION_PROMPT)],
)
@pytest.mark.asyncio
async def test_abort(
    output_kind: RequestOutputKind,
    engine_args: AsyncEngineArgs,
    prompt: PromptType,
):
    with ExitStack() as after:
        with set_default_torch_num_threads(1):
            engine = AsyncLLM.from_engine_args(engine_args)
        after.callback(engine.shutdown)

        NUM_REQUESTS = 100
        NUM_EXPECTED_TOKENS = 100
        NUM_EXPECTED_TOKENS_LONG = 50000
        REQUEST_IDS_TO_ABORT = range(1, 100, 10)
        PARALLEL_SAMPLE_REQ_IDS = range(1, 100, 15)

        request_ids = [f"request-{i}" for i in range(NUM_REQUESTS)]

        # Create concurrent requests.
        tasks: list[asyncio.Task] = []
        for idx, request_id in enumerate(request_ids):
            max_tokens = (
                NUM_EXPECTED_TOKENS_LONG
                if (idx in REQUEST_IDS_TO_ABORT)
                else NUM_EXPECTED_TOKENS
            )
            n = 3 if idx in PARALLEL_SAMPLE_REQ_IDS else 1
            tasks.append(
                asyncio.create_task(
                    generate(engine, request_id, prompt, output_kind, max_tokens, n)
                )
            )

        # API server cancels requests when they disconnect.
        for idx in REQUEST_IDS_TO_ABORT:
            tasks[idx].cancel()
            await asyncio.sleep(0.1)

        # Confirm the other requests are okay.
        for idx, task in enumerate(tasks):
            # Confirm that it was actually canceled.
            if idx in REQUEST_IDS_TO_ABORT:
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                # Otherwise, make sure the request was not impacted.
                num_generated_tokens, request_id = await task
                n = 3 if idx in PARALLEL_SAMPLE_REQ_IDS else 1
                expected_tokens = NUM_EXPECTED_TOKENS * n
                assert num_generated_tokens == expected_tokens, (
                    f"{request_id} generated {num_generated_tokens} but "
                    f"expected {expected_tokens}"
                )

        # Make sure all aborted requests were really aborted.
        assert not engine.output_processor.has_unfinished_requests()

        # Confirm we can do another generation.
        request_id = f"request-{REQUEST_IDS_TO_ABORT[0]}"
        task = asyncio.create_task(
            generate(engine, request_id, prompt, output_kind, NUM_EXPECTED_TOKENS)
        )
        num_generated_tokens, request_id = await task
        assert num_generated_tokens == NUM_EXPECTED_TOKENS
        assert not engine.output_processor.has_unfinished_requests()


@pytest.mark.parametrize(
    "output_kind", [RequestOutputKind.DELTA, RequestOutputKind.FINAL_ONLY]
)
@pytest.mark.asyncio
async def test_multi_abort(output_kind: RequestOutputKind):
    with ExitStack() as after:
        with set_default_torch_num_threads(1):
            engine = AsyncLLM.from_engine_args(TEXT_ENGINE_ARGS)
        after.callback(engine.shutdown)

        NUM_REQUESTS = 50
        NUM_EXPECTED_TOKENS = 100
        NUM_EXPECTED_TOKENS_LONG = 50000
        REQUEST_IDS_TO_ABORT = [5, 10, 15, 20, 25]
        PARALLEL_SAMPLE_REQ_IDS = [5, 15, 30, 35]

        request_ids = [f"request-{i}" for i in range(NUM_REQUESTS)]

        # Create concurrent requests.
        tasks: list[asyncio.Task] = []
        for idx, request_id in enumerate(request_ids):
            max_tokens = (
                NUM_EXPECTED_TOKENS_LONG
                if (idx in REQUEST_IDS_TO_ABORT)
                else NUM_EXPECTED_TOKENS
            )
            n = 3 if idx in PARALLEL_SAMPLE_REQ_IDS else 1
            tasks.append(
                asyncio.create_task(
                    generate(
                        engine, request_id, TEXT_PROMPT, output_kind, max_tokens, n
                    )
                )
            )

        # Let requests start
        await asyncio.sleep(0.5)

        # Use multi-abort to abort multiple requests at once
        abort_request_ids = [request_ids[i] for i in REQUEST_IDS_TO_ABORT]
        await engine.abort(abort_request_ids, internal=False)

        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Verify results
        for idx, result in enumerate(results):
            if idx in REQUEST_IDS_TO_ABORT:
                # Aborted requests should return partial results
                assert isinstance(result, tuple), (
                    f"Request {idx} should have completed with partial results"
                )
                num_generated_tokens, request_id = result
                # Should have generated some tokens before abort
                assert num_generated_tokens > 0, (
                    f"Aborted request {request_id} should have generated some tokens"
                )
            else:
                # Non-aborted requests should complete normally
                assert isinstance(result, tuple), (
                    f"Request {idx} should have completed successfully"
                )
                num_generated_tokens, request_id = result
                n = 3 if idx in PARALLEL_SAMPLE_REQ_IDS else 1
                expected_tokens = NUM_EXPECTED_TOKENS * n
                assert num_generated_tokens == expected_tokens, (
                    f"{request_id} generated {num_generated_tokens} but "
                    f"expected {expected_tokens}"
                )

        # Make sure all aborted requests were cleaned up
        assert not engine.output_processor.has_unfinished_requests()


@pytest.mark.parametrize("n", [1, 3])
@pytest.mark.parametrize(
    "engine_args,prompt",
    [(TEXT_ENGINE_ARGS, TEXT_PROMPT), (VISION_ENGINE_ARGS, VISION_PROMPT)],
)
@pytest.mark.asyncio
async def test_finished_flag(
    n: int,
    engine_args: AsyncEngineArgs,
    prompt: PromptType,
):
    with ExitStack() as after:
        with set_default_torch_num_threads(1):
            engine = AsyncLLM.from_engine_args(engine_args)
        after.callback(engine.shutdown)

        sampling_params = SamplingParams(
            max_tokens=100,
            output_kind=RequestOutputKind.DELTA,
            temperature=1.0,
            seed=33,
            n=n,
        )
        outputs = [
            out
            async for out in engine.generate(
                request_id="request-33", prompt=prompt, sampling_params=sampling_params
            )
        ]

        # Assert only the last output has the finished flag set
        assert all(not out.finished for out in outputs[:-1])
        assert outputs[-1].finished


@pytest.mark.parametrize(
    "engine_args,prompt",
    [(TEXT_ENGINE_ARGS, TEXT_PROMPT), (VISION_ENGINE_ARGS, VISION_PROMPT)],
)
@pytest.mark.asyncio
async def test_mid_stream_cancellation(
    engine_args: AsyncEngineArgs, prompt: PromptType
):
    """Test that requests can be cancelled mid-stream."""
    with ExitStack() as after:
        with set_default_torch_num_threads(1):
            engine = AsyncLLM.from_engine_args(engine_args)
        after.callback(engine.shutdown)

        NUM_REQUESTS = 100
        NUM_TOKENS = 1000
        NUM_EXPECTED_TOKENS = 20

        request_ids = [f"request-{i}" for i in range(NUM_REQUESTS)]

        # Create concurrent requests that will be cancelled mid-stream
        tasks = []
        for request_id in request_ids:
            tasks.append(
                asyncio.create_task(
                    generate(
                        engine,
                        request_id,
                        prompt,
                        RequestOutputKind.DELTA,
                        NUM_TOKENS,
                        cancel_after=NUM_EXPECTED_TOKENS,
                    )
                )
            )

        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)

        # Verify all tasks were cancelled at the expected point
        for num_generated_tokens, request_id in results:
            assert num_generated_tokens == NUM_EXPECTED_TOKENS, (
                f"{request_id} generated {num_generated_tokens} tokens but "
                f"expected to cancel after {NUM_EXPECTED_TOKENS}"
            )

        # Make sure no requests are left hanging
        assert not engine.output_processor.has_unfinished_requests()

        # Confirm we can reuse the request id after the cancellations.
        request_id = request_ids[0]
        task = asyncio.create_task(
            generate(
                engine, request_id, prompt, RequestOutputKind.DELTA, NUM_EXPECTED_TOKENS
            )
        )
        num_generated_tokens, request_id = await task
        assert num_generated_tokens == NUM_EXPECTED_TOKENS
        assert not engine.output_processor.has_unfinished_requests()


class MockLoggingStatLogger(LoggingStatLogger):
    def __init__(self, vllm_config: VllmConfig, engine_index: int = 0):
        super().__init__(vllm_config, engine_index)
        self.log = MagicMock()


class MockAggregatedStatLogger(AggregatedLoggingStatLogger):
    def __init__(self, vllm_config: VllmConfig, engine_indexes: list[int]):
        super().__init__(vllm_config, engine_indexes)
        self.log = MagicMock()


@pytest.mark.asyncio
async def test_customize_loggers(monkeypatch):
    """Test that we can customize the loggers.
    If a customized logger is provided at the init, it should
    be added to the default loggers.
    """

    with ExitStack() as after:
        with set_default_torch_num_threads(1):
            engine = AsyncLLM.from_engine_args(
                TEXT_ENGINE_ARGS,
                stat_loggers=[MockLoggingStatLogger],
            )
        after.callback(engine.shutdown)

        await engine.do_log_stats()

        stat_loggers = engine.logger_manager.stat_loggers
        assert (
            len(stat_loggers) == 3
        )  # MockLoggingStatLogger + LoggingStatLogger +  Promethus Logger
        print(f"{stat_loggers=}")
        stat_loggers[0].per_engine_stat_loggers[0].log.assert_called_once()
        assert isinstance(stat_loggers[1], PerEngineStatLoggerAdapter)
        assert isinstance(stat_loggers[1].per_engine_stat_loggers[0], LoggingStatLogger)
        assert isinstance(stat_loggers[2], PrometheusStatLogger)


@pytest.mark.asyncio
async def test_customize_aggregated_loggers():
    """Test that we can customize the aggregated loggers.
    If a customized logger is provided at the init, it should
    be added to the default loggers.
    """
    with ExitStack() as after:
        with set_default_torch_num_threads(1):
            engine = AsyncLLM.from_engine_args(
                TEXT_ENGINE_ARGS,
                stat_loggers=[MockLoggingStatLogger, MockAggregatedStatLogger],
            )
        after.callback(engine.shutdown)

        await engine.do_log_stats()

        stat_loggers = engine.logger_manager.stat_loggers
        assert len(stat_loggers) == 4
        #  MockLoggingStatLogger + MockAggregatedStatLogger
        # + LoggingStatLogger + PrometheusStatLogger
        stat_loggers[0].per_engine_stat_loggers[0].log.assert_called_once()
        stat_loggers[1].log.assert_called_once()
        assert isinstance(stat_loggers[2], PerEngineStatLoggerAdapter)
        assert isinstance(stat_loggers[2].per_engine_stat_loggers[0], LoggingStatLogger)
        assert isinstance(stat_loggers[3], PrometheusStatLogger)


@pytest.mark.asyncio(scope="module")
async def test_dp_rank_argument():
    with ExitStack() as after:
        with set_default_torch_num_threads(1):
            engine = AsyncLLM.from_engine_args(TEXT_ENGINE_ARGS)
        after.callback(engine.shutdown)

        sampling_params = SamplingParams(
            max_tokens=100,
            output_kind=RequestOutputKind.DELTA,
            temperature=1.0,
            seed=33,
        )

        # Test with valid DP rank.
        async for _ in engine.generate(
            request_id="request-34",
            prompt=TEXT_PROMPT,
            sampling_params=sampling_params,
            data_parallel_rank=0,
        ):
            pass

        # Test with out-of-range DP rank.
        with pytest.raises(ValueError):
            async for _ in engine.generate(
                request_id="request-35",
                prompt=TEXT_PROMPT,
                sampling_params=sampling_params,
                data_parallel_rank=1,
            ):
                pass


@pytest.mark.asyncio(scope="module")
async def test_header_dp_rank_argument():
    with ExitStack() as after:
        with set_default_torch_num_threads(1):
            engine = AsyncLLM.from_engine_args(TEXT_ENGINE_ARGS)
        after.callback(engine.shutdown)

        MODEL_NAME = "test-model"
        BASE_MODEL_PATHS = [BaseModelPath(name=MODEL_NAME, model_path=MODEL_NAME)]

        # Create models first
        models = OpenAIServingModels(
            engine_client=engine,
            base_model_paths=BASE_MODEL_PATHS,
        )

        # Create serving chat instance
        serving_chat = OpenAIServingChat(
            engine_client=engine,
            models=models,
            response_role="assistant",
            chat_template=None,
            chat_template_content_format="auto",
            request_logger=None,
        )
        # Create a chat completion request
        req = ChatCompletionRequest(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": TEXT_PROMPT}],
            max_tokens=100,
            temperature=1.0,
            seed=33,
        )
        # Test 1: Valid DP rank (0)
        mock_raw_request = MagicMock()
        mock_raw_request.headers = {"X-data-parallel-rank": "0"}
        mock_raw_request.state = MagicMock()

        # Should succeed with valid rank
        response = await serving_chat.create_chat_completion(req, mock_raw_request)
        assert isinstance(response, ChatCompletionResponse), (
            "Expected a ChatCompletionResponse for valid DP rank"
        )

        # Test 2: Out-of-range DP rank (1)
        mock_raw_request.headers = {"X-data-parallel-rank": "1"}

        # should return ErrorResponse for out-of-range rank
        response2 = await serving_chat.create_chat_completion(req, mock_raw_request)
        assert isinstance(response2, ErrorResponse), (
            "Expected an ErrorResponse for out-of-range DP rank"
        )


@pytest.mark.asyncio
async def test_check_health():
    """Test that check_health returns normally for healthy engine
    and raises EngineDeadError when the engine is dead.
    """
    from unittest.mock import patch

    from vllm.v1.engine.exceptions import EngineDeadError

    with ExitStack() as after:
        with set_default_torch_num_threads(1):
            engine = AsyncLLM.from_engine_args(TEXT_ENGINE_ARGS)
        after.callback(engine.shutdown)

        # Test 1: Healthy engine should not raise any exception
        await engine.check_health()

        # Test 2: Mock the errored property to simulate a dead engine
        with (
            patch.object(
                type(engine),
                "errored",
                new_callable=lambda: property(lambda self: True),
            ),
            pytest.raises(EngineDeadError),
        ):
            await engine.check_health()

        # Test 3: Verify healthy engine still works after mock
        await engine.check_health()


@pytest.mark.parametrize(
    "output_kind", [RequestOutputKind.DELTA, RequestOutputKind.FINAL_ONLY]
)
@pytest.mark.asyncio
async def test_abort_final_output(output_kind: RequestOutputKind):
    """Test that abort() returns a final output with correct information."""

    with ExitStack() as after:
        with set_default_torch_num_threads(1):
            engine = AsyncLLM.from_engine_args(TEXT_ENGINE_ARGS)
        after.callback(engine.shutdown)

        request_id = "test-abort-final-output"

        # Start a long-running request
        sampling_params = SamplingParams(
            max_tokens=3000,  # Long enough to allow abort
            ignore_eos=True,
            output_kind=output_kind,
            temperature=0.5,
            seed=42,
        )

        outputs: list[RequestOutput] = []
        generated = asyncio.create_task(
            collect_outputs(engine, request_id, TEXT_PROMPT, sampling_params, outputs)
        )

        # Let it generate some tokens
        await asyncio.sleep(0.5)

        # Abort the request
        await engine.abort(request_id, internal=False)

        # Wait for generation to complete and return final output
        final_output = await generated

        # Verify we got a final output
        assert final_output is not None
        assert final_output.finished
        assert len(final_output.outputs) == 1

        assert final_output.outputs[0].finish_reason == "abort"
        assert final_output.outputs[0].stop_reason is None

        # Verify num_cached_tokens is set correctly
        assert hasattr(final_output, "num_cached_tokens")
        assert final_output.num_cached_tokens >= 0

        # If we got intermediate outputs, verify they are consistent
        if output_kind == RequestOutputKind.DELTA:
            # For DELTA, sum all intermediate tokens should <= final tokens
            token_count = sum(len(output.outputs[0].token_ids) for output in outputs)
            assert token_count > 0
            # This would ordinarily be 0, but could end up > 0 if the
            # final abort is coalesced with another chunk in the output queue.
            assert len(final_output.outputs[0].token_ids) >= 0
        else:
            # For FINAL_ONLY, we should only get the final output
            assert len(outputs) == 0
            assert len(final_output.outputs[0].token_ids) > 0

        assert not engine.output_processor.has_unfinished_requests()


async def collect_outputs(
    engine: AsyncLLM,
    request_id: str,
    prompt: PromptType,
    sampling_params: SamplingParams,
    outputs_list: list[RequestOutput],
) -> RequestOutput | None:
    """Helper to collect outputs and return the final one."""
    final_output: RequestOutput | None = None
    async for output in engine.generate(
        request_id=request_id, prompt=prompt, sampling_params=sampling_params
    ):
        if not output.finished:
            outputs_list.append(output)
        final_output = output
    return final_output


# =============================================================================
# Pause/Resume Tests
# =============================================================================


@pytest.mark.asyncio
async def test_pause_resume_basic():
    """Test basic pause/resume flag behavior and idempotency.

    Tests:
    - pause_generation sets the paused flag
    - resume_generation clears the paused flag
    - calling pause when already paused is a no-op
    - calling resume when not paused is safe
    - all pause modes work with no requests in flight
    - rapid pause/resume cycles don't break the engine
    """
    with ExitStack() as after:
        with set_default_torch_num_threads(1):
            engine = AsyncLLM.from_engine_args(TEXT_ENGINE_ARGS)
        after.callback(engine.shutdown)

        # Initially not paused
        assert not await engine.is_paused()

        # Resume when not paused should be safe
        await engine.resume_generation()
        assert not await engine.is_paused()

        # Pause sets flag
        await engine.pause_generation(mode="abort")
        assert await engine.is_paused()

        # Pause when already paused is a no-op
        await engine.pause_generation(mode="abort")
        assert await engine.is_paused()

        # Resume clears flag
        await engine.resume_generation()
        assert not await engine.is_paused()

        # Test all modes with no requests in flight
        for mode in ("abort", "wait", "keep"):
            await engine.pause_generation(mode=mode)
            assert await engine.is_paused()
            await engine.resume_generation()
            assert not await engine.is_paused()

        # Rapid pause/resume cycles
        for _ in range(10):
            await engine.pause_generation(mode="abort")
            assert await engine.is_paused()
            await engine.resume_generation()
            assert not await engine.is_paused()

        # Engine should still work after all cycles
        sampling_params = SamplingParams(max_tokens=5)
        async for out in engine.generate(
            request_id="post-cycles",
            prompt=TEXT_PROMPT,
            sampling_params=sampling_params,
        ):
            pass
        assert out.finished


@pytest.mark.asyncio
async def test_pause_abort():
    """Test that mode='abort' aborts in-flight requests immediately."""
    with ExitStack() as after:
        with set_default_torch_num_threads(1):
            engine = AsyncLLM.from_engine_args(TEXT_ENGINE_ARGS)
        after.callback(engine.shutdown)

        # Start a long-running request
        sampling_params = SamplingParams(max_tokens=1000, ignore_eos=True)
        outputs: list[RequestOutput] = []

        async def gen():
            async for out in engine.generate(
                request_id="test-abort-pause",
                prompt=TEXT_PROMPT,
                sampling_params=sampling_params,
            ):
                outputs.append(out)
            return outputs[-1] if outputs else None

        # Start generation task
        gen_task = asyncio.create_task(gen())

        # Wait a bit for some tokens
        await asyncio.sleep(0.3)

        # Pause with abort mode
        await engine.pause_generation(mode="abort")

        # Wait for task to complete (should be aborted)
        final_output = await gen_task

        # Request should be finished (aborted)
        assert final_output is not None
        assert final_output.finished
        assert final_output.outputs[0].finish_reason == "abort"

        # Also test that new requests are blocked while paused, then resume
        assert await engine.is_paused()

        request_completed = False

        async def gen_blocked():
            nonlocal request_completed
            async for out in engine.generate(
                request_id="test-blocked",
                prompt=TEXT_PROMPT,
                sampling_params=SamplingParams(max_tokens=5),
            ):
                pass
            request_completed = True
            return out

        # Start a request (should block)
        gen_task2 = asyncio.create_task(gen_blocked())

        # Wait a bit - request should not have completed
        await asyncio.sleep(0.3)
        assert not request_completed, "Request should be blocked while paused"

        # Resume
        await engine.resume_generation()

        # Now request should complete
        final_output2 = await asyncio.wait_for(gen_task2, timeout=10.0)
        assert request_completed
        assert final_output2.finished


@pytest.mark.asyncio
async def test_pause_wait():
    """Test that mode='wait' waits for in-flight requests to complete."""
    with ExitStack() as after:
        with set_default_torch_num_threads(1):
            engine = AsyncLLM.from_engine_args(TEXT_ENGINE_ARGS)
        after.callback(engine.shutdown)

        # Start a request - use fewer tokens since wait mode waits for completion
        sampling_params = SamplingParams(max_tokens=10, ignore_eos=True)
        got_first_token = asyncio.Event()

        async def gen():
            async for out in engine.generate(
                request_id="test-wait",
                prompt=TEXT_PROMPT,
                sampling_params=sampling_params,
            ):
                got_first_token.set()
            return out

        # Start generation
        gen_task = asyncio.create_task(gen())

        # Wait for generation to start (event-driven)
        await asyncio.wait_for(got_first_token.wait(), timeout=30.0)

        # Pause with wait mode - should wait for request to finish
        await engine.pause_generation(mode="wait")

        # By now the request should be done (wait mode waits for completion)
        final_output = await asyncio.wait_for(gen_task, timeout=30.0)

        assert final_output.finished
        # Should complete normally, not aborted
        assert final_output.outputs[0].finish_reason != "abort"


@pytest.mark.asyncio
async def test_pause_keep():
    """Test that mode='keep' freezes requests and they resume later.

    Tests:
    - Single request is frozen and resumes with timing gap
    - Multiple concurrent requests all resume correctly
    """
    with ExitStack() as after:
        with set_default_torch_num_threads(1):
            engine = AsyncLLM.from_engine_args(TEXT_ENGINE_ARGS)
        after.callback(engine.shutdown)

        # --- Test 1: Single request with timing verification ---
        sampling_params = SamplingParams(max_tokens=30, ignore_eos=True)
        token_times: list[tuple[int, float]] = []
        pause_duration = 5.0

        async def generator_task():
            """Generate tokens and record timestamps."""
            async for output in engine.generate(
                request_id="test-keep-single",
                prompt=TEXT_PROMPT,
                sampling_params=sampling_params,
            ):
                token_count = len(output.outputs[0].token_ids)
                token_times.append((token_count, time.monotonic()))
            return output

        async def controller_task():
            """Pause and resume the engine."""
            # Wait for some tokens (event-driven, handles slow token generation)
            while len(token_times) < 5:
                await asyncio.sleep(0.01)

            # Pause with keep mode
            await engine.pause_generation(mode="keep")

            # Sleep while paused
            await asyncio.sleep(pause_duration)

            # Resume
            await engine.resume_generation()

        # Run both tasks with timeout for slow generation
        gen_task = asyncio.create_task(generator_task())
        ctrl_task = asyncio.create_task(controller_task())

        final_output, _ = await asyncio.wait_for(
            asyncio.gather(gen_task, ctrl_task), timeout=60.0
        )

        # Request should complete with all tokens
        assert final_output.finished
        assert len(final_output.outputs[0].token_ids) == 30

        # Analyze timing gaps - should see a gap matching pause duration
        max_gap = 0.0
        for i in range(1, len(token_times)):
            gap = token_times[i][1] - token_times[i - 1][1]
            max_gap = max(max_gap, gap)

        # The max gap should be close to the pause duration
        assert max_gap >= pause_duration * 0.8, (
            f"Expected gap of ~{pause_duration}s, got {max_gap:.3f}s"
        )

        # --- Test 2: Multiple concurrent requests ---
        num_requests = 3
        sampling_params2 = SamplingParams(max_tokens=10, ignore_eos=True)
        completed_requests: list[str] = []
        any_token_generated = asyncio.Event()

        async def gen_multi(request_id: str):
            async for out in engine.generate(
                request_id=request_id,
                prompt=TEXT_PROMPT,
                sampling_params=sampling_params2,
            ):
                any_token_generated.set()
            completed_requests.append(request_id)
            return out

        # Start multiple requests
        tasks = [
            asyncio.create_task(gen_multi(f"req-multi-{i}"))
            for i in range(num_requests)
        ]

        # Wait for at least one token across any request (event-driven)
        await asyncio.wait_for(any_token_generated.wait(), timeout=30.0)

        # Pause with keep mode
        await engine.pause_generation(mode="keep")

        # Wait while paused
        await asyncio.sleep(0.5)

        # Resume
        await engine.resume_generation()

        # All requests should complete
        results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=60.0)

        assert len(completed_requests) == num_requests
        for result in results:
            assert result.finished
            assert len(result.outputs[0].token_ids) == 10
