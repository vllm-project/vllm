import asyncio
from typing import List, Optional, Tuple

import pytest

from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.platforms import current_platform
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.utils import STR_ASYNC_LLM_PROMPT_LP_APC_UNSUPPORTED

if not current_platform.is_cuda():
    pytest.skip(reason="V1 currently only supported on CUDA.",
                allow_module_level=True)

ENGINE_ARGS = AsyncEngineArgs(model="meta-llama/Llama-3.2-1B",
                              enforce_eager=True,
                              disable_log_requests=True)


async def generate(
    engine: AsyncLLM,
    request_id: str,
    max_tokens: Optional[int] = None,
    sampling_params: Optional[SamplingParams] = None,
) -> Tuple[int, str]:
    """Wrapper for `AsyncLLM` generation.

    At least one of `max_tokens` and `sampling_params` must
    not be `None`. If `sampling_params` is `None`, `max_tokens`
    is used to create a `SamplingParams` instance. If
    `sampling_params` is provided, `max_tokens` is not used.
    
    Args:
      engine: AsyncLLM instance
      request_id: AsyncLLM request ID
      max_tokens: (optional) max number of tokens to generate
      sampling_params: (optional) request sampling params

    Returns:
      count: number of returns from engine.generate()
      request_id
    """
    assert not (max_tokens is None and sampling_params is None), (
        "At least one of max_tokens and sampling_params"
        " must not be None.")
    if sampling_params is None:
        sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0)
    count = 0
    async for _ in engine.generate(request_id=request_id,
                                   prompt="Hello my name is Robert and",
                                   sampling_params=sampling_params):

        count += 1
        await asyncio.sleep(0.)

    return count, request_id


@pytest.mark.asyncio
async def test_async_llm_refuses_prompt_logprobs_with_apc(monkeypatch):
    """Test passes if AsyncLLM raises an exception when it is configured
    for automatic prefix caching and it receives a request with
    prompt_logprobs enabled, which is incompatible."""
    # TODO(rickyx): Remove monkeypatch VLLM_USE_V1 setting once we have a
    # better way to test V1 so that in the future when we switch, we don't
    # have to change all the tests.
    monkeypatch.setenv("VLLM_USE_V1", "1")
    # Create AsyncLLM engine with APC
    apc_engine_args = AsyncEngineArgs(model="facebook/opt-125m",
                                      enable_prefix_caching=True,
                                      gpu_memory_utilization=0.8,
                                      disable_log_requests=True)
    engine = AsyncLLM.from_engine_args(apc_engine_args)
    try:
        with pytest.raises(ValueError) as excinfo:
            # Issue a request with prompt logprobs enabled, which should fail
            await asyncio.create_task(
                generate(engine,
                         "request-0",
                         sampling_params=SamplingParams(max_tokens=10,
                                                        temperature=0,
                                                        prompt_logprobs=5)))
        # Validate exception string is correct
        assert str(excinfo.value) == STR_ASYNC_LLM_PROMPT_LP_APC_UNSUPPORTED
    finally:
        # Shut down engine
        engine.shutdown()


@pytest.mark.asyncio
async def test_load(monkeypatch):
    # TODO(rickyx): Remove monkeypatch once we have a better way to test V1
    # so that in the future when we switch, we don't have to change all the
    # tests.
    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "1")

        engine = AsyncLLM.from_engine_args(ENGINE_ARGS)

        NUM_REQUESTS = 10000
        NUM_EXPECTED_TOKENS = 10

        request_ids = [f"request-{i}" for i in range(NUM_REQUESTS)]

        # Create concurrent requests.
        tasks = []
        for request_id in request_ids:
            tasks.append(
                asyncio.create_task(
                    generate(engine, request_id, NUM_EXPECTED_TOKENS)))

        # Confirm that we got all the EXPECTED tokens from the requests.
        for task in tasks:
            num_generated_tokens, request_id = await task
            assert num_generated_tokens == NUM_EXPECTED_TOKENS, (
                f"{request_id} generated {num_generated_tokens} but "
                f"expected {NUM_EXPECTED_TOKENS}")

        assert not engine.output_processor.has_unfinished_requests()
        engine.shutdown()


@pytest.mark.asyncio
async def test_abort(monkeypatch):

    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "1")

        engine = AsyncLLM.from_engine_args(ENGINE_ARGS)

        NUM_REQUESTS = 100
        NUM_EXPECTED_TOKENS = 100
        REQUEST_IDS_TO_ABORT = range(1, 100, 10)

        request_ids = [f"request-{i}" for i in range(NUM_REQUESTS)]

        # Create concurrent requests.
        tasks: List[asyncio.Task] = []
        for request_id in request_ids:
            tasks.append(
                asyncio.create_task(
                    generate(engine, request_id, NUM_EXPECTED_TOKENS)))

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
                assert num_generated_tokens == NUM_EXPECTED_TOKENS, (
                    f"{request_id} generated {num_generated_tokens} but "
                    f"expected {NUM_EXPECTED_TOKENS}")

        assert not engine.output_processor.has_unfinished_requests()

        # Confirm we can do another generation.
        request_id = f"request-{REQUEST_IDS_TO_ABORT[0]}"
        task = asyncio.create_task(
            generate(engine, request_id, NUM_EXPECTED_TOKENS))
        num_generated_tokens, request_id = await task
        assert num_generated_tokens == NUM_EXPECTED_TOKENS
        assert not engine.output_processor.has_unfinished_requests()

        engine.shutdown()
