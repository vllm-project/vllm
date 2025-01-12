import asyncio
from typing import Tuple

import pytest

from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.platforms import current_platform
from vllm.v1.engine.async_llm import AsyncLLM

if not current_platform.is_cuda():
    pytest.skip(reason="V1 currently only supported on CUDA.",
                allow_module_level=True)

ENGINE_ARGS = AsyncEngineArgs(model="meta-llama/Llama-3.2-1B",
                              enforce_eager=True,
                              disable_log_requests=True)


async def run_example(
    engine: AsyncLLM,
    request_id: str,
    num_tokens: int,
    abort_after: int = 0
) -> Tuple[int, int, str]:
    
    generator = engine.generate(
        request_id=request_id,
        prompt="Hello my name is Robert and",
        sampling_params=SamplingParams(max_tokens=num_tokens, temperature=0))

    count = 0
    try:
        async for _ in generator():
            count += 1
            print(f"{request_id=}, {count=}, {abort_after=}")
            if count == abort_after:
                # Simulate request cancellation.
                print(f"{request_id=}")
                asyncio.current_task().cancel()
    except asyncio.CancelledError:
        print(f"{request_id=}")
        assert request_id not in engine.request_states
    finally:
        
        expected_count = num_tokens if abort_after == 0 else abort_after
        return count, expected_count, request_id


@pytest.mark.asyncio
async def test_load(monkeypatch):
    # TODO(rickyx): Remove monkeypatch once we have a better way to test V1
    # so that in the future when we switch, we don't have to change all the
    # tests.
    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "1")

        engine = AsyncLLM.from_engine_args(ENGINE_ARGS)

        NUM_REQUESTS = 100
        NUM_EXPECTED_TOKENS = 10
        # Abort 1/100 requests after 5 tokens.
        ABORT_RATE = 100
        ABORT_AFTER = 5

        request_ids = [f"request-{i}" for i in range(NUM_REQUESTS)]

        # Create concurrent requests.
        tasks = [
            asyncio.create_task(run_example(
                engine=engine,
                request_id=request_id,
                num_tokens=NUM_EXPECTED_TOKENS,
                abort_after=(ABORT_AFTER if idx % ABORT_RATE == 0 else 0)
            )) for idx, request_id in enumerate(request_ids)
        ]

        # Confirm that we got all the EXPECTED tokens from the requests.
        failed_request_id = None
        tokens = None
        for task in tasks:
            num_generated_tokens, expected_tokens, request_id = await task
            if (num_generated_tokens != expected_tokens
                    and failed_request_id is None):
                failed_request_id = request_id
                tokens = num_generated_tokens

        assert failed_request_id is None, (
            f"{failed_request_id} generated {tokens} but "
            f"expected {NUM_EXPECTED_TOKENS}")

        engine.shutdown()
