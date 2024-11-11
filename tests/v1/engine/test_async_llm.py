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
                              disable_log_requests=True)


async def generate(engine: AsyncLLM, request_id: str,
                   max_tokens: int) -> Tuple[int, str]:
    count = 0
    async for _ in engine.generate(request_id=request_id,
                                   prompt="Hello my name is Robert and",
                                   sampling_params=SamplingParams(
                                       max_tokens=max_tokens, temperature=0)):

        count += 1
        await asyncio.sleep(0.)

    return count, request_id


@pytest.mark.asyncio
async def test_load(monkeypatch):
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
        failed_request_id = None
        tokens = None
        for task in tasks:
            num_generated_tokens, request_id = await task
            if (num_generated_tokens != NUM_EXPECTED_TOKENS
                    and failed_request_id is None):
                failed_request_id = request_id
                tokens = num_generated_tokens

        assert failed_request_id is None, (
            f"{failed_request_id} generated {tokens} but "
            f"expected {NUM_EXPECTED_TOKENS}")

        engine.shutdown()
