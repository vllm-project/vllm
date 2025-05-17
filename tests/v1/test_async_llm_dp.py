# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
from contextlib import ExitStack

import pytest

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.platforms import current_platform
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.core_client import DPAsyncMPClient

from .utils import generate_dp

engine_args = AsyncEngineArgs(
    model="ibm-research/PowerMoE-3b",
    enforce_eager=True,
    disable_log_requests=True,
    tensor_parallel_size=int(os.getenv("TP_SIZE", 1)),
    data_parallel_size=int(os.getenv("DP_SIZE", 2)),
)

if not current_platform.supports_v1(engine_args.create_model_config()):
    pytest.skip(reason="Requires V1-supporting platform.",
                allow_module_level=True)


@pytest.mark.parametrize(
    "output_kind", [RequestOutputKind.DELTA, RequestOutputKind.FINAL_ONLY])
@pytest.mark.asyncio
async def test_load(output_kind: RequestOutputKind):

    with ExitStack() as after:

        prompt = "This is a test of data parallel"

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
                    generate_dp(engine, request_id, prompt, output_kind,
                                NUM_EXPECTED_TOKENS)))

        # Confirm that we got all the EXPECTED tokens from the requests.
        done, pending = await asyncio.wait(tasks,
                                           return_when=asyncio.FIRST_EXCEPTION)
        for task in pending:
            task.cancel()
        for task in done:
            num_generated_tokens, request_id = await task
            assert num_generated_tokens == NUM_EXPECTED_TOKENS, (
                f"{request_id} generated {num_generated_tokens} but "
                f"expected {NUM_EXPECTED_TOKENS}")

        assert not engine.output_processor.has_unfinished_requests()

        # testing internals here which may break
        core_client: DPAsyncMPClient = engine.engine_core
        # the engines only synchronize stopping every N steps so
        # allow a small amount of time here.
        for _ in range(10):
            if not core_client.engines_running:
                break
            await asyncio.sleep(0.5)

        assert not core_client.engines_running
        assert not core_client.reqs_in_flight
