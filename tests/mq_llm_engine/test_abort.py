import asyncio
import tempfile
import uuid

import pytest

from tests.mq_llm_engine.utils import RemoteMQLLMEngine
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs

MODEL = "Qwen/Qwen2-0.5B-Instruct"
ENGINE_ARGS = AsyncEngineArgs(model=MODEL)
RAISED_ERROR = KeyError
RAISED_VALUE = "foo"


@pytest.fixture(scope="function")
def tmp_socket():
    with tempfile.TemporaryDirectory() as td:
        yield f"ipc://{td}/{uuid.uuid4()}"


@pytest.mark.asyncio
async def test_abort(tmp_socket):
    with RemoteMQLLMEngine(engine_args=ENGINE_ARGS,
                           ipc_path=tmp_socket) as engine:

        client = await engine.make_client()

        request_id_to_be_aborted = "request-aborted"
        request_ids_a = [f"request-a-{idx}" for idx in range(10)]
        request_ids_b = [f"request-b-{idx}" for idx in range(10)]

        async def run_to_completion(request_id) -> bool:
            EXPECTED = 250
            count = 0
            async for _ in client.generate(inputs="Hello my name is",
                                           sampling_params=SamplingParams(
                                               max_tokens=EXPECTED,
                                               temperature=0),
                                           request_id=request_id):
                count += 1
                await asyncio.sleep(0.)

            # Confirm we generated all the tokens we expected.
            return count == EXPECTED

        async def run_to_be_aborted(request_id):
            EXPECTED = 250
            count = 0
            try:
                async for _ in client.generate(inputs="Hello my name is",
                                               sampling_params=SamplingParams(
                                                   max_tokens=EXPECTED,
                                                   temperature=0),
                                               request_id=request_id):
                    count += 1
                    await asyncio.sleep(0.)

            # Confirm this was actually stopped.
            except asyncio.CancelledError:
                assert (count < EXPECTED)

        # Create concurrent requests.
        tasks_a = [
            asyncio.create_task(run_to_completion(request_id))
            for request_id in request_ids_a
        ]
        task_aborted = asyncio.create_task(
            run_to_be_aborted(request_id_to_be_aborted))
        tasks_b = [
            asyncio.create_task(run_to_completion(request_id))
            for request_id in request_ids_b
        ]

        await asyncio.sleep(0.5)
        await client.abort(request_id_to_be_aborted)

        # Confirm that we got all the EXPECTED tokens from the requests.
        for task in tasks_a:
            assert (await task), "Expected this task to run to completion."
        for task in tasks_b:
            assert (await task), "Expected this task to run to completion."

        # Cancel task (this will hang indefinitely if not).
        task_aborted.cancel()

        # Shutdown.
        client.close()
