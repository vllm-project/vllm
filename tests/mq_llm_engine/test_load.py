"""Test that the MQLLMEngine is able to handle 10k concurrent requests."""

import asyncio
import tempfile
import uuid

import pytest

from tests.mq_llm_engine.utils import RemoteMQLLMEngine, generate
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import RequestOutputKind, SamplingParams

MODEL = "google/gemma-1.1-2b-it"
NUM_EXPECTED_TOKENS = 10
NUM_REQUESTS = 10000

# Scenarios to test for num generated token.
ENGINE_ARGS = AsyncEngineArgs(model=MODEL, disable_log_requests=True)


@pytest.fixture(scope="function")
def tmp_socket():
    with tempfile.TemporaryDirectory() as td:
        yield f"ipc://{td}/{uuid.uuid4()}"


@pytest.mark.asyncio
async def test_load(tmp_socket):
    with RemoteMQLLMEngine(engine_args=ENGINE_ARGS,
                           ipc_path=tmp_socket) as engine:

        client = await engine.make_client()

        request_ids = [f"request-{i}" for i in range(NUM_REQUESTS)]

        # Create concurrent requests.
        tasks = []
        for request_id in request_ids:
            tasks.append(
                asyncio.create_task(
                    generate(client, request_id, NUM_EXPECTED_TOKENS)))

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

        # Shutdown.
        client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("min_chunk_size", [512, 64])
async def test_chunked_prefill(tmp_socket, min_chunk_size):
    ENGINE_ARGS = AsyncEngineArgs(
        model=MODEL,
        disable_log_requests=True,
        enable_chunked_prefill=True,
        max_num_batched_tokens=512,
        min_chunk_size=min_chunk_size,
    )
    with RemoteMQLLMEngine(engine_args=ENGINE_ARGS,
                           ipc_path=tmp_socket) as engine:

        client = await engine.make_client()

        large_request = "hello" * 1000
        small_request = "hello"

        async def generate(prompt, req_id):
            async for out in client.generate(
                    request_id=req_id,
                    prompt=prompt,
                    sampling_params=SamplingParams(
                        max_tokens=1,
                        output_kind=RequestOutputKind.FINAL_ONLY),
            ):
                return out

        large_task = asyncio.create_task(generate(large_request, "one"))

        small_task = asyncio.create_task(generate(small_request, "two"))

        done, _ = await asyncio.wait((large_task, small_task),
                                     return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            if min_chunk_size == 512:
                assert large_task in done
                assert len(done) == 2
            else:
                assert small_task in done
                assert len(done) == 1
            assert task.exception() is None
        # Shutdown.
        client.close()
