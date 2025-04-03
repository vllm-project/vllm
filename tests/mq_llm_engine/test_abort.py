# SPDX-License-Identifier: Apache-2.0
"""Test that aborting is handled properly."""

import asyncio
import tempfile
import uuid

import pytest

from tests.mq_llm_engine.utils import RemoteMQLLMEngine, generate
from vllm.engine.arg_utils import AsyncEngineArgs

MODEL = "google/gemma-1.1-2b-it"
ENGINE_ARGS = AsyncEngineArgs(model=MODEL)
RAISED_ERROR = KeyError
RAISED_VALUE = "foo"
EXPECTED_TOKENS = 250


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

        # Requests started before one to be aborted.
        tasks = []
        for request_id in request_ids_a:
            tasks.append(
                asyncio.create_task(
                    generate(client, request_id, EXPECTED_TOKENS)))

        # Aborted.
        task_aborted = asyncio.create_task(
            generate(client, request_id_to_be_aborted, EXPECTED_TOKENS))

        # Requests started after one to be aborted.
        for request_id in request_ids_b:
            tasks.append(
                asyncio.create_task(
                    generate(client, request_id, EXPECTED_TOKENS)))

        # Actually abort.
        await asyncio.sleep(0.5)
        await client.abort(request_id_to_be_aborted)

        # Confirm that we got all the EXPECTED tokens from the requests.
        for task in tasks:
            count, request_id = await task
            assert count == EXPECTED_TOKENS, (
                f"{request_id} generated only {count} tokens")

        # Cancel task (this will hang indefinitely if not).
        task_aborted.cancel()

        # Shutdown.
        client.close()
