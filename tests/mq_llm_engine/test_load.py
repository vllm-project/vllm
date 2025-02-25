# SPDX-License-Identifier: Apache-2.0
"""Test that the MQLLMEngine is able to handle 10k concurrent requests."""

import asyncio
import tempfile
import uuid

import pytest

from tests.mq_llm_engine.utils import RemoteMQLLMEngine, generate
from vllm.engine.arg_utils import AsyncEngineArgs

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
