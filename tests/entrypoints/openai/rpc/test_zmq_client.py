import asyncio
import tempfile
import unittest
import unittest.mock
import uuid

import pytest
import pytest_asyncio

from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.rpc.client import (AsyncEngineRPCClient,
                                                ClientClosedError)
from vllm.entrypoints.openai.rpc.server import AsyncEngineRPCServer


@pytest.fixture(scope="function")
def tmp_socket():
    with tempfile.TemporaryDirectory() as td:
        yield f"ipc://{td}/{uuid.uuid4()}"


@pytest_asyncio.fixture(scope="function")
async def dummy_server(tmp_socket, monkeypatch):
    dummy_engine = unittest.mock.AsyncMock()

    def dummy_engine_builder(*args):
        return dummy_engine

    with monkeypatch.context() as m:
        m.setattr(AsyncLLMEngine, "from_engine_args", dummy_engine_builder)
        server = AsyncEngineRPCServer(None, None, rpc_path=tmp_socket)

    loop = asyncio.get_running_loop()
    server_task = loop.create_task(server.run_server_loop())

    try:
        yield server
    finally:
        server_task.cancel()
        server.cleanup()


@pytest_asyncio.fixture(scope="function")
async def client(tmp_socket):
    client = AsyncEngineRPCClient(rpc_path=tmp_socket)
    # Sanity check: the server is connected
    await client.wait_for_server()

    try:
        yield client
    finally:
        client.close()


@pytest.mark.asyncio
async def test_client_data_methods_use_timeouts(monkeypatch, dummy_server,
                                                client: AsyncEngineRPCClient):
    with monkeypatch.context() as m:
        # Make the server _not_ reply with a model config
        m.setattr(dummy_server, "get_model_config", lambda x: None)
        m.setattr(client, "_data_timeout", 10)

        # And ensure the task completes anyway
        client_task = asyncio.get_running_loop().create_task(client.setup())
        done, pending = await asyncio.wait([client_task], timeout=0.05)
        assert len(done) > 0
        with pytest.raises(TimeoutError):
            list(done)[0].result()


# TODO: needs changes from https://github.com/vllm-project/vllm/pull/7394
# to work
@pytest.mark.asyncio
@pytest.mark.skip
async def test_client_data_methods_reraise_exceptions(
        monkeypatch, dummy_server, client: AsyncEngineRPCClient):
    with monkeypatch.context() as m:
        # Make the server raise some random exception
        exception = RuntimeError("Client test exception")

        def raiser():
            raise exception

        m.setattr(dummy_server.engine, "get_model_config", raiser)
        m.setattr(client, "_data_timeout", 10)

        client_task = asyncio.get_running_loop().create_task(client.setup())
        done, pending = await asyncio.wait([client_task], timeout=0.05)
        assert len(done) > 0

        # And ensure the task completes anyway
        with pytest.raises(RuntimeError, match=str(exception)):
            for t in done:
                t.result()


@pytest.mark.asyncio
async def test_client_health_check_times_out(monkeypatch, dummy_server,
                                             client: AsyncEngineRPCClient):
    with monkeypatch.context() as m:
        # Make the server _not_ reply with a health check
        m.setattr(dummy_server, "check_health", lambda x: None)
        m.setattr(client, "_data_timeout", 10)

        # And ensure the health check times out
        client_task = asyncio.get_running_loop().create_task(
            client.check_health())
        done, pending = await asyncio.wait([client_task], timeout=0.05)
        assert len(done) > 0
        with pytest.raises(TimeoutError):
            list(done)[0].result()


@pytest.mark.asyncio
async def test_client_errors_after_closing(monkeypatch, dummy_server,
                                           client: AsyncEngineRPCClient):

    client.close()

    # Healthchecks and generate requests will fail with explicit errors
    with pytest.raises(ClientClosedError):
        await client.check_health()
    with pytest.raises(ClientClosedError):
        async for _ in client.generate(None, None, None):
            pass

    # But no-ops like aborting will pass
    await client.abort("test-request-id")
    await client.do_log_stats()
