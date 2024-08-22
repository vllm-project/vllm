import asyncio
import tempfile
import unittest
import unittest.mock
import uuid

import pytest
import pytest_asyncio

from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.rpc.client import (AsyncEngineRPCClient,
                                                RPCClientClosedError)
from vllm.entrypoints.openai.rpc.server import AsyncEngineRPCServer


@pytest.fixture(scope="function")
def tmp_socket():
    with tempfile.TemporaryDirectory() as td:
        yield f"ipc://{td}/{uuid.uuid4()}"


@pytest_asyncio.fixture(scope="function")
async def dummy_server(tmp_socket, monkeypatch):
    dummy_engine = unittest.mock.AsyncMock()

    def dummy_engine_builder(*args, **kwargs):
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
    await client._wait_for_server_rpc()

    try:
        yield client
    finally:
        client.close()


@pytest.mark.asyncio
async def test_client_data_methods_use_timeouts(monkeypatch, dummy_server,
                                                client: AsyncEngineRPCClient):
    with monkeypatch.context() as m:
        # Make the server _not_ reply with a model config
        m.setattr(dummy_server, "get_config", lambda x: None)
        m.setattr(client, "_data_timeout", 10)

        # And ensure the task completes anyway
        # (client.setup() invokes server.get_config())
        client_task = asyncio.get_running_loop().create_task(client.setup())
        with pytest.raises(TimeoutError, match="Server didn't reply within"):
            await asyncio.wait_for(client_task, timeout=0.05)


@pytest.mark.asyncio
async def test_client_aborts_use_timeouts(monkeypatch, dummy_server,
                                          client: AsyncEngineRPCClient):
    with monkeypatch.context() as m:
        # Hang all abort requests
        m.setattr(dummy_server, "abort", lambda x: None)
        m.setattr(client, "_data_timeout", 10)

        # The client should suppress timeouts on `abort`s
        # and return normally, assuming the server will eventually
        # abort the request.
        client_task = asyncio.get_running_loop().create_task(
            client.abort("test request id"))
        await asyncio.wait_for(client_task, timeout=0.05)


@pytest.mark.asyncio
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
        # And ensure the task completes, raising the exception
        with pytest.raises(RuntimeError, match=str(exception)):
            await asyncio.wait_for(client_task, timeout=0.05)


@pytest.mark.asyncio
async def test_client_errors_after_closing(monkeypatch, dummy_server,
                                           client: AsyncEngineRPCClient):

    client.close()

    # Healthchecks and generate requests will fail with explicit errors
    with pytest.raises(RPCClientClosedError):
        await client.check_health()
    with pytest.raises(RPCClientClosedError):
        async for _ in client.generate(None, None, None):
            pass

    # But no-ops like aborting will pass
    await client.abort("test-request-id")
    await client.do_log_stats()
