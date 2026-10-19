# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Management-call failure tests without model loading or tokenizer downloads."""

import asyncio
import uuid
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm.v1.engine.core_client import AsyncMPClient, MPClient, SyncMPClient


@pytest.mark.parametrize("explicit_shutdown", [False, True])
def test_utility_waiters_fail_when_sync_receiver_stops(monkeypatch, explicit_shutdown):
    import zmq

    from vllm.v1.engine.exceptions import EngineDeadError

    ctx = zmq.Context()
    peer = ctx.socket(zmq.PAIR)
    address = f"inproc://utility-failure-{uuid.uuid4()}"
    peer.bind(address)
    receiver = ctx.socket(zmq.PAIR)
    receiver.connect(address)
    futures = [Future(), Future(), Future()]
    futures[-1].cancel()

    def fail_validation(frames):
        raise EngineDeadError()

    resources = SimpleNamespace(
        output_socket=receiver, engine_dead=False, validate_alive=fail_validation
    )

    def initialize(client, **kwargs):
        client.ctx = ctx
        client.resources = resources
        client.decoder = MagicMock()
        client.utility_results = dict(enumerate(futures))
        client.vllm_config = SimpleNamespace(
            parallel_config=SimpleNamespace(data_parallel_size=1)
        )

    monkeypatch.setattr(MPClient, "__init__", initialize)
    client = SyncMPClient(MagicMock(), MagicMock(), False)
    shutdown = None
    try:
        if explicit_shutdown:
            shutdown = ctx.socket(zmq.PAIR)
            shutdown.connect(resources.shutdown_path)
            shutdown.send(b"")
        else:
            peer.send(b"engine failed")
        for future in futures[:-1]:
            with pytest.raises(EngineDeadError):
                future.result(timeout=2)
        assert resources.engine_dead
        assert not client.utility_results
        assert futures[-1].cancelled()
    finally:
        client.output_queue_thread.join(timeout=2)
        if shutdown is not None:
            shutdown.close(linger=0)
        peer.close(linger=0)
        ctx.term()


@pytest.mark.asyncio
async def test_utility_waiters_fail_when_async_receiver_stops():
    from unittest.mock import AsyncMock

    from vllm.v1.engine.exceptions import EngineDeadError

    client = AsyncMPClient.__new__(AsyncMPClient)
    client.resources = SimpleNamespace(
        output_queue_task=None,
        output_socket=SimpleNamespace(
            recv_multipart=AsyncMock(side_effect=EngineDeadError())
        ),
        engine_dead=False,
    )
    client.decoder = MagicMock()
    client.outputs_queue = asyncio.Queue()
    futures = [asyncio.get_running_loop().create_future() for _ in range(3)]
    futures[-1].cancel()
    client.utility_results = dict(enumerate(futures))
    client._ensure_output_queue_task()
    await client.resources.output_queue_task
    for future in futures[:-1]:
        with pytest.raises(EngineDeadError):
            await asyncio.wait_for(future, timeout=2)
    assert client.resources.engine_dead
    assert not client.utility_results
    assert futures[-1].cancelled()


@pytest.mark.parametrize("error_type", [RuntimeError, ValueError])
def test_utility_send_failure_removes_waiter(error_type):
    client = SyncMPClient.__new__(SyncMPClient)
    client.utility_results = {}
    client._send_input = MagicMock(side_effect=error_type("send failed"))
    with pytest.raises(error_type, match="send failed"):
        client.call_utility("sleep")
    assert not client.utility_results


@pytest.mark.asyncio
async def test_utility_cancel_during_send_and_late_reply():
    from vllm.v1.engine.core_client import _process_utility_output

    client = AsyncMPClient.__new__(AsyncMPClient)
    client.utility_results = {}
    client.client_index = 0
    client.encoder = MagicMock()
    client.encoder.encode.return_value = ()
    entered = asyncio.Event()
    never = asyncio.Event()

    async def send(*args):
        entered.set()
        await never.wait()

    client._send_input_message = send
    task = asyncio.create_task(client._call_utility_async("sleep", engine=b"0"))
    await entered.wait()
    call_id = next(iter(client.utility_results))
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not client.utility_results
    _process_utility_output(
        SimpleNamespace(
            call_id=call_id,
            failure_message=None,
            result=SimpleNamespace(result=None),
        ),
        client.utility_results,
    )


@pytest.mark.asyncio
async def test_utility_send_failure_after_receiver_stops():
    import gc

    from vllm.v1.engine.core_client import _fail_utility_calls

    client = AsyncMPClient.__new__(AsyncMPClient)
    client.utility_results = {}
    client.client_index = 0
    client.encoder = MagicMock()
    client.encoder.encode.return_value = ()
    loop = asyncio.get_running_loop()
    old_handler = loop.get_exception_handler()
    unhandled = []
    loop.set_exception_handler(lambda loop, context: unhandled.append(context))

    async def send(*args):
        _fail_utility_calls(client.utility_results)
        raise RuntimeError("send interrupted by receiver failure")

    client._send_input_message = send
    try:
        with pytest.raises(RuntimeError, match="send interrupted"):
            await client._call_utility_async("sleep", engine=b"0")
        gc.collect()
        await asyncio.sleep(0)
        assert not unhandled
        assert not client.utility_results
    finally:
        loop.set_exception_handler(old_handler)
