# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Exercise utility RPC lifecycles without starting a model or EngineCore."""

import asyncio
import weakref
from types import SimpleNamespace
from unittest.mock import AsyncMock

import msgspec
import pytest
import pytest_asyncio
import zmq
import zmq.asyncio

from vllm.v1.engine import (
    EEP_NOTIFICATION_CALL_ID,
    EngineCoreOutputs,
    UtilityOutput,
    UtilityResult,
)
from vllm.v1.engine.core_client import (
    AsyncMPClient,
    BackgroundResources,
    DPLBAsyncMPClient,
    MPClient,
    SyncMPClient,
)
from vllm.v1.engine.exceptions import EngineDeadError
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder


@pytest_asyncio.fixture
async def utility_client(monkeypatch):
    clients = []

    def init(self, asyncio_mode, vllm_config, *args, **kwargs):
        ctx = zmq.Context()
        self.ctx = zmq.asyncio.Context(ctx) if asyncio_mode else ctx
        self.vllm_config = vllm_config
        self.resources = BackgroundResources(ctx=ctx)
        self._finalizer = weakref.finalize(self, self.resources)
        self.input_socket = self.resources.input_socket = self.ctx.socket(zmq.ROUTER)
        self.resources.output_socket = self.ctx.socket(zmq.PULL)
        self.input_socket.bind("inproc://utility-input")
        self.resources.output_socket.bind("inproc://utility-output")
        self.encoder = MsgpackEncoder()
        self.decoder = MsgpackDecoder(EngineCoreOutputs)
        self.core_engine = b"\x00\x00"
        self.utility_results = {}

        peer_input, peer_output = ctx.socket(zmq.DEALER), ctx.socket(zmq.PUSH)
        peer_input.setsockopt(zmq.IDENTITY, self.core_engine)
        peer_input.setsockopt(zmq.RCVTIMEO, 5000)
        peer_input.connect("inproc://utility-input")
        peer_output.connect("inproc://utility-output")
        peer_input.send(b"ready")
        assert zmq.Socket.shadow(self.input_socket).recv_multipart() == [
            self.core_engine,
            b"ready",
        ]
        clients.append((self, peer_input, peer_output))

    monkeypatch.setattr(MPClient, "__init__", init)

    def create(asyncio_mode):
        config = SimpleNamespace(
            parallel_config=SimpleNamespace(
                data_parallel_size=1, enable_fault_tolerance=False
            )
        )
        client_cls = AsyncMPClient if asyncio_mode else SyncMPClient
        client_cls(config, None, False)
        return clients[-1]

    yield create

    for client, peer_input, peer_output in clients:
        # Release waiters even when testing the broken implementation.
        for future in list(client.utility_results.values()):
            if not future.done():
                future.cancel()
        if isinstance(client, SyncMPClient):
            if not client.output_queue_thread.is_alive():
                client.resources.shutdown_path = None
            client.shutdown()
            await asyncio.to_thread(client.output_queue_thread.join, 5)
            assert not client.output_queue_thread.is_alive()
        else:
            client.shutdown()
            await asyncio.gather(
                client.resources.output_queue_task, return_exceptions=True
            )
        peer_input.close(linger=0)
        peer_output.close(linger=0)
        client.resources.ctx.term()


async def _receive_call(peer_input):
    frames = await asyncio.to_thread(peer_input.recv_multipart)
    return msgspec.msgpack.decode(frames[1])[1]


def _reply(peer_output, call_id):
    output = EngineCoreOutputs(
        utility_output=UtilityOutput(call_id=call_id, result=UtilityResult("ok"))
    )
    peer_output.send_multipart(MsgpackEncoder().encode(output))


@pytest.mark.asyncio
@pytest.mark.parametrize("asyncio_mode", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("stop", ["shutdown", "engine_dead", "decode_error"])
async def test_receiver_stop_fails_pending_utility(utility_client, asyncio_mode, stop):
    client, peer_input, peer_output = utility_client(asyncio_mode)

    async def call():
        if asyncio_mode:
            return await client.call_utility_async("echo")
        return await asyncio.to_thread(client.call_utility, "echo")

    completed = asyncio.create_task(call())
    _reply(peer_output, await _receive_call(peer_input))
    assert await asyncio.wait_for(completed, 5) == "ok"

    pending = asyncio.create_task(call())
    await _receive_call(peer_input)
    error = EngineDeadError
    if stop == "shutdown":
        client.shutdown()
    else:
        peer_output.send(b"ENGINE_CORE_DEAD" if stop == "engine_dead" else b"\xc1")
        if stop == "decode_error":
            error = msgspec.DecodeError
    try:
        with pytest.raises(error):
            await asyncio.wait_for(asyncio.shield(pending), 5)
        assert not client.utility_results
    finally:
        for future in list(client.utility_results.values()):
            if not future.done():
                future.cancel()
        await asyncio.gather(pending, return_exceptions=True)


@pytest.mark.asyncio
async def test_receiver_cancelled_before_start_fails_pending_utility(utility_client):
    client, _, _ = utility_client(True)
    pending = asyncio.get_running_loop().create_future()
    client.utility_results[1] = pending
    client.resources.output_queue_task.cancel()

    with pytest.raises(EngineDeadError):
        await asyncio.wait_for(asyncio.shield(pending), 5)
    assert not client.utility_results


@pytest.mark.asyncio
async def test_cancelled_utility_late_response_keeps_receiver_alive(utility_client):
    client, peer_input, peer_output = utility_client(True)
    cancelled = asyncio.create_task(client.call_utility_async("echo"))
    call_id = await _receive_call(peer_input)
    cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled
    _reply(peer_output, call_id)

    completed = asyncio.create_task(client.call_utility_async("echo"))
    _reply(peer_output, await _receive_call(peer_input))
    assert await asyncio.wait_for(completed, 5) == "ok"
    assert not client.resources.output_queue_task.done()
    assert not client.utility_results


@pytest.mark.asyncio
async def test_send_error_after_receiver_failure_cleans_pending_utility(
    utility_client, monkeypatch
):
    client, _, peer_output = utility_client(True)
    sending, release = asyncio.Event(), asyncio.Event()

    async def send(*args):
        sending.set()
        await release.wait()
        raise RuntimeError("send failed")

    monkeypatch.setattr(client, "_send_input_message", send)
    pending = asyncio.create_task(client.call_utility_async("echo"))
    await asyncio.wait_for(sending.wait(), 5)
    peer_output.send(b"ENGINE_CORE_DEAD")
    await asyncio.wait_for(client.resources.output_queue_task, 5)
    release.set()
    with pytest.raises(RuntimeError, match="send failed"):
        await pending
    assert not client.utility_results


@pytest.mark.asyncio
async def test_cancelled_eep_commit_cancels_notification_waiter(monkeypatch):
    client = DPLBAsyncMPClient.__new__(DPLBAsyncMPClient)
    client.core_engines = [b"\x00\x00"]
    client.utility_results = {}
    monkeypatch.setattr(client, "_ensure_output_queue_task", lambda: None)
    monkeypatch.setattr(client, "pause_scheduler_async", AsyncMock())
    committing = asyncio.Event()

    async def commit(*args, **kwargs):
        committing.set()
        await asyncio.get_running_loop().create_future()

    monkeypatch.setattr(client, "_call_utility_async", commit)
    pending = asyncio.create_task(client._commit_scale_up_elastic_ep(2))
    await asyncio.wait_for(committing.wait(), 5)
    notification = client.utility_results[EEP_NOTIFICATION_CALL_ID]
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert notification.cancelled()
