# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shutdown sequencing of the gRPC launcher (``_drain_and_stop``)."""

import asyncio
import threading
import time

import pytest

grpc = pytest.importorskip("grpc")
pytest.importorskip("smg_grpc_servicer")

from vllm.entrypoints.launchers import grpc_server  # noqa: E402


class FakeAsyncLLM:
    """``shutdown(timeout)`` blocks like EngineCore's drain mode: it returns
    once no request is in flight, or counts as killed when the timeout passes
    first. Like EngineCore, it rejects a request it reads after shutdown has
    started, even one that was added before."""

    def __init__(self):
        self.lock = threading.Lock()
        self.inflight = 0
        self.rejected = 0
        self.started = None
        self.timeout = "not called"
        self.outcome = None

    def get_num_unfinished_requests(self):
        with self.lock:
            return self.inflight

    def shutdown(self, timeout=None):
        self.started = time.monotonic()
        self.timeout = timeout
        deadline = self.started + (5.0 if timeout is None else timeout)
        while time.monotonic() < deadline:
            with self.lock:
                if self.inflight == 0:
                    self.outcome = "drained"
                    return
            time.sleep(0.01)
        self.outcome = "killed"


def make_server(llm):
    async def generate(request, context):
        tokens, _, queued = request.decode().partition(":")
        with llm.lock:
            llm.inflight += 1
        try:
            # The request can wait in EngineCore's input queue, e.g. behind a
            # long prefill step, before EngineCore reads it.
            await asyncio.sleep(float(queued or 0))
            with llm.lock:
                if llm.started is not None:
                    llm.rejected += 1
                    return
            for _ in range(int(tokens)):
                await asyncio.sleep(0.01)
                yield b"t"
        finally:  # completion, or the abort a cancelled RPC sends
            with llm.lock:
                llm.inflight -= 1

    async def hold(request, context):
        # Like SubscribeKvEvents: open until the client or the server ends it.
        yield b"e"
        await asyncio.Event().wait()

    server = grpc.aio.server()
    handlers = {
        "Generate": grpc.unary_stream_rpc_method_handler(generate),
        "Hold": grpc.unary_stream_rpc_method_handler(hold),
    }
    server.add_generic_rpc_handlers(
        (grpc.method_handlers_generic_handler("test.Svc", handlers),)
    )
    port = server.add_insecure_port("127.0.0.1:0")
    return server, port


async def call(port, method, arg=b""):
    """Run one streaming RPC; return the number of messages and the status."""
    async with grpc.aio.insecure_channel(f"127.0.0.1:{port}") as channel:
        n = 0
        try:
            async for _ in channel.unary_stream(f"/test.Svc/{method}")(arg):
                n += 1
            return n, grpc.StatusCode.OK
        except grpc.aio.AioRpcError as e:
            return n, e.code()


async def shut_down(rpcs, shutdown_timeout):
    """Start the RPCs, then time ``_drain_and_stop`` as SIGTERM would run it."""
    llm = FakeAsyncLLM()
    server, port = make_server(llm)
    await server.start()
    clients = [asyncio.create_task(call(port, *rpc)) for rpc in rpcs]
    await asyncio.sleep(0.3)
    t0 = time.monotonic()
    await grpc_server._drain_and_stop(server, llm, shutdown_timeout)
    elapsed = time.monotonic() - t0
    return llm, t0, elapsed, await asyncio.gather(*clients)


@pytest.mark.asyncio
async def test_engine_keeps_serving_until_requests_finish():
    llm, t0, elapsed, results = await shut_down([("Generate", b"100")], 10)
    # EngineCore is told to stop only once the request is done, and gets the
    # rest of the budget plus the teardown floor.
    assert llm.started - t0 > 0.5
    remaining = 10 - (llm.started - t0)
    assert llm.timeout == pytest.approx(
        remaining + grpc_server._ENGINE_TEARDOWN_S, abs=0.1
    )
    assert llm.outcome == "drained"
    assert results == [(100, grpc.StatusCode.OK)]
    assert elapsed < 3


@pytest.mark.asyncio
async def test_requests_queued_at_sigterm_are_served():
    # Added before SIGTERM, read by EngineCore 0.2 s after it.
    rpcs = [("Generate", b"50:0.5") for _ in range(8)]
    llm, _, _, results = await shut_down(rpcs, 10)
    assert llm.rejected == 0
    assert results == [(50, grpc.StatusCode.OK)] * 8


@pytest.mark.asyncio
async def test_open_rpc_without_request_does_not_hold_shutdown(monkeypatch):
    monkeypatch.setattr(grpc_server, "_RPC_FLUSH_S", 0.5)
    llm, t0, elapsed, results = await shut_down([("Hold",)], 30)
    # EngineCore stops right away, and the open RPC is cancelled after the
    # flush instead of holding shutdown for the whole grace period.
    assert llm.started - t0 < 0.5
    assert llm.outcome == "drained"
    assert 0.5 <= elapsed < 2
    assert results[0][1] != grpc.StatusCode.OK


@pytest.mark.asyncio
async def test_rpc_over_budget_aborts_before_engine_deadline():
    llm, _, elapsed, results = await shut_down([("Generate", b"1000")], 1)
    # The RPC is cancelled at the deadline and aborts its request; EngineCore
    # still gets the teardown floor.
    n, code = results[0]
    assert n < 1000 and code != grpc.StatusCode.OK
    assert llm.timeout == pytest.approx(grpc_server._ENGINE_TEARDOWN_S, abs=0.2)
    assert llm.outcome == "drained"
    assert 1 <= elapsed < 1 + grpc_server._ENGINE_TEARDOWN_S


@pytest.mark.asyncio
async def test_zero_timeout_keeps_abort_behavior():
    llm, _, _, _ = await shut_down([], 0)
    # RPCs get the fixed grace and EngineCore its default join, as before.
    assert llm.timeout is None
    assert llm.outcome == "drained"
