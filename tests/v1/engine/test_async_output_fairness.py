# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from contextlib import suppress
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm.v1.engine import EngineCoreOutputs
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.core_client import AsyncMPClient

pytestmark = pytest.mark.cpu_test


class _ReadyOutputSocket:
    def __init__(self, ready_outputs: int):
        self.ready_outputs = ready_outputs
        self.recv_count = 0
        self._blocked = asyncio.Event()

    async def recv_multipart(self, copy: bool):
        assert not copy
        self.recv_count += 1
        if self.recv_count > self.ready_outputs:
            await self._blocked.wait()
        return [b"output"]


class _ReadyEngineCore:
    def __init__(self, ready_outputs: int):
        self.ready_outputs = ready_outputs
        self.recv_count = 0
        self._blocked = asyncio.Event()

    async def get_output_async(self):
        self.recv_count += 1
        if self.recv_count > self.ready_outputs:
            await self._blocked.wait()
        return EngineCoreOutputs(outputs=[SimpleNamespace(mm_cache_miss_hashes=None)])

    def shutdown(self, timeout: float | None = None) -> None:
        pass


async def _cancel(task: asyncio.Task):
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task


async def _check_async_mp_client_yields_while_outputs_are_ready():
    socket = _ReadyOutputSocket(ready_outputs=10)
    client = object.__new__(AsyncMPClient)
    client.resources = SimpleNamespace(
        output_queue_task=None,
        output_socket=socket,
        validate_alive=lambda frames: None,
    )
    client.decoder = SimpleNamespace(
        decode=lambda frames: EngineCoreOutputs(scheduler_stats=MagicMock())
    )
    client.utility_results = {}
    client.outputs_queue = asyncio.Queue()

    client._ensure_output_queue_task()
    task = client.resources.output_queue_task
    assert task is not None
    recv_count_at_control: list[int] = []
    asyncio.get_running_loop().call_soon(
        lambda: recv_count_at_control.append(socket.recv_count)
    )

    try:
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert recv_count_at_control == [1]
    finally:
        await _cancel(task)


def test_async_mp_client_yields_while_outputs_are_ready():
    asyncio.run(_check_async_mp_client_yields_while_outputs_are_ready())


async def _check_async_llm_yields_between_ready_output_batches():
    engine_core = _ReadyEngineCore(ready_outputs=10)
    output_processor = SimpleNamespace(
        process_outputs=lambda *args: SimpleNamespace(
            request_outputs=[], reqs_to_abort=[]
        ),
        update_scheduler_stats=lambda stats: None,
        propagate_error=MagicMock(),
    )
    llm = object.__new__(AsyncLLM)
    llm.output_handler = None
    llm.engine_core = engine_core
    llm.output_processor = output_processor
    llm.log_stats = False
    llm.logger_manager = None
    llm.renderer = SimpleNamespace(mm_processor_cache=None, shutdown=lambda: None)

    llm._run_output_handler()
    task = llm.output_handler
    assert task is not None
    recv_count_at_control: list[int] = []
    asyncio.get_running_loop().call_soon(
        lambda: recv_count_at_control.append(engine_core.recv_count)
    )

    try:
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert recv_count_at_control == [1]
    finally:
        await _cancel(task)
        llm.output_handler = None


def test_async_llm_yields_between_ready_output_batches():
    asyncio.run(_check_async_llm_yields_between_ready_output_batches())
