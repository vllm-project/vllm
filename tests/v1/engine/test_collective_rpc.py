# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import deque
from concurrent.futures import Future
from functools import partial
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.core import EngineCore
from vllm.v1.engine.core_client import AsyncMPClient, DPLBAsyncMPClient


class OneShotFuture(Future):
    """Model the single consumption required by Ray compiled graph outputs."""

    def __init__(self, output, order, name, error=None):
        super().__init__()
        self.output = output
        self.order = order
        self.name = name
        self.error = error
        self.consumed = False

    def result(self, timeout=None):
        assert not self.consumed
        self.consumed = True
        self.order.append(self.name)
        if self.error is not None:
            raise self.error
        return self.output


@pytest.mark.parametrize("wait", [False, True])
@pytest.mark.parametrize("separate_exec_future", [False, True])
def test_collective_rpc_preserves_pending_outputs(wait, separate_exec_future):
    order: list[int | str] = []
    outputs = [object(), object()]
    futures: list[Future] = [
        OneShotFuture(output, order, i) for i, output in enumerate(outputs)
    ]
    exec_futures: list[Future] = (
        [Future(), Future()] if separate_exec_future else futures
    )
    scheduled = [object(), object()]
    queue = deque(reversed(list(zip(futures, scheduled, exec_futures))), maxlen=2)
    original = list(queue)
    executor = MagicMock()
    executor.collective_rpc.side_effect = lambda *args: order.append("rpc")
    core = SimpleNamespace(batch_queue=queue, model_executor=executor)

    EngineCore.collective_rpc(core, "release", wait_for_inflight_batches=wait)

    assert order == ([0, 1, "rpc"] if wait else ["rpc"])
    assert queue.maxlen == 2
    if not wait:
        assert list(queue) == original
        return
    for i, (future, scheduler_output, exec_future) in enumerate(reversed(queue)):
        assert scheduler_output is scheduled[i]
        assert future.result() is outputs[i]
        assert future.result() is outputs[i]
        assert exec_future is (exec_futures[i] if separate_exec_future else future)
    executor.collective_rpc.assert_called_once_with("release", None, (), None)


def test_collective_rpc_preserves_failed_output():
    error = RuntimeError("model execution failed")
    future = OneShotFuture(None, [], "query", error)
    core = SimpleNamespace(
        batch_queue=deque([(future, object(), future)]), model_executor=MagicMock()
    )

    with pytest.raises(RuntimeError, match="model execution failed"):
        EngineCore.collective_rpc(core, "release", wait_for_inflight_batches=True)

    saved, _, exec_future = core.batch_queue[0]
    assert saved is exec_future
    for _ in range(2):
        with pytest.raises(RuntimeError) as exc:
            saved.result()
        assert exc.value is error
    core.model_executor.collective_rpc.assert_not_called()


@pytest.mark.asyncio
async def test_collective_rpc_barrier_reaches_each_dp_engine():
    order: list[str] = []
    engines = {}
    for name in ("engine-0", "engine-1"):
        future = OneShotFuture(object(), order, name)
        executor = MagicMock()
        executor.collective_rpc.side_effect = lambda *args, n=name: order.append(
            f"{n}-rpc"
        )
        engines[name] = SimpleNamespace(
            batch_queue=deque([(future, object(), future)]), model_executor=executor
        )

    async def call_utility(method, *args, engine):
        assert method == "collective_rpc"
        return EngineCore.collective_rpc(engines[engine], *args)

    client = SimpleNamespace(
        core_engines=list(engines), _call_utility_async=call_utility
    )
    client.call_utility_async = partial(DPLBAsyncMPClient.call_utility_async, client)
    client.collective_rpc_async = partial(AsyncMPClient.collective_rpc_async, client)
    llm = SimpleNamespace(engine_core=client)

    await AsyncLLM.collective_rpc(llm, "release", wait_for_inflight_batches=True)

    assert order == ["engine-0", "engine-0-rpc", "engine-1", "engine-1-rpc"]
