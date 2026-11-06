# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""How the AsyncLLM output handler reports an engine core that is already gone."""

import asyncio
import time

import pytest

from vllm.config import KernelConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.outputs import RequestOutput
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.v1.engine import async_llm as async_llm_module
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.exceptions import EngineDeadError

MODEL = "hmellor/tiny-random-LlamaForCausalLM"
TIMEOUT_SEC = 30
FAILURE_LOG = "AsyncLLM output_handler failed."
STOP_LOG = "Engine core gone during shutdown; output handler exiting"


def _make_engine() -> AsyncLLM:
    return AsyncLLM.from_engine_args(
        AsyncEngineArgs(
            model=MODEL,
            enforce_eager=True,
            kernel_config=KernelConfig(enable_jit_warmup=False),
        )
    )


async def _start_long_request(engine: AsyncLLM) -> asyncio.Task:
    """Start a request, returning once it is streaming, that outlives the test."""
    stream = engine.generate(
        "Hello my name is",
        request_id="in-flight",
        sampling_params=SamplingParams(
            max_tokens=2000,
            ignore_eos=True,
            output_kind=RequestOutputKind.DELTA,
        ),
    )
    last_output = await anext(stream)

    async def drain() -> RequestOutput:
        nonlocal last_output
        async for output in stream:
            last_output = output
        return last_output

    return asyncio.create_task(drain())


@pytest.mark.asyncio
async def test_requested_stop_is_not_reported_as_a_failure(caplog_vllm, monkeypatch):
    real_cancel = async_llm_module.cancel_task_threadsafe

    def cancel_after_handler_exits(task: asyncio.Task) -> None:
        # Force the race: the handler sees the stopped core before its cancel.
        deadline = time.monotonic() + TIMEOUT_SEC
        while not task.done() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert task.done()
        real_cancel(task)

    monkeypatch.setattr(
        async_llm_module, "cancel_task_threadsafe", cancel_after_handler_exits
    )

    engine = _make_engine()
    try:
        request = await _start_long_request(engine)
        await asyncio.get_running_loop().run_in_executor(None, engine.shutdown)
    finally:
        engine.shutdown()

    assert FAILURE_LOG not in caplog_vllm.text
    assert STOP_LOG in caplog_vllm.text
    # The request must end, not hang: finished by the core's abort, or failed.
    [outcome] = await asyncio.wait_for(
        asyncio.gather(request, return_exceptions=True), TIMEOUT_SEC
    )
    assert isinstance(outcome, EngineDeadError) or outcome.finished


@pytest.mark.asyncio
async def test_death_outside_shutdown_is_still_reported(caplog_vllm):
    engine = _make_engine()
    try:
        request = await _start_long_request(engine)
        engine.engine_core.resources.engine_manager.processes[0].kill()
        with pytest.raises(EngineDeadError):
            await asyncio.wait_for(request, TIMEOUT_SEC)
        assert engine.output_handler is not None
        await asyncio.wait_for(engine.output_handler, TIMEOUT_SEC)
    finally:
        engine.shutdown()

    assert FAILURE_LOG in caplog_vllm.text
    assert STOP_LOG not in caplog_vllm.text
