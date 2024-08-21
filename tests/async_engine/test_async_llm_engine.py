import asyncio
import os
from asyncio import CancelledError
from dataclasses import dataclass
from typing import Optional

import pytest
import pytest_asyncio
import torch

from vllm import SamplingParams
from vllm.config import ParallelConfig
from vllm.engine.async_llm_engine import AsyncEngineArgs, AsyncLLMEngine
from vllm.outputs import RequestOutput as RealRequestOutput

from ..conftest import cleanup
from ..utils import wait_for_gpu_memory_to_clear


@dataclass
class RequestOutput:
    request_id: int
    finished: bool = False


class MockEngine:

    def __init__(self):
        self.step_calls = 0
        self.add_request_calls = 0
        self.abort_request_calls = 0
        self.request_id = None
        # Ugly, remove dependency when possible
        self.parallel_config = ParallelConfig(1, 1, False)

    async def step_async(self, virtual_engine):
        # PP size is 1, ignore virtual engine
        self.step_calls += 1
        return [RequestOutput(
            request_id=self.request_id)] if self.request_id else []

    async def process_model_inputs_async(self, *args, **kwargs):
        pass

    async def stop_remote_worker_execution_loop_async(self):
        pass

    def generate(self, request_id):
        self.request_id = request_id

    def stop_generating(self):
        self.request_id = None

    def add_request(self, **kwargs):
        del kwargs  # Unused
        self.add_request_calls += 1
        print(f'Request calls: {self.add_request_calls}')

    async def add_request_async(self, **kwargs):
        self.add_request_calls += 1
        return

    def abort_request(self, request_id):
        del request_id  # Unused
        self.abort_request_calls += 1

    def has_unfinished_requests(self):
        return self.request_id is not None

    def has_unfinished_requests_for_virtual_engine(self, virtual_engine):
        return self.request_id is not None


class MockAsyncLLMEngine(AsyncLLMEngine):

    def _init_engine(self, *args, **kwargs):
        return MockEngine()


@pytest.mark.asyncio
async def test_new_requests_event():
    engine = MockAsyncLLMEngine(worker_use_ray=False, engine_use_ray=False)
    engine.start_background_loop()
    await asyncio.sleep(0.01)
    assert engine.engine.step_calls == 0

    await engine.add_request("1", "", None)
    await asyncio.sleep(0.01)
    assert engine.engine.add_request_calls == 1
    assert engine.engine.step_calls == 1

    await engine.add_request("2", "", None)
    engine.engine.generate("2")
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert engine.engine.add_request_calls == 2
    assert engine.engine.step_calls >= 2
    await asyncio.sleep(0.001)
    assert engine.engine.step_calls >= 3
    engine.engine.stop_generating()
    await asyncio.sleep(0.001)
    old_step_calls = engine.engine.step_calls
    await asyncio.sleep(0.001)
    assert engine.engine.step_calls == old_step_calls

    await engine.add_request("3", "", None)
    await asyncio.sleep(0.01)
    assert engine.engine.add_request_calls == 3
    assert engine.engine.step_calls == old_step_calls + 1
    await asyncio.sleep(0.01)
    assert engine.engine.add_request_calls == 3
    assert engine.engine.step_calls == old_step_calls + 1

    # Allow deprecated engine_use_ray to not raise exception
    os.environ["VLLM_ALLOW_ENGINE_USE_RAY"] = "1"

    engine = MockAsyncLLMEngine(worker_use_ray=True, engine_use_ray=True)
    assert engine.get_model_config() is not None
    assert engine.get_tokenizer() is not None
    assert engine.get_decoding_config() is not None

    os.environ.pop("VLLM_ALLOW_ENGINE_USE_RAY")


def start_engine():
    wait_for_gpu_memory_to_clear(
        devices=list(range(torch.cuda.device_count())),
        threshold_bytes=2 * 2**30,
        timeout_s=60,
    )

    return AsyncLLMEngine.from_engine_args(
        AsyncEngineArgs(model="facebook/opt-125m", enforce_eager=True))


@pytest_asyncio.fixture(scope="module")
async def async_engine():
    engine = await asyncio.get_event_loop().run_in_executor(executor=None,
                                                            func=start_engine)
    try:
        yield engine
    finally:
        engine.shutdown_background_loop()
        del engine
        await asyncio.sleep(0.1)
        cleanup()


@pytest.fixture()
def should_do_global_cleanup_after_test(request) -> bool:
    # So we can share the async engine fixture between these tests
    return False


@pytest.mark.asyncio(scope="module")
async def test_asyncio_run(async_engine):

    async def run(prompt: str):
        sampling_params = SamplingParams(
            temperature=0,
            max_tokens=32,
        )

        async for output in async_engine.generate(prompt,
                                                  sampling_params,
                                                  request_id=prompt):
            final_output = output
        return final_output

    results = await asyncio.gather(
        run("test0"),
        run("test1"),
    )
    assert len(results) == 2


@pytest.mark.asyncio(scope="module")
async def test_cancellation(async_engine):
    sampling_params = SamplingParams(
        temperature=0,
        min_tokens=10,
        max_tokens=10,
    )

    i = 0
    with pytest.raises(CancelledError):
        async for output in async_engine.generate("test2",
                                                  sampling_params,
                                                  request_id="test2"):
            assert not output.finished
            i += 1
            if i == 5:
                await async_engine.abort("test2")

    assert i == 5


@pytest.mark.asyncio(scope="module")
async def test_delayed_generator(async_engine):
    sampling_params = SamplingParams(
        temperature=0,
        min_tokens=10,
        max_tokens=10,
    )

    stream = async_engine.generate("test3",
                                   sampling_params,
                                   request_id="test3")
    i = 0
    final_output: Optional[RealRequestOutput] = None
    async for output in stream:
        final_output = output
        if i == 0:
            # wait for generation to complete before consuming
            # the remaining messages
            await asyncio.sleep(1)
        if i < 9:
            assert not output.finished
        i += 1

    assert i == 10
    assert final_output is not None
    assert len(final_output.outputs[0].token_ids) == 10
    assert final_output.finished
