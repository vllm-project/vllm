import asyncio
from dataclasses import dataclass

import pytest
import torch

from vllm import SamplingParams
from vllm.engine.async_llm_engine import AsyncEngineArgs, AsyncLLMEngine

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

    async def step_async(self):
        self.step_calls += 1
        return [RequestOutput(
            request_id=self.request_id)] if self.request_id else []

    async def process_model_inputs_async(self, *args, **kwargs):
        pass

    def generate(self, request_id):
        self.request_id = request_id

    def stop_generating(self):
        self.request_id = None

    def add_request(self, **kwargs):
        del kwargs  # Unused
        self.add_request_calls += 1

    async def add_request_async(self, **kwargs):
        self.add_request_calls += 1
        return

    def abort_request(self, request_id):
        del request_id  # Unused
        self.abort_request_calls += 1

    def has_unfinished_requests(self):
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

    engine = MockAsyncLLMEngine(worker_use_ray=True, engine_use_ray=True)
    assert engine.get_model_config() is not None
    assert engine.get_tokenizer() is not None
    assert engine.get_decoding_config() is not None


def test_asyncio_run():
    wait_for_gpu_memory_to_clear(
        devices=list(range(torch.cuda.device_count())),
        threshold_bytes=2 * 2**30,
        timeout_s=60,
    )

    engine = AsyncLLMEngine.from_engine_args(
        AsyncEngineArgs(model="facebook/opt-125m"))

    async def run(prompt: str):
        sampling_params = SamplingParams(
            temperature=0,
            max_tokens=32,
        )

        async for output in engine.generate(prompt,
                                            sampling_params,
                                            request_id=prompt):
            final_output = output
        return final_output

    async def generate():
        return await asyncio.gather(
            run("test0"),
            run("test1"),
        )

    results = asyncio.run(generate())
    assert len(results) == 2
