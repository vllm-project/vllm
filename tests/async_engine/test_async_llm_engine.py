import asyncio
from dataclasses import dataclass

import pytest

from vllm.engine.async_llm_engine import AsyncLLMEngine


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

    async def encode_request_async(self, *args, **kwargs):
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
