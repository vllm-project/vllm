# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
import uuid
from asyncio import CancelledError
from copy import copy
from dataclasses import dataclass
from typing import Optional

import pytest
import pytest_asyncio
import torch

from vllm import SamplingParams
from vllm.config import ParallelConfig
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.engine.async_llm_engine import AsyncEngineArgs, AsyncLLMEngine
from vllm.outputs import RequestOutput as RealRequestOutput
from vllm.sampling_params import RequestOutputKind

from ..utils import wait_for_gpu_memory_to_clear


@dataclass
class RequestOutput:
    request_id: int
    finished: bool = False


@dataclass
class MockModelConfig:
    use_async_output_proc = True


class MockEngine:

    def __init__(self):
        self.step_calls = 0
        self.add_request_calls = 0
        self.abort_request_calls = 0
        self.request_id = None
        # Ugly, remove dependency when possible
        self.parallel_config = ParallelConfig(1, 1, False)
        self.model_config = MockModelConfig()

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
    _engine_class = MockEngine


@pytest.mark.asyncio
async def test_new_requests_event():
    params = SamplingParams()

    engine = MockAsyncLLMEngine()
    engine.start_background_loop()
    await asyncio.sleep(0.01)
    assert engine.engine.step_calls == 0

    await engine.add_request("1", "", params)
    await asyncio.sleep(0.01)
    assert engine.engine.add_request_calls == 1
    assert engine.engine.step_calls == 1

    await engine.add_request("2", "", params)
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

    await engine.add_request("3", "", params)
    await asyncio.sleep(0.01)
    assert engine.engine.add_request_calls == 3
    assert engine.engine.step_calls == old_step_calls + 1
    await asyncio.sleep(0.01)
    assert engine.engine.add_request_calls == 3
    assert engine.engine.step_calls == old_step_calls + 1

    engine = MockAsyncLLMEngine()
    assert engine.get_model_config() is not None
    assert engine.get_tokenizer() is not None
    assert engine.get_decoding_config() is not None


def start_engine():
    wait_for_gpu_memory_to_clear(
        devices=list(range(torch.cuda.device_count())),
        threshold_bytes=2 * 2**30,
        timeout_s=60,
    )

    num_scheduler_steps = int(os.getenv("NUM_SCHEDULER_STEPS", "1"))
    print(f"Starting engine with num_scheduler_steps={num_scheduler_steps}")

    return AsyncLLMEngine.from_engine_args(
        AsyncEngineArgs(model="facebook/opt-125m",
                        enforce_eager=True,
                        num_scheduler_steps=num_scheduler_steps))


def uid() -> str:
    return str(uuid.uuid4())


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
        cleanup_dist_env_and_memory()


@pytest.fixture()
def should_do_global_cleanup_after_test(request) -> bool:
    # So we can share the async engine fixture between these tests
    return False


@pytest.mark.asyncio(scope="module")
@pytest.mark.parametrize("stop", [None, ["a stop string"]])
async def test_asyncio_run(async_engine, stop):

    scheduler_config = await async_engine.get_scheduler_config()
    num_scheduler_steps = scheduler_config.num_scheduler_steps

    async def run(prompt: str):
        sampling_params = SamplingParams(
            temperature=0,
            max_tokens=32,
            min_tokens=32,
            stop=stop,
        )

        output_count = 0
        final_output = None
        async for output in async_engine.generate(prompt,
                                                  sampling_params,
                                                  request_id=uid()):
            output_count += 1
            final_output = output
        return final_output, output_count

    results = await asyncio.gather(
        run("test0"),
        run("test0"),
    )
    assert len(results) == 2
    first, second = results

    # remove nondeterministic fields for comparison
    first[0].metrics = None
    second[0].metrics = None
    first[0].request_id = None
    second[0].request_id = None

    assert str(first) == str(second)

    output_count = results[0][1]
    if num_scheduler_steps == 1:
        assert output_count == 32
    else:
        assert 1 < output_count < 32


@pytest.mark.asyncio(scope="module")
@pytest.mark.parametrize("stop", [None, ["a stop string"]])
async def test_output_kinds(async_engine, stop):
    """Test that output_kind works as expected and that
    results are equivalent across different kinds."""

    scheduler_config = await async_engine.get_scheduler_config()
    num_scheduler_steps = scheduler_config.num_scheduler_steps

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=32,
        min_tokens=32,
        stop=stop,
    )

    async def run(prompt: str, kind: RequestOutputKind):
        params = copy(sampling_params)
        params.output_kind = kind

        output_count = 0
        final_output = None
        async for output in async_engine.generate(prompt,
                                                  params,
                                                  request_id=uid()):
            output_count += 1
            final_output = output

        assert final_output is not None
        assert final_output.finished

        return (final_output.prompt_token_ids,
                final_output.outputs[0].token_ids,
                final_output.outputs[0].text, output_count)

    async def run_deltas(prompt: str):
        params = copy(sampling_params)
        params.output_kind = RequestOutputKind.DELTA

        prompt_tokens = None
        output_tokens: list[int] = []
        output_text = ""
        output_count = 0
        final_output = None
        async for output in async_engine.generate(prompt,
                                                  params,
                                                  request_id=uid()):
            token_ids = output.outputs[0].token_ids
            text = output.outputs[0].text
            final_output = output

            # Ensure we get prompt ids iff we haven't yet received output tokens
            if output_tokens:
                assert 1 <= len(token_ids) <= num_scheduler_steps
                assert stop or text
                assert not output.prompt_token_ids
            else:
                assert output.prompt_token_ids
                prompt_tokens = output.prompt_token_ids

            output_tokens.extend(token_ids)
            output_text += text

            output_count += 1

        assert final_output is not None
        assert final_output.finished

        return prompt_tokens, output_tokens, output_text, output_count

    results = await asyncio.gather(
        run("common input prompt", RequestOutputKind.CUMULATIVE),
        run("common input prompt", RequestOutputKind.FINAL_ONLY),
        run_deltas("common input prompt"))

    # Make sure outputs are the same
    prompt_set = set(tuple(prompt_ids) for prompt_ids, _, _, _ in results)
    assert len(prompt_set) == 1

    text_set = set(text for _, _, text, _ in results)
    assert len(text_set) == 1

    tokens_set = set(tuple(ids) for _, ids, _, _ in results)
    assert len(tokens_set) == 1

    cumulative, final, deltas = results

    # output message counts
    assert cumulative[3] == deltas[3]

    if num_scheduler_steps == 1:
        assert cumulative[3] == 32
    else:
        assert 1 < cumulative[3] < 32

    assert final[3] == 1


@pytest.mark.asyncio(scope="module")
@pytest.mark.parametrize("stop", [None, ["a stop string"]])
async def test_cancellation(async_engine, stop):
    scheduler_config = await async_engine.get_scheduler_config()
    num_scheduler_steps = scheduler_config.num_scheduler_steps

    sampling_params = SamplingParams(
        temperature=0,
        min_tokens=13,
        max_tokens=13,
        stop=stop,
    )

    stop_at = 5 if num_scheduler_steps == 1 else 1

    request_id = uid()

    i = 0
    with pytest.raises(CancelledError):
        async for output in async_engine.generate("test2",
                                                  sampling_params,
                                                  request_id=request_id):
            assert not output.finished
            i += 1
            if i == stop_at:
                await async_engine.abort(request_id)

    assert i == stop_at


@pytest.mark.asyncio(scope="module")
@pytest.mark.parametrize("stop", [None, ["a stop string"]])
async def test_delayed_generator(async_engine, stop):
    scheduler_config = await async_engine.get_scheduler_config()

    if scheduler_config.num_scheduler_steps != 1:
        pytest.skip("no need to test this one with multistep")

    sampling_params = SamplingParams(
        temperature=0,
        min_tokens=10,
        max_tokens=10,
        stop=stop,
    )

    stream = async_engine.generate("test3", sampling_params, request_id=uid())
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
