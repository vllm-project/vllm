import asyncio
import os

import pytest

from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.llm_engine import LLMEngine
from vllm.executor.gpu_executor import GPUExecutor, GPUExecutorAsync
from vllm.sampling_params import SamplingParams


class Mock:
    ...


class CustomGPUExecutor(GPUExecutor):

    def execute_model(self, *args, **kwargs):
        # Drop marker to show that this was ran
        with open(".marker", "w"):
            ...
        return super().execute_model(*args, **kwargs)


class CustomGPUExecutorAsync(GPUExecutorAsync):

    async def execute_model_async(self, *args, **kwargs):
        with open(".marker", "w"):
            ...
        return await super().execute_model_async(*args, **kwargs)


@pytest.mark.parametrize("model", ["facebook/opt-125m"])
def test_custom_executor_type_checking(model):
    with pytest.raises(ValueError):
        engine_args = EngineArgs(model=model,
                                 distributed_executor_backend=Mock)
        LLMEngine.from_engine_args(engine_args)
    with pytest.raises(ValueError):
        engine_args = AsyncEngineArgs(model=model,
                                      distributed_executor_backend=Mock)
        AsyncLLMEngine.from_engine_args(engine_args)
    with pytest.raises(TypeError):
        engine_args = AsyncEngineArgs(
            model=model, distributed_executor_backend=CustomGPUExecutor)
        AsyncLLMEngine.from_engine_args(engine_args)


@pytest.mark.parametrize("model", ["facebook/opt-125m"])
def test_custom_executor(model, tmpdir):
    cwd = os.path.abspath(".")
    os.chdir(tmpdir)
    try:
        assert not os.path.exists(".marker")

        engine_args = EngineArgs(
            model=model, distributed_executor_backend=CustomGPUExecutor)
        engine = LLMEngine.from_engine_args(engine_args)
        sampling_params = SamplingParams(max_tokens=1)

        engine.add_request("0", "foo", sampling_params)
        engine.step()

        assert os.path.exists(".marker")
    finally:
        os.chdir(cwd)


@pytest.mark.parametrize("model", ["facebook/opt-125m"])
def test_custom_executor_async(model, tmpdir):
    cwd = os.path.abspath(".")
    os.chdir(tmpdir)
    try:
        assert not os.path.exists(".marker")

        engine_args = AsyncEngineArgs(
            model=model, distributed_executor_backend=CustomGPUExecutorAsync)
        engine = AsyncLLMEngine.from_engine_args(engine_args)
        sampling_params = SamplingParams(max_tokens=1)

        async def t():
            stream = await engine.add_request("0", "foo", sampling_params)
            async for x in stream:
                ...

        asyncio.run(t())

        assert os.path.exists(".marker")
    finally:
        os.chdir(cwd)
