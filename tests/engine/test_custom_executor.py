import asyncio
import os

import pytest

from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.llm_engine import LLMEngine
from vllm.sampling_params import SamplingParams


@pytest.mark.parametrize("model", ["facebook/opt-125m"])
def test_custom_executor(model, tmpdir):
    cwd = os.path.abspath(".")
    os.chdir(tmpdir)
    old_env = os.environ.get("VLLM_PLUGINS", None)
    try:
        os.environ["VLLM_PLUGINS"] = "switch_executor"
        assert not os.path.exists(".marker")

        engine_args = EngineArgs(model=model,
                                 enforce_eager=True,
                                 gpu_memory_utilization=0.3)
        engine = LLMEngine.from_engine_args(engine_args)
        sampling_params = SamplingParams(max_tokens=1)

        engine.add_request("0", "foo", sampling_params)
        engine.step()

        assert os.path.exists(".marker")
    finally:
        os.chdir(cwd)
        if old_env is not None:
            os.environ["VLLM_PLUGINS"] = old_env
        else:
            del os.environ["VLLM_PLUGINS"]


@pytest.mark.parametrize("model", ["facebook/opt-125m"])
def test_custom_executor_async(model, tmpdir):
    cwd = os.path.abspath(".")
    os.chdir(tmpdir)
    old_env = os.environ.get("VLLM_PLUGINS", None)
    try:
        os.environ["VLLM_PLUGINS"] = "switch_executor"
        assert not os.path.exists(".marker")

        engine_args = AsyncEngineArgs(model=model,
                                      enforce_eager=True,
                                      gpu_memory_utilization=0.3)
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
        if old_env is not None:
            os.environ["VLLM_PLUGINS"] = old_env
        else:
            del os.environ["VLLM_PLUGINS"]
