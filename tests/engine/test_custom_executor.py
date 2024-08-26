import asyncio
import os

import pytest

from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
from vllm.engine.llm_engine import AsyncLLMEngine, LLMEngine
from vllm.sampling_params import SamplingParams


@pytest.mark.parametrize("model", ["facebook/opt-125m"])
def test_custom_executor(model, tmpdir):
    cwd = os.path.abspath(".")
    os.chdir(tmpdir)
    old_env = os.environ["VLLM_PLUGINS"]
    try:
        os.environ["VLLM_PLUGINS"] = "switch_executor"
        assert not os.path.exists(".marker")

        engine_args = EngineArgs(model=model, enforce_eager=True)
        engine = LLMEngine.from_engine_args(engine_args)
        sampling_params = SamplingParams(max_tokens=1)

        engine.add_request("0", "foo", sampling_params)
        engine.step()

        assert os.path.exists(".marker")
    finally:
        os.chdir(cwd)
        os.environ["VLLM_PLUGINS"] = old_env


@pytest.mark.parametrize("model", ["facebook/opt-125m"])
def test_custom_executor_async(model, tmpdir):
    cwd = os.path.abspath(".")
    os.chdir(tmpdir)
    old_env = os.environ["VLLM_PLUGINS"]
    try:
        os.environ["VLLM_PLUGINS"] = "switch_executor"
        assert not os.path.exists(".marker")

        engine_args = AsyncEngineArgs(model=model, enforce_eager=True)
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
        os.environ["VLLM_PLUGINS"] = old_env
