import asyncio
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pytest

from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.llm_engine import LLMEngine
from vllm.executor.uniproc_executor import UniProcExecutor
from vllm.sampling_params import SamplingParams


class Mock:
    ...


class CustomUniExecutor(UniProcExecutor):

    def collective_rpc(self,
                       method: Union[str, Callable],
                       timeout: Optional[float] = None,
                       args: Tuple = (),
                       kwargs: Optional[Dict] = None) -> List[Any]:
        # Drop marker to show that this was ran
        with open(".marker", "w"):
            ...
        return super().collective_rpc(method, timeout, args, kwargs)


CustomUniExecutorAsync = CustomUniExecutor


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


@pytest.mark.parametrize("model", ["facebook/opt-125m"])
def test_custom_executor(model, tmp_path):
    cwd = os.path.abspath(".")
    os.chdir(tmp_path)
    try:
        assert not os.path.exists(".marker")

        engine_args = EngineArgs(
            model=model,
            distributed_executor_backend=CustomUniExecutor,
        )
        engine = LLMEngine.from_engine_args(engine_args)
        sampling_params = SamplingParams(max_tokens=1)

        engine.add_request("0", "foo", sampling_params)
        engine.step()

        assert os.path.exists(".marker")
    finally:
        os.chdir(cwd)


@pytest.mark.parametrize("model", ["facebook/opt-125m"])
def test_custom_executor_async(model, tmp_path):
    cwd = os.path.abspath(".")
    os.chdir(tmp_path)
    try:
        assert not os.path.exists(".marker")

        engine_args = AsyncEngineArgs(
            model=model, distributed_executor_backend=CustomUniExecutorAsync)
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
