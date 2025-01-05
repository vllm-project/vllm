"""Test error handling in Processor. Should not impact other reqs."""

import asyncio

import pytest

from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.inputs.data import TokensPrompt
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.exceptions import EngineGenerateError


@pytest.mark.asyncio
async def test_async_llm_processor_error(monkeypatch):

    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "1")

        engine_args = AsyncEngineArgs(model="meta-llama/Llama-3.2-1B",
                                      enforce_eager=True)
        async_llm = AsyncLLM.from_engine_args(engine_args)

        async def generate(request_id: str):
            # [] is not allowed and will raise a ValueError in Processor.
            generator = async_llm.generate(TokensPrompt([]),
                                           request_id=request_id,
                                           sampling_params=SamplingParams())
            try:
                async for _ in generator:
                    pass
            except Exception as e:
                return e

        NUM_REQS = 3
        tasks = [generate(f"request-{idx}") for idx in range(NUM_REQS)]
        outputs = await asyncio.gather(*tasks)

        # Every request should have get an EngineGenerateError.
        for output in outputs:
            with pytest.raises(EngineGenerateError):
                raise output

        # AsyncLLM should be errored.
        assert not async_llm.errored

        # This should be no problem.
        outputs = []
        async for out in async_llm.generate(
                "Hello my name is",
                request_id="abc",
                sampling_params=SamplingParams(max_tokens=5)):
            outputs.append(out)
        assert len(outputs) == 5

        async_llm.shutdown()
