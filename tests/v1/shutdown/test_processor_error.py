# SPDX-License-Identifier: Apache-2.0
"""Test error handling in Processor. Should not impact other reqs."""

import asyncio

import pytest

from tests.v1.shutdown.utils import SHUTDOWN_TEST_TIMEOUT_SEC
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.inputs.data import TokensPrompt
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.exceptions import EngineGenerateError

MODELS = ["meta-llama/Llama-3.2-1B"]


@pytest.mark.asyncio
@pytest.mark.timeout(SHUTDOWN_TEST_TIMEOUT_SEC)
@pytest.mark.parametrize("model", MODELS)
async def test_async_llm_processor_error(model: str) -> None:
    """Test that AsyncLLM propagates a processor error.
    Test empty tokens prompt (failure) and non-empty prompt (no failure.)
    AsyncLLM always uses an MP client.
    """
    engine_args = AsyncEngineArgs(model=model, enforce_eager=True)
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
    EXPECTED_TOKENS = 5
    outputs = []
    async for out in async_llm.generate(
            "Hello my name is",
            request_id="abc",
            sampling_params=SamplingParams(
                max_tokens=EXPECTED_TOKENS,
                output_kind=RequestOutputKind.DELTA)):
        outputs.append(out)

    generated_tokens = []
    for out in outputs:
        generated_tokens.extend(out.outputs[0].token_ids)
    assert len(generated_tokens) == EXPECTED_TOKENS

    async_llm.shutdown()
