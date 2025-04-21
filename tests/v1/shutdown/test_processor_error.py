# SPDX-License-Identifier: Apache-2.0
"""Test error handling in Processor. Should not impact other reqs."""

import asyncio
import os

import pytest

from tests.v1.shutdown.utils import SHUTDOWN_TEST_TIMEOUT_SEC
from tests.v1.utils import generate_dp
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.inputs.data import TokensPrompt
from vllm.sampling_params import RequestOutputKind
from vllm.utils import cuda_device_count_stateless
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


@pytest.mark.asyncio
@pytest.mark.timeout(SHUTDOWN_TEST_TIMEOUT_SEC)
@pytest.mark.parametrize("data_parallel_size", [2])
@pytest.mark.parametrize("model", MODELS)
async def test_async_llm_dp_processor_error(
    model: str,
    data_parallel_size: int,
) -> None:
    """Test that AsyncLLM w/ data parallelism propagates a processor error.
    Test empty tokens prompt (failure) and non-empty prompt (no failure.)
    AsyncLLM always uses an MP client.

    Args:
      model: model under test
      data_parallel_size: degree of data parallelism
    """
    if cuda_device_count_stateless() < data_parallel_size:
        pytest.skip(reason="Not enough CUDA devices")

    engine_args = AsyncEngineArgs(
        model=model,
        enforce_eager=True,
        disable_log_requests=True,
        tensor_parallel_size=int(os.getenv("TP_SIZE", 1)),
        data_parallel_size=int(os.getenv("DP_SIZE", data_parallel_size)),
    )

    # [] is not allowed and will raise a ValueError in Processor.
    async_llm = AsyncLLM.from_engine_args(engine_args)

    async def generate(request_id: str):
        # [] is not allowed and will raise a ValueError in Processor.
        generator = generate_dp(engine=async_llm,
                                prompt=TokensPrompt([]),
                                request_id=request_id,
                                sampling_params=SamplingParams(),
                                output_kind=RequestOutputKind.FINAL_ONLY,
                                max_tokens=NUM_EXPECTED_TOKENS)
        try:
            async for _ in generator:
                pass
        except Exception as e:
            return e

    # Create concurrent requests.
    NUM_REQUESTS = 10
    NUM_EXPECTED_TOKENS = 10

    tasks = [generate(f"request-{idx}") for idx in range(NUM_REQUESTS)]
    outputs = await asyncio.gather(*tasks)

    # Every request should have got an EngineGenerateError.
    for output in outputs:
        with pytest.raises(EngineGenerateError):
            raise output

    # AsyncLLM should not be errored.
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
