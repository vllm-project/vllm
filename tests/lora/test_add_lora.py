# SPDX-License-Identifier: Apache-2.0
import time
from typing import Callable, List, Optional

import pytest

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.protocol import EngineClient
from vllm.inputs import TextPrompt
from vllm.lora.request import LoRARequest
from vllm.sampling_params import SamplingParams
from vllm.utils import merge_async_iterators

MODEL_PATH = "meta-llama/Llama-2-7b-hf"
LORA_MODULE_PATH = "yard1/llama-2-7b-sql-lora-test"
LORA_RANK = 8
DEFAULT_MAX_LORAS = 4


@pytest.fixture(autouse=True)
def v1(run_with_both_engines_lora):
    # Simple autouse wrapper to run both engines for each test
    # This can be promoted up to conftest.py to run for every
    # test in a package
    pass


async def add_lora_callable(llm: EngineClient,
                            lora_requests=List[LoRARequest]):
    for lr in lora_requests:
        await llm.add_lora(lr)


async def requests_processing_time(
    lora_requests=List[LoRARequest],
    warmup_function: Optional[Callable[[EngineClient, List[LoRARequest]],
                                       None]] = None
) -> float:
    """
    Utility function to measure LoRA requests processing time. The primary
    usecase is to test the difference between a cold start
    (no warmup functions) vs a hot start.
    """

    max_loras = len(set([lr.lora_int_id for lr in lora_requests]))
    engine_args = AsyncEngineArgs(model=MODEL_PATH,
                                  enable_lora=True,
                                  max_loras=max_loras,
                                  max_lora_rank=LORA_RANK)
    sampling_params = SamplingParams(n=1,
                                     temperature=0.0,
                                     top_p=1.0,
                                     ignore_eos=True,
                                     max_tokens=1)

    # The run_with_both_engines_lora fixture sets up the `VLLM_USE_V1`
    # environment variable. reload vllm.enging.async_llm_engine as
    # vllm.engine.async_llm_engine.AsyncLLMEgnine changes depending on the
    # env var
    import importlib

    import vllm.engine.async_llm_engine
    importlib.reload(vllm.engine.async_llm_engine)
    from vllm.entrypoints.openai.api_server import (
        build_async_engine_client_from_engine_args)

    async with build_async_engine_client_from_engine_args(engine_args) as llm:

        if warmup_function:
            await warmup_function(llm, lora_requests)
            # Wait for the warmup functions complete
            time.sleep(10)

        generators = []
        start = time.perf_counter()

        for idx, lora_request in enumerate(lora_requests):
            generator = llm.generate(prompt=TextPrompt(prompt=f"hello {idx}",
                                                       multi_modal_data=None),
                                     sampling_params=sampling_params,
                                     lora_request=lora_request,
                                     request_id=f"test{idx}")
            generators.append(generator)

        all_gens = merge_async_iterators(*generators)
        async for i, res in all_gens:
            pass

        end = time.perf_counter()
        return end - start


def get_lora_requests() -> List[LoRARequest]:
    lora_requests: List[LoRARequest] = [
        LoRARequest(lora_name=f"{i}",
                    lora_int_id=i,
                    lora_path=LORA_MODULE_PATH)
        for i in range(1, DEFAULT_MAX_LORAS + 1)
    ]
    return lora_requests


@pytest.mark.asyncio
async def test_add_lora():
    """ 
    The add_lora function is used to pre-load some LoRA adapters into the
    engine in anticipation of future requests using these adapters. To test
    this functionality, we use the async engine to process some requests - We
    do it twice, once with add_lora() pre-loading and once without.

    We measure the request processing time in both cases and expect the time 
    to be lesser in the case with add_lora() calls.
    """
    lora_requests: List[LoRARequest] = get_lora_requests()

    # Dummy run - So any 1-time functionality like triton kernel compilation
    # is complete here.
    await requests_processing_time(lora_requests)

    # Run with warmup
    time_with_add_lora = await requests_processing_time(
        lora_requests, warmup_function=add_lora_callable)

    # Run without any warmup
    time_cold_start = await requests_processing_time(lora_requests)

    print(f"time hot-start {time_with_add_lora} vs "
          f"time cold-start {time_cold_start} ")

    assert time_with_add_lora < time_cold_start, (
        f"time_with_add_lora={time_with_add_lora}, "
        f"time_cold_start={time_cold_start}"
        "The engine request processing time with LoRA pre-loading "
        "must be less than the version that does on-demand LoRA loading."
        f"time with add_lora ={time}")
