# SPDX-License-Identifier: Apache-2.0
import asyncio
import time
from pathlib import Path
from typing import List

import pytest
from huggingface_hub import snapshot_download

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.inputs import TextPrompt
from vllm.lora.request import LoRARequest
from vllm.sampling_params import SamplingParams
from vllm.utils import merge_async_iterators

MODEL_PATH = "meta-llama/Llama-2-7b-hf"
LORA_MODULE_DOWNLOAD_PATH = None  # Populated by download_and_prepare_lora_module() #noqa
LORA_RANK = 8
DEFAULT_MAX_LORAS = 16 * 3


def download_and_prepare_lora_module():
    """
    Request submission is expensive when the LoRA adapters have their own
    tokenizers. This is because, for each request with a new LoRA adapter ID,
    the front-end loads the tokenizer from disk.

    In this test, as we are comparing request processing times, we want to
    minimize any extra activity. To this effect, we download the LoRA
    adapter and remove all the tokenizer files, so the engine will default
    to the base model tokenizer.
    """
    global LORA_MODULE_DOWNLOAD_PATH

    LORA_MODULE_HF_PATH = "yard1/llama-2-7b-sql-lora-test"
    LORA_MODULE_DOWNLOAD_PATH = snapshot_download(repo_id=LORA_MODULE_HF_PATH)

    tokenizer_files = [
        'added_tokens.json', 'tokenizer_config.json', 'tokenizer.json',
        'tokenizer.model'
    ]
    for tokenizer_file in tokenizer_files:
        del_path = Path(LORA_MODULE_DOWNLOAD_PATH) / tokenizer_file
        del_path.unlink(missing_ok=True)


@pytest.fixture(autouse=True)
def v1(run_with_both_engines_lora):
    # Simple autouse wrapper to run both engines for each test
    # This can be promoted up to conftest.py to run for every
    # test in a package
    pass


def get_lora_requests() -> List[LoRARequest]:
    lora_requests: List[LoRARequest] = [
        LoRARequest(lora_name=f"{i}",
                    lora_int_id=i,
                    lora_path=LORA_MODULE_DOWNLOAD_PATH)
        for i in range(1, DEFAULT_MAX_LORAS + 1)
    ]
    return lora_requests


async def requests_processing_time(llm,
                                   lora_requests: List[LoRARequest]) -> float:

    sampling_params = SamplingParams(n=1,
                                     temperature=0.0,
                                     top_p=1.0,
                                     ignore_eos=True,
                                     max_tokens=1)

    generators = []
    start = time.perf_counter()

    for lora_request in lora_requests:
        lora_int_id = lora_request.lora_int_id
        generator = llm.generate(
            prompt=TextPrompt(prompt=f"hello {lora_int_id}",
                              multi_modal_data=None),  # type: ignore 
            sampling_params=sampling_params,
            lora_request=lora_request,
            request_id=f"test{lora_int_id}")
        generators.append(generator)

    all_gens = merge_async_iterators(*generators)
    async for i, res in all_gens:
        pass

    end = time.perf_counter()
    return end - start


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

    download_and_prepare_lora_module()

    lora_requests: List[LoRARequest] = get_lora_requests()

    max_loras = len(set([lr.lora_int_id for lr in lora_requests]))
    # Create engine in eager-mode. Due to high max_loras, the CI can
    # OOM during cuda-graph capture.
    engine_args = AsyncEngineArgs(
        model=MODEL_PATH,
        enable_lora=True,
        max_loras=max_loras,
        max_lora_rank=LORA_RANK,
        max_model_len=128,
        gpu_memory_utilization=0.8,  #avoid OOM
        enforce_eager=True)

    # The run_with_both_engines_lora fixture sets up the `VLLM_USE_V1`
    # environment variable. reload vllm.enging.async_llm_engine as
    # vllm.engine.async_llm_engine.AsyncLLMEgnine changes depending on the
    # env var.
    import importlib

    import vllm.engine.async_llm_engine
    importlib.reload(vllm.engine.async_llm_engine)
    from vllm.entrypoints.openai.api_server import (
        build_async_engine_client_from_engine_args)

    # split lora_requests into 3 parts
    part_size = len(lora_requests) // 3
    dummy_run_requests = lora_requests[:part_size]
    warmup_run_requests = lora_requests[part_size:part_size * 2]
    cold_run_requests = lora_requests[part_size * 2:]

    async with build_async_engine_client_from_engine_args(engine_args) as llm:

        # Dummy run - So any 1-time functionality like triton kernel compilation
        # is complete here.
        await requests_processing_time(llm, dummy_run_requests)

        # Run with warmup
        for lr in warmup_run_requests:
            await llm.add_lora(lr)
        # Wait for the add_lora function to complete on the server side.
        await asyncio.sleep(30)
        time_with_add_lora = await requests_processing_time(
            llm, warmup_run_requests)

        # Run without any warmup
        time_cold_start = await requests_processing_time(
            llm, cold_run_requests)

    print(f"time hot-start {time_with_add_lora} vs "
          f"time cold-start {time_cold_start} ")

    assert time_with_add_lora < time_cold_start, (
        f"time_with_add_lora={time_with_add_lora}, "
        f"time_cold_start={time_cold_start}"
        "The engine request processing time with LoRA pre-loading "
        "must be less than the version that does on-demand LoRA loading.")
