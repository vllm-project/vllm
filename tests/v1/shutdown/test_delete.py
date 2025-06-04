# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test that we handle a startup Error and shutdown."""

import pytest

from tests.utils import wait_for_gpu_memory_to_clear
from tests.v1.shutdown.utils import (SHUTDOWN_TEST_THRESHOLD_BYTES,
                                     SHUTDOWN_TEST_TIMEOUT_SEC)
from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import RequestOutputKind
from vllm.utils import cuda_device_count_stateless
from vllm.v1.engine.async_llm import AsyncLLM

MODELS = ["meta-llama/Llama-3.2-1B"]


@pytest.mark.asyncio
@pytest.mark.timeout(SHUTDOWN_TEST_TIMEOUT_SEC)
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tensor_parallel_size", [2, 1])
@pytest.mark.parametrize("send_one_request", [False, True])
async def test_async_llm_delete(model: str, tensor_parallel_size: int,
                                send_one_request: bool) -> None:
    """Test that AsyncLLM frees GPU memory upon deletion.
    AsyncLLM always uses an MP client.

    Args:
      model: model under test
      tensor_parallel_size: degree of tensor parallelism
      send_one_request: send one request to engine before deleting
    """
    if cuda_device_count_stateless() < tensor_parallel_size:
        pytest.skip(reason="Not enough CUDA devices")

    engine_args = AsyncEngineArgs(model=model,
                                  enforce_eager=True,
                                  tensor_parallel_size=tensor_parallel_size)

    # Instantiate AsyncLLM; make request to complete any deferred
    # initialization; then delete instance
    async_llm = AsyncLLM.from_engine_args(engine_args)
    if send_one_request:
        async for _ in async_llm.generate(
                "Hello my name is",
                request_id="abc",
                sampling_params=SamplingParams(
                    max_tokens=1, output_kind=RequestOutputKind.DELTA)):
            pass
    del async_llm

    # Confirm all the processes are cleaned up.
    wait_for_gpu_memory_to_clear(
        devices=list(range(tensor_parallel_size)),
        threshold_bytes=SHUTDOWN_TEST_THRESHOLD_BYTES,
    )


@pytest.mark.timeout(SHUTDOWN_TEST_TIMEOUT_SEC)
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tensor_parallel_size", [2, 1])
@pytest.mark.parametrize("enable_multiprocessing", [True])
@pytest.mark.parametrize("send_one_request", [False, True])
def test_llm_delete(monkeypatch, model: str, tensor_parallel_size: int,
                    enable_multiprocessing: bool,
                    send_one_request: bool) -> None:
    """Test that LLM frees GPU memory upon deletion.
    TODO(andy) - LLM without multiprocessing.

    Args:
      model: model under test
      tensor_parallel_size: degree of tensor parallelism
      enable_multiprocessing: enable workers in separate process(es)
      send_one_request: send one request to engine before deleting
    """
    if cuda_device_count_stateless() < tensor_parallel_size:
        pytest.skip(reason="Not enough CUDA devices")

    with monkeypatch.context() as m:
        MP_VALUE = "1" if enable_multiprocessing else "0"
        m.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", MP_VALUE)

        # Instantiate LLM; make request to complete any deferred
        # initialization; then delete instance
        llm = LLM(model=model,
                  enforce_eager=True,
                  tensor_parallel_size=tensor_parallel_size)
        if send_one_request:
            llm.generate("Hello my name is",
                         sampling_params=SamplingParams(max_tokens=1))
        del llm

        # Confirm all the processes are cleaned up.
        wait_for_gpu_memory_to_clear(
            devices=list(range(tensor_parallel_size)),
            threshold_bytes=SHUTDOWN_TEST_THRESHOLD_BYTES,
        )
