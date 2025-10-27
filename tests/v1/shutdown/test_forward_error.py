# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test that we handle an Error in model forward and shutdown."""

import asyncio

import pytest

from tests.utils import wait_for_gpu_memory_to_clear
from tests.v1.shutdown.utils import (
    SHUTDOWN_TEST_THRESHOLD_BYTES,
    SHUTDOWN_TEST_TIMEOUT_SEC,
)
from vllm import LLM, AsyncEngineArgs, SamplingParams
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.model_executor.models.llama import LlamaForCausalLM
from vllm.utils.torch_utils import cuda_device_count_stateless
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.exceptions import EngineDeadError

MODELS = ["hmellor/tiny-random-LlamaForCausalLM"]


def evil_forward(self, *args, **kwargs):
    """Evil forward method that raise an exception after 10 calls."""
    NUMBER_OF_GOOD_PASSES = 10

    if not hasattr(self, "num_calls"):
        self.num_calls = 0

    if (
        self.num_calls == NUMBER_OF_GOOD_PASSES
        and get_tensor_model_parallel_rank() == 0
    ):
        raise Exception("Simulated illegal memory access on Rank 0!")
    self.num_calls += 1

    return self.model(*args, **kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize("tensor_parallel_size", [2, 1])
@pytest.mark.parametrize("model", MODELS)
async def test_async_llm_model_error(
    monkeypatch, tensor_parallel_size: int, model: str
) -> None:
    """Test that AsyncLLM propagates a forward pass error and frees memory.

    AsyncLLM always uses an MP client.
    """
    if cuda_device_count_stateless() < tensor_parallel_size:
        pytest.skip(reason="Not enough CUDA devices")

    # Monkeypatch an error in the model.
    monkeypatch.setattr(LlamaForCausalLM, "forward", evil_forward)

    engine_args = AsyncEngineArgs(
        model=model, enforce_eager=True, tensor_parallel_size=tensor_parallel_size
    )
    async_llm = AsyncLLM.from_engine_args(engine_args)

    async def generate(request_id: str):
        generator = async_llm.generate(
            "Hello my name is", request_id=request_id, sampling_params=SamplingParams()
        )
        try:
            async for _ in generator:
                pass
        except Exception as e:
            return e

    NUM_REQS = 3
    tasks = [generate(f"request-{idx}") for idx in range(NUM_REQS)]
    outputs = await asyncio.gather(*tasks)

    # Every request should get an EngineDeadError.
    for output in outputs:
        assert isinstance(output, EngineDeadError)

    # AsyncLLM should be errored.
    assert async_llm.errored

    # We should not be able to make another request.
    with pytest.raises(EngineDeadError):
        async for _ in async_llm.generate(
            "Hello my name is", request_id="abc", sampling_params=SamplingParams()
        ):
            raise Exception("We should not get here.")

    # Confirm all the processes are cleaned up.
    wait_for_gpu_memory_to_clear(
        devices=list(range(tensor_parallel_size)),
        threshold_bytes=2 * 2**30,
        timeout_s=60,
    )

    # NOTE: shutdown is handled by the API Server if an exception
    # occurs, so it is expected that we would need to call this.
    async_llm.shutdown()


@pytest.mark.timeout(SHUTDOWN_TEST_TIMEOUT_SEC)
@pytest.mark.parametrize("enable_multiprocessing", [True])
@pytest.mark.parametrize("tensor_parallel_size", [2, 1])
@pytest.mark.parametrize("model", MODELS)
def test_llm_model_error(
    monkeypatch, tensor_parallel_size: int, enable_multiprocessing: bool, model: str
) -> None:
    """Test that LLM propagates a forward pass error and frees memory.
    TODO(andy) - LLM without multiprocessing; LLM with multiprocessing
    and >1 rank
    """
    if cuda_device_count_stateless() < tensor_parallel_size:
        pytest.skip(reason="Not enough CUDA devices")

    with monkeypatch.context() as m:
        MP_VALUE = "1" if enable_multiprocessing else "0"
        m.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", MP_VALUE)

        # Monkeypatch an error in the model.
        m.setattr(LlamaForCausalLM, "forward", evil_forward)

        llm = LLM(
            model=model, enforce_eager=True, tensor_parallel_size=tensor_parallel_size
        )

        with pytest.raises(EngineDeadError if enable_multiprocessing else Exception):
            llm.generate("Hello my name is Robert and I")

        # Confirm all the processes are cleaned up.
        wait_for_gpu_memory_to_clear(
            devices=list(range(tensor_parallel_size)),
            threshold_bytes=SHUTDOWN_TEST_THRESHOLD_BYTES,
        )
