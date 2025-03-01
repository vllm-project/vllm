# SPDX-License-Identifier: Apache-2.0
"""Test that we handle an Error in model forward and shutdown."""

import asyncio

import pytest

from tests.utils import wait_for_gpu_memory_to_clear
from vllm import LLM, SamplingParams
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.model_executor.models.llama import LlamaForCausalLM
from vllm.utils import GiB_bytes, cuda_device_count_stateless
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.exceptions import EngineDeadError


def evil_forward(self, *args, **kwargs):
    """Evil forward method that raises an exception after 10 calls."""
    NUMBER_OF_GOOD_PASSES = 10

    if not hasattr(self, "num_calls"):
        self.num_calls = 0

    if (self.num_calls == NUMBER_OF_GOOD_PASSES
            and get_tensor_model_parallel_rank() == 0):
        raise Exception("Simulated illegal memory access on Rank 0!")
    self.num_calls += 1

    return self.model(*args, **kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize("tensor_parallel_size", [2, 1])
async def test_async_llm_model_error(monkeypatch, tensor_parallel_size):

    if cuda_device_count_stateless() < tensor_parallel_size:
        pytest.skip(reason="Not enough CUDA devices")

    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "1")

        # Monkeypatch an error in the model.
        m.setattr(LlamaForCausalLM, "forward", evil_forward)

        engine_args = AsyncEngineArgs(
            model="meta-llama/Llama-3.2-1B",
            enforce_eager=True,
            tensor_parallel_size=tensor_parallel_size)
        async_llm = AsyncLLM.from_engine_args(engine_args)

        async def generate(request_id: str):
            generator = async_llm.generate("Hello my name is",
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

        # Every request should get an EngineDeadError.
        for output in outputs:
            assert isinstance(output, EngineDeadError)

        # AsyncLLM should be errored.
        assert async_llm.errored

        # We should not be able to make another request.
        with pytest.raises(EngineDeadError):
            async for _ in async_llm.generate(
                    "Hello my name is",
                    request_id="abc",
                    sampling_params=SamplingParams()):
                raise Exception("We should not get here.")

        # Confirm all the processes are cleaned up.
        wait_for_gpu_memory_to_clear(
            devices=list(range(tensor_parallel_size)),
            threshold_bytes=2 * GiB_bytes,
            timeout_s=60,
        )

        # NOTE: shutdown is handled by the API Server if an exception
        # occurs, so it is expected that we would need to call this.
        async_llm.shutdown()


@pytest.mark.parametrize("tensor_parallel_size", [2, 1])
def test_llm_model_error(monkeypatch, tensor_parallel_size):

    if cuda_device_count_stateless() < tensor_parallel_size:
        pytest.skip(reason="Not enough CUDA devices")

    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "1")

        # Monkeypatch an error in the model.
        m.setattr(LlamaForCausalLM, "forward", evil_forward)

        llm = LLM(model="meta-llama/Llama-3.2-1B",
                  enforce_eager=True,
                  tensor_parallel_size=tensor_parallel_size)

        with pytest.raises(EngineDeadError):
            llm.generate("Hello my name is Robert and I")

    # Confirm all the processes are cleaned up.
    wait_for_gpu_memory_to_clear(
        devices=list(range(tensor_parallel_size)),
        threshold_bytes=2 * GiB_bytes,
        timeout_s=60,
    )
