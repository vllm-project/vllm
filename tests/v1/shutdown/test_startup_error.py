# SPDX-License-Identifier: Apache-2.0
"""Test that we handle a startup Error and shutdown."""

import pytest

from tests.utils import wait_for_gpu_memory_to_clear
from vllm import LLM
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.model_executor.models.llama import LlamaForCausalLM
from vllm.utils import cuda_device_count_stateless
from vllm.v1.engine.async_llm import AsyncLLM


def evil_forward(self, *args, **kwargs):
    """Evil forward method that raise an exception."""

    if get_tensor_model_parallel_rank() == 0:
        raise Exception("Simulated Error in startup!")

    return self.model(*args, **kwargs, intermediate_tensors=None)


MODELS = [
    "meta-llama/Llama-3.2-1B",  # Raises on first fwd pass.
    "mistralai/Mixtral-8x22B-Instruct-v0.1"  # Causes OOM.
]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tensor_parallel_size", [2, 1])
def test_async_llm_startup_error(monkeypatch, model, tensor_parallel_size):

    if cuda_device_count_stateless() < tensor_parallel_size:
        pytest.skip(reason="Not enough CUDA devices")

    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "1")

        # Monkeypatch an error in the model.
        monkeypatch.setattr(LlamaForCausalLM, "forward", evil_forward)

        engine_args = AsyncEngineArgs(
            model=model,
            enforce_eager=True,
            tensor_parallel_size=tensor_parallel_size)

        # Confirm we get an exception.
        with pytest.raises(Exception, match="initialization failed"):
            _ = AsyncLLM.from_engine_args(engine_args)

        # Confirm all the processes are cleaned up.
        wait_for_gpu_memory_to_clear(
            devices=list(range(tensor_parallel_size)),
            threshold_bytes=2 * 2**30,
            timeout_s=60,
        )


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tensor_parallel_size", [2, 1])
@pytest.mark.parametrize("enable_multiprocessing", [True, False])
def test_llm_startup_error(monkeypatch, model, tensor_parallel_size,
                           enable_multiprocessing):

    if cuda_device_count_stateless() < tensor_parallel_size:
        pytest.skip(reason="Not enough CUDA devices")

    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "1")

        MP_VALUE = "1" if enable_multiprocessing else "0"
        m.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", MP_VALUE)

        # Monkeypatch an error in the model.
        monkeypatch.setattr(LlamaForCausalLM, "forward", evil_forward)

        with pytest.raises(Exception, match="initialization failed"):
            _ = LLM(model="meta-llama/Llama-3.2-1B",
                    enforce_eager=True,
                    tensor_parallel_size=tensor_parallel_size)

        # Confirm all the processes are cleaned up.
        wait_for_gpu_memory_to_clear(
            devices=list(range(tensor_parallel_size)),
            threshold_bytes=2 * 2**30,
            timeout_s=60,
        )
