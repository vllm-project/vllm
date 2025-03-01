# SPDX-License-Identifier: Apache-2.0
"""Test that we handle a startup Error and shutdown."""

import pytest

from tests.utils import wait_for_gpu_memory_to_clear
from vllm import LLM
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.model_executor.models.llama import LlamaForCausalLM
from vllm.utils import GiB_bytes, cuda_device_count_stateless
from vllm.v1.engine.async_llm import AsyncLLM


def evil_forward(self, *args, **kwargs):
    """Evil forward method that raises an exception."""

    if get_tensor_model_parallel_rank() == 0:
        raise RuntimeError("Simulated Error during forward pass!")

    return self.model(*args, **kwargs)


def evil_load_weights(self, *args, **kwargs):
    """Evil load_weights method that raises an exception."""

    raise RuntimeError("Simulated OOM Error during weight loading!")


MODELS = [
    "meta-llama/Llama-3.2-1B",
]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tensor_parallel_size", [2, 1])
def test_async_llm_forward_pass_error(monkeypatch, model,
                                      tensor_parallel_size):
    """Test failure during first forward pass"""

    if cuda_device_count_stateless() < tensor_parallel_size:
        pytest.skip(reason="Not enough CUDA devices")

    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "1")

        # Monkeypatch an error in the model.
        m.setattr(LlamaForCausalLM, "load_weights", evil_load_weights)

        engine_args = AsyncEngineArgs(
            model=model,
            enforce_eager=True,
            tensor_parallel_size=tensor_parallel_size)

        # Confirm we get an exception.
        with pytest.raises(Exception,
                           match="EngineCore initialization failed"):
            _ = AsyncLLM.from_engine_args(engine_args)

        # Confirm all the processes are cleaned up.
        wait_for_gpu_memory_to_clear(
            devices=list(range(tensor_parallel_size)),
            threshold_bytes=2 * GiB_bytes,
            timeout_s=60,
        )


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tensor_parallel_size", [2, 1])
def test_async_llm_weight_loading_failure(monkeypatch, model,
                                          tensor_parallel_size):
    """Test failure during first forward pass"""

    if cuda_device_count_stateless() < tensor_parallel_size:
        pytest.skip(reason="Not enough CUDA devices")

    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "1")

        # Monkeypatch an error in the model.
        m.setattr(LlamaForCausalLM, "forward", evil_forward)

        engine_args = AsyncEngineArgs(
            model=model,
            enforce_eager=True,
            tensor_parallel_size=tensor_parallel_size)

        # Confirm we get an exception.
        with pytest.raises(Exception,
                           match="EngineCore initialization failed"):
            _ = AsyncLLM.from_engine_args(engine_args)

        # Confirm all the processes are cleaned up.
        wait_for_gpu_memory_to_clear(
            devices=list(range(tensor_parallel_size)),
            threshold_bytes=2 * GiB_bytes,
            timeout_s=60,
        )


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tensor_parallel_size", [2, 1])
def test_llm_forward_pass_failure(monkeypatch, model, tensor_parallel_size):
    """Test failure during first forward pass (after IPC setup)."""

    if cuda_device_count_stateless() < tensor_parallel_size:
        pytest.skip(reason="Not enough CUDA devices")

    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "1")

        # Simulate error during forward pass
        m.setattr(LlamaForCausalLM, "forward", evil_forward)

        with pytest.raises(Exception,
                           match="EngineCore initialization failed"):
            _ = LLM(model=model,
                    enforce_eager=True,
                    tensor_parallel_size=tensor_parallel_size)

        wait_for_gpu_memory_to_clear(
            devices=list(range(tensor_parallel_size)),
            threshold_bytes=2 * GiB_bytes,
            timeout_s=60,
        )


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tensor_parallel_size", [2, 1])
def test_llm_weight_loading_failure(monkeypatch, model, tensor_parallel_size):
    """Test failure during weight loading (before IPC setup)."""

    if cuda_device_count_stateless() < tensor_parallel_size:
        pytest.skip(reason="Not enough CUDA devices")

    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "1")

        # Simulate error during weight loading
        m.setattr(LlamaForCausalLM, "load_weights", evil_load_weights)

        with pytest.raises(Exception,
                           match="EngineCore initialization failed"):
            _ = LLM(model=model,
                    enforce_eager=True,
                    tensor_parallel_size=tensor_parallel_size)

        wait_for_gpu_memory_to_clear(
            devices=list(range(tensor_parallel_size)),
            threshold_bytes=2 * GiB_bytes,
            timeout_s=60,
        )
