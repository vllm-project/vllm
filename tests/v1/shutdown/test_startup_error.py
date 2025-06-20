# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test that we handle a startup Error and shutdown."""

import pytest

from tests.utils import wait_for_gpu_memory_to_clear
from tests.v1.shutdown.utils import (SHUTDOWN_TEST_THRESHOLD_BYTES,
                                     SHUTDOWN_TEST_TIMEOUT_SEC)
from vllm import LLM
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.model_executor.models.llama import LlamaForCausalLM
from vllm.utils import cuda_device_count_stateless
from vllm.v1.engine.async_llm import AsyncLLM

MODELS = ["meta-llama/Llama-3.2-1B"]


def evil_method(self, *args, **kwargs):
    """Evil method that raises an exception."""

    if get_tensor_model_parallel_rank() == 0:
        raise Exception("Simulated Error in startup!")

    return self.model(*args, **kwargs, intermediate_tensors=None)


@pytest.mark.timeout(SHUTDOWN_TEST_TIMEOUT_SEC)
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tensor_parallel_size", [2, 1])
@pytest.mark.parametrize("failing_method", ["forward", "load_weights"])
def test_async_llm_startup_error(monkeypatch, model: str,
                                 tensor_parallel_size: int,
                                 failing_method: str) -> None:
    """Test that AsyncLLM propagates an __init__ error & frees memory.
    Test profiling (forward()) and load weights failures.
    AsyncLLM always uses an MP client.
    """
    if cuda_device_count_stateless() < tensor_parallel_size:
        pytest.skip(reason="Not enough CUDA devices")

    # Monkeypatch an error in the model.
    monkeypatch.setattr(LlamaForCausalLM, failing_method, evil_method)

    engine_args = AsyncEngineArgs(model=model,
                                  enforce_eager=True,
                                  tensor_parallel_size=tensor_parallel_size)

    # Confirm we get an exception.
    with pytest.raises(Exception, match="initialization failed"):
        _ = AsyncLLM.from_engine_args(engine_args)

    # Confirm all the processes are cleaned up.
    wait_for_gpu_memory_to_clear(
        devices=list(range(tensor_parallel_size)),
        threshold_bytes=SHUTDOWN_TEST_THRESHOLD_BYTES,
    )


@pytest.mark.timeout(SHUTDOWN_TEST_TIMEOUT_SEC)
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tensor_parallel_size", [2, 1])
@pytest.mark.parametrize("enable_multiprocessing", [True])
@pytest.mark.parametrize("failing_method", ["forward", "load_weights"])
def test_llm_startup_error(monkeypatch, model: str, tensor_parallel_size: int,
                           enable_multiprocessing: bool,
                           failing_method: str) -> None:
    """Test that LLM propagates an __init__ error and frees memory.
    Test profiling (forward()) and load weights failures.
    TODO(andy) - LLM without multiprocessing.
    """
    if model != "meta-llama/Llama-3.2-1B":
        pytest.skip(reason="Only test meta-llama/Llama-3.2-1B")
    if cuda_device_count_stateless() < tensor_parallel_size:
        pytest.skip(reason="Not enough CUDA devices")

    with monkeypatch.context() as m:

        MP_VALUE = "1" if enable_multiprocessing else "0"
        m.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", MP_VALUE)

        # Monkeypatch an error in the model.
        monkeypatch.setattr(LlamaForCausalLM, failing_method, evil_method)

        with pytest.raises(
                Exception,
                match="initialization failed"
                if enable_multiprocessing else "Simulated Error in startup!"):
            _ = LLM(model=model,
                    enforce_eager=True,
                    tensor_parallel_size=tensor_parallel_size)

        # Confirm all the processes are cleaned up.
        wait_for_gpu_memory_to_clear(
            devices=list(range(tensor_parallel_size)),
            threshold_bytes=SHUTDOWN_TEST_THRESHOLD_BYTES,
        )
