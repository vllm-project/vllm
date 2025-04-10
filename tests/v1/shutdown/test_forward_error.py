# SPDX-License-Identifier: Apache-2.0
"""Test that we handle an Error in model forward and shutdown."""

import pytest

from tests.utils import wait_for_gpu_memory_to_clear
from tests.v1.shutdown.utils import SHUTDOWN_TEST_TIMEOUT
from vllm import LLM
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.model_executor.models.llama import LlamaForCausalLM
from vllm.utils import cuda_device_count_stateless
from vllm.v1.engine.exceptions import EngineDeadError

MODELS = ["meta-llama/Llama-3.2-1B"]


def evil_forward(self, *args, **kwargs):
    """Evil forward method that raise an exception after 10 calls."""
    NUMBER_OF_GOOD_PASSES = 10

    if not hasattr(self, "num_calls"):
        self.num_calls = 0

    if (self.num_calls == NUMBER_OF_GOOD_PASSES
            and get_tensor_model_parallel_rank() == 0):
        raise Exception("Simulated illegal memory access on Rank 0!")
    self.num_calls += 1

    return self.model(*args, **kwargs)


@pytest.mark.timeout(SHUTDOWN_TEST_TIMEOUT)
@pytest.mark.parametrize("enable_multiprocessing", [True])
@pytest.mark.parametrize("tensor_parallel_size", [1])
@pytest.mark.parametrize("model", MODELS)
def test_llm_model_error(monkeypatch, tensor_parallel_size: int,
                         enable_multiprocessing: bool, model: str) -> None:
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

        llm = LLM(model=model,
                  enforce_eager=True,
                  tensor_parallel_size=tensor_parallel_size)

        with pytest.raises(
                EngineDeadError if enable_multiprocessing else Exception):
            llm.generate("Hello my name is Robert and I")

        # Confirm all the processes are cleaned up.
        wait_for_gpu_memory_to_clear(
            devices=list(range(tensor_parallel_size)),
            threshold_bytes=2 * 2**30,
            timeout_s=60,
        )
