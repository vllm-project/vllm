"""Compares the outputs of gptq vs gptq_marlin when tp > 1
Note: GPTQ and Marlin do not have bitwise correctness.
As a result, in this test, we just confirm that the top selected tokens of the
Marlin/GPTQ models are in the top 5 selections of each other.
Note: Marlin internally uses locks to synchronize the threads. This can
result in very slight nondeterminism for Marlin. As a result, we re-run the test
up to 3 times to see if we pass.

Run `pytest tests/models/test_distributed_gptq_marlin.py`.
"""
import os

import pytest

from tests.models.test_gptq_marlin import MODELS, run_test
from tests.quantization.utils import (cuda_device_count_stateless,
                                      is_quant_method_supported)


@pytest.mark.parametrize("tensor_parallel_size", [2])
@pytest.mark.flaky(reruns=3)
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half", "bfloat16"])
@pytest.mark.parametrize("max_tokens", [32])
@pytest.mark.parametrize("num_logprobs", [5])
def test_models(vllm_runner, example_prompts, model, dtype: str,
                max_tokens: int, num_logprobs: int,
                tensor_parallel_size: int) -> None:

    if cuda_device_count_stateless() < tensor_parallel_size:
        pytest.skip("gptq_marlin is not supported on this GPU type.")

    if not is_quant_method_supported("gptq_marlin"):
        pytest.skip(
            f"Need at least {tensor_parallel_size} GPUs to run the test.")

    distributed_executor_backend = os.getenv("DISTRIBUTED_EXECUTOR_BACKEND")
    run_test(vllm_runner,
             example_prompts,
             model,
             dtype,
             max_tokens,
             num_logprobs,
             tensor_parallel_size=tensor_parallel_size,
             distributed_executor_backend=distributed_executor_backend)