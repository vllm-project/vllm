"""Compare the outputs of HF and distributed vLLM when using greedy sampling.
The second test will hang if more than one test is run per command, so we need
to run the tests one by one. The solution is to pass arguments (model name) by
environment variables.

Run:
```sh
TEST_DIST_MODEL=llava-hf/llava-1.5-7b-hf \
    test_multimodal_broadcast.py
TEST_DIST_MODEL=microsoft/Phi-3-vision-128k-instruct \
    test_multimodal_broadcast.py
```
"""
import os

import pytest

from vllm.utils import cuda_device_count_stateless

model = os.environ["TEST_DIST_MODEL"]

if model.startswith("llava-hf/llava"):
    from ..models.test_llava import models, run_test
elif model.startswith("microsoft/Phi-3-vision"):
    from ..models.test_phi3v import models, run_test
else:
    raise NotImplementedError(f"Unsupported model: {model}")


@pytest.mark.parametrize("tensor_parallel_size", [2])
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [128])
@pytest.mark.parametrize("num_logprobs", [5])
def test_models(hf_runner, vllm_runner, image_assets,
                tensor_parallel_size: int, dtype: str, max_tokens: int,
                num_logprobs: int) -> None:
    if cuda_device_count_stateless() < tensor_parallel_size:
        pytest.skip(
            f"Need at least {tensor_parallel_size} GPUs to run the test.")

    distributed_executor_backend = os.getenv("DISTRIBUTED_EXECUTOR_BACKEND")

    run_test(
        hf_runner,
        vllm_runner,
        image_assets,
        model=models[0],
        size_factors=[1.0],
        dtype=dtype,
        max_tokens=max_tokens,
        num_logprobs=num_logprobs,
        tensor_parallel_size=tensor_parallel_size,
        distributed_executor_backend=distributed_executor_backend,
    )
