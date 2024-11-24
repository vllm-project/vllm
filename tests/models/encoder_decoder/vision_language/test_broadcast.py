import pytest

from ....utils import multi_gpu_test


@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize("distributed_executor_backend", ["ray", "mp"])
@pytest.mark.parametrize("model", [
    "meta-llama/Llama-3.2-11B-Vision-Instruct",
])
def test_models(hf_runner, vllm_runner, image_assets,
                distributed_executor_backend, model) -> None:

    dtype = "half"
    max_tokens = 5
    num_logprobs = 5
    tensor_parallel_size = 2

    if model.startswith("meta-llama/Llama-3.2-11B-Vision-Instruct"):
        from .test_mllama import models, run_test
    else:
        raise NotImplementedError(f"Unsupported model: {model}")

    run_test(
        hf_runner,
        vllm_runner,
        image_assets,
        model=models[0],
        size_factors=[0.25, 0.5, 1.0],
        dtype=dtype,
        max_tokens=max_tokens,
        num_logprobs=num_logprobs,
        tensor_parallel_size=tensor_parallel_size,
        distributed_executor_backend=distributed_executor_backend,
    )
