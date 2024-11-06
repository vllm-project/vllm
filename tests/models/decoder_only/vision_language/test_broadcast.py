import pytest

from ....utils import multi_gpu_test


@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize("distributed_executor_backend", ["ray", "mp"])
@pytest.mark.parametrize("model", [
    "llava-hf/llava-1.5-7b-hf",
    "llava-hf/llava-v1.6-mistral-7b-hf",
    "facebook/chameleon-7b",
])
def test_models(hf_runner, vllm_runner, image_assets,
                distributed_executor_backend, model) -> None:

    dtype = "half"
    max_tokens = 5
    num_logprobs = 5
    tensor_parallel_size = 2

    if model.startswith("llava-hf/llava-1.5"):
        from .test_llava import models, run_test
    elif model.startswith("llava-hf/llava-v1.6"):
        from .test_llava_next import models, run_test  # type: ignore[no-redef]
    elif model.startswith("facebook/chameleon"):
        from .test_chameleon import models, run_test  # type: ignore[no-redef]
    else:
        raise NotImplementedError(f"Unsupported model: {model}")

    run_test(
        hf_runner,
        vllm_runner,
        image_assets,
        model=models[0],
        # So that LLaVA-NeXT processor may return nested list
        size_factors=[0.25, 0.5, 1.0],
        dtype=dtype,
        max_tokens=max_tokens,
        num_logprobs=num_logprobs,
        tensor_parallel_size=tensor_parallel_size,
        distributed_executor_backend=distributed_executor_backend,
    )
