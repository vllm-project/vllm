import pytest

from vllm.utils import cuda_device_count_stateless

from ..models.test_llava import model_and_vl_config, run_llava_test
from ..utils import override_env


@pytest.fixture(autouse=True)
def tensor_parallel_ctx(tensor_parallel_size: int):
    if cuda_device_count_stateless() < tensor_parallel_size:
        pytest.skip(
            f"Need at least {tensor_parallel_size} GPUs to run the test.")

    if tensor_parallel_size > 1:
        with override_env("VLLM_WORKER_MULTIPROC_METHOD", "spawn"):
            yield
    else:
        yield


@pytest.mark.parametrize("model_and_config", [model_and_vl_config[0]])
@pytest.mark.parametrize("tensor_parallel_size", [2])
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [128])
def test_llava_tp(hf_runner, vllm_runner, image_assets, model_and_config,
                  tensor_parallel_size: int, dtype: str,
                  max_tokens: int) -> None:
    run_llava_test(
        hf_runner,
        vllm_runner,
        image_assets,
        model_and_config,
        dtype=dtype,
        max_tokens=max_tokens,
        tensor_parallel_size=tensor_parallel_size,
    )
