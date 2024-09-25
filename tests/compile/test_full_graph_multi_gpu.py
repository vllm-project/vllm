import pytest

from vllm.compilation.backends import vllm_backend
from vllm.utils import cuda_device_count_stateless

from ..utils import fork_new_process_for_each_test
from .utils import TEST_MODELS_SMOKE, check_full_graph_support


@pytest.mark.parametrize("model_info", TEST_MODELS_SMOKE)
@pytest.mark.parametrize("tp_size", [2])
@pytest.mark.parametrize("backend", ["eager", vllm_backend])
@fork_new_process_for_each_test
def test_full_graph_multi_gpu(model_info, tp_size, backend):
    model = model_info[0]
    model_kwargs = model_info[1]

    # Skip the test if there are not enough CUDA devices.
    if cuda_device_count_stateless() < tp_size:
        pytest.skip("Not enough CUDA devices for the test.")

    check_full_graph_support(model, model_kwargs, backend, tp_size=tp_size)
