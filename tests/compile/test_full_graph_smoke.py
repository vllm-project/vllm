import pytest

from vllm.compilation.backends import vllm_backend

from .utils import TEST_MODELS_SMOKE, check_full_graph_support


@pytest.mark.parametrize("model_info", TEST_MODELS_SMOKE)
@pytest.mark.parametrize("backend", ["eager", vllm_backend])
def test_full_graph(model_info, backend):
    model = model_info[0]
    model_kwargs = model_info[1]
    check_full_graph_support(model, model_kwargs, backend, tp_size=1)
