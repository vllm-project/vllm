import pytest

from vllm.config import CompilationLevel

from ..utils import fork_new_process_for_each_test
from .utils import TEST_MODELS, check_full_graph_support


@pytest.mark.parametrize("model_info", TEST_MODELS)
@pytest.mark.parametrize(
    "optimization_level",
    [CompilationLevel.DYNAMO_ONCE, CompilationLevel.PIECEWISE])
@fork_new_process_for_each_test
def test_full_graph(model_info, optimization_level):
    model = model_info[0]
    model_kwargs = model_info[1]
    check_full_graph_support(model,
                             model_kwargs,
                             optimization_level,
                             tp_size=1)
