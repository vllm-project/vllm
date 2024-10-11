import pytest

from vllm.compilation.levels import CompilationLevel

from .utils import TEST_MODELS, check_full_graph_support


@pytest.mark.parametrize("model_info", TEST_MODELS)
@pytest.mark.parametrize(
    "optimization_level",
    [CompilationLevel.DYNAMO_ONCE, CompilationLevel.INDUCTOR])
def test_full_graph(model_info, optimization_level):
    model = model_info[0]
    model_kwargs = model_info[1]
    check_full_graph_support(model,
                             model_kwargs,
                             optimization_level,
                             tp_size=1)
