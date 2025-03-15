# SPDX-License-Identifier: Apache-2.0

import pytest

from vllm.config import CompilationLevel
from vllm.platforms import current_platform

from ..utils import create_new_process_for_each_test
from .utils import TEST_MODELS, check_full_graph_support


@pytest.mark.parametrize("model_info", TEST_MODELS)
@pytest.mark.parametrize(
    "optimization_level",
    [CompilationLevel.DYNAMO_ONCE, CompilationLevel.PIECEWISE])
@create_new_process_for_each_test(
    "spawn" if current_platform.is_rocm() else "fork")
def test_full_graph(model_info, optimization_level):
    model = model_info[0]
    model_kwargs = model_info[1]
    check_full_graph_support(model,
                             model_kwargs,
                             optimization_level,
                             tp_size=1)
