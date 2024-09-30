from typing import Dict, List, Optional

import pytest

from ..utils import compare_all_settings
from .utils import TEST_MODELS_SMOKE


@pytest.mark.parametrize("model_info", TEST_MODELS_SMOKE)
def test_compile_correctness(model_info):
    model = model_info[0]
    model_args = model_info[1]
    all_args = [
        ["--enforce-eager"] + model_args,
        ["--enforce-eager"] + model_args,
        ["--enforce-eager"] + model_args,
    ]
    all_envs: List[Optional[Dict[str, str]]] = [{
        "VLLM_TORCH_COMPILE_LEVEL":
        str(i)
    } for i in range(3)]
    compare_all_settings(model, all_args, all_envs)
