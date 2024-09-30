from typing import Dict, List, Optional

import pytest

from vllm.utils import cuda_device_count_stateless

from ..utils import compare_all_settings
from .utils import TEST_MODELS_SMOKE


@pytest.mark.parametrize("model_info", TEST_MODELS_SMOKE)
@pytest.mark.parametrize("pp_size", [1, 2])
def test_compile_correctness(model_info, pp_size):
    if cuda_device_count_stateless() < pp_size:
        pytest.skip("Not enough CUDA devices for the test.")
    model = model_info[0]
    model_args = model_info[1]
    all_args = [
        ["--enforce-eager"] + model_args + ["--max_model_len", "1024"] +
        ["-pp", str(pp_size)],
    ] * 3
    all_envs: List[Optional[Dict[str, str]]] = [{
        "VLLM_TORCH_COMPILE_LEVEL":
        str(i)
    } for i in range(3)]
    compare_all_settings(model, all_args, all_envs)
