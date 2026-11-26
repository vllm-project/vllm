# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

import vllm.envs as envs
from vllm._aiter_ops import is_aiter_found_and_supported, rocm_aiter_ops

AITER_CONFIGS = {
    "aiter_off": {"VLLM_ROCM_USE_AITER": "0"},
    "aiter_on": {"VLLM_ROCM_USE_AITER": "1", "VLLM_ROCM_USE_AITER_RMSNORM": "1"},
    "aiter_on_rmsnorm_off": {
        "VLLM_ROCM_USE_AITER": "1",
        "VLLM_ROCM_USE_AITER_RMSNORM": "0",
    },
}


def _refresh_aiter_envs():
    envs.disable_envs_cache()
    rocm_aiter_ops.refresh_env_variables()


@pytest.fixture(
    params=list(AITER_CONFIGS) if is_aiter_found_and_supported() else ["aiter_off"]
)
def aiter_config(request):
    with pytest.MonkeyPatch.context() as mp:
        for name, value in AITER_CONFIGS[request.param].items():
            mp.setenv(name, value)
        _refresh_aiter_envs()
        yield request.param
    _refresh_aiter_envs()
