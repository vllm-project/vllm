# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

AITER_CONFIGS = {
    "aiter_off": {"VLLM_ROCM_USE_AITER": "0"},
    "aiter_on": {"VLLM_ROCM_USE_AITER": "1", "VLLM_ROCM_USE_AITER_RMSNORM": "1"},
    "aiter_on_rmsnorm_off": {
        "VLLM_ROCM_USE_AITER": "1",
        "VLLM_ROCM_USE_AITER_RMSNORM": "0",
    },
}


def _refresh_envs():
    import vllm.envs as envs
    from vllm._aiter_ops import rocm_aiter_ops

    # Env values are cached after the first engine starts in this process.
    envs.disable_envs_cache()
    rocm_aiter_ops.refresh_env_variables()


@pytest.fixture(params=list(AITER_CONFIGS))
def aiter_config(request, monkeypatch):
    """Run the test once per AITER setting; AITER-on variants are ROCm-only."""
    from vllm._aiter_ops import is_aiter_found_and_supported

    env = AITER_CONFIGS[request.param]
    if env["VLLM_ROCM_USE_AITER"] == "1" and not is_aiter_found_and_supported():
        pytest.skip("AITER is not available on this platform")

    for name, value in env.items():
        monkeypatch.setenv(name, value)
    _refresh_envs()
    yield request.param
    monkeypatch.undo()
    _refresh_envs()
