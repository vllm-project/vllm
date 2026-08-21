# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os

from vllm.ray.ray_env import RAY_NON_CARRY_OVER_ENV_VARS

_BREAKABLE_CUDAGRAPH_ENV_VAR = "VLLM_USE_BREAKABLE_CUDAGRAPH"


def get_driver_env_vars(
    worker_specific_vars: set[str],
) -> dict[str, str]:
    """Return driver env vars to propagate to Ray workers.

    Returns everything from ``os.environ`` except ``worker_specific_vars``
    and user-configured exclusions (``RAY_NON_CARRY_OVER_ENV_VARS``).
    """
    exclude_vars = worker_specific_vars | RAY_NON_CARRY_OVER_ENV_VARS

    return {key: value for key, value in os.environ.items() if key not in exclude_vars}


def update_runtime_env_for_breakable_cudagraph(runtime_env: dict) -> dict:
    """Expose the effective graph setting before Ray actor import.

    VllmConfig can auto-enable it after Ray starts, while
    eager_break_during_capture resolves it during model import.
    """
    if _BREAKABLE_CUDAGRAPH_ENV_VAR in os.environ:
        env_vars = runtime_env.setdefault("env_vars", {})
        env_vars[_BREAKABLE_CUDAGRAPH_ENV_VAR] = os.environ[
            _BREAKABLE_CUDAGRAPH_ENV_VAR
        ]
    return runtime_env
