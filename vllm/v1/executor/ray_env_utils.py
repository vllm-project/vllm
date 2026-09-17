# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os

from vllm.ray.ray_env import RAY_NON_CARRY_OVER_ENV_VARS

_RAY_WORKER_PRE_IMPORT_ENV_VARS = frozenset({"VLLM_USE_BREAKABLE_CUDAGRAPH"})


def get_driver_env_vars(
    worker_specific_vars: set[str],
) -> dict[str, str]:
    """Return driver env vars to propagate to Ray workers.

    Returns everything from ``os.environ`` except ``worker_specific_vars``
    and user-configured exclusions (``RAY_NON_CARRY_OVER_ENV_VARS``).
    """
    exclude_vars = worker_specific_vars | RAY_NON_CARRY_OVER_ENV_VARS

    return {key: value for key, value in os.environ.items() if key not in exclude_vars}


def update_runtime_env_for_worker_import(runtime_env: dict) -> dict:
    """Expose driver environment variables needed before Ray actor import.

    VllmConfig can auto-enable breakable CUDA graphs after Ray starts, while
    eager_break_during_capture resolves the setting during model import.
    """
    for env_var in _RAY_WORKER_PRE_IMPORT_ENV_VARS:
        if env_var in os.environ:
            env_vars = runtime_env.setdefault("env_vars", {})
            env_vars[env_var] = os.environ[env_var]
    return runtime_env
