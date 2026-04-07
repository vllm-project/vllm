# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os

from vllm.ray.ray_env import RAY_NON_CARRY_OVER_ENV_VARS


def get_driver_env_vars(
    worker_specific_vars: set[str],
) -> dict[str, str]:
    """Return driver env vars to propagate to Ray workers.

    Returns everything from ``os.environ`` except ``worker_specific_vars``
    and user-configured exclusions (``RAY_NON_CARRY_OVER_ENV_VARS``).
    """
    exclude_vars = worker_specific_vars | RAY_NON_CARRY_OVER_ENV_VARS

    return {key: value for key, value in os.environ.items() if key not in exclude_vars}
