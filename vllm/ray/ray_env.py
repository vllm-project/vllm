# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import os

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)

CONFIG_HOME = envs.VLLM_CONFIG_ROOT

# This file contains a list of env vars that should not be copied
# from the driver to the Ray workers.
RAY_NON_CARRY_OVER_ENV_VARS_FILE = os.path.join(
    CONFIG_HOME, "ray_non_carry_over_env_vars.json"
)

try:
    if os.path.exists(RAY_NON_CARRY_OVER_ENV_VARS_FILE):
        with open(RAY_NON_CARRY_OVER_ENV_VARS_FILE) as f:
            RAY_NON_CARRY_OVER_ENV_VARS = set(json.load(f))
    else:
        RAY_NON_CARRY_OVER_ENV_VARS = set()
except json.JSONDecodeError:
    logger.warning(
        "Failed to parse %s. Using an empty set for non-carry-over env vars.",
        RAY_NON_CARRY_OVER_ENV_VARS_FILE,
    )
    RAY_NON_CARRY_OVER_ENV_VARS = set()


def get_env_vars_to_copy(
    exclude_vars: set[str] | None = None,
    additional_vars: set[str] | None = None,
    destination: str | None = None,
) -> set[str]:
    """
    Get the environment variables to copy to downstream Ray actors.

    Example use cases:
    - Copy environment variables from RayDistributedExecutor to Ray workers.
    - Copy environment variables from RayDPClient to Ray DPEngineCoreActor.

    Args:
        exclude_vars: A set of vllm defined environment variables to exclude
            from copying.
        additional_vars: A set of additional environment variables to copy.
            If a variable is in both exclude_vars and additional_vars, it will
            be excluded.
        destination: The destination of the environment variables.
    Returns:
        A set of environment variables to copy.
    """
    exclude_vars = exclude_vars or set()
    additional_vars = additional_vars or set()

    env_vars_to_copy = {
        v
        for v in set(envs.environment_variables).union(additional_vars)
        if v not in exclude_vars and v not in RAY_NON_CARRY_OVER_ENV_VARS
    }

    to_destination = " to " + destination if destination is not None else ""

    logger.info(
        "RAY_NON_CARRY_OVER_ENV_VARS from config: %s", RAY_NON_CARRY_OVER_ENV_VARS
    )
    logger.info(
        "Copying the following environment variables%s: %s",
        to_destination,
        [v for v in env_vars_to_copy if v in os.environ],
    )
    logger.info(
        "If certain env vars should NOT be copied, add them to %s file",
        RAY_NON_CARRY_OVER_ENV_VARS_FILE,
    )

    return env_vars_to_copy
