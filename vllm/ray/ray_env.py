# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import os
import warnings

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

# ---------------------------------------------------------------------------
# Prefix-based env var propagation
# ---------------------------------------------------------------------------
# In addition to env vars registered in ``envs.environment_variables``
# (VLLM_* vars), we also copy any env var whose name starts with one of
# these prefixes.  This ensures third-party integrations (KV connectors,
# NCCL tuning knobs, etc.) propagate from the driver to Ray workers
# without requiring each integration to be hard-coded here.
#
# Users can extend this at deploy time by setting
# ``VLLM_RAY_ENV_VAR_PREFIXES_TO_COPY`` (comma-separated) on the driver.
DEFAULT_ENV_VAR_PREFIXES_TO_COPY: set[str] = {
    "VLLM_",
    "LMCACHE_",
    "NCCL_",
    "UCX_",
    "HF_",
    "HUGGING_FACE_",
}

# Individual env vars (without a common prefix) that should always be
# copied.  Users can extend via ``VLLM_RAY_EXTRA_ENV_VARS_TO_COPY``.
DEFAULT_EXTRA_ENV_VARS_TO_COPY: set[str] = {
    "PYTHONHASHSEED",
}


def _get_env_var_prefixes() -> set[str]:
    """Return the merged set of env var prefixes to copy.

    Combines ``DEFAULT_ENV_VAR_PREFIXES_TO_COPY`` with any user-supplied
    prefixes in ``VLLM_RAY_ENV_VAR_PREFIXES_TO_COPY`` (comma-separated).
    """
    prefixes = set(DEFAULT_ENV_VAR_PREFIXES_TO_COPY)
    extra = envs.VLLM_RAY_ENV_VAR_PREFIXES_TO_COPY
    if extra:
        for p in extra.split(","):
            p = p.strip()
            if p:
                prefixes.add(p)
    return prefixes


def _get_extra_env_vars() -> set[str]:
    """Return the merged set of individual env var names to copy.

    Combines ``DEFAULT_EXTRA_ENV_VARS_TO_COPY`` with any user-supplied
    names in ``VLLM_RAY_EXTRA_ENV_VARS_TO_COPY`` (comma-separated).
    """
    extra_vars = set(DEFAULT_EXTRA_ENV_VARS_TO_COPY)
    extra = envs.VLLM_RAY_EXTRA_ENV_VARS_TO_COPY
    if extra:
        for v in extra.split(","):
            v = v.strip()
            if v:
                extra_vars.add(v)
    return extra_vars


def _collect_prefix_matched_vars(prefixes: set[str]) -> set[str]:
    """Scan ``os.environ`` and return var names matching *prefixes*."""
    matched: set[str] = set()
    for name in os.environ:
        for prefix in prefixes:
            if name.startswith(prefix):
                matched.add(name)
                break
    return matched


def get_env_vars_to_copy(
    exclude_vars: set[str] | None = None,
    additional_vars: set[str] | None = None,
    destination: str | None = None,
) -> set[str]:
    """
    Get the environment variables to copy to downstream Ray actors.

    The result is the union of:
    1. Env vars registered in ``vllm.envs.environment_variables``.
    2. Env vars in ``os.environ`` whose name matches a prefix in
       ``DEFAULT_ENV_VAR_PREFIXES_TO_COPY`` (or user-configured via
       ``VLLM_RAY_ENV_VAR_PREFIXES_TO_COPY``).
    3. Individual vars listed in ``DEFAULT_EXTRA_ENV_VARS_TO_COPY``.

    Minus any vars in *exclude_vars* or ``RAY_NON_CARRY_OVER_ENV_VARS``.

    Example use cases:
    - Copy environment variables from RayDistributedExecutor to Ray workers.
    - Copy environment variables from RayDPClient to Ray DPEngineCoreActor.

    Args:
        exclude_vars: A set of environment variables to exclude from copying.
        additional_vars: **Deprecated.** Previously used to specify extra
            env vars to copy.  All non-VLLM env vars should now be covered
            by ``DEFAULT_ENV_VAR_PREFIXES_TO_COPY``,
            ``DEFAULT_EXTRA_ENV_VARS_TO_COPY``, or the
            ``VLLM_RAY_ENV_VAR_PREFIXES_TO_COPY`` /
            ``VLLM_RAY_EXTRA_ENV_VARS_TO_COPY`` env vars.  This parameter
            is ignored and will be removed in a future release.
        destination: The destination of the environment variables (for
            logging purposes only).
    Returns:
        A set of environment variable names to copy.
    """
    if additional_vars:
        warnings.warn(
            "The 'additional_vars' parameter of get_env_vars_to_copy() is "
            "deprecated and ignored. Add prefixes to "
            "DEFAULT_ENV_VAR_PREFIXES_TO_COPY in vllm/ray/ray_env.py or set "
            "VLLM_RAY_ENV_VAR_PREFIXES_TO_COPY / "
            "VLLM_RAY_EXTRA_ENV_VARS_TO_COPY on the driver. "
            "This parameter will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )

    exclude_vars = exclude_vars or set()

    # 1. vLLM's own registered env vars
    env_vars_to_copy = set(envs.environment_variables)

    # 2. Prefix-matched vars from os.environ
    prefixes = _get_env_var_prefixes()
    env_vars_to_copy |= _collect_prefix_matched_vars(prefixes)

    # 3. Individual extra vars
    env_vars_to_copy |= _get_extra_env_vars()

    # 4. Exclude worker-specific and user-blacklisted vars
    env_vars_to_copy -= exclude_vars
    env_vars_to_copy -= RAY_NON_CARRY_OVER_ENV_VARS

    to_destination = " to " + destination if destination is not None else ""

    logger.info(
        "RAY_NON_CARRY_OVER_ENV_VARS from config: %s", RAY_NON_CARRY_OVER_ENV_VARS
    )
    logger.info(
        "Env var prefixes to copy: %s", prefixes,
    )
    logger.info(
        "Copying the following environment variables%s: %s",
        to_destination,
        sorted(v for v in env_vars_to_copy if v in os.environ),
    )
    logger.info(
        "If certain env vars should NOT be copied, add them to %s file",
        RAY_NON_CARRY_OVER_ENV_VARS_FILE,
    )

    return env_vars_to_copy
