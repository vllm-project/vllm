# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import os

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)

CONFIG_HOME = envs.VLLM_CONFIG_ROOT

# Env vars that should NOT be copied from the driver to Ray workers.
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
# Built-in defaults for env var propagation.
# Users can add more via VLLM_RAY_EXTRA_ENV_VAR_PREFIXES_TO_COPY and
# VLLM_RAY_EXTRA_ENV_VARS_TO_COPY (additive, not replacing).
# ---------------------------------------------------------------------------
DEFAULT_ENV_VAR_PREFIXES: set[str] = {
    "VLLM_",
    "LMCACHE_",
    "NCCL_",
    "UCX_",
    "HF_",
    "HUGGING_FACE_",
}

DEFAULT_EXTRA_ENV_VARS: set[str] = {
    "PYTHONHASHSEED",
}


def _parse_csv(value: str) -> set[str]:
    """Split a comma-separated string into a set of stripped, non-empty tokens."""
    return {tok.strip() for tok in value.split(",") if tok.strip()}


def get_env_vars_to_copy(
    exclude_vars: set[str] | None = None,
    additional_vars: set[str] | None = None,
    destination: str | None = None,
) -> set[str]:
    """Return the env var names to copy from the driver to Ray actors.

    The result is the union of:

    1. Env vars registered in ``vllm.envs.environment_variables``.
    2. Env vars in ``os.environ`` matching a prefix in
       ``DEFAULT_ENV_VAR_PREFIXES`` + ``VLLM_RAY_EXTRA_ENV_VAR_PREFIXES_TO_COPY``.
    3. Individual names in ``DEFAULT_EXTRA_ENV_VARS`` +
       ``VLLM_RAY_EXTRA_ENV_VARS_TO_COPY``.
    4. Caller-supplied *additional_vars* (e.g. platform-specific).

    Minus any names in *exclude_vars* or ``RAY_NON_CARRY_OVER_ENV_VARS``.

    Args:
        exclude_vars: Env vars to exclude (e.g. worker-specific ones).
        additional_vars: Extra individual env var names to copy.  Useful
            for caller-specific vars (e.g. platform env vars).
        destination: Label used in log messages only.
    """
    exclude = (exclude_vars or set()) | RAY_NON_CARRY_OVER_ENV_VARS

    # -- prefixes (built-in + user-supplied, additive) ----------------------
    prefixes = DEFAULT_ENV_VAR_PREFIXES | _parse_csv(
        envs.VLLM_RAY_EXTRA_ENV_VAR_PREFIXES_TO_COPY
    )

    # -- collect env var names ----------------------------------------------
    # 1. vLLM's registered env vars
    result = set(envs.environment_variables)
    # 2. Prefix-matched vars present in the current environment
    result |= {name for name in os.environ if any(name.startswith(p) for p in prefixes)}
    # 3. Individual extra vars (built-in + user-supplied, additive)
    result |= DEFAULT_EXTRA_ENV_VARS | _parse_csv(envs.VLLM_RAY_EXTRA_ENV_VARS_TO_COPY)
    # 4. Caller-supplied extra vars (e.g. platform-specific)
    result |= additional_vars or set()
    # 5. Exclude worker-specific and user-blacklisted vars
    result -= exclude

    # -- logging ------------------------------------------------------------
    dest = f" to {destination}" if destination else ""
    logger.info("Env var prefixes to copy: %s", sorted(prefixes))
    logger.info(
        "Copying the following environment variables%s: %s",
        dest,
        sorted(v for v in result if v in os.environ),
    )
    if RAY_NON_CARRY_OVER_ENV_VARS:
        logger.info(
            "RAY_NON_CARRY_OVER_ENV_VARS from config: %s",
            RAY_NON_CARRY_OVER_ENV_VARS,
        )
    logger.info(
        "To exclude env vars from copying, add them to %s",
        RAY_NON_CARRY_OVER_ENV_VARS_FILE,
    )

    return result
