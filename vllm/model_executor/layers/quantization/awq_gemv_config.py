# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Configuration loading for AWQ GEMV split-k selection on ROCm.

The AWQ GEMV HIP kernel uses split-k parallelism to improve occupancy
for small N dimensions. The optimal split-k value depends on N (output
dimension) and the target device. This module loads per-device config
files that specify N-threshold rules for split-k selection.

Config file format (JSON):
{
    "rules": [
        [N_threshold, split_k],   // first rule where N <= threshold wins
        ...
    ],
    "default_split_k": <int>      // used when N exceeds all thresholds
}

Config files are stored in awq_gemv_configs/ directory, named by device:
    device_name=Radeon_8060S_Graphics.json
"""

import functools
import json
import os
from typing import Any

from vllm import envs
from vllm.logger import init_logger

logger = init_logger(__name__)


def _get_config_file_name() -> str:
    """Construct the config file name for the current device."""
    from vllm.platforms import current_platform

    device_name = current_platform.get_device_name().replace(" ", "_")
    return f"device_name={device_name}.json"


@functools.lru_cache
def get_awq_gemv_config() -> dict[str, Any] | None:
    """Load the AWQ GEMV split-k config for the current device.

    Checks (in order):
    1. User-defined folder via VLLM_TUNED_CONFIG_FOLDER env var
    2. Built-in awq_gemv_configs/ directory

    Returns:
        Config dict with "rules" and "default_split_k", or None if not found.
    """
    json_file_name = _get_config_file_name()
    config_file_paths: list[str] = []

    # User-defined config takes priority
    user_defined_config_folder = envs.VLLM_TUNED_CONFIG_FOLDER
    if user_defined_config_folder is not None:
        config_file_paths.append(
            os.path.join(user_defined_config_folder, json_file_name)
        )

    # Built-in configs
    default_config_dir = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "awq_gemv_configs"
    )
    config_file_paths.append(os.path.join(default_config_dir, json_file_name))

    for config_file_path in config_file_paths:
        if os.path.exists(config_file_path):
            with open(config_file_path) as f:
                logger.info(
                    "Using AWQ GEMV split-k config from %s",
                    config_file_path,
                )
                config = json.load(f)
                # Validate structure
                if "rules" not in config or "default_split_k" not in config:
                    logger.warning(
                        "Invalid AWQ GEMV config (missing 'rules' or "
                        "'default_split_k'): %s",
                        config_file_path,
                    )
                    continue
                return config

    logger.info(
        "No AWQ GEMV split-k config found for device. "
        "Using default heuristic. Config file not found at: %s",
        ", ".join(config_file_paths),
    )
    return None


def _default_split_k_heuristic(N: int) -> int:
    """Default split-k heuristic matching the original C++ logic.

    This is the fallback when no config file exists for the device.
    """
    if N <= 4096:
        return 16
    elif N <= 16384:
        return 8
    else:
        return 4


def get_awq_gemv_split_k(N: int) -> int:
    """Get the optimal split-k value for AWQ GEMV given output dimension N.

    Checks (in priority order):
    1. AWQ_GEMV_SPLIT_K environment variable (for manual tuning)
    2. Device-specific config file (threshold rules)
    3. Default heuristic

    Args:
        N: Output dimension of the GEMV operation.

    Returns:
        The split-k value to use (1, 2, 4, 8, or 16).
    """
    # Priority 1: Environment variable override (for tuning sweeps)
    env_split_k = os.environ.get("AWQ_GEMV_SPLIT_K")
    if env_split_k is not None:
        return int(env_split_k)

    # Priority 2: Device config file
    config = get_awq_gemv_config()
    if config is not None:
        rules = config["rules"]
        default = config["default_split_k"]
        for n_threshold, split_k in rules:
            if n_threshold >= N:
                return split_k
        return default

    # Priority 3: Default heuristic
    return _default_split_k_heuristic(N)
