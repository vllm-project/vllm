# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mooncake requester config helpers."""

from collections.abc import Mapping
from typing import Any

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)


def normalize_string_override(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def get_requester_local_hostname(local_ip: str) -> str:
    override = normalize_string_override(envs.MOONCAKE_REQUESTER_LOCAL_HOSTNAME)
    if override is not None:
        return override
    return local_ip


def get_configured_preferred_segment(
    extra_config: Mapping[str, Any],
) -> str | None:
    preferred_segment = normalize_string_override(extra_config.get("preferred_segment"))
    if preferred_segment is not None:
        return preferred_segment
    if extra_config.get("preferred_segment") is not None:
        raise ValueError(
            "Mooncake preferred_segment override must be a non-empty string"
        )

    env_value = normalize_string_override(envs.MOONCAKE_PREFERRED_SEGMENT)
    if env_value is not None:
        logger.info(
            "Mooncake preferred_segment from MOONCAKE_PREFERRED_SEGMENT: %s",
            env_value,
        )
        return env_value
    return None
