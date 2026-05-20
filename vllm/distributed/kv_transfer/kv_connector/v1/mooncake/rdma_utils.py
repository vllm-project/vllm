# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mooncake requester config helpers."""

from collections.abc import Mapping
from typing import Any

import torch

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)


def normalize_string_override(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def get_current_physical_gpu_index() -> int | None:
    try:
        from vllm.platforms import current_platform
    except ImportError:
        return None

    try:
        device_index = torch.accelerator.current_device_index()
        physical_device_id = current_platform.device_id_to_physical_device_id(
            device_index
        )
        return int(physical_device_id)
    except Exception:
        return None


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


def _get_explicit_worker_rnic(device_list: str) -> str:
    entries = [entry.strip() for entry in device_list.split(",")]
    if any(not entry for entry in entries):
        raise ValueError(
            "Mooncake worker device_name contains an empty RDMA device entry"
        )
    if len(entries) == 1:
        return entries[0]

    gpu_index = get_current_physical_gpu_index()
    if gpu_index is None:
        raise RuntimeError(
            "Mooncake RDMA requester could not determine the local physical GPU index"
        )
    if gpu_index >= len(entries):
        raise ValueError(
            "Mooncake worker device list does not cover local GPU "
            f"{gpu_index}: {device_list}"
        )
    device_name = entries[gpu_index]
    logger.info(
        "Mooncake selected worker RNIC %s from explicit device list for local GPU %s",
        device_name,
        gpu_index,
    )
    return device_name


def get_configured_worker_rnic(
    *,
    protocol: str,
    configured_device: str,
) -> str:
    normalized_device = normalize_string_override(configured_device)
    if normalized_device is not None:
        return _get_explicit_worker_rnic(normalized_device)

    if protocol not in {"rdma", "efa"}:
        return ""

    logger.warning(
        "No RDMA devices specified for Mooncake backend (protocol=%s). "
        "Set 'device_name' in mooncake_config.json to a single RNIC name "
        "or a comma-separated CSV indexed by physical GPU; falling back to "
        "Mooncake's built-in auto-selection, which may converge on the same "
        "NIC across all DP ranks and saturate bandwidth.",
        protocol,
    )
    return ""
