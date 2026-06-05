# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mooncake requester config helpers."""

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)

_SYSFS_INFINIBAND_ROOT = Path("/sys/class/infiniband")
_DEFAULT_MOONCAKE_PKEY_INDEX = 0


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


def _normalize_explicit_worker_rnics(device_list: str) -> str:
    entries = [entry.strip() for entry in device_list.split(",")]
    if any(not entry for entry in entries):
        raise ValueError(
            "Mooncake worker device_name contains an empty RDMA device entry"
        )
    return ",".join(entries)


def _read_sysfs_text(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def _get_mooncake_pkey_index() -> int:
    raw_index = os.environ.get("MC_PKEY_INDEX")
    if raw_index is None:
        return _DEFAULT_MOONCAKE_PKEY_INDEX
    try:
        pkey_index = int(raw_index)
    except ValueError:
        return _DEFAULT_MOONCAKE_PKEY_INDEX
    if not 0 <= pkey_index <= 65535:
        return _DEFAULT_MOONCAKE_PKEY_INDEX
    return pkey_index


def _has_compatible_pkey(port_path: Path, link_layer: str) -> bool:
    if link_layer != "InfiniBand":
        return True

    pkey = _read_sysfs_text(port_path / "pkeys" / str(_get_mooncake_pkey_index()))
    if pkey is None:
        return False
    try:
        pkey_value = int(pkey, 16)
    except ValueError:
        return False

    # 0x7FFF: check bits 0-14 for valid partition number
    # 0x8000: check bit 15 for membership type (full vs limited)
    # Limited P_Keys can fail peer-to-peer transfers
    return bool(pkey_value & 0x8000) and bool(pkey_value & 0x7FFF)


def _is_compatible_port(port_path: Path) -> bool:
    link_layer = _read_sysfs_text(port_path / "link_layer")
    state = _read_sysfs_text(port_path / "state")

    if link_layer not in {"InfiniBand", "Ethernet"}:
        return False

    if state != "4: ACTIVE":
        return False
    return _has_compatible_pkey(port_path, link_layer)


def _is_compatible_rdma_device(device_path: Path) -> bool:
    # node_type must be CA (Channel Adapter)
    if _read_sysfs_text(device_path / "node_type") != "1: CA":
        return False

    # Inspect every port: a device is usable if any of its ports is ACTIVE with
    # a compatible link layer and P_Key.
    try:
        port_paths = sorted(
            path for path in (device_path / "ports").iterdir() if path.is_dir()
        )
    except OSError:
        return False

    return any(_is_compatible_port(port_path) for port_path in port_paths)


def _get_auto_discovered_worker_rnics() -> str:
    try:
        device_paths = sorted(
            path
            for path in _SYSFS_INFINIBAND_ROOT.iterdir()
            if path.is_dir() or path.is_symlink()
        )
    except OSError:
        return ""

    device_names = [
        path.name for path in device_paths if _is_compatible_rdma_device(path)
    ]
    return ",".join(device_names)


def get_configured_worker_rnic(
    *,
    protocol: str,
    configured_device: str,
) -> str:
    normalized_device = normalize_string_override(configured_device)
    if normalized_device is not None:
        return _normalize_explicit_worker_rnics(normalized_device)

    if protocol not in {"rdma", "efa"}:
        return ""

    if protocol == "rdma":
        logger.info(
            "No RDMA devices specified for Mooncake backend (protocol=%s). "
            "Running automatic device selection."
        )
        discovered_devices = _get_auto_discovered_worker_rnics()
        if discovered_devices:
            logger.info(
                "Mooncake auto-selected compatible RDMA devices for protocol=%s: %s",
                protocol,
                discovered_devices,
            )
            return discovered_devices

        logger.warning(
            "No compatible RDMA devices were discovered in /sys/class/infiniband. "
            "Falling back to Mooncake's built-in auto-selection.",
            protocol,
        )
        return ""

    logger.warning(
        "No devices specified for Mooncake backend (protocol=%s); falling back "
        "to Mooncake's built-in auto-selection.",
        protocol,
    )
    return ""
