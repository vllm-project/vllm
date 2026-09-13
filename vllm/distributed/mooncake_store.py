# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared configuration and setup for EC and KV Mooncake Store clients."""

import json
import os
from dataclasses import dataclass
from typing import Any, Literal

import regex as re

DEFAULT_GLOBAL_SEGMENT_SIZE = 4 * 1024 * 1024 * 1024  # 4 GiB
DEFAULT_LOCAL_BUFFER_SIZE = 4 * 1024 * 1024 * 1024  # 4 GiB
DEFAULT_TENANT_ID = "default"


MooncakeMode = Literal["embedded", "standalone-store"]


@dataclass
class MooncakeStoreConfig:
    """Configuration for MooncakeDistributedStore.

    ``mode`` selects the topology: ``embedded`` (each rank contributes
    ``global_segment_size`` in-process) or ``standalone-store`` (rank
    contributes 0; an external ``mooncake_client`` process owns the pool
    and the SSD tier).
    """

    metadata_server: str
    master_server_address: str
    protocol: str
    device_name: str
    mode: MooncakeMode = "embedded"
    global_segment_size: int = DEFAULT_GLOBAL_SEGMENT_SIZE
    local_buffer_size: int = DEFAULT_LOCAL_BUFFER_SIZE
    enable_offload: bool = False
    tenant_id: str = DEFAULT_TENANT_ID

    def __post_init__(self) -> None:
        if self.mode not in ("embedded", "standalone-store"):
            raise ValueError(f"unknown Mooncake mode: {self.mode!r}")
        if self.local_buffer_size <= 0:
            raise ValueError("local_buffer_size must be > 0")
        if self.mode == "embedded" and self.global_segment_size == 0:
            raise ValueError("embedded mode requires global_segment_size > 0")
        if self.mode == "standalone-store" and self.global_segment_size != 0:
            raise ValueError("standalone-store mode requires global_segment_size == 0")

    @staticmethod
    def from_file(file_path: str) -> "MooncakeStoreConfig":
        with open(file_path) as file:
            config = json.load(file)
        return MooncakeStoreConfig(
            metadata_server=config.get("metadata_server", ""),
            master_server_address=config.get("master_server_address", ""),
            protocol=config.get("protocol", "rdma"),
            device_name=config.get("device_name", ""),
            mode=config.get("mode", "embedded"),
            global_segment_size=_parse_size(
                config.get("global_segment_size", DEFAULT_GLOBAL_SEGMENT_SIZE)
            ),
            local_buffer_size=_parse_size(
                config.get("local_buffer_size", DEFAULT_LOCAL_BUFFER_SIZE)
            ),
            enable_offload=bool(config.get("enable_offload", False)),
            tenant_id=_normalize_tenant_id(config.get("tenant_id", DEFAULT_TENANT_ID)),
        )

    @staticmethod
    def load_from_config() -> "MooncakeStoreConfig":
        config_path = os.getenv("MOONCAKE_CONFIG_PATH")
        if not config_path:
            raise ValueError(
                "The environment variable 'MOONCAKE_CONFIG_PATH' is not set."
            )
        return MooncakeStoreConfig.from_file(config_path)


def _normalize_tenant_id(value: Any) -> str:
    if value is None:
        return DEFAULT_TENANT_ID
    if not isinstance(value, str):
        raise TypeError(
            f"tenant_id must be a string or null, got {type(value).__name__}: {value!r}"
        )
    tenant_id = value.strip()
    return tenant_id if tenant_id else DEFAULT_TENANT_ID


def _parse_size(value: Any) -> int:
    """Parse storage size strings with units: GB, MB, KB, B."""
    if isinstance(value, int):
        return value
    if not isinstance(value, str):
        try:
            return int(value)
        except (TypeError, ValueError) as e:
            raise TypeError(f"Unsupported type for size: {type(value)}") from e

    cleaned = value.strip().lower()
    if not cleaned:
        raise ValueError("Size cannot be empty.")

    unit_multipliers = {
        "gb": 1024**3,
        "mb": 1024**2,
        "kb": 1024,
        "b": 1,
    }
    match = re.match(r"^\s*([\d.]+)\s*(gb|mb|kb|b)?\s*$", cleaned)
    if not match:
        raise ValueError(f"Invalid format: '{value}'")

    number_str = match.group(1)
    unit = match.group(2) or "b"
    multiplier = unit_multipliers[unit]

    try:
        numeric_value = float(number_str)
    except ValueError as exc:
        raise ValueError(f"Invalid numeric value '{number_str}' in: '{value}'") from exc
    return int(numeric_value * multiplier)


def setup_mooncake_store(
    store: Any, config: MooncakeStoreConfig, local_hostname: str
) -> None:
    """Initialize a native Store, preserving explicit tenant configuration."""
    setup_kwargs: dict[str, str] = {}
    if config.tenant_id != DEFAULT_TENANT_ID:
        setup_kwargs["tenant_id"] = config.tenant_id
    ret = store.setup(
        local_hostname,
        config.metadata_server,
        config.global_segment_size,
        config.local_buffer_size,
        config.protocol,
        config.device_name,
        config.master_server_address,
        **setup_kwargs,
    )
    if ret != 0:
        raise RuntimeError("Initialize MooncakeDistributedStore failed.")
