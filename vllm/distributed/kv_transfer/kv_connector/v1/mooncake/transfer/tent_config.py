# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Mapping
from typing import Any

SUPPORTED_PROTOCOLS = ("rdma", "tcp")

# TENT installs these by itself depending on how it was built, so a deployment
# that asked for one transport would still serve the others. NVLink stays
# available next to RDMA and is picked for same-host peers.
DISABLED_TRANSPORTS = ("gds", "hp_tcp", "io_uring", "mnnvl", "shm")


def build_tent_config(
    *,
    hostname: str,
    protocol: str,
    device_name: str,
    enable_sleep_mode: bool = False,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Builds the TENT config vLLM owns.

    Only the selected transports are installed, so the engine cannot answer on a
    network the peers did not agree on. NVLink joins RDMA for same-host peers.
    """
    if protocol not in SUPPORTED_PROTOCOLS:
        raise RuntimeError(
            "The TENT transfer engine supports "
            f"{' and '.join(SUPPORTED_PROTOCOLS)}, got protocol {protocol!r}."
        )
    disabled = set(DISABLED_TRANSPORTS)
    if protocol != "rdma" or enable_sleep_mode:
        # NVLink is only used with RDMA, and sleep mode disables it until
        # kvcache-ai/Mooncake#5886 preserves sub-allocated buffer ranges.
        disabled.add("nvlink")
    config: dict[str, Any] = {
        # Peers are addressed through the connector, so no metadata service.
        "metadata_type": "p2p",
        "rpc_server_hostname": hostname,
        "transports": {
            **{name: {"enable": False} for name in disabled},
            "tcp": {"enable": protocol == "tcp"},
            "rdma": {"enable": protocol == "rdma"},
        },
        "metrics": {"enabled": False, "http_port": 0},
    }
    devices = [name.strip() for name in device_name.split(",") if name.strip()]
    if devices:
        config["topology"] = {"rdma_whitelist": devices}
    if overrides is not None and not isinstance(overrides, Mapping):
        raise RuntimeError(
            f"tent_config must be a mapping of TENT settings, got {type(overrides)}."
        )
    return merge_tent_config(config, overrides or {})


def merge_tent_config(
    base: Mapping[str, Any], overrides: Mapping[str, Any]
) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = merge_tent_config(current, value)
        else:
            merged[key] = value
    return merged
