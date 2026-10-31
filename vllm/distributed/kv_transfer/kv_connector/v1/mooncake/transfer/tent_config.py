# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Mapping
from typing import Any

TENT_METRICS_BASE_PORT = 9100

SUPPORTED_PROTOCOLS = ("rdma", "tcp")


def build_tent_config(
    *,
    hostname: str,
    protocol: str,
    device_name: str,
    metrics_port: int,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Builds the TENT config vLLM owns.

    Only the requested transport is installed, so the engine cannot answer on a
    network the peers did not agree on.
    """
    if protocol not in SUPPORTED_PROTOCOLS:
        raise RuntimeError(
            "The TENT transfer engine supports "
            f"{' and '.join(SUPPORTED_PROTOCOLS)}, got protocol {protocol!r}."
        )
    config: dict[str, Any] = {
        # Peers are addressed through the connector, so no metadata service.
        "metadata_type": "p2p",
        "rpc_server_hostname": hostname,
        # RDMA is installed by default, so the unused transport is switched off.
        "transports": {
            "tcp": {"enable": protocol == "tcp"},
            "rdma": {"enable": protocol == "rdma"},
        },
        "metrics": {"enabled": metrics_port > 0, "http_port": metrics_port},
    }
    devices = [name for name in device_name.split(",") if name]
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
