# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mooncake requester config helpers."""

import ipaddress
import os
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)

_SYSFS_INFINIBAND_ROOT = Path("/sys/class/infiniband")
_PROC_NET_ROUTE = Path("/proc/net/route")
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


def _ipv4_from_gid(gid: str) -> ipaddress.IPv4Address | None:
    """Extract IPv4 from an IPv4-mapped RoCE GID."""
    try:
        ip = ipaddress.IPv6Address(gid)
    except ValueError:
        return None
    return ip.ipv4_mapped


def _first_netdev_name(device_path: Path) -> str | None:
    """Return the first Linux netdev backing an RDMA device."""
    try:
        netdevs = sorted(path.name for path in (device_path / "device/net").iterdir())
    except OSError:
        return None
    return netdevs[0] if netdevs else None


def _first_roce_v2_ipv4_gid(device_path: Path) -> ipaddress.IPv4Address | None:
    """Return the first non-link-local RoCE v2 IPv4 GID for an RDMA device."""
    try:
        port_paths = sorted(
            path for path in (device_path / "ports").iterdir() if path.is_dir()
        )
    except OSError:
        return None

    for port_path in port_paths:
        try:
            gid_paths = sorted(
                path for path in (port_path / "gids").iterdir() if path.name.isdigit()
            )
        except OSError:
            continue
        for gid_path in gid_paths:
            gid_type = _read_sysfs_text(
                port_path / "gid_attrs" / "types" / gid_path.name
            )
            if gid_type != "RoCE v2":
                continue
            gid_ip = _ipv4_from_gid(_read_sysfs_text(gid_path) or "")
            if gid_ip is not None and not gid_ip.is_link_local:
                return gid_ip
    return None


def _ipv4_from_proc_route_hex(value: str) -> ipaddress.IPv4Address | None:
    """Decode /proc/net/route IPv4 hex, e.g. 000010AC -> 172.16.0.0."""
    try:
        return ipaddress.IPv4Address(int.from_bytes(bytes.fromhex(value), "little"))
    except ValueError:
        return None


def _read_ipv4_routes() -> dict[str, list[ipaddress.IPv4Network]]:
    """Read non-default IPv4 routes from /proc/net/route grouped by netdev."""
    try:
        lines = _PROC_NET_ROUTE.read_text().splitlines()
    except OSError:
        return {}

    # Map each Linux netdev to its non-default IPv4 routes, e.g.
    # {"spx_p1n1": [IPv4Network("172.16.0.0/13")]}.
    routes: dict[str, list[ipaddress.IPv4Network]] = defaultdict(list)
    for line in lines[1:]:
        fields = line.split()
        if len(fields) < 8:
            continue
        iface = fields[0]
        dest_hex = fields[1]
        flags_hex = fields[3]
        mask_hex = fields[7]
        try:
            flags = int(flags_hex, 16)
        except ValueError:
            continue
        # ensure the route is active
        if not flags & 0x1:
            continue
        dest = _ipv4_from_proc_route_hex(dest_hex)
        mask = _ipv4_from_proc_route_hex(mask_hex)
        if dest is None or mask is None:
            continue
        try:
            network = ipaddress.IPv4Network(f"{dest}/{mask}", strict=False)
        except ValueError:
            continue
        if network.prefixlen == 0:
            continue
        routes[iface].append(network)
    return routes


def _route_group_for_device(
    device_path: Path,
    routes_by_netdev: Mapping[str, list[ipaddress.IPv4Network]],
) -> str | None:
    """Return the broadest route containing the device's RoCE v2 GID IP."""
    netdev = _first_netdev_name(device_path)
    gid_ip = _first_roce_v2_ipv4_gid(device_path)
    if netdev is None or gid_ip is None:
        return None

    candidates = [
        route for route in routes_by_netdev.get(netdev, []) if gid_ip in route
    ]
    if not candidates:
        return None

    # The least-specific non-default route captures the routed rail family. On
    # split-rail RoCE hosts, direct host/link routes differ per NIC while the
    # broad rail route is shared by mutually reachable NICs.
    return str(min(candidates, key=lambda route: route.prefixlen))


def _filter_to_largest_routable_rnic_group(
    device_names: list[str],
    route_groups: Mapping[str, str],
) -> list[str]:
    """Choose the largest routable RNIC group for RDMA device auto discovery."""
    if not device_names:
        return []

    # Group devices by routed rail.
    by_route_group: dict[str, list[str]] = defaultdict(list)
    for name in device_names:
        route_group = route_groups.get(name)
        if route_group is not None:
            by_route_group[route_group].append(name)
    if not by_route_group:
        return sorted(device_names)

    # Pick the largest routable rail group, with a deterministic tie-break.
    selected = min(
        by_route_group.values(),
        key=lambda names: (-len(names), sorted(names)[0]),
    )
    return sorted(selected)


def _get_auto_discovered_worker_rnics() -> str:
    try:
        device_paths = sorted(
            path
            for path in _SYSFS_INFINIBAND_ROOT.iterdir()
            if path.is_dir() or path.is_symlink()
        )
    except OSError:
        return ""

    compatible = [path for path in device_paths if _is_compatible_rdma_device(path)]
    device_names = [path.name for path in compatible]
    routes_by_netdev = _read_ipv4_routes()
    route_groups: dict[str, str] = {}
    for path in compatible:
        route_group = _route_group_for_device(path, routes_by_netdev)
        if route_group is not None:
            route_groups[path.name] = route_group

    selected = _filter_to_largest_routable_rnic_group(device_names, route_groups)
    return ",".join(selected)


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
            "Running automatic device selection.",
            protocol,
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
        )
        return ""

    logger.warning(
        "No devices specified for Mooncake backend (protocol=%s); falling back "
        "to Mooncake's built-in auto-selection.",
        protocol,
    )
    return ""
