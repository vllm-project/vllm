# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NUMA helpers for prefetch weight offloading.

The prefetch offloader allocates pinned CPU buffers and launches H2D copies
from one worker process per GPU. Binding that process to the CPU NUMA node
local to its GPU keeps pinned allocation and copy submission local.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def _parse_cpu_list(raw: str) -> list[int]:
    cpus: list[int] = []
    for part in raw.strip().split(","):
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            cpus.extend(range(int(start), int(end) + 1))
        else:
            cpus.append(int(part))
    return cpus


def _sysfs_pci_id(bus_id: str) -> str:
    domain, rest = bus_id.split(":", 1)
    return f"{domain[-4:]}:{rest}".lower()


def _visible_gpu_identifier(gpu_index: int) -> str:
    visible_devices = os.getenv("CUDA_VISIBLE_DEVICES")
    if not visible_devices:
        return str(gpu_index)

    identifiers = [item.strip() for item in visible_devices.split(",") if item.strip()]
    if gpu_index >= len(identifiers):
        return str(gpu_index)
    return identifiers[gpu_index]


def _query_gpu_pci_bus_ids() -> tuple[dict[int, str], dict[str, str]]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,uuid,pci.bus_id",
        "--format=csv,noheader",
    ]
    out = subprocess.check_output(cmd, text=True)
    by_index: dict[int, str] = {}
    by_uuid: dict[str, str] = {}
    for line in out.strip().splitlines():
        if not line.strip():
            continue
        idx_raw, uuid_raw, bus_raw = [item.strip() for item in line.split(",", 2)]
        by_index[int(idx_raw)] = bus_raw
        by_uuid[uuid_raw] = bus_raw
    return by_index, by_uuid


def _gpu_pci_bus_id(gpu_index: int) -> str | None:
    by_index, by_uuid = _query_gpu_pci_bus_ids()
    identifier = _visible_gpu_identifier(gpu_index)
    try:
        return by_index.get(int(identifier))
    except ValueError:
        return by_uuid.get(identifier)


def _gpu_numa_node_from_bus_id(bus_id: str) -> int | None:
    path = Path("/sys/bus/pci/devices") / _sysfs_pci_id(bus_id) / "numa_node"
    node = int(path.read_text().strip())
    return node if node >= 0 else None


def _numa_node_cpus(node: int) -> list[int]:
    cpulist = Path(f"/sys/devices/system/node/node{node}/cpulist")
    cpus = _parse_cpu_list(cpulist.read_text())
    allowed = os.sched_getaffinity(0)
    return [cpu for cpu in cpus if cpu in allowed]


def bind_process_to_gpu_numa(gpu_index: int) -> bool:
    """Bind the current process to CPUs local to ``gpu_index``.

    ``gpu_index`` is a CUDA-visible ordinal. It is translated through
    ``CUDA_VISIBLE_DEVICES`` before looking up the PCI bus id from
    ``nvidia-smi`` so remapped device lists bind to the correct NUMA node.

    Returns ``True`` when affinity was changed. Returns ``False`` when the GPU
    has no NUMA node or no CPUs from that node are allowed by the process cpuset.
    Other environment errors are left to the caller to catch and log.
    """
    bus_id = _gpu_pci_bus_id(gpu_index)
    if bus_id is None:
        return False
    node = _gpu_numa_node_from_bus_id(bus_id)
    if node is None:
        return False
    cpus = _numa_node_cpus(node)
    if not cpus:
        return False
    os.sched_setaffinity(0, cpus)
    return True
