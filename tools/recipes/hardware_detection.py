# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CPU hardware discovery for optional vLLM Recipes runtime tuning.

Keep this module focused on facts exposed to the current process/container.
Tuning decisions belong in runtime_tuning.py.
"""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path

from vllm.utils.cpu_resource_utils import (
    get_allowed_cpu_list,
    get_memory_affinity,
    get_memory_node_info,
)


@dataclass(frozen=True)
class NumaNodeInfo:
    node_id: int
    logical_cpu_ids: tuple[int, ...]
    physical_core_count: int
    total_memory_bytes: int
    available_memory_bytes: int


@dataclass(frozen=True)
class HardwareInfo:
    architecture: str
    socket_count: int | None
    numa_nodes: tuple[NumaNodeInfo, ...]
    allowed_logical_cpu_count: int
    physical_core_count: int

    @property
    def numa_node_count(self) -> int:
        return len(self.numa_nodes)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _read_socket_id(cpu_id: int) -> int | None:
    path = Path(f"/sys/devices/system/cpu/cpu{cpu_id}/topology/physical_package_id")
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def detect_hardware() -> HardwareInfo:
    """Return CPU resources effectively available to this process.

    vLLM's CPU helpers already honor process CPU affinity and cgroup-aware
    memory limits. We intentionally use get_memory_affinity() rather than
    get_visible_memory_node() so CPU_VISIBLE_MEMORY_NODES does not become an
    input to the detector; auto binding remains owned by the CPU backend.
    """
    cpus = get_allowed_cpu_list()
    if not cpus:
        raise RuntimeError("vLLM reported no CPUs available to this process.")

    cpu_numa_nodes = {cpu.numa_node for cpu in cpus if cpu.numa_node >= 0}
    memory_numa_nodes = set(get_memory_affinity())
    effective_numa_nodes = sorted(cpu_numa_nodes & memory_numa_nodes)
    if not effective_numa_nodes:
        effective_numa_nodes = sorted(cpu_numa_nodes)

    physical_cores = {
        (cpu.numa_node, cpu.physical_core) for cpu in cpus if cpu.physical_core >= 0
    }

    numa_nodes = []
    for node_id in effective_numa_nodes:
        node_cpus = [cpu for cpu in cpus if cpu.numa_node == node_id]
        node_physical_cores = {
            cpu.physical_core for cpu in node_cpus if cpu.physical_core >= 0
        }
        memory = get_memory_node_info(node_id)
        numa_nodes.append(
            NumaNodeInfo(
                node_id=node_id,
                logical_cpu_ids=tuple(sorted(cpu.id for cpu in node_cpus)),
                physical_core_count=len(node_physical_cores),
                total_memory_bytes=memory.total_memory,
                available_memory_bytes=memory.available_memory,
            )
        )

    socket_ids = {
        socket_id for cpu in cpus if (socket_id := _read_socket_id(cpu.id)) is not None
    }

    return HardwareInfo(
        architecture=platform.machine(),
        socket_count=len(socket_ids) if socket_ids else None,
        numa_nodes=tuple(numa_nodes),
        allowed_logical_cpu_count=len(cpus),
        physical_core_count=len(physical_cores),
    )


def main() -> int:
    print(json.dumps(detect_hardware().to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
