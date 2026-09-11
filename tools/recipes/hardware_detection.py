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


def _compress_cpu_ids(cpu_ids: list[int] | tuple[int, ...]) -> str:
    """Return a compact Linux CPU-list string while preserving CPU IDs."""
    values = sorted(set(cpu_ids))
    if not values:
        return ""

    ranges: list[str] = []
    start = previous = values[0]
    for cpu_id in values[1:]:
        if cpu_id == previous + 1:
            previous = cpu_id
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = cpu_id
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


def build_numa_omp_threads_bind(
    hardware: HardwareInfo,
    *,
    reserved_cores_per_numa: int = 1,
) -> str:
    """Build explicit x86 OMP CPU lists for the effective NUMA nodes.

    Temporary TP/DP sweep workaround:
    - select one logical CPU per physical core, matching vLLM x86 auto-binding;
    - reserve cores by omitting them from each explicit NUMA CPU list;
    - emit one pipe-separated list per effective NUMA node.

    Explicit VLLM_CPU_OMP_THREADS_BIND lists bypass vLLM's auto-binding reserve
    logic, so the reserved cores must be excluded here.
    """
    if reserved_cores_per_numa < 0:
        raise ValueError("reserved_cores_per_numa must be at least 0")

    effective_nodes = {node.node_id for node in hardware.numa_nodes}
    allowed_cpus = get_allowed_cpu_list()

    groups: list[str] = []
    for node in hardware.numa_nodes:
        node_cpus = [
            cpu
            for cpu in allowed_cpus
            if cpu.numa_node == node.node_id and cpu.numa_node in effective_nodes
        ]

        core_to_logical_ids: dict[int, list[int]] = {}
        for cpu in node_cpus:
            if cpu.physical_core < 0:
                continue
            core_to_logical_ids.setdefault(cpu.physical_core, []).append(cpu.id)

        if not core_to_logical_ids:
            raise RuntimeError(
                f"NUMA node {node.node_id} has no physical-core topology "
                "available for explicit OMP binding."
            )

        # Match vLLM's x86 auto-binding selector: for SMT siblings on a physical
        # core, use the highest logical CPU ID.
        selected_cpu_ids = sorted(
            max(logical_ids) for _, logical_ids in sorted(core_to_logical_ids.items())
        )

        if len(selected_cpu_ids) <= reserved_cores_per_numa:
            raise RuntimeError(
                f"NUMA node {node.node_id} has only {len(selected_cpu_ids)} "
                "usable physical cores, which is not enough after reserving "
                f"{reserved_cores_per_numa} core(s)."
            )

        if reserved_cores_per_numa:
            selected_cpu_ids = selected_cpu_ids[:-reserved_cores_per_numa]

        groups.append(_compress_cpu_ids(selected_cpu_ids))

    if not groups:
        raise RuntimeError("No effective NUMA CPU lists are available for OMP binding.")

    return "|".join(groups)


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
