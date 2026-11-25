# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CPU hardware discovery and runtime policies for vLLM Recipes.

Discovery records process-visible resources; policies refine the recipe baseline.
"""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from core.model_constraints import load_head_constraints, suggest_tensor_parallel_size
from core.tuning import Policy, TuningResult, WorkloadHints, record_override


@dataclass(frozen=True, slots=True)
class NumaNodeInfo:
    node_id: int
    logical_cpu_ids: tuple[int, ...]
    physical_core_count: int
    total_memory_bytes: int
    available_memory_bytes: int


@dataclass(frozen=True, slots=True)
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
    from vllm.utils.cpu_resource_utils import (
        get_allowed_cpu_list,
        get_memory_affinity,
        get_memory_node_info,
    )

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


def _resolve_tensor_parallel_size(
    config: dict[str, Any],
    hardware: HardwareInfo | None,
    workload: WorkloadHints,
    result: TuningResult,
) -> None:
    del workload
    if hardware is None or hardware.numa_node_count < 1:
        return

    if any(
        config.get(key, 1) != 1
        for key in ("data-parallel-size", "pipeline-parallel-size")
    ):
        result.notes.append(
            "tensor-parallel-size: kept recipe value; automatic TP selection "
            "currently requires DP=1 and PP=1"
        )
        return

    # Draft policy: use the largest power-of-two TP that does not exceed the
    # effective NUMA-node count. This deliberately avoids hard-coding unusual
    # non-power-of-two topologies into the converter.
    candidate = 1 << (hardware.numa_node_count.bit_length() - 1)
    try:
        constraints = load_head_constraints(config)
    except Exception as exc:
        result.notes.append(
            "tensor-parallel-size: kept recipe value (vLLM default TP=1 if absent); "
            f"model head detection failed: {exc}"
        )
        return

    for constraint in constraints:
        result.notes.append(
            f"detected {constraint.name}: "
            f"attention_heads={constraint.attention_heads}, "
            f"kv_heads={constraint.kv_heads}"
        )
    selected = suggest_tensor_parallel_size(candidate, constraints)
    if selected != candidate:
        result.notes.append(
            f"tensor-parallel-size: reduced NUMA-derived TP={candidate} to "
            f"TP={selected} for model head partitioning/replication"
        )
    record_override(
        result,
        "tensor-parallel-size",
        selected,
        "derived from effective NUMA nodes and detected model head constraints",
    )


def _resolve_gpu_memory_utilization(
    config: dict[str, Any],
    hardware: HardwareInfo | None,
    workload: WorkloadHints,
    result: TuningResult,
) -> None:
    del workload
    if hardware is None or not hardware.numa_nodes:
        return

    reserve_fraction = 0.10
    safe_fractions = [
        max(
            0.0,
            node.available_memory_bytes / node.total_memory_bytes - reserve_fraction,
        )
        for node in hardware.numa_nodes
        if node.total_memory_bytes > 0
    ]
    if not safe_fractions:
        return

    safe_fraction = round(min(0.90, min(safe_fractions)), 2)
    if safe_fraction <= 0:
        return

    recipe_value = config.get("gpu-memory-utilization")
    if isinstance(recipe_value, int | float):
        candidate = min(float(recipe_value), safe_fraction)
    else:
        candidate = min(0.80, safe_fraction)

    record_override(
        result,
        "gpu-memory-utilization",
        round(candidate, 2),
        "constrained by effective per-NUMA available memory",
    )


def _resolve_data_parallel_size(
    config: dict[str, Any],
    hardware: HardwareInfo | None,
    workload: WorkloadHints,
    result: TuningResult,
) -> None:
    del config, hardware
    if workload.target_qps is None:
        return

    result.notes.append(
        "data-parallel-size: automatic DP tuning is disabled; kept recipe value "
        "because CPU DP auto-binding and per-replica capacity need validation"
    )


DEFAULT_POLICIES: tuple[Policy, ...] = (
    _resolve_tensor_parallel_size,
    _resolve_gpu_memory_utilization,
    _resolve_data_parallel_size,
)


def sweep_kwargs(hardware: HardwareInfo) -> dict[str, Any]:
    return {"numa_node_count": hardware.numa_node_count}


def main() -> int:
    print(json.dumps(detect_hardware().to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
