# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from functools import cache

import psutil
import regex as re

from vllm.logger import init_logger

DEVICE_CONTROL_ENV_VAR = "CPU_VISIBLE_MEMORY_NODES"

logger = init_logger(__name__)


@dataclass
class LogicalCPUInfo:
    id: int = -1
    physical_core: int = -1
    numa_node: int = -1

    @classmethod
    def _int(cls, value: str) -> int:
        try:
            int_value = int(value)
        except Exception:
            int_value = -1
        return int_value

    @staticmethod
    def json_decoder(obj_dict: dict):
        id = obj_dict.get("cpu")
        physical_core = obj_dict.get("core")
        numa_node = obj_dict.get("node")

        if not (id is None or physical_core is None or numa_node is None):
            return LogicalCPUInfo(
                id=LogicalCPUInfo._int(id),
                physical_core=LogicalCPUInfo._int(physical_core),
                numa_node=LogicalCPUInfo._int(numa_node),
            )
        else:
            return obj_dict


@dataclass
class MemoryNodeInfo:
    total_memory: int = -1
    available_memory: int = -1


def _read_int_file(path: str) -> int | None:
    try:
        with open(path) as f:
            value = f.read().strip()
        if not value or value == "max":
            return None
        return int(value)
    except (OSError, ValueError):
        return None


def get_cgroup_memory_limit() -> tuple[int | None, int | None]:
    """Return (limit, usage) in bytes from cgroup, or (None, None).

    Supports both cgroup v2 (unified) and v1. Returns (None, None) when
    not running under a constrained cgroup (e.g. bare metal, or limit
    reported as `max`/an unrealistically large value).
    """
    if sys.platform != "linux":
        return None, None

    # cgroup v2 unified hierarchy
    v2_limit = _read_int_file("/sys/fs/cgroup/memory.max")
    if v2_limit is not None:
        v2_usage = _read_int_file("/sys/fs/cgroup/memory.current")
        return v2_limit, v2_usage

    # cgroup v1
    v1_limit = _read_int_file("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    if v1_limit is not None:
        # cgroup v1 reports a huge sentinel (close to PAGE_COUNTER_MAX)
        # when unlimited. Treat absurdly large values as "no limit".
        if v1_limit >= (1 << 62):
            return None, None
        v1_usage = _read_int_file("/sys/fs/cgroup/memory/memory.usage_in_bytes")
        return v1_limit, v1_usage

    return None, None


def check_cgroup_memory_available(
    required_bytes: int,
    allocation_name: str,
) -> None:
    """Raise if a memory allocation would exceed the cgroup limit.

    Args:
        required_bytes: Bytes required by the allocation.
        allocation_name: Human-readable name used in the error message.

    Raises:
        RuntimeError: If the cgroup has a finite limit and insufficient
            memory remains for the allocation.

    If the cgroup limit or usage cannot be read, the check is skipped.
    """
    cgroup_limit, cgroup_usage = get_cgroup_memory_limit()
    if cgroup_limit is None or cgroup_usage is None:
        return

    cgroup_available = max(0, cgroup_limit - cgroup_usage)
    mib = 1 << 20
    if required_bytes <= cgroup_available:
        logger.info(
            "Cgroup memory preflight passed for %s: %.0f MiB required, %.0f MiB "
            "current usage, %.0f MiB available under %.0f MiB limit, %.0f MiB "
            "estimated remaining after allocation.",
            allocation_name,
            required_bytes / mib,
            cgroup_usage / mib,
            cgroup_available / mib,
            cgroup_limit / mib,
            (cgroup_available - required_bytes) / mib,
        )
        return

    raise RuntimeError(
        f"Insufficient cgroup memory for {allocation_name}: "
        f"{required_bytes / mib:.0f} MiB required, "
        f"{cgroup_available / mib:.0f} MiB available under the "
        f"{cgroup_limit / mib:.0f} MiB limit "
        f"({cgroup_usage / mib:.0f} MiB current usage). "
        "Increase the container memory limit or reduce the allocation size."
    )


def get_memory_affinity(pid: int = 0) -> list[int]:
    pid = os.getpid() if pid == 0 else pid
    path = f"/proc/{pid}/status"
    with open(path) as f:
        for line in f:
            if line.startswith("Mems_allowed_list:"):
                # Extract the string part (e.g., "0-1,3")
                raw_list = line.split(":")[1].strip()
                return parse_id_list(raw_list)
    return []


def parse_id_list(raw_str: str) -> list[int]:
    """Parses strings like '0-2,4,7-8' into [0, 1, 2, 4, 7, 8]"""
    result: list[int] = []
    if not raw_str:
        return result

    for part in raw_str.split(","):
        if "-" in part:
            start, end = map(int, part.split("-"))
            result.extend(range(start, end + 1))
        else:
            result.append(int(part))
    return sorted(list(set(result)))


def get_memory_node_info(node_id: int = 0) -> MemoryNodeInfo:
    if sys.platform == "darwin":
        # MacOS has no memory node
        return MemoryNodeInfo(
            total_memory=psutil.virtual_memory().total,
            available_memory=psutil.virtual_memory().available,
        )

    meminfo_path = f"/sys/devices/system/node/node{node_id}/meminfo"
    if not os.path.exists(meminfo_path):
        # Non-NUMA systems (e.g. many RISC-V boards) don't expose per-node
        # meminfo. Fall back to system-wide numbers from psutil.
        vm = psutil.virtual_memory()
        return MemoryNodeInfo(
            total_memory=vm.total,
            available_memory=vm.available,
        )

    meminfo = {}
    with open(meminfo_path) as f:
        for line in f:
            # Each line looks like: "Node 0 MemTotal: 97421888 kB"
            parts = line.split()
            key = parts[2].rstrip(":")
            # convert to Bytes
            value = int(parts[3]) * 1024
            meminfo[key] = value

    total_memory = meminfo["MemTotal"]
    free_memory = meminfo["MemFree"]
    active_file_memory = meminfo["Active(file)"]
    inactive_file_memory = meminfo["Inactive(file)"]
    reclaimable_memory = meminfo["SReclaimable"]
    available_memory = (
        free_memory + active_file_memory + inactive_file_memory + reclaimable_memory
    )

    # Honor cgroup memory limit (containers / k8s pods). NUMA meminfo
    # reflects host-wide numbers; without this, gpu_memory_utilization
    # would be applied to host RAM instead of the pod's limit. cgroup
    # does not expose per-NUMA-node limits, so we just clamp the totals
    # against the pod-wide limit here.
    cgroup_limit, cgroup_usage = get_cgroup_memory_limit()
    if cgroup_limit is not None and cgroup_limit < total_memory:
        total_memory = cgroup_limit
        cgroup_available = cgroup_limit - (cgroup_usage or 0)
        available_memory = max(0, min(available_memory, cgroup_available))

    return MemoryNodeInfo(
        total_memory=total_memory,
        available_memory=available_memory,
    )


def get_allowed_cpu_list() -> list[LogicalCPUInfo]:
    cpu_list = _get_cpu_list()
    if sys.platform == "linux":
        allowed = os.sched_getaffinity(0)
        return [x for x in cpu_list if x.id in allowed]
    return cpu_list


def get_visible_memory_node() -> list[int]:
    if sys.platform == "darwin":
        return [0]

    allowed_memory_node_list = get_memory_affinity()

    env_key = DEVICE_CONTROL_ENV_VAR
    if (
        ("VLLM_CPU_SIM_MULTI_NUMA" not in os.environ)
        and env_key in os.environ
        and os.environ[env_key] != ""
    ):
        visible_nodes = [int(s) for s in os.environ[env_key].split(",")]
        visible_nodes = [
            node for node in visible_nodes if node in allowed_memory_node_list
        ]
        return visible_nodes

    return allowed_memory_node_list


@cache
def _synthesize_cpu_list() -> list[LogicalCPUInfo]:
    """Synthesize a flat CPU list: each logical CPU is its own core on
    NUMA node 0.  Used when lscpu output is unavailable or unparsable
    (e.g. macOS, RISC-V)."""
    cpu_count = os.cpu_count()
    assert cpu_count
    return [LogicalCPUInfo(i, i, 0) for i in range(cpu_count)]


def _get_cpu_list() -> list[LogicalCPUInfo]:
    if sys.platform == "darwin":
        # For MacOS, no user-level CPU affinity and SMT, return all CPUs
        return _synthesize_cpu_list()

    if platform.machine() == "s390x":
        lscpu_output = subprocess.check_output(
            "lscpu -J -e=CPU,CORE,NODE,SOCKET,BOOK", shell=True, text=True
        )
    else:
        lscpu_output = subprocess.check_output(
            "lscpu --json --extended=CPU,CORE,NODE --online", shell=True, text=True
        )

    # For platforms without NUMA, map bare `-` node to 0 so non-NUMA
    # systems keep the existing behavior from #39781.
    lscpu_output = re.sub(r'"node":\s*-\s*(,|\n|\})', r'"node": 0\1', lscpu_output)

    # On some architectures (notably RISC-V), lscpu also emits bare `-`
    # for cpu/core.  Quote them so the JSON parses; they will decode to
    # -1 and be filtered out below, triggering the synthesized fallback.
    lscpu_output = re.sub(
        r'("(?:cpu|core)":\s*)-\s*(,|\n|\})',
        r'\1"-"\2',
        lscpu_output,
    )

    if platform.machine() == "s390x":
        # On s390x, use the best topology level as the abstract CPU group
        # key via numa_node. The hierarchy is book > socket > core; prefer
        # book, fall back to socket if only one book exists.
        lscpu_output = re.sub(r'"book":\s*-\s*(,|\n|\})', r'"book": 0\1', lscpu_output)
        lscpu_output = re.sub(
            r'"socket":\s*-\s*(,|\n|\})', r'"socket": 0\1', lscpu_output
        )

        raw_cpus = json.loads(lscpu_output)["cpus"]
        book_values = set()
        socket_values = set()
        for entry in raw_cpus:
            book_values.add(LogicalCPUInfo._int(str(entry.get("book", "-1"))))
            socket_values.add(LogicalCPUInfo._int(str(entry.get("socket", "-1"))))

        # Pick the best partitioning level: book first, then socket
        if len(book_values - {-1}) > 1:
            group_key = "book"
        elif len(socket_values - {-1}) > 1:
            group_key = "socket"
        else:
            group_key = None

        if group_key is not None:
            logical_book_socket_list: list[LogicalCPUInfo] = []
            for entry in raw_cpus:
                cpu_id = LogicalCPUInfo._int(str(entry.get("cpu", "-1")))
                core = LogicalCPUInfo._int(str(entry.get("core", "-1")))
                group = LogicalCPUInfo._int(str(entry.get(group_key, "-1")))
                if -1 not in (cpu_id, core, group):
                    logical_book_socket_list.append(
                        LogicalCPUInfo(id=cpu_id, physical_core=core, numa_node=group)
                    )

            if not logical_book_socket_list:
                return _synthesize_cpu_list()
            return logical_book_socket_list

    logical_cpu_list: list[LogicalCPUInfo] = json.loads(
        lscpu_output, object_hook=LogicalCPUInfo.json_decoder
    )["cpus"]

    # Filter CPUs with invalid attributes (only require id, core, node)
    logical_cpu_list = [
        x for x in logical_cpu_list if -1 not in (x.id, x.physical_core, x.numa_node)
    ]

    # If lscpu returned no valid entries (e.g. RISC-V where all fields
    # are bare `-`), fall back to synthesized topology.
    if not logical_cpu_list:
        logical_cpu_list = _synthesize_cpu_list()

    return logical_cpu_list
