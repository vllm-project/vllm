# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from functools import cache

import psutil
import regex as re

DEVICE_CONTROL_ENV_VAR = "CPU_VISIBLE_MEMORY_NODES"


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


@cache
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


@cache
def _get_cgroup_numa_used() -> dict[int, int] | None:
    """Return per-NUMA-node bytes used by the current cgroup, or None.

    Parses ``memory.numa_stat`` for v2 (``key N0=.. N1=..`` in bytes) or
    v1 (``key=val N0=.. N1=..`` in pages). The ``anon`` + ``file`` rows
    are summed to approximate the cgroup's resident footprint per node.
    """
    if sys.platform != "linux":
        return None

    v2_path = "/sys/fs/cgroup/memory.numa_stat"
    v1_path = "/sys/fs/cgroup/memory/memory.numa_stat"
    if os.path.exists(v2_path):
        path, is_v2 = v2_path, True
    elif os.path.exists(v1_path):
        path, is_v2 = v1_path, False
    else:
        return None

    wanted = {"anon", "file"}
    page_size = os.sysconf("SC_PAGE_SIZE") if not is_v2 else 1
    per_node: dict[int, int] = {}
    try:
        with open(path) as f:
            for line in f:
                parts = line.split()
                if not parts:
                    continue
                # v1: "anon=123 N0=10 N1=20" -> head = "anon"
                # v2: "anon 12345 N0=10 N1=20" -> head = "anon"
                head = parts[0].split("=", 1)[0]
                if head not in wanted:
                    continue
                for tok in parts[1:]:
                    if not tok.startswith("N") or "=" not in tok:
                        continue
                    node_str, val_str = tok[1:].split("=", 1)
                    try:
                        node_id = int(node_str)
                        val = int(val_str) * page_size
                    except ValueError:
                        continue
                    per_node[node_id] = per_node.get(node_id, 0) + val
    except OSError:
        return None

    return per_node or None


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
    # would be applied to host RAM instead of the pod's limit. Prefer
    # ``memory.numa_stat`` so the per-node figures stay accurate on
    # multi-NUMA pods instead of clamping every node to the global limit.
    cgroup_limit, cgroup_usage = get_cgroup_memory_limit()
    if cgroup_limit is not None:
        numa_used = _get_cgroup_numa_used()
        if numa_used is not None:
            total_used = sum(numa_used.values())
            node_used = numa_used.get(node_id, 0)
            # Distribute the pod-wide headroom proportionally to each
            # node's host capacity so the sum stays within ``cgroup_limit``.
            node_total_cap = min(total_memory, cgroup_limit)
            total_memory = node_total_cap
            cgroup_headroom = max(0, cgroup_limit - total_used)
            available_memory = max(
                0, min(available_memory - node_used, cgroup_headroom)
            )
        elif cgroup_limit < total_memory:
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

    logical_cpu_list: list[LogicalCPUInfo] = json.loads(
        lscpu_output, object_hook=LogicalCPUInfo.json_decoder
    )["cpus"]

    # Filter CPUs with invalid attributes
    logical_cpu_list = [
        x for x in logical_cpu_list if -1 not in (x.id, x.physical_core, x.numa_node)
    ]

    # If lscpu returned no valid entries (e.g. RISC-V where all fields
    # are bare `-`), fall back to synthesized topology.
    if not logical_cpu_list:
        logical_cpu_list = _synthesize_cpu_list()

    return logical_cpu_list
