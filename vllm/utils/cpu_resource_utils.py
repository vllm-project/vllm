# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
import platform
import subprocess
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
    if platform.system() == "Darwin":
        # MacOS has no memory node
        return MemoryNodeInfo(
            total_memory=psutil.virtual_memory().total,
            available_memory=psutil.virtual_memory().available,
        )

    meminfo_path = f"/sys/devices/system/node/node{node_id}/meminfo"
    if not os.path.exists(meminfo_path):
        raise RuntimeError(f"{meminfo_path} doesn't exit.")

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

    return MemoryNodeInfo(
        total_memory=total_memory,
        available_memory=available_memory,
    )


def get_allowed_cpu_list() -> list[LogicalCPUInfo]:
    cpu_list = _get_cpu_list()
    if platform.system() == "Darwin":
        return cpu_list

    global_allowed_cpu_id_list = os.sched_getaffinity(0)
    logical_cpu_list = [x for x in cpu_list if x.id in global_allowed_cpu_id_list]

    return logical_cpu_list


def get_visible_memory_node() -> list[int]:
    if platform.system() == "Darwin":
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
def _get_cpu_list() -> list[LogicalCPUInfo]:
    if platform.system() == "Darwin":
        # For MacOS, no user-level CPU affinity and SMT, return all CPUs
        cpu_count = os.cpu_count()
        assert cpu_count
        return [LogicalCPUInfo(i, i, 0) for i in range(cpu_count)]

    lscpu_output = subprocess.check_output(
        "lscpu --json --extended=CPU,CORE,NODE --online", shell=True, text=True
    )

    # For platform without NUMA, replace '-' to '0'
    lscpu_output = re.sub(r'"node":\s*-\s*(,|\n)', r'"node": 0\1', lscpu_output)

    logical_cpu_list: list[LogicalCPUInfo] = json.loads(
        lscpu_output, object_hook=LogicalCPUInfo.json_decoder
    )["cpus"]

    # Filter CPUs with invalid attributes
    logical_cpu_list = [
        x for x in logical_cpu_list if -1 not in (x.id, x.physical_core, x.numa_node)
    ]

    return logical_cpu_list
