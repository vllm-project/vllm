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

import vllm.envs as envs
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


_CGROUP_ROOT = "/sys/fs/cgroup"
_PROC_SELF_CGROUP = "/proc/self/cgroup"
# Release default for resolving the process cgroup instead of reading the
# cgroup root only. Flip this one line to change the default behavior;
# VLLM_RESOLVE_PROCESS_CGROUP=1/0 overrides it either way.
RESOLVE_PROCESS_CGROUP_BY_DEFAULT = False
# cgroup v1 reports "no limit" as a huge sentinel (close to PAGE_COUNTER_MAX).
_CGROUP_UNLIMITED = 1 << 62


def _resolve_process_cgroup() -> bool:
    override = envs.VLLM_RESOLVE_PROCESS_CGROUP
    return RESOLVE_PROCESS_CGROUP_BY_DEFAULT if override is None else override


def _cgroup_dirs(controller: str | None) -> list[str]:
    """Directories of this process's cgroup, nearest first, mount root last.

    When resolving is off (``RESOLVE_PROCESS_CGROUP_BY_DEFAULT`` unless
    ``VLLM_RESOLVE_PROCESS_CGROUP`` overrides it) only the root is returned,
    which is the previous behavior. Otherwise the cgroup is resolved from
    /proc/self/cgroup: the ``0::`` entry for cgroup v2, the entry naming
    ``controller`` for v1. The root is always included, so a path that is not
    visible inside the mount (host cgroup namespace) yields the root alone.
    """
    root = (
        _CGROUP_ROOT if controller is None else os.path.join(_CGROUP_ROOT, controller)
    )
    if not _resolve_process_cgroup():
        return [root]
    rel_path = ""
    try:
        with open(_PROC_SELF_CGROUP) as f:
            for line in f:
                parts = line.strip().split(":", 2)
                if len(parts) != 3:
                    continue
                _, controllers, path = parts
                is_v2 = controller is None and controllers == ""
                is_v1 = controller is not None and controller in controllers.split(",")
                if is_v2 or is_v1:
                    rel_path = path
                    break
    except OSError:
        pass
    components = [c for c in rel_path.split("/") if c]
    if ".." in components:
        # A cgroup outside this namespace cannot be resolved from here.
        components = []
    dirs = [
        os.path.join(root, *components[:depth])
        for depth in range(len(components), 0, -1)
    ]
    dirs.append(root)
    return dirs


def _binding_cgroup(controller: str | None, limit_file: str) -> tuple[str, int] | None:
    """The cgroup directory with the smallest limit, and that limit.

    The kernel enforces the smallest limit from the process's cgroup up to
    the root, so usage has to be read from that same directory.
    """
    binding: tuple[str, int] | None = None
    for cg_dir in _cgroup_dirs(controller):
        limit = _read_int_file(os.path.join(cg_dir, limit_file))
        if limit is None or limit >= _CGROUP_UNLIMITED:
            continue
        if binding is None or limit < binding[1]:
            binding = (cg_dir, limit)
    return binding


def _binding_memory_cgroup() -> tuple[str, str, int] | None:
    """(directory, usage file, limit) of the memory cgroup binding this process."""
    v2 = _binding_cgroup(None, "memory.max")
    if v2 is not None:
        return v2[0], "memory.current", v2[1]
    v1 = _binding_cgroup("memory", "memory.limit_in_bytes")
    if v1 is not None:
        return v1[0], "memory.usage_in_bytes", v1[1]
    return None


@cache
def get_cgroup_memory_limit() -> int | None:
    """Return the cgroup memory limit in bytes, or None.

    Supports both cgroup v2 (unified) and v1. When resolving the process
    cgroup is enabled, the limit is the smallest one on the process's own
    cgroup and its parents, which is the one the kernel enforces; otherwise
    only the cgroup root is read. Returns None when no cgroup limits the
    process (e.g. bare metal, or every level reports `max`/an unrealistically
    large value).
    """
    if sys.platform != "linux":
        return None
    binding = _binding_memory_cgroup()
    return None if binding is None else binding[2]


def get_cgroup_memory_usage() -> int | None:
    """Return the current cgroup memory usage in bytes, or None.

    Usage is read from the cgroup that holds the binding limit, so it is the
    number the kernel compares against that limit. It is intentionally read
    on every call because cgroup memory usage changes while the process is
    running.
    """
    if sys.platform != "linux":
        return None
    binding = _binding_memory_cgroup()
    if binding is None:
        return None
    cg_dir, usage_file, _ = binding
    return _read_int_file(os.path.join(cg_dir, usage_file))


def check_cgroup_memory_available(
    required_bytes: int,
    allocation_name: str,
) -> None:
    """Log cgroup memory headroom for an upcoming allocation.

    Args:
        required_bytes: Bytes required by the allocation.
        allocation_name: Human-readable name used in log messages.

    Low headroom logs a warning, but does not reject the allocation because
    cgroup usage can include reclaimable memory. If the cgroup limit or usage
    cannot be read, the check is skipped.

    """
    cgroup_limit = get_cgroup_memory_limit()
    cgroup_usage = get_cgroup_memory_usage()
    if cgroup_limit is None or cgroup_usage is None:
        return

    cgroup_available = max(0, cgroup_limit - cgroup_usage)
    mib = 1 << 20
    remaining_bytes = cgroup_available - required_bytes
    log_fn = logger.debug if remaining_bytes >= 0 else logger.warning
    status = (
        "current headroom meets the requested allocation"
        if remaining_bytes >= 0
        else "current headroom is below the requested allocation; allocation "
        "will still be attempted because cgroup usage may be reclaimable"
    )
    log_fn(
        "Cgroup memory preflight for %s: %.0f MiB required, %.0f MiB current "
        "usage, %.0f MiB available under %.0f MiB limit, %.0f MiB remaining "
        "after allocation based on current usage; %s.",
        allocation_name,
        required_bytes / mib,
        cgroup_usage / mib,
        cgroup_available / mib,
        cgroup_limit / mib,
        remaining_bytes / mib,
        status,
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
    """Parses strings like '0-2,4,7-8' into [0, 1, 2, 4, 7, 8]."""
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
    cgroup_limit = get_cgroup_memory_limit()
    cgroup_usage = get_cgroup_memory_usage()
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


def get_visible_memory_node(assigned_node_ids: list[int] | None = None) -> list[int]:
    """Memory nodes visible to this process.

    assigned_node_ids is the per-DP-rank list of NUMA node ids from
    parallel_config.assigned_physical_gpu_ids. It takes priority over the
    CPU_VISIBLE_MEMORY_NODES env var, which still applies when it is None.
    """
    if sys.platform == "darwin":
        return [0]

    allowed_memory_node_list = get_memory_affinity()

    if "VLLM_CPU_SIM_MULTI_NUMA" in os.environ:
        return allowed_memory_node_list

    if assigned_node_ids is not None:
        return [node for node in assigned_node_ids if node in allowed_memory_node_list]

    env_key = DEVICE_CONTROL_ENV_VAR
    if env_key in os.environ and os.environ[env_key] != "":
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
