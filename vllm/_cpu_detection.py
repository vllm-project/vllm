# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CFS-aware CPU detection utilities.

Provides container-aware CPU counting that respects cgroup CFS quotas,
for use by subsystems that size thread pools (structured output grammar
compilation, numba parallel processing, CPU backend OMP configuration).
"""

import functools
import math
import os


def _get_cgroup_v2_path() -> str:
    """Get the cgroup v2 path for the current process.

    Returns:
        The cgroup path from /proc/self/cgroup, or empty string if not found.
    """
    try:
        with open("/proc/self/cgroup") as f:
            # cgroup v2 format: "0::/path/to/cgroup"
            for line in f:
                parts = line.strip().split(":", 2)
                if len(parts) == 3 and parts[0] == "0":
                    return parts[2].lstrip("/")
    except (FileNotFoundError, PermissionError, OSError):
        pass
    return ""


def _get_cfs_cpu_limit() -> int | None:
    """Get CPU limit from CFS quota (cgroup v2/v1).

    Only returns a value when a real CFS quota is set (i.e., inside a
    container or cgroup with CPU limits). Returns None on bare metal
    or when no quota is configured.

    Returns:
        Number of CPUs from CFS quota, or None if no quota is set.
    """
    # 1. CFS quota (cgroup v2)
    try:
        cgroup_path = _get_cgroup_v2_path()
        if cgroup_path:
            cpu_max_file = f"/sys/fs/cgroup/{cgroup_path}/cpu.max"
        else:
            cpu_max_file = "/sys/fs/cgroup/cpu.max"

        with open(cpu_max_file) as f:
            parts = f.read().strip().split()
            if parts[0] != "max":
                quota = int(parts[0])
                period = int(parts[1])
                return max(1, math.floor(quota / period))
    except (FileNotFoundError, PermissionError, ValueError, IndexError, OSError):
        pass

    # 2. CFS quota (cgroup v1)
    try:
        cgroup_cpu_path = ""
        try:
            with open("/proc/self/cgroup") as f:
                for line in f:
                    parts = line.strip().split(":", 2)
                    if len(parts) == 3 and "cpu" in parts[1]:
                        cgroup_cpu_path = parts[2].lstrip("/")
                        break
        except (FileNotFoundError, PermissionError, OSError):
            pass

        if cgroup_cpu_path:
            quota_file = f"/sys/fs/cgroup/cpu/{cgroup_cpu_path}/cpu.cfs_quota_us"
            period_file = f"/sys/fs/cgroup/cpu/{cgroup_cpu_path}/cpu.cfs_period_us"
        else:
            quota_file = "/sys/fs/cgroup/cpu/cpu.cfs_quota_us"
            period_file = "/sys/fs/cgroup/cpu/cpu.cfs_period_us"

        with open(quota_file) as f:
            quota = int(f.read().strip())
        if quota > 0:
            with open(period_file) as f:
                period = int(f.read().strip())
            return max(1, math.floor(quota / period))
    except (FileNotFoundError, PermissionError, ValueError, OSError):
        pass

    return None


@functools.lru_cache(maxsize=1)
def get_available_cpus() -> int:
    """Get the number of CPUs available, respecting container limits.

    Detection order:
        1. CFS quota from cgroup v2 (process's cgroup path)
        2. CFS quota from cgroup v1 (/sys/fs/cgroup/cpu/cpu.cfs_quota_us)
        3. CPU affinity via os.sched_getaffinity (respects cpuset)
        4. os.cpu_count() (host-level fallback)

    Returns:
        Number of CPUs available to this process.
    """
    cfs_limit = _get_cfs_cpu_limit()
    if cfs_limit is not None:
        return cfs_limit

    # 3. CPU affinity (respects cpuset limits)
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        pass

    # 4. Host-level fallback
    return os.cpu_count() or 1
