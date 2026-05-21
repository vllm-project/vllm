# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NUMA binding utilities for vLLM worker processes.

Adapted in part from SGLang's NUMA helper implementation:
https://github.com/sgl-project/sglang/blob/ba6d54d0f08f82f42b8224908ae2459a496b31b3/python/sglang/srt/utils/numa_utils.py
"""

import ctypes
import logging
import multiprocessing
import os
import subprocess
from contextlib import contextmanager
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING

import psutil

from vllm import envs

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = logging.getLogger(__name__)
_NUMACTL_ARGS_ENV = "_VLLM_INTERNAL_NUMACTL_ARGS"
_NUMACTL_PYTHON_EXECUTABLE_ENV = "_VLLM_INTERNAL_NUMACTL_PYTHON_EXECUTABLE"


@cache
def get_libnuma():
    libnuma = None
    for libnuma_so in ["libnuma.so", "libnuma.so.1"]:
        try:
            libnuma = ctypes.CDLL(libnuma_so)
        except OSError:
            libnuma = None
        if libnuma is not None:
            break
    return libnuma


def _can_set_mempolicy() -> bool:
    """Check whether the current process can use NUMA memory policy syscalls."""
    try:
        libnuma = get_libnuma()
        if libnuma is None or libnuma.numa_available() < 0:
            return False
        mode = ctypes.c_int()
        ret = libnuma.get_mempolicy(
            ctypes.byref(mode), None, ctypes.c_ulong(0), None, ctypes.c_ulong(0)
        )
        return ret == 0
    except Exception:
        return False


def _is_auto_numa_available() -> bool:
    """Check whether automatic GPU-to-NUMA detection should be attempted."""
    from vllm.platforms import current_platform

    if not current_platform.is_cuda_alike():
        return False

    if not os.path.isdir("/sys/devices/system/node/node1"):
        return False

    try:
        process = psutil.Process(os.getpid())
        cpu_affinity = process.cpu_affinity()
        cpu_count = psutil.cpu_count()
        if cpu_count is not None and cpu_affinity != list(range(cpu_count)):
            logger.warning(
                "CPU affinity is already constrained for this process. "
                "Skipping automatic NUMA binding; pass --numa-bind-nodes "
                "explicitly to override."
            )
            return False
    except (AttributeError, NotImplementedError, psutil.Error):
        pass

    if not _can_set_mempolicy():
        logger.warning(
            "User lacks permission to set NUMA memory policy. "
            "Automatic NUMA detection may not work; if you are using Docker, "
            "try adding --cap-add SYS_NICE."
        )
        return False

    if not hasattr(current_platform, "get_all_device_numa_nodes"):
        logger.warning(
            "Platform %s does not support automatic NUMA detection",
            type(current_platform).__name__,
        )
        return False

    return True


@cache
def get_auto_numa_nodes() -> list[int] | None:
    """Auto-detect NUMA nodes for all visible GPUs."""
    from vllm.platforms import current_platform

    if not _is_auto_numa_available():
        return None

    numa_nodes = current_platform.get_all_device_numa_nodes()
    if numa_nodes is not None:
        logger.info("Auto-detected NUMA nodes for GPUs: %s", numa_nodes)
    return numa_nodes


_PCT_CPU_MODEL_SUBSTRING = "6776P"
_PCT_HIGHEST_PERF_SIGNAL = 46
_PCT_HIGHEST_PERF_PATH = "/sys/devices/system/cpu/cpu0/acpi_cppc/highest_perf"
_PROC_CPUINFO_PATH = "/proc/cpuinfo"


@cache
def _is_xeon_6776p_with_pct() -> bool:
    """Detect Xeon 6776P (DGX B300) with Priority Core Turbo enabled.

    Gates: ``/proc/cpuinfo`` ``model name`` contains ``6776P`` and
    ``/sys/devices/system/cpu/cpu0/acpi_cppc/highest_perf`` reads ``46``.
    """
    try:
        with open(_PROC_CPUINFO_PATH) as f:
            has_model = any(
                line.lstrip().lower().startswith("model name")
                and _PCT_CPU_MODEL_SUBSTRING in line
                for line in f
            )
    except OSError:
        return False
    if not has_model:
        return False

    try:
        with open(_PCT_HIGHEST_PERF_PATH) as f:
            highest_perf = int(f.read().strip())
    except (OSError, ValueError):
        return False
    return highest_perf == _PCT_HIGHEST_PERF_SIGNAL


def _get_gpu_index(
    parallel_config, local_rank: int, dp_local_rank: int | None = None
) -> int:
    """Compute the physical GPU index used for NUMA lookup."""
    if (
        parallel_config.distributed_executor_backend not in ("ray", "external_launcher")
        and parallel_config.data_parallel_backend != "ray"
        and parallel_config.nnodes_within_dp == 1
    ):
        if dp_local_rank is None:
            dp_local_rank = parallel_config.data_parallel_rank_local
            if dp_local_rank is None:
                dp_local_rank = parallel_config.data_parallel_index

        tp_pp_world_size = (
            parallel_config.pipeline_parallel_size
            * parallel_config.tensor_parallel_size
        )
        return local_rank + dp_local_rank * tp_pp_world_size

    return local_rank


def _get_numa_node(parallel_config, gpu_index: int) -> int:
    numa_nodes = parallel_config.numa_bind_nodes
    if numa_nodes is None:
        numa_nodes = get_auto_numa_nodes()
        if numa_nodes is None:
            raise RuntimeError(
                "NUMA binding was requested, but vLLM could not detect the "
                "GPU-to-NUMA topology automatically. Pass --numa-bind-nodes "
                "explicitly or disable --numa-bind."
            )
        parallel_config.numa_bind_nodes = numa_nodes

    if gpu_index >= len(numa_nodes):
        raise ValueError(
            f"GPU index {gpu_index} exceeds numa_bind_nodes size {len(numa_nodes)}. "
            "Ensure the binding lists cover every visible GPU."
        )

    return numa_nodes[gpu_index]


def _maybe_get_pct_cpu_binding(numa_nodes: list[int]) -> str | None:
    """Return the union of PCT priority cores across ``numa_nodes`` (or None).

    PCT (Priority Core Turbo) lets a subset of cores boost above the rest;
    we want workers and the EngineCore on those cores. Linux kernel does not
    expose PCT membership yet, so we only handle the DGX B300 case (Xeon
    6776P), where priority cores are ``cpu_id % 16 in (0, 1)`` intersected
    with each node's ``cpulist``.
    """
    if not _is_xeon_6776p_with_pct():
        return None

    from vllm.utils.cpu_resource_utils import parse_id_list

    union_cpus: set[int] = set()
    for numa_node in numa_nodes:
        cpulist_path = Path(f"/sys/devices/system/node/node{numa_node}/cpulist")
        try:
            cpulist_raw = cpulist_path.read_text().strip()
        except OSError:
            continue
        if not cpulist_raw:
            continue
        try:
            node_cpus = parse_id_list(cpulist_raw)
        except ValueError:
            continue

        priority = [cpu for cpu in node_cpus if cpu % 16 in (0, 1)]
        if not priority:
            continue
        union_cpus.update(priority)
        logger.info(
            "Detected Xeon 6776P with PCT (highest_perf=%d); NUMA node %d "
            "priority cores: %s",
            _PCT_HIGHEST_PERF_SIGNAL,
            numa_node,
            ",".join(str(c) for c in priority),
        )

    if not union_cpus:
        return None
    return ",".join(str(c) for c in sorted(union_cpus))


def _get_cpu_binding(
    parallel_config, gpu_index: int, numa_nodes: list[int]
) -> str | None:
    """Return the CPU list a process should be pinned to (or None)."""
    cpu_bindings = parallel_config.numa_bind_cpus
    if cpu_bindings is None:
        return _maybe_get_pct_cpu_binding(numa_nodes)

    if gpu_index >= len(cpu_bindings):
        raise ValueError(
            f"GPU index {gpu_index} exceeds numa_bind_cpus size "
            f"{len(cpu_bindings)}. Ensure the binding lists cover every visible GPU."
        )
    return cpu_bindings[gpu_index]


def _get_numactl_worker_args(
    parallel_config, local_rank: int, dp_local_rank: int | None, process_kind: str
) -> str:
    """Compute the numactl args for a single TP/PP worker subprocess."""
    gpu_index = _get_gpu_index(parallel_config, local_rank, dp_local_rank)
    numa_node = _get_numa_node(parallel_config, gpu_index)
    cpu_binding = _get_cpu_binding(parallel_config, gpu_index, [numa_node])

    if cpu_binding is not None:
        logger.info(
            "Binding %s subprocess (local_rank=%s, gpu_index=%s) to CPUs %s and NUMA node %s",  # noqa: E501
            process_kind,
            local_rank,
            gpu_index,
            cpu_binding,
            numa_node,
        )
        return f"--physcpubind={cpu_binding} --membind={numa_node}"

    logger.info(
        "Binding %s subprocess (local_rank=%s, gpu_index=%s) to NUMA node %s",
        process_kind,
        local_rank,
        gpu_index,
        numa_node,
    )
    return f"--cpunodebind={numa_node} --membind={numa_node}"


def _get_enginecore_numa_nodes(
    parallel_config, dp_local_rank: int | None = None
) -> list[int]:
    """Return the sorted, unique NUMA nodes of the EngineCore's DP shard."""
    numa_nodes = parallel_config.numa_bind_nodes
    if numa_nodes is None:
        # Trigger auto-detection (it caches into parallel_config).
        _get_numa_node(parallel_config, 0)
        numa_nodes = parallel_config.numa_bind_nodes

    if (
        parallel_config.distributed_executor_backend not in ("ray", "external_launcher")
        and parallel_config.data_parallel_backend != "ray"
        and parallel_config.nnodes_within_dp == 1
    ):
        if dp_local_rank is None:
            dp_local_rank = parallel_config.data_parallel_rank_local
            if dp_local_rank is None:
                dp_local_rank = parallel_config.data_parallel_index

        tp_pp_world_size = (
            parallel_config.pipeline_parallel_size
            * parallel_config.tensor_parallel_size
        )
        shard_start = dp_local_rank * tp_pp_world_size
        shard_end = min(shard_start + tp_pp_world_size, len(numa_nodes))
        shard_indices: range | tuple[int, ...] = range(shard_start, shard_end)
    else:
        shard_indices = range(len(numa_nodes))

    if not shard_indices:
        return [numa_nodes[0]]
    return sorted({numa_nodes[i] for i in shard_indices})


def _get_numactl_enginecore_args(
    parallel_config, local_rank: int, dp_local_rank: int | None
) -> str:
    """Compute the numactl args for an EngineCore subprocess."""
    shard_nodes = _get_enginecore_numa_nodes(parallel_config, dp_local_rank)
    gpu_index = _get_gpu_index(parallel_config, local_rank, dp_local_rank)
    cpu_binding = _get_cpu_binding(parallel_config, gpu_index, shard_nodes)
    membind_arg = ",".join(str(n) for n in shard_nodes)

    if cpu_binding is not None:
        logger.info(
            "Binding EngineCore subprocess (local_rank=%s) to CPUs %s "
            "and NUMA nodes %s",
            local_rank,
            cpu_binding,
            membind_arg,
        )
        return f"--physcpubind={cpu_binding} --membind={membind_arg}"

    logger.info(
        "Binding EngineCore subprocess (local_rank=%s) to NUMA nodes %s",
        local_rank,
        membind_arg,
    )
    return f"--cpunodebind={membind_arg} --membind={membind_arg}"


def _get_numactl_args(
    vllm_config: "VllmConfig",
    local_rank: int,
    dp_local_rank: int | None = None,
    process_kind: str = "worker",
) -> str | None:
    parallel_config = vllm_config.parallel_config
    if not parallel_config.numa_bind:
        return None

    if process_kind == "EngineCore":
        return _get_numactl_enginecore_args(parallel_config, local_rank, dp_local_rank)
    return _get_numactl_worker_args(
        parallel_config, local_rank, dp_local_rank, process_kind
    )


def _log_numactl_show(label: str) -> bool:
    try:
        result = subprocess.run(
            ["numactl", "--show"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        logger.warning("Failed to run `numactl --show` for %s: %s", label, e)
        return False

    output = result.stdout.strip()
    if not output:
        logger.warning("`numactl --show` returned no output for %s", label)
        return False

    summary = ", ".join(line.strip() for line in output.splitlines() if line.strip())
    logger.debug("%s affinity: %s", label, summary)
    return True


def log_current_affinity_state(label: str) -> None:
    """Log the process's effective NUMA affinity state."""
    _log_numactl_show(label)


@contextmanager
def configure_subprocess(
    vllm_config: "VllmConfig",
    local_rank: int,
    dp_local_rank: int | None = None,
    process_kind: str = "worker",
):
    """Temporarily replace the multiprocessing executable with a numactl wrapper."""
    numactl_args = _get_numactl_args(
        vllm_config, local_rank, dp_local_rank, process_kind
    )
    if numactl_args is None:
        yield
        return

    executable, debug_str = _get_numactl_executable()
    python_executable = os.fsdecode(multiprocessing.spawn.get_executable())
    with (
        _set_numa_wrapper_env(numactl_args, python_executable),
        _mp_set_executable(executable, debug_str),
    ):
        yield


def _get_numactl_executable() -> tuple[str, str]:
    """Return the fixed wrapper executable used to launch numactl."""
    from shutil import which

    if which("numactl") is None:
        raise RuntimeError(
            "numactl is required for NUMA binding but is not installed or "
            "not available on PATH."
        )

    script_path = Path(__file__).with_name("numa_wrapper.sh")
    return str(script_path), f"{script_path} via {_NUMACTL_ARGS_ENV}"


@contextmanager
def _set_numa_wrapper_env(numactl_args: str, python_executable: str):
    old_numactl_args = os.environ.get(_NUMACTL_ARGS_ENV)
    old_python_executable = os.environ.get(_NUMACTL_PYTHON_EXECUTABLE_ENV)
    os.environ[_NUMACTL_ARGS_ENV] = numactl_args
    os.environ[_NUMACTL_PYTHON_EXECUTABLE_ENV] = python_executable
    try:
        yield
    finally:
        if old_numactl_args is None:
            os.environ.pop(_NUMACTL_ARGS_ENV, None)
        else:
            os.environ[_NUMACTL_ARGS_ENV] = old_numactl_args

        if old_python_executable is None:
            os.environ.pop(_NUMACTL_PYTHON_EXECUTABLE_ENV, None)
        else:
            os.environ[_NUMACTL_PYTHON_EXECUTABLE_ENV] = old_python_executable


@contextmanager
def _mp_set_executable(executable: str, debug_str: str):
    start_method = envs.VLLM_WORKER_MULTIPROC_METHOD
    if start_method != "spawn":
        logger.warning(
            "NUMA binding requires spawn method but got '%s'. "
            "NUMA binding will be ineffective. "
            "Set VLLM_WORKER_MULTIPROC_METHOD=spawn to enable NUMA binding.",
            start_method,
        )
        yield
        return

    old_executable = os.fsdecode(multiprocessing.spawn.get_executable())
    multiprocessing.spawn.set_executable(executable)
    try:
        yield
    finally:
        assert os.fsdecode(multiprocessing.spawn.get_executable()) == executable, (
            "Executable was changed during NUMA binding context: "
            f"expected {executable}, got {multiprocessing.spawn.get_executable()}"
        )
        multiprocessing.spawn.set_executable(old_executable)
