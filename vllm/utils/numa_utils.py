# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NUMA binding utilities for vLLM worker processes.

This module provides NUMA (Non-Uniform Memory Access) binding support for vLLM
worker processes to optimize performance on multi-socket systems. It uses a
technique of wrapping the Python executable with numactl to ensure all memory
allocations happen on the correct NUMA node.

Supports two modes:
1. Manual binding: User specifies NUMA nodes via --numa-node 0 0 1 1
2. Auto binding: Use --numa-node-auto to detect GPU-to-NUMA topology via NVML

Based on SGLang's NUMA binding implementation.
"""

import atexit
import logging
import multiprocessing
import os
import tempfile
from contextlib import contextmanager
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING

from vllm import envs

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = logging.getLogger(__name__)

# Cache of temporary wrapper scripts: {numactl_args: script_path}
# Reuse the same wrapper for all ranks with the same NUMA configuration
_wrapper_cache: dict[str, str] = {}


def _cleanup_all_wrappers():
    """Clean up all cached wrapper scripts at process exit."""
    for script_path in _wrapper_cache.values():
        try:
            Path(script_path).unlink(missing_ok=True)
            logger.debug("Cleaned up wrapper: %s", script_path)
        except Exception as e:
            logger.warning("Failed to clean up wrapper %s: %s", script_path, e)
    _wrapper_cache.clear()


# Register cleanup handler
atexit.register(_cleanup_all_wrappers)


@cache
def get_auto_numa_nodes() -> list[int] | None:
    """Auto-detect NUMA nodes for all GPUs using NVML.

    Returns:
        List of NUMA node IDs (one per GPU), or None if detection fails
    """
    from vllm.platforms import current_platform

    # Check if platform supports NUMA node detection
    if not hasattr(current_platform, "get_all_device_numa_nodes"):
        logger.warning(
            "Platform %s does not support NUMA node detection",
            type(current_platform).__name__,
        )
        return None

    numa_nodes = current_platform.get_all_device_numa_nodes()
    if numa_nodes is not None:
        logger.info("Auto-detected NUMA nodes for GPUs: %s", numa_nodes)
    return numa_nodes


@contextmanager
def configure_subprocess(vllm_config: "VllmConfig", local_rank: int):
    """Configure NUMA binding for worker subprocess before it starts.

    This context manager wraps process creation to ensure the worker process
    is bound to the correct NUMA node. It uses a technique of temporarily
    replacing the Python executable with a shell script that invokes numactl
    before running Python.

    This approach is superior to post-spawn binding because:
    1. The Python interpreter itself is allocated on the correct NUMA node
    2. All module imports happen with correct NUMA policy
    3. Initial PyTorch allocations are on the correct node
    4. ~100% effective vs ~70% for post-spawn binding

    Args:
        vllm_config: vLLM configuration containing NUMA node mapping
        local_rank: Local GPU rank (0-based on this node)

    Example:
        proc = mp.Process(target=worker_main, ...)
        with numa_utils.configure_subprocess(config, local_rank):
            proc.start()  # Now bound to correct NUMA node

    Requires:
        - numactl must be installed on the system
        - Multiprocessing must use 'spawn' start method (not fork)
        - vllm_config.parallel_config.numa_node or numa_node_auto must be set
    """
    parallel_config = vllm_config.parallel_config

    # Get NUMA nodes from config or auto-detect
    numa_nodes = parallel_config.numa_node
    if numa_nodes is None and parallel_config.numa_node_auto:
        numa_nodes = get_auto_numa_nodes()

    if numa_nodes is not None:
        # Validate array bounds
        if local_rank >= len(numa_nodes):
            raise ValueError(
                f"local_rank {local_rank} exceeds numa_node array size "
                f"{len(numa_nodes)}. Ensure --numa-node has enough elements "
                "or check that CUDA_VISIBLE_DEVICES matches your GPU count."
            )

        numa_node = numa_nodes[local_rank]
        logger.info(
            "Binding worker (local_rank=%s) to NUMA node %s", local_rank, numa_node
        )
        numactl_args = f"--cpunodebind={numa_node} --membind={numa_node}"
        executable, debug_str = _create_numactl_executable(numactl_args)

        with _mp_set_executable(executable, debug_str):
            yield
    else:
        yield


def _create_numactl_executable(numactl_args: str) -> tuple[str, str]:
    """Create or retrieve cached temporary shell script that wraps Python with numactl.

    Temp files are cached per unique numactl_args to avoid creating multiple files
    for the same NUMA configuration. Files are cleaned up at process exit via atexit.

    The generated script looks like:
        #!/bin/sh
        exec numactl --cpunodebind=0 --membind=0 /usr/bin/python3 "$@"

    Args:
        numactl_args: Arguments to pass to numactl (e.g., "--cpunodebind=0 --membind=0")

    Returns:
        Tuple of (script_path, debug_string)
    """
    # Check cache first - reuse wrapper for same NUMA config
    if numactl_args in _wrapper_cache:
        return _wrapper_cache[numactl_args], f"numactl {numactl_args}"

    old_executable = os.fsdecode(multiprocessing.spawn.get_executable())

    script = f'''#!/bin/sh
exec numactl {numactl_args} {old_executable} "$@"'''

    # Use tempfile.mkstemp for secure file creation (CWE-377 mitigation)
    # Creates file with mode 0o600, uses cryptographically secure random names
    fd, script_path = tempfile.mkstemp(
        suffix=".sh", prefix="vllm_numa_wrapper_", text=True
    )

    # Write using file descriptor (no TOCTOU race condition)
    with os.fdopen(fd, "w") as f:
        f.write(script)

    # Set executable permissions (0o755 = rwxr-xr-x)
    Path(script_path).chmod(0o755)

    # Cache for reuse
    _wrapper_cache[numactl_args] = script_path

    return script_path, f"numactl {numactl_args}"


@contextmanager
def _mp_set_executable(executable: str, debug_str: str):
    """Temporarily replace multiprocessing executable.

    This is the core of the NUMA binding technique. By replacing the
    executable that multiprocessing uses to spawn workers, we can inject
    numactl into the process creation without modifying the worker code.

    The flow is:
    1. Save current Python executable
    2. Replace with our numactl wrapper script
    3. Spawn process (which now runs through numactl)
    4. Restore original executable

    Args:
        executable: Path to wrapper script
        debug_str: Debug information for logging

    Requires:
        Multiprocessing must use 'spawn' start method (not fork).
        Fork doesn't use the executable path.
    """
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
    logger.info("NUMA binding: %s -> %s (%s)", old_executable, executable, debug_str)

    try:
        yield
    finally:
        # Verify it wasn't changed by someone else
        assert os.fsdecode(multiprocessing.spawn.get_executable()) == executable, (
            f"Executable was changed during context: "
            f"expected {executable}, got {multiprocessing.spawn.get_executable()}"
        )
        multiprocessing.spawn.set_executable(old_executable)
        logger.info("NUMA binding restored: %s", old_executable)
