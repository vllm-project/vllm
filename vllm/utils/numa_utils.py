# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NUMA binding utilities for vLLM worker processes.

This module provides NUMA (Non-Uniform Memory Access) binding support for vLLM
worker processes to optimize performance on multi-socket systems. It uses a
technique of wrapping the Python executable with numactl to ensure all memory
allocations happen on the correct NUMA node.

Based on SGLang's NUMA binding implementation.
"""

import logging
import multiprocessing
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = logging.getLogger(__name__)


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
        - vllm_config.parallel_config.numa_node must be set
    """
    parallel_config = vllm_config.parallel_config

    if (numa_nodes := parallel_config.numa_node) is not None:
        # Validate array bounds
        if local_rank >= len(numa_nodes):
            raise ValueError(
                f"local_rank {local_rank} exceeds numa_node array size "
                f"{len(numa_nodes)}. Ensure --numa-node has enough elements."
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
    """Create a temporary shell script that wraps Python with numactl.

    The generated script looks like:
        #!/bin/sh
        exec numactl --cpunodebind=0 --membind=0 /usr/bin/python3 "$@"

    Args:
        numactl_args: Arguments to pass to numactl (e.g., "--cpunodebind=0 --membind=0")

    Returns:
        Tuple of (script_path, debug_string)
    """
    old_executable = os.fsdecode(multiprocessing.spawn.get_executable())

    script = f'''#!/bin/sh
exec numactl {numactl_args} {old_executable} "$@"'''

    # Write to unique temporary file
    path = Path(
        f"/tmp/vllm_numa_wrapper_{time.time()}_{random.randrange(0, 10000000)}.sh"
    )
    path.write_text(script)
    path.chmod(0o777)

    return str(path), f"numactl {numactl_args}"


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
    start_method = multiprocessing.get_start_method()
    if start_method != "spawn":
        logger.warning(
            "NUMA binding requires spawn method but got %s. "
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
