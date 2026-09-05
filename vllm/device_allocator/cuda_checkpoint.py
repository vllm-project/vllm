# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CUDA checkpoint/restore wrapper using cuCheckpointProcess* APIs.

Provides in-process GPU state preservation (compiled kernels, torch.compile
artifacts, CUDA graphs) across suspend/resume cycles for near-zero cold
start times. Requires NVIDIA driver >= 570.
"""

import os

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

# CUprocessState values returned by the driver (see cuda.h).
PROCESS_STATE_RUNNING = 0
PROCESS_STATE_LOCKED = 1
PROCESS_STATE_CHECKPOINTED = 2
PROCESS_STATE_FAILED = 3

cuda_checkpoint_available = False
try:
    from vllm.cuda_checkpoint import (
        get_state,
        is_available,
        process_checkpoint,
        process_lock,
        process_restore,
        process_unlock,
    )

    cuda_checkpoint_available = is_available()
except ModuleNotFoundError:
    # C extension not built (non-CUDA platform or build without it)
    process_lock = None
    process_checkpoint = None
    process_restore = None
    process_unlock = None
    get_state = None
    is_available = None


class CudaCheckpointer:
    """Singleton class for CUDA process checkpoint/restore.

    Wraps the cuCheckpointProcess* driver APIs to suspend and resume
    GPU state, preserving compiled kernels, CUDA graphs, and
    torch.compile artifacts across cycles.
    """

    instance: "CudaCheckpointer | None" = None

    @staticmethod
    def get_instance() -> "CudaCheckpointer":
        assert cuda_checkpoint_available, (
            "CUDA checkpoint is not available. "
            "Requires NVIDIA driver >= 570 and CUDA platform."
        )
        if CudaCheckpointer.instance is None:
            CudaCheckpointer.instance = CudaCheckpointer()
        return CudaCheckpointer.instance

    def __init__(self):
        self._is_suspended = False
        # The checkpoint API is keyed by process id; there is no opaque
        # handle. We record the pid that was checkpointed and return it
        # from suspend() so callers have a stable token to pass to resume().
        self._checkpoint_pid: int | None = None
        # Retain the restored process until unlocking succeeds.
        self._restored_pid: int | None = None

    @property
    def is_suspended(self) -> bool:
        return self._is_suspended

    def suspend(self) -> int:
        """Suspend the CUDA process, preserving GPU state.

        Synchronizes all CUDA streams, then locks and checkpoints the
        process (RUNNING -> LOCKED -> CHECKPOINTED), releasing GPU memory
        to host.

        Returns:
            The checkpointed process id, usable as the ``handle`` for a
            later resume() call.

        Raises:
            RuntimeError: If already suspended or a CUDA API fails.
        """
        if self._is_suspended:
            raise RuntimeError(
                "CUDA process is already suspended. "
                "Call resume() before suspending again."
            )

        # Synchronize all CUDA streams before checkpoint
        torch.cuda.synchronize()

        pid = os.getpid()
        logger.info("Suspending CUDA process (pid=%d)...", pid)
        # Two-step checkpoint sequence: lock blocks further CUDA API calls,
        # checkpoint moves GPU state to host memory.
        process_lock(pid)
        process_checkpoint(pid)
        self._checkpoint_pid = pid
        self._is_suspended = True
        logger.info("CUDA process suspended (pid=%d).", pid)
        return pid

    def recover(self) -> None:
        """Return the CUDA process to RUNNING after a failed suspension.

        A failed driver call can leave the process locked or checkpointed
        before suspend() records its state. Query the driver so recovery
        also handles these partially completed suspensions.
        """
        pid = os.getpid()
        state = self.get_state(pid)
        if state == PROCESS_STATE_RUNNING:
            self._restored_pid = None
            self._is_suspended = False
            return

        self._checkpoint_pid = pid
        self._is_suspended = True
        if state == PROCESS_STATE_LOCKED:
            self._restored_pid = pid
        elif state == PROCESS_STATE_CHECKPOINTED:
            self._restored_pid = None
        else:
            raise RuntimeError(
                f"Cannot recover CUDA process {pid} from driver state {state}."
            )
        self.resume(pid)

    def resume(self, handle: int | None = None) -> None:
        """Resume the CUDA process from a checkpoint.

        Restores GPU state and unlocks the process
        (CHECKPOINTED -> LOCKED -> RUNNING). If unlocking fails, a retry
        only unlocks the already restored process.

        Args:
            handle: The pid returned by suspend(). If None, uses the pid
                from the last suspend() call.

        Raises:
            RuntimeError: If not suspended or a CUDA API fails.
        """
        if not self._is_suspended:
            raise RuntimeError(
                "CUDA process is not suspended. Call suspend() before resume()."
            )

        if handle is None:
            handle = (
                self._restored_pid
                if self._restored_pid is not None
                else self._checkpoint_pid
            )

        if handle is None:
            raise RuntimeError("No checkpoint pid available for resume.")

        logger.info("Resuming CUDA process (pid=%s)...", handle)
        # Inverse of suspend: restore GPU state, then unlock.
        if self._restored_pid is None:
            process_restore(handle)
            self._restored_pid = handle
        elif handle != self._restored_pid:
            raise RuntimeError(
                f"CUDA process {self._restored_pid} is restored but still locked. "
                "Retry resume() for that process before resuming another."
            )
        process_unlock(handle)
        self._restored_pid = None
        self._is_suspended = False
        logger.info("CUDA process resumed.")

    def get_state(self, handle: int | None = None) -> int:
        """Query the CUprocessState of the process.

        Args:
            handle: The pid to query. If None, uses the pid from the last
                suspend() call, falling back to the current process.

        Returns:
            Integer CUprocessState value from the CUDA driver (see the
            ``PROCESS_STATE_*`` constants in this module).
        """
        if handle is None:
            handle = self._checkpoint_pid
        if handle is None:
            handle = os.getpid()

        return get_state(handle)
