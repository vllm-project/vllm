# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CUDA checkpoint/restore wrapper using cuCheckpointProcess* APIs.

Provides in-process GPU state preservation (compiled kernels, torch.compile
artifacts, CUDA graphs) across suspend/resume cycles for near-zero cold
start times. Requires NVIDIA driver >= 570.
"""

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

cuda_checkpoint_available = False
try:
    from vllm.cuda_checkpoint import (
        checkpoint_get_state,
        checkpoint_resume,
        checkpoint_suspend,
        is_available,
    )

    cuda_checkpoint_available = is_available()
except ModuleNotFoundError:
    # C extension not built (non-CUDA platform or build without it)
    checkpoint_suspend = None
    checkpoint_resume = None
    checkpoint_get_state = None
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
        self._checkpoint_handle: int | None = None

    @property
    def is_suspended(self) -> bool:
        return self._is_suspended

    def suspend(self) -> int:
        """Suspend the CUDA process, preserving GPU state.

        Synchronizes all CUDA streams before suspending.

        Returns:
            Checkpoint handle (integer) for later resume.

        Raises:
            RuntimeError: If already suspended or CUDA API fails.
        """
        if self._is_suspended:
            raise RuntimeError(
                "CUDA process is already suspended. "
                "Call resume() before suspending again."
            )

        # Synchronize all CUDA streams before checkpoint
        torch.cuda.synchronize()

        logger.info("Suspending CUDA process...")
        handle = checkpoint_suspend()
        self._checkpoint_handle = handle
        self._is_suspended = True
        logger.info("CUDA process suspended with handle %s.", handle)
        return handle

    def resume(self, handle: int | None = None) -> None:
        """Resume the CUDA process from a checkpoint.

        Args:
            handle: Checkpoint handle from suspend(). If None, uses
                the handle from the last suspend() call.

        Raises:
            RuntimeError: If not suspended or CUDA API fails.
        """
        if not self._is_suspended:
            raise RuntimeError(
                "CUDA process is not suspended. Call suspend() before resume()."
            )

        if handle is None:
            handle = self._checkpoint_handle

        if handle is None:
            raise RuntimeError("No checkpoint handle available for resume.")

        logger.info("Resuming CUDA process from handle %s...", handle)
        checkpoint_resume(handle)
        self._is_suspended = False
        logger.info("CUDA process resumed.")

    def get_state(self, handle: int | None = None) -> int:
        """Query the state of a checkpoint.

        Args:
            handle: Checkpoint handle to query. If None, uses the
                handle from the last suspend() call.

        Returns:
            Integer state value from the CUDA driver.
        """
        if handle is None:
            handle = self._checkpoint_handle

        if handle is None:
            raise RuntimeError("No checkpoint handle available to query.")

        return checkpoint_get_state(handle)
