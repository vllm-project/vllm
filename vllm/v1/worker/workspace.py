# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import inspect
import os
from itertools import accumulate
from math import prod
from typing import Optional

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.utils.math_utils import round_up
from vllm.v1.worker.ubatching import dbo_current_ubatch_id

logger = init_logger(__name__)


def _compute_bytes(shape: tuple[int, ...], dtype: torch.dtype) -> int:
    return prod(shape) * dtype.itemsize


# Constants
_MB = 1024**2
_GiB = 1024**3

# Global workspace manager instance
_manager: Optional["WorkspaceManager"] = None


class WorkspaceManager:
    """Manager for workspace allocation.

    Manages workspace buffers for DBO (Dual Batch Overlap) execution.
    Can be locked to prevent further growth during execution.
    """

    def __init__(self, device: torch.device, num_ubatches: int | None = None):
        self._device = device
        # Cache num ubatches at init based on configuration (default to 1)
        self._num_ubatches = num_ubatches if num_ubatches is not None else 1
        self._current_workspaces: list[torch.Tensor | None] = [None, None]
        self._locked: bool = False

    @staticmethod
    def _workspace_size_bytes(workspace: torch.Tensor | None) -> int:
        """Get size of workspace in bytes."""
        if workspace is None:
            return 0
        return workspace.numel() * workspace.element_size()

    def lock(self) -> None:
        """Lock the workspace to prevent further growth.

        After locking, any attempt to allocate a larger workspace will raise
        an assertion error. This ensures workspace size is fixed during execution.
        """
        self._locked = True
        if envs.VLLM_DEBUG_WORKSPACE:
            logger.info(
                "[WORKSPACE DEBUG] Workspace locked. Current sizes: %s",
                [
                    self._workspace_size_bytes(ws) / _MB
                    for ws in self._current_workspaces
                    if ws is not None
                ],
            )

    def is_locked(self) -> bool:
        """Check if workspace is locked."""
        return self._locked

    def get_simultaneous(
        self, *shapes_and_dtypes: tuple[tuple[int, ...], torch.dtype]
    ) -> list[torch.Tensor]:
        """Get multiple workspace tensors simultaneously from a single allocation.

        Args:
            *shapes_and_dtypes: One or more (shape, dtype) tuples.

        Returns:
            List of tensor views into the workspace buffer, one per shape/dtype pair.
        """
        actual_bytes = [_compute_bytes(s, d) for s, d in shapes_and_dtypes]
        aligned_bytes = [round_up(actual, 256) for actual in actual_bytes]
        total_bytes = sum(aligned_bytes)

        # Calculate cumulative offsets using itertools.accumulate
        offsets = list(accumulate([0] + aligned_bytes[:-1]))

        current_workspace = self._ensure_workspace_size(total_bytes)

        return [
            current_workspace[offsets[i] : offsets[i] + actual_bytes[i]]
            .view(shapes_and_dtypes[i][1])
            .reshape(shapes_and_dtypes[i][0])
            for i in range(len(shapes_and_dtypes))
        ]

    def _ensure_workspace_size(self, required_bytes: int) -> torch.Tensor:
        """Ensure workspace is allocated and large enough, return current workspace.

        Args:
            required_bytes: The number of bytes required.

        Returns:
            The current workspace tensor.
        """
        ubatch_id = dbo_current_ubatch_id()
        current_workspace = self._current_workspaces[ubatch_id]
        current_size = self._workspace_size_bytes(current_workspace)

        if current_size < required_bytes:

            def get_caller_info() -> str:
                """Find first frame outside WorkspaceManager."""
                curr_frame = inspect.currentframe()
                if curr_frame is None:
                    return "unknown"
                # Walk up the stack skipping WorkspaceManager frames
                curr_frame = curr_frame.f_back
                while curr_frame is not None:
                    # TODO: This only catches instance methods (self), missing
                    # classmethods and staticmethods. Once Python 3.11+ is the
                    # minimum supported version, use co_qualname instead:
                    #   qualname = curr_frame.f_code.co_qualname
                    #   if qualname.startswith("WorkspaceManager."):
                    if isinstance(curr_frame.f_locals.get("self"), WorkspaceManager):
                        curr_frame = curr_frame.f_back
                        continue
                    filename = os.path.basename(curr_frame.f_code.co_filename)
                    return (
                        f"{filename}:{curr_frame.f_lineno}:{curr_frame.f_code.co_name}"
                    )
                return "unknown"

            if self._locked:
                raise AssertionError(
                    f"Workspace is locked but allocation from '{get_caller_info()}' "
                    f"requires {required_bytes / _MB:.2f} MB, current size is "
                    f"{current_size / _MB:.2f} MB. "
                    "Workspace growth is not allowed after locking."
                )

            for ubatch_id in range(self._num_ubatches):
                current_workspace = self._current_workspaces[ubatch_id]
                if (
                    current_workspace is None
                    or self._workspace_size_bytes(current_workspace) < required_bytes
                ):
                    # Delete old tensor before allocating new one to avoid
                    # memory spike from resize_(). resize_() allocates new
                    # memory before freeing old, which can cause OOM.
                    # Must clear the list reference first since local var
                    # is just a copy of the reference.
                    self._current_workspaces[ubatch_id] = None
                    del current_workspace
                    self._current_workspaces[ubatch_id] = torch.empty(
                        (required_bytes,), dtype=torch.uint8, device=self._device
                    )

            if envs.VLLM_DEBUG_WORKSPACE:
                logger.info(
                    "[WORKSPACE DEBUG] Resized workspace from '%s': %.2f MB -> "
                    "%.2f MB (%d ubatches, total memory %.2f MB)",
                    get_caller_info(),
                    current_size / _MB,
                    required_bytes / _MB,
                    self._num_ubatches,
                    required_bytes * self._num_ubatches / _MB,
                )

            current_workspace = self._current_workspaces[dbo_current_ubatch_id()]

        return current_workspace


def is_workspace_manager_initialized() -> bool:
    """Check if workspace manager has been initialized.

    Returns:
        True if workspace manager is initialized, False otherwise.
    """
    return _manager is not None


def current_workspace_manager() -> "WorkspaceManager":
    """Get the current workspace manager instance.

    Raises:
        AssertionError: If workspace manager has not been initialized.
    """
    assert _manager is not None, (
        "WorkspaceManager not initialized. Call init_workspace_manager() "
        "with a device before using workspace functions."
    )
    return _manager


def init_workspace_manager(
    device: torch.device, num_ubatches: int | None = None
) -> None:
    """Initialize the workspace manager with a device.

    Must be called before using any workspace functions. Typically called
    from GPUModelRunner.__init__.

    Args:
        device: The device to allocate workspace on.
        num_ubatches: Number of micro-batches. Defaults to 1.
    """
    global _manager
    if _manager is not None:
        logger.warning(
            "WorkspaceManager already initialized on device %s, "
            "reinitializing on device %s",
            _manager._device,
            device,
        )
    _manager = WorkspaceManager(device, num_ubatches)


def lock_workspace() -> None:
    """Lock the workspace to prevent further growth.

    After calling this function, any attempt to allocate a workspace larger
    than the current size will raise an AssertionError. This ensures that
    workspace size is fixed during execution and prevents unexpected memory
    allocations in the hot path.

    Example:
        # During initialization
        init_workspace_manager(device)
        reserve_workspace(shape1, dtype1)
        reserve_workspace(shape2, dtype2)

        # Lock after warmup/profiling
        lock_workspace()

        # Now all get_workspace calls must fit in pre-allocated size
    """
    current_workspace_manager().lock()


def reset_workspace_manager() -> None:
    """Reset the workspace manager to uninitialized state.

    This is primarily intended for testing purposes to allow tests
    to reinitialize the workspace manager cleanly.
    """
    global _manager
    _manager = None
