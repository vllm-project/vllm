# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from itertools import accumulate
from math import prod
from typing import Optional

import torch

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils import round_up
from vllm.v1.worker.ubatching import dbo_current_ubatch_id

logger = init_logger(__name__)


@dataclass(frozen=True)
class WorkspaceSpec:
    """Specification of a workspace to be allocated.

    Attributes:
        shape: The shape of the workspace.
        dtype: The data type of the workspace.
        name: Optional name for debugging.
    """

    shape: tuple[int, ...]
    dtype: torch.dtype
    name: str = "unnamed"

    def num_bytes(self) -> int:
        return prod(self.shape) * self.dtype.itemsize


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

    def __init__(self, device: torch.device, vllm_config):
        self._device = device
        self._vllm_config = vllm_config
        # Cache num ubatches at init based on configuration
        self._num_ubatches = 2 if vllm_config.parallel_config.enable_dbo else 1
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

    def current_allocated_size_bytes(self) -> int:
        """Get the size of the current workspace in bytes."""
        return self._workspace_size_bytes(
            self._current_workspaces[dbo_current_ubatch_id()]
        )

    def reserve(self, spec: "WorkspaceSpec") -> None:
        """Reserve workspace memory for a given spec.

        This is a convenience wrapper around get() that makes it easier to grep
        for workspace reservations in the codebase for auditing purposes.

        Args:
            spec: The workspace specification.
        """
        # TODO(Lucas): Assert that only reserves (reserve/reserve_simultaneous) can
        # increase the workspace size, so that reserve must be called before `get`.
        # This will encourage the use of reserve which is mostly just useful for
        # grepping/auditing the codebase.
        self.get(spec)

    def reserve_simultaneous(self, *specs: "WorkspaceSpec") -> None:
        """Reserve workspace memory for multiple specs simultaneously.

        This is a convenience wrapper around get_simultaneous() that makes it easier
        to grep for workspace reservations in the codebase for auditing purposes.

        Args:
            *specs: One or more workspace specifications.
        """
        # TODO(Lucas): Assert that only reserves (ds/reserve_simultaneous) can
        # increase the workspace size, so that reserve must be called before `get`.
        # This will encourage the use of reserve which is mostly just useful for
        # grepping/auditing the codebase.
        self.get_simultaneous(*specs)

    def get(self, spec: "WorkspaceSpec") -> torch.Tensor:
        """Get a workspace tensor for the given spec.

        Args:
            spec: The workspace specification.

        Returns:
            A tensor view into the workspace buffer with the requested shape and dtype.
        """
        num_bytes = spec.num_bytes()
        current_workspace = self._ensure_workspace_size(num_bytes, spec.name)
        return current_workspace[:num_bytes].view(spec.dtype).reshape(spec.shape)

    def get_simultaneous(self, *specs: "WorkspaceSpec") -> list[torch.Tensor]:
        """Get multiple workspace tensors simultaneously from a single allocation.

        Args:
            *specs: One or more workspace specifications.

        Returns:
            List of tensor views into the workspace buffer, one per spec.
        """
        actual_bytes = [spec.num_bytes() for spec in specs]
        aligned_bytes = [round_up(actual, 256) for actual in actual_bytes]
        total_bytes = sum(aligned_bytes)

        # Calculate cumulative offsets using itertools.accumulate
        offsets = list(accumulate([0] + aligned_bytes[:-1]))

        workspace_names = ", ".join(spec.name for spec in specs)
        current_workspace = self._ensure_workspace_size(
            total_bytes, f"[{workspace_names}]"
        )

        return [
            current_workspace[offsets[i] : offsets[i] + actual_bytes[i]]
            .view(specs[i].dtype)
            .reshape(specs[i].shape)
            for i in range(len(specs))
        ]

    def _ensure_workspace_size(self, num_bytes: int, name: str) -> torch.Tensor:
        """Ensure workspace is allocated and large enough, return current workspace."""
        ubatch_id = dbo_current_ubatch_id()
        current_workspace = self._current_workspaces[ubatch_id]

        # Manager owns a single device; no cross-device assertions needed

        if self._workspace_size_bytes(current_workspace) < num_bytes:
            self._increase_size(num_bytes, name)
            current_workspace = self._current_workspaces[ubatch_id]

        return current_workspace

    def _increase_size(
        self,
        required_bytes: int,
        name: str = "unnamed",
    ) -> None:
        """Allocate or resize workspace for all ubatches.

        If DBO is enabled, allocates for both ubatches. Otherwise, allocates for
        ubatch 0. Uses PyTorch's resize_() for efficient in-place resizing when
        possible.

        Invariant: Both ubatches always have the same size after this function
        completes.

        Args:
            required_bytes: The number of bytes required.
            name: Name for debugging/logging.
        """
        current_size = self._workspace_size_bytes(self._current_workspaces[0])
        if self._locked and current_size < required_bytes:
            raise AssertionError(
                f"Workspace is locked but allocation for '{name}' requires "
                f"{required_bytes / _MB:.2f} MB, current size is "
                f"{current_size / _MB:.2f} MB. "
                "Workspace growth is not allowed after locking."
            )

        for ubatch_id in range(self._num_ubatches):
            current_workspace = self._current_workspaces[ubatch_id]

            if current_workspace is None:
                self._current_workspaces[ubatch_id] = torch.empty(
                    (required_bytes,), dtype=torch.uint8, device=self._device
                )
            elif self._workspace_size_bytes(current_workspace) < required_bytes:
                # Use resize_() for efficient in-place resizing
                current_workspace.resize_(required_bytes)

        if envs.VLLM_DEBUG_WORKSPACE:
            total_mb = required_bytes * self._num_ubatches / _MB
            logger.info(
                "[WORKSPACE DEBUG] Resized workspace '%s': %.2f MB -> %.2f "
                "MB (%d ubatches, total memory %.2f MB)",
                name,
                current_size / _MB,
                required_bytes / _MB,
                self._num_ubatches,
                total_mb,
            )


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


def init_workspace_manager(device: torch.device, vllm_config: VllmConfig) -> None:
    """Initialize the workspace manager with a device.

    Must be called before using any workspace functions. Typically called
    from GPUModelRunner.__init__.

    Args:
        device: The device to allocate workspace on.
    """
    global _manager
    if _manager is not None:
        logger.warning(
            "WorkspaceManager already initialized on device %s, "
            "reinitializing on device %s",
            _manager._device,
            device,
        )
    _manager = WorkspaceManager(device, vllm_config)


def lock_workspace() -> None:
    """Lock the workspace to prevent further growth.

    After calling this function, any attempt to allocate a workspace larger
    than the current size will raise an AssertionError. This ensures that
    workspace size is fixed during execution and prevents unexpected memory
    allocations in the hot path.

    Example:
        # During initialization
        init_workspace_manager(device)
        reserve_workspace(spec1)
        reserve_workspace(spec2)

        # Lock after warmup/profiling
        lock_workspace()

        # Now all get_workspace calls must fit in pre-allocated size
    """
    current_workspace_manager().lock()
