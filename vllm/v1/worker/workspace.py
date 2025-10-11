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


class PerKVCacheTokenWorkspace(WorkspaceSpec):
    """Workspaces for per-key-value caching.

    Attributes:
        key: The workspace for the key cache.
        value: The workspace for the value cache.
    """

    pass


# Constants
_MB = 1024**2
_GiB = 1024**3

# Global workspace manager instance
_manager: Optional["WorkspaceManager"] = None


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


class WorkspaceManager:
    """Manager for workspace allocation.

    Manages workspace buffers for DBO (Dual Batch Overlap) execution.
    Can be locked to prevent further growth during execution.
    """

    def __init__(self, device: torch.device, vllm_config):
        self._device = device
        self._vllm_config = vllm_config
        self._cache_config = vllm_config.cache_config
        # Cache num ubatches at init based on configuration
        self._num_ubatches = 2 if vllm_config.parallel_config.enable_dbo else 1
        self._reserved_workspaces: dict[str, WorkspaceSpec] = {}
        self._current_workspaces: list[Optional[torch.Tensor]] = [None, None]
        self._num_kv_cache_tokens: Optional[int] = None
        self._locked: bool = False

    @staticmethod
    def _workspace_size_bytes(workspace: Optional[torch.Tensor]) -> int:
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

    def adjust_available_memory(
        self,
        available_memory: int,
        per_token_workspace_bytes: int,
        estimated_max_kv_tokens: int,
    ) -> int:
        """Reserve workspace memory by shrinking available KV memory.

        Args:
            available_memory: Memory available for KV cache in bytes.
            per_token_workspace_bytes: Workspace memory required per KV cache token.
            estimated_max_kv_tokens: Estimated maximum number of KV cache tokens.

        Returns:
            Remaining memory for KV cache after reserving workspace.
        """
        if per_token_workspace_bytes == 0:
            return available_memory

        already_allocated = self.current_allocated_size_bytes()
        expected_workspace = per_token_workspace_bytes * estimated_max_kv_tokens

        if already_allocated > expected_workspace:
            return available_memory

        workspace_to_reserve = expected_workspace - already_allocated
        adjusted_available_memory = max(available_memory - workspace_to_reserve, 0)

        return adjusted_available_memory

    def reserve(self, spec: "WorkspaceSpec") -> None:
        """Reserve workspace memory for a given spec.

        For PerKVCacheTokenWorkspace, this just registers the spec.
        For regular WorkspaceSpec, this registers the spec and allocates immediately
        if workspace needs to grow.

        Args:
            spec: The workspace specification.
        """
        # Store the spec by name
        if spec.name in self._reserved_workspaces:
            logger.warning("Workspace '%s' already reserved, overwriting", spec.name)
        self._reserved_workspaces[spec.name] = spec

        # PerKVCacheTokenWorkspace allocation is deferred until first use
        if isinstance(spec, PerKVCacheTokenWorkspace):
            return

        # Allocate if workspace needs resize
        # Note: both ubatches always have the same size, so we only check the first
        num_bytes = spec.num_bytes()
        if self._workspace_size_bytes(self._current_workspaces[0]) < num_bytes:
            self._increase_size(num_bytes, spec.name)

    def get(self, spec: "WorkspaceSpec") -> torch.Tensor:
        """Get a workspace tensor for the given spec.

        Args:
            spec: The workspace specification.

        Returns:
            A tensor view into the workspace buffer with the requested shape and dtype.
        """
        shape, num_bytes = self._shape_and_bytes_for_spec(spec)
        current_workspace = self._ensure_workspace_size(num_bytes, spec.name)
        return current_workspace[:num_bytes].view(spec.dtype).reshape(shape)

    def _shape_and_bytes_for_spec(
        self, spec: "WorkspaceSpec"
    ) -> tuple[tuple[int, ...], int]:
        """Return adjusted shape and actual size for a workspace spec."""
        num_bytes = spec.num_bytes()
        shape = spec.shape

        if isinstance(spec, PerKVCacheTokenWorkspace):
            num_tokens, multiplier = self._get_kv_cache_multiplier(spec.name)
            shape = (num_tokens, *spec.shape)
            num_bytes *= multiplier

        return shape, num_bytes

    def get_multiple(self, *specs: "WorkspaceSpec") -> list[torch.Tensor]:
        """Get multiple workspace tensors efficiently from a single allocation.

        Args:
            *specs: One or more workspace specifications.

        Returns:
            List of tensor views into the workspace buffer, one per spec.
        """
        adjusted = [self._shape_and_bytes_for_spec(spec) for spec in specs]
        adjusted_shapes = [shape for shape, _ in adjusted]
        actual_bytes = [actual for _, actual in adjusted]
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
            .reshape(adjusted_shapes[i])
            for i in range(len(specs))
        ]

    def per_kv_cache_token_workspace_size_bytes(self) -> int:
        """Get the maximum per-KV-cache-token workspace size in bytes."""
        return max(
            (
                spec.num_bytes()
                for spec in self._reserved_workspaces.values()
                if isinstance(spec, PerKVCacheTokenWorkspace)
            ),
            default=0,
        )

    def _get_kv_cache_multiplier(self, spec_name: str) -> tuple[int, int]:
        """Get KV cache token count and multiplier for shape calculation.

        Returns:
            Tuple of (num_tokens, multiplier) where shape becomes
            (num_tokens, *spec.shape) and num_bytes is multiplied by
            multiplier.
        """
        if (
            self._num_kv_cache_tokens is None
            and self._cache_config.num_gpu_blocks is not None
        ):
            self._num_kv_cache_tokens = (
                self._cache_config.num_gpu_blocks * self._cache_config.block_size
            )

        if self._num_kv_cache_tokens is None:
            # KV cache not initialized - use minimal workspace
            import warnings

            warnings.warn(
                f"PerKVCacheTokenWorkspace '{spec_name}' requested before "
                "KV cache initialization. Allocating minimal workspace.",
                stacklevel=4,
            )
            return (1, 1)

        return (self._num_kv_cache_tokens, self._num_kv_cache_tokens)

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
        # Manager owns a single device; no cross-device assertions needed

        # Check if we need to grow the workspace
        current_size = self._workspace_size_bytes(self._current_workspaces[0])
        if self._locked and current_size < required_bytes:
            raise AssertionError(
                f"Workspace is locked but allocation for '{name}' requires "
                f"{required_bytes / _MB:.2f} MB, current size is "
                f"{current_size / _MB:.2f} MB. "
                "Workspace growth is not allowed after locking."
            )

        was_unallocated = self._current_workspaces[0] is None

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
            if was_unallocated:
                logger.info(
                    "[WORKSPACE DEBUG] Allocated workspace '%s': %.2f MB "
                    "(%d ubatches, total memory %.2f MB)",
                    name,
                    required_bytes / _MB,
                    self._num_ubatches,
                    total_mb,
                )
            else:
                logger.info(
                    "[WORKSPACE DEBUG] Resized workspace '%s': %.2f MB -> %.2f "
                    "MB (%d ubatches, total memory %.2f MB)",
                    name,
                    current_size / _MB,
                    required_bytes / _MB,
                    self._num_ubatches,
                    total_mb,
                )


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
