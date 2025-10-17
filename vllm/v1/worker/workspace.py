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
        # List of workspace groups, where each group is a tuple of specs
        # that were reserved together (via reserve or reserve_simultaneous)
        self._reserved_workspaces: list[tuple[WorkspaceSpec, ...]] = []
        # Number of layers in the model (set during adjust_available_kv_cache_memory)
        self._num_layers: int | None = None
        self._current_workspaces: list[torch.Tensor | None] = [None, None]
        self._num_kv_cache_tokens: int | None = None
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

    def _estimate_required_workspace_memory(
        self,
        estimated_max_kv_tokens: int | None = None,
    ) -> int | None:
        # Calculate max workspace bytes across all groups
        # PerKVCacheTokenWorkspace specs are scaled by estimated_max_kv_tokens in-place
        # Regular WorkspaceSpec specs are included as-is (fixed overhead)
        max_workspace_bytes = 0
        for group in self._reserved_workspaces:
            group_bytes = 0
            for spec in group:
                if isinstance(spec, PerKVCacheTokenWorkspace):
                    if estimated_max_kv_tokens is None:
                        # Cannot estimate required workspace memory without
                        # estimated_max_kv_tokens
                        return None

                    # Scale by number of KV cache tokens, then align the total buffer
                    group_bytes += round_up(
                        spec.num_bytes() * estimated_max_kv_tokens, 256
                    )
                else:
                    # Fixed overhead, aligned
                    group_bytes += round_up(spec.num_bytes(), 256)

            max_workspace_bytes = max(max_workspace_bytes, group_bytes)
        return max_workspace_bytes

    def requires_memory_adjustment(self) -> bool:
        """Check if workspace requires memory adjustment."""
        return (
            self._estimate_required_workspace_memory()
            != self.current_allocated_size_bytes()
        )

    def adjust_available_kv_cache_memory(
        self,
        available_memory: int,
        estimated_max_kv_tokens: int,
    ) -> int:
        """Reserve workspace memory by shrinking available KV cache memory.

        Args:
            available_memory: Memory available for KV cache in bytes.
            estimated_max_kv_tokens: Estimated maximum number of KV cache tokens.

        Returns:
            Remaining memory for KV cache after reserving workspace.
        """

        already_allocated = self.current_allocated_size_bytes()
        required_workspace_size = self._estimate_required_workspace_memory(
            estimated_max_kv_tokens
        )
        assert required_workspace_size is not None

        # If already allocated >= reserved, no need to reserve more memory.
        # This can happen if:
        # 1. Workspace already sized appropriately from previous profiling
        # 2. A .get() was called for a buffer not reserved during profiling,
        #    causing the workspace to resize larger than expected. This is fine
        #    since .get() acts as an implicit reserve in that case.
        if already_allocated >= required_workspace_size:
            return available_memory

        workspace_to_reserve = required_workspace_size - already_allocated
        return max(available_memory - workspace_to_reserve, 0)

    def reserve(self, spec: "WorkspaceSpec") -> None:
        """Reserve workspace memory for a given spec.

        For PerKVCacheTokenWorkspace, this just registers the spec.
        For regular WorkspaceSpec, this registers the spec and allocates immediately
        if workspace needs to grow.

        Args:
            spec: The workspace specification.
        """
        # Store as a single-spec group
        self._reserved_workspaces.append((spec,))

        # PerKVCacheTokenWorkspace allocation is deferred until first use
        if isinstance(spec, PerKVCacheTokenWorkspace):
            return

        # Allocate if workspace needs resize
        # Note: both ubatches always have the same size, so we only check the first
        num_bytes = spec.num_bytes()
        if self._workspace_size_bytes(self._current_workspaces[0]) < num_bytes:
            self._increase_size(num_bytes, spec.name)

    def reserve_simultaneous(self, *specs: "WorkspaceSpec") -> None:
        """Reserve workspace memory for multiple specs simultaneously.

        For PerKVCacheTokenWorkspace specs, this just registers them.
        For regular WorkspaceSpec specs, this registers them and allocates
        a single workspace large enough for all if needed.

        Args:
            *specs: One or more workspace specifications.
        """
        # Store as a multi-spec group
        self._reserved_workspaces.append(specs)

        # Separate PerKVCacheTokenWorkspace from regular WorkspaceSpec
        regular_specs = [
            spec for spec in specs if not isinstance(spec, PerKVCacheTokenWorkspace)
        ]

        # PerKVCacheTokenWorkspace allocation is deferred until first use
        if not regular_specs:
            return

        # Calculate total bytes needed for regular specs
        spec_bytes = [spec.num_bytes() for spec in regular_specs]
        aligned_bytes = [round_up(byte_count, 256) for byte_count in spec_bytes]
        total_bytes = sum(aligned_bytes)

        # Allocate if workspace needs resize
        if self._workspace_size_bytes(self._current_workspaces[0]) < total_bytes:
            workspace_names = ", ".join(spec.name for spec in regular_specs)
            self._increase_size(total_bytes, f"[{workspace_names}]")

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

    def get_simultaneous(self, *specs: "WorkspaceSpec") -> list[torch.Tensor]:
        """Get multiple workspace tensors simultaneously from a single allocation.

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
            raise RuntimeError(
                f"PerKVCacheTokenWorkspace '{spec_name}' requested before "
                "KV cache initialization. Allocating minimal workspace.",
            )

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
