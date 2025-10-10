# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from dataclasses import dataclass
from math import prod
from typing import Optional

import torch

from vllm.config import get_current_vllm_config
from vllm.logger import init_logger
from vllm.utils import round_up
from vllm.v1.worker.ubatching import dbo_current_ubatch_id

logger = init_logger(__name__)

_REQUIRED_WORKSPACES: dict[str, "WorkspaceSpec"] = {}
_CURRENT_WORKSPACES: list[torch.Tensor] = [None, None]
_NUM_KV_CACHE_TOKENS: Optional[int] = None
_DEBUG_WORKSPACE: bool = os.environ.get("VLLM_DEBUG_WORKSPACE", "0").lower() in (
    "1",
    "true",
    "yes",
    "all",
)


def _increase_size(
    required_bytes: int,
    device: torch.device,
    name: str = "unnamed",
) -> None:
    """Allocate or resize workspace for all ubatches.

    If DBO is enabled, allocates for both ubatches. Otherwise, allocates for ubatch 0.
    Uses PyTorch's resize_() for efficient in-place resizing when possible.

    Args:
        required_bytes: The number of bytes required.
        device: The device to allocate on.
        name: Name for debugging/logging.
    """
    global _CURRENT_WORKSPACES

    # Determine number of ubatches based on DBO configuration
    from vllm.config import get_current_vllm_config

    vllm_config = get_current_vllm_config()
    num_ubatches = 2 if vllm_config.parallel_config.enable_dbo else 1

    for ubatch_id in range(num_ubatches):
        current_workspace = _CURRENT_WORKSPACES[ubatch_id]

        # Assert device matches if workspace already exists
        if current_workspace is not None:
            assert current_workspace.device == device, (
                f"Workspace device mismatch for '{name}': "
                f"existing={current_workspace.device}, requested={device}"
            )

        # Check if we need to allocate/resize
        if current_workspace is None:
            # First allocation
            if _DEBUG_WORKSPACE:
                logger.info(
                    "[WORKSPACE DEBUG] Allocating workspace '%s' "
                    "for ubatch %d: %.2f MB",
                    name,
                    ubatch_id,
                    required_bytes / (1024**2),
                )

            current_workspace = torch.empty(
                (required_bytes,), dtype=torch.uint8, device=device
            )
            _CURRENT_WORKSPACES[ubatch_id] = current_workspace
        elif (
            current_workspace.numel() * current_workspace.element_size()
            < required_bytes
        ):
            # Resize existing workspace using resize_()
            if _DEBUG_WORKSPACE:
                old_size = current_workspace.numel() * current_workspace.element_size()
                logger.info(
                    "[WORKSPACE DEBUG] Resizing workspace '%s' "
                    "for ubatch %d: %.2f MB -> %.2f MB",
                    name,
                    ubatch_id,
                    old_size / (1024**2),
                    required_bytes / (1024**2),
                )

            # Use resize_() for efficient in-place resizing
            current_workspace.resize_(required_bytes)


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
        return prod(self.shape) * torch.tensor([], dtype=self.dtype).element_size()


class PerKVCacheTokenWorkspace(WorkspaceSpec):
    """Workspaces for per-key-value caching.

    Attributes:
        key: The workspace for the key cache.
        value: The workspace for the value cache.
    """

    pass


def current_workspace_size_bytes() -> int:
    current_workspace = _CURRENT_WORKSPACES[dbo_current_ubatch_id()]
    if current_workspace is None:
        return 0
    return current_workspace.element_size() * current_workspace.numel()


def adjust_available_memory_to_account_for_workspaces(
    available_memory: int,
    per_token_workspace_bytes: int,
    estimated_max_kv_tokens: int,
) -> int:
    """Adjust available memory accounting for reserved workspace.

    If current workspace is not large enough yet, adds back the already-allocated
    workspace memory to avoid double-counting during KV cache allocation.

    Args:
        available_memory: Memory available for KV cache in bytes.
        per_token_workspace_bytes: Workspace memory required per KV cache token.
        estimated_max_kv_tokens: Estimated maximum number of KV cache tokens.

    Returns:
        Adjusted available memory in bytes.
    """
    if per_token_workspace_bytes == 0:
        return available_memory

    already_allocated_workspace_bytes = current_workspace_size_bytes()

    # Calculate the maximum expected workspace size based on estimated KV cache tokens
    max_expected_workspace_bytes = per_token_workspace_bytes * estimated_max_kv_tokens

    # Only add back workspace memory up to the expected amount to avoid over-allocation
    workspace_to_add_back = min(
        already_allocated_workspace_bytes, max_expected_workspace_bytes
    )

    if workspace_to_add_back < already_allocated_workspace_bytes:
        # Convert to GiB for logging
        GiB_bytes = 1024**3
        logger.warning(
            "Already-allocated workspace (%.2f GB) exceeds expected "
            "workspace size (%.2f GB). Only adding back %.2f GB to "
            "avoid over-allocating KV cache.",
            already_allocated_workspace_bytes / GiB_bytes,
            max_expected_workspace_bytes / GiB_bytes,
            workspace_to_add_back / GiB_bytes,
        )

    return available_memory + workspace_to_add_back


def reserve_workspace(spec: WorkspaceSpec, device: Optional[torch.device] = None):
    """Reserve workspace memory for a given spec.

    For PerKVCacheTokenWorkspace, this just registers the spec.
    For regular WorkspaceSpec, this registers the spec and optionally allocates
    immediately if device is provided. If device is None, allocation is deferred
    until first get_workspace() call.

    If workspace is already allocated (any ubatch), uses that device if device=None.
    If DBO is enabled and device is provided, allocates workspace for both ubatches.

    Args:
        spec: The workspace specification.
        device: The device to allocate on (optional, allocation deferred if None).
    """
    global _REQUIRED_WORKSPACES, _CURRENT_WORKSPACES

    # Store the spec by name
    if spec.name in _REQUIRED_WORKSPACES:
        logger.warning("Workspace '%s' already reserved, overwriting", spec.name)
    _REQUIRED_WORKSPACES[spec.name] = spec

    # For non-PerKVCacheTokenWorkspace, allocate immediately if device is provided
    # or if workspace is already allocated (use existing device)
    if not isinstance(spec, PerKVCacheTokenWorkspace):
        # If device not specified, check if workspace already exists and use its device
        if device is None:
            for ws in _CURRENT_WORKSPACES:
                if ws is not None:
                    device = ws.device
                    break

        # Allocate if we have a device
        if device is not None:
            num_bytes = spec.num_bytes()

            # Determine which ubatch(s) to check
            from vllm.config import get_current_vllm_config

            vllm_config = get_current_vllm_config()
            num_ubatches = 2 if vllm_config.parallel_config.enable_dbo else 1

            # Check if any ubatch needs allocation/resize
            needs_resize = False
            for ubatch_id in range(num_ubatches):
                ws = _CURRENT_WORKSPACES[ubatch_id]
                if ws is None or ws.numel() * ws.element_size() < num_bytes:
                    needs_resize = True
                    break

            # Only call _increase_size if actually needed
            if needs_resize:
                _increase_size(num_bytes, device, spec.name)


def per_kv_cache_token_workspace_size_bytes() -> int:
    _max_per_kv_cache_token = 0
    for spec in _REQUIRED_WORKSPACES.values():
        if isinstance(spec, PerKVCacheTokenWorkspace):
            _per_kv_cache_token_workspace_size = spec.num_bytes()
            if _per_kv_cache_token_workspace_size > _max_per_kv_cache_token:
                _max_per_kv_cache_token = _per_kv_cache_token_workspace_size
    return _max_per_kv_cache_token


def get_workspace(spec: WorkspaceSpec, device: torch.device) -> torch.Tensor:
    global _NUM_KV_CACHE_TOKENS
    shape = spec.shape
    num_bytes = spec.num_bytes()
    ubatch_id = dbo_current_ubatch_id()

    if isinstance(spec, PerKVCacheTokenWorkspace):
        if _NUM_KV_CACHE_TOKENS is None:
            cache_config = get_current_vllm_config().cache_config
            if cache_config.num_gpu_blocks is not None:
                _NUM_KV_CACHE_TOKENS = (
                    cache_config.num_gpu_blocks * cache_config.block_size
                )

        if _NUM_KV_CACHE_TOKENS is None:
            # KV cache still not initialized
            # This should only happen during initialization/dummy runs
            # Allocate a minimal workspace as a fallback, but don't cache this value
            import warnings

            warnings.warn(
                f"PerKVCacheTokenWorkspace '{spec.name}' requested before "
                "KV cache initialization. Allocating minimal workspace.",
                stacklevel=2,
            )
            shape = (1, *spec.shape)  # Minimal 1-token workspace
            num_bytes *= 1
        else:
            shape = (_NUM_KV_CACHE_TOKENS, *spec.shape)
            num_bytes *= _NUM_KV_CACHE_TOKENS

    # Get the workspace for current ubatch
    current_workspace = _CURRENT_WORKSPACES[ubatch_id]

    # Only allocate/resize if needed
    if (
        current_workspace is None
        or current_workspace.numel() * current_workspace.element_size() < num_bytes
    ):
        _increase_size(num_bytes, device, spec.name)
        current_workspace = _CURRENT_WORKSPACES[ubatch_id]

    return current_workspace[:num_bytes].view(spec.dtype).reshape(shape)


def get_workspaces(*args: WorkspaceSpec, device: torch.device) -> list[torch.Tensor]:
    # Align to 256 bytes for better memory allocation performance.
    byte_sizes = [round_up(spec.num_bytes(), 256) for spec in args]
    total_bytes = sum(byte_sizes)
    offsets = [0]
    for i in range(len(byte_sizes) - 1):
        offsets.append(offsets[-1] + byte_sizes[i])

    ubatch_id = dbo_current_ubatch_id()

    # Get the workspace for current ubatch
    current_workspace = _CURRENT_WORKSPACES[ubatch_id]

    # Only allocate/resize if needed
    if (
        current_workspace is None
        or current_workspace.numel() * current_workspace.element_size() < total_bytes
    ):
        # Create a combined name for logging
        workspace_names = ", ".join(spec.name for spec in args)
        _increase_size(total_bytes, device, f"[{workspace_names}]")
        current_workspace = _CURRENT_WORKSPACES[ubatch_id]

    return [
        current_workspace[offsets[i] : offsets[i] + byte_sizes[i]]
        .view(args[i].dtype)
        .reshape(args[i].shape)
        for i in range(len(args))
    ]
