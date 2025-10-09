# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from math import prod
from typing import Optional

import torch

from vllm.config import get_current_vllm_config
from vllm.utils import round_up
from vllm.v1.worker.ubatching import dbo_current_ubatch_id

_REQUIRED_WORKSPACES: set["WorkspaceSpec"] = set()
_CURRENT_WORKSPACES: list[torch.Tensor] = [None, None]
_NUM_KV_CACHE_TOKENS: Optional[int] = None


@dataclass(frozen=True)
class WorkspaceSpec:
    """Specification of a workspace to be allocated.

    Attributes:
        shape: The shape of the workspace.
        dtype: The data type of the workspace.
        device: The device of the workspace.
    """

    shape: tuple[int, ...]
    dtype: torch.dtype

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


def register_workspace(spec: WorkspaceSpec):
    global _REQUIRED_WORKSPACES
    _REQUIRED_WORKSPACES.add(spec)


def per_kv_cache_token_workspace_size_bytes() -> int:
    _max_per_kv_cache_token = 0
    for spec in _REQUIRED_WORKSPACES:
        if isinstance(spec, PerKVCacheTokenWorkspace):
            _per_kv_cache_token_workspace_size = spec.num_bytes()
            if _per_kv_cache_token_workspace_size > _max_per_kv_cache_token:
                _max_per_kv_cache_token = _per_kv_cache_token_workspace_size
    return _max_per_kv_cache_token


def get_workspace(spec: WorkspaceSpec, device: torch.device) -> torch.Tensor:
    global _CURRENT_WORKSPACES, _NUM_KV_CACHE_TOKENS
    shape = spec.shape
    num_bytes = spec.num_bytes()

    current_workspace = _CURRENT_WORKSPACES[dbo_current_ubatch_id()]

    if isinstance(spec, PerKVCacheTokenWorkspace):
        if _NUM_KV_CACHE_TOKENS is None:
            cache_config = get_current_vllm_config().cache_config
            _NUM_KV_CACHE_TOKENS = cache_config.num_gpu_blocks * cache_config.block_size

        shape = (_NUM_KV_CACHE_TOKENS, *spec.shape)
        num_bytes *= _NUM_KV_CACHE_TOKENS

    if (
        current_workspace is None
        or current_workspace.numel() * current_workspace.element_size() < num_bytes
    ):
        current_workspace = torch.empty((num_bytes,), dtype=torch.uint8, device=device)
        _CURRENT_WORKSPACES[dbo_current_ubatch_id()] = current_workspace

    return current_workspace[:num_bytes].view(spec.dtype).reshape(shape)


def get_workspaces(*args: WorkspaceSpec, device: torch.device) -> list[torch.Tensor]:
    # Align to 256 bytes for better memory allocation performance.
    byte_sizes = [round_up(spec.num_bytes(), 256) for spec in args]
    total_bytes = sum(byte_sizes)
    offsets = [0]
    for i in range(len(byte_sizes) - 1):
        offsets.append(offsets[-1] + byte_sizes[i])

    current_workspace = _CURRENT_WORKSPACES[dbo_current_ubatch_id()]

    if (
        current_workspace is None
        or current_workspace.numel() * current_workspace.element_size() < total_bytes
    ):
        current_workspace = torch.empty(
            (total_bytes,), dtype=torch.uint8, device=device
        )
        _CURRENT_WORKSPACES[dbo_current_ubatch_id()] = current_workspace

    return [
        current_workspace[offsets[i] : offsets[i] + byte_sizes[i]]
        .view(args[i].dtype)
        .reshape(args[i].shape)
        for i in range(len(args))
    ]
