# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Registry for all_reduce backends.

This module provides a plugin system for all_reduce backends, allowing
users to register their own custom all_reduce kernels.
"""

import threading
from typing import Any, Optional, Protocol

import torch
from torch.distributed import ProcessGroup

from vllm.logger import init_logger

logger = init_logger(__name__)


class AllReduceBackend(Protocol):
    """Protocol for all_reduce backends."""

    def __call__(self, tensor: torch.Tensor,
                 group: ProcessGroup) -> torch.Tensor:
        """Perform all_reduce operation on the given tensor."""
        ...


class AllReduceRegistry:
    """Registry for all_reduce backends with thread-safe operations."""

    def __init__(self):
        self._backends: dict[str, AllReduceBackend] = {}
        self._default_backend: Optional[str] = None
        self._lock = threading.RLock()

    def register(self,
                 name: str,
                 backend: AllReduceBackend,
                 is_default: bool = False) -> None:
        """Register an all_reduce backend."""
        if not callable(backend):
            raise ValueError("Backend must be callable")

        with self._lock:
            self._backends[name] = backend
            if is_default or self._default_backend is None:
                self._default_backend = name

    def get(self, name: Optional[str] = None) -> Optional[AllReduceBackend]:
        """Get an all_reduce backend by name or return default."""
        with self._lock:
            backend_name = name or self._default_backend
            return self._backends.get(backend_name) if backend_name else None

    def list_backends(self) -> dict[str, AllReduceBackend]:
        """List all registered backends."""
        with self._lock:
            return self._backends.copy()


# Global registry instance
_registry = AllReduceRegistry()


def register_allreduce_backend(name: str,
                               backend: AllReduceBackend,
                               is_default: bool = False) -> None:
    """Register an all_reduce backend globally."""
    _registry.register(name, backend, is_default)


def get_allreduce_backend(
        name: Optional[str] = None) -> Optional[AllReduceBackend]:
    """Get an all_reduce backend by name."""
    return _registry.get(name)


def list_allreduce_backends() -> dict[str, AllReduceBackend]:
    """List all registered all_reduce backends."""
    return _registry.list_backends()


def get_allreduce_info() -> dict[str, Any]:
    """Get detailed information about all registered backends."""
    with _registry._lock:
        return {
            "backends": list(_registry._backends.keys()),
            "default": _registry._default_backend,
            "count": len(_registry._backends)
        }


def torch_all_reduce(tensor: torch.Tensor,
                     group: ProcessGroup) -> torch.Tensor:
    """Default torch.distributed.all_reduce backend."""
    output = tensor.clone()
    torch.distributed.all_reduce(output, group=group)
    return output


# Register the default backend
register_allreduce_backend("torch_distributed",
                           torch_all_reduce,
                           is_default=True)
