# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Registry for all_reduce backends.

This module provides a plugin system for all_reduce backends, allowing
companies and users to register their own custom all_reduce kernels.
"""

import inspect
import threading
from typing import Any, Dict, Optional, Protocol

import torch
from torch.distributed import ProcessGroup

from vllm.logger import init_logger

logger = init_logger(__name__)


class AllReduceBackend(Protocol):
    """Protocol for all_reduce backends."""

    def __call__(self, tensor: torch.Tensor,
                 group: ProcessGroup) -> torch.Tensor:
        """
        Perform all_reduce operation on the given tensor.

        Args:
            tensor: Input tensor to reduce
            group: Process group for the reduction

        Returns:
            Reduced tensor
        """
        ...


class AllReduceRegistry:
    """
    Registry for all_reduce backends.

    This registry allows plugins to register custom all_reduce backends
    that can be selected at runtime via the VLLM_ALLREDUCE_BACKEND
    environment variable.

    Thread Safety:
        All methods are thread-safe and can be called from multiple threads.

    Example:
        >>> registry = AllReduceRegistry()
        >>> registry.register("my_backend", my_function)
        >>> backend = registry.get("my_backend")
    """

    def __init__(self):
        self._backends: Dict[str, AllReduceBackend] = {}
        self._default_backend: Optional[str] = None
        self._lock = threading.RLock()

    def register(self,
                 name: str,
                 backend: AllReduceBackend,
                 is_default: bool = False,
                 validate: bool = True) -> None:
        """
        Register an all_reduce backend.

        Args:
            name: Name of the backend
            backend: The all_reduce backend function
            is_default: Whether this should be the default backend
            validate: Whether to validate the backend signature
        """
        with self._lock:
            if validate:
                self._validate_backend(backend)

            self._backends[name] = backend
            logger.info(f"Registered all_reduce backend: {name}")

            if is_default or self._default_backend is None:
                self._default_backend = name
                logger.info(f"Set default all_reduce backend: {name}")

    def unregister(self, name: str) -> bool:
        """
        Unregister a backend.

        Args:
            name: Name of the backend to remove

        Returns:
            True if found and removed, False otherwise
        """
        with self._lock:
            if name in self._backends:
                del self._backends[name]
                if self._default_backend == name:
                    # Reset default to first available or None
                    self._default_backend = next(iter(self._backends), None)
                logger.info(f"Unregistered all_reduce backend: {name}")
                return True
            return False

    def get(self, name: Optional[str] = None) -> Optional[AllReduceBackend]:
        """
        Get an all_reduce backend by name.

        Args:
            name: Name of the backend. If None, returns the default.

        Returns:
            The backend function, or None if not found
        """
        with self._lock:
            if name is None:
                name = self._default_backend

            if name is None:
                return None

            return self._backends.get(name)

    def list_backends(self) -> Dict[str, AllReduceBackend]:
        """List all registered backends."""
        with self._lock:
            return self._backends.copy()

    def get_default_name(self) -> Optional[str]:
        """Get the name of the default backend."""
        with self._lock:
            return self._default_backend

    def get_info(self) -> Dict[str, Any]:
        """Get detailed information about all registered backends."""
        with self._lock:
            return {
                "backends": list(self._backends.keys()),
                "default": self._default_backend,
                "count": len(self._backends)
            }

    def _validate_backend(self, backend: AllReduceBackend) -> None:
        """Basic validation that the backend is callable with correct signature."""
        if not callable(backend):
            raise ValueError("Backend must be callable")

        try:
            sig = inspect.signature(backend)
            if len(sig.parameters) != 2:
                raise ValueError(
                    f"Backend must accept exactly 2 parameters (tensor, group), "
                    f"got {len(sig.parameters)}")
        except (ValueError, TypeError) as e:
            logger.warning(f"Could not validate backend signature: {e}")


# Global registry instance
_allreduce_registry = AllReduceRegistry()


def register_allreduce_backend(name: str,
                               backend: AllReduceBackend,
                               is_default: bool = False,
                               validate: bool = True) -> None:
    """
    Register an all_reduce backend globally.

    Args:
        name: Name of the backend
        backend: The all_reduce backend function
        is_default: Whether this should be the default backend
        validate: Whether to validate the backend signature
    """
    _allreduce_registry.register(name, backend, is_default, validate)


def get_allreduce_backend(
        name: Optional[str] = None) -> Optional[AllReduceBackend]:
    """
    Get an all_reduce backend by name with better error context.

    Args:
        name: Name of the backend. If None, returns the default.

    Returns:
        The backend function, or None if not found
    """
    try:
        return _allreduce_registry.get(name)
    except Exception as e:
        available = list(_allreduce_registry.list_backends().keys())
        logger.error(f"Failed to get all_reduce backend '{name}'. "
                     f"Available backends: {available}. Error: {e}")
        return None


def unregister_allreduce_backend(name: str) -> bool:
    """
    Unregister an all_reduce backend.

    Args:
        name: Name of the backend to remove

    Returns:
        True if found and removed, False otherwise
    """
    return _allreduce_registry.unregister(name)


def list_allreduce_backends() -> Dict[str, AllReduceBackend]:
    """List all registered all_reduce backends."""
    return _allreduce_registry.list_backends()


def get_default_allreduce_backend_name() -> Optional[str]:
    """Get the name of the default all_reduce backend."""
    return _allreduce_registry.get_default_name()


def get_allreduce_info() -> Dict[str, Any]:
    """Get detailed information about all registered backends."""
    return _allreduce_registry.get_info()


# Built-in backends
def torch_all_reduce(tensor: torch.Tensor,
                     group: ProcessGroup) -> torch.Tensor:
    """
    Default torch.distributed.all_reduce backend.

    Args:
        tensor: Input tensor to reduce
        group: Process group for the reduction

    Returns:
        Reduced tensor
    """
    output = tensor.clone()
    torch.distributed.all_reduce(output, group=group)
    return output


# Register the default torch.distributed backend
register_allreduce_backend("torch_distributed",
                           torch_all_reduce,
                           is_default=True)
