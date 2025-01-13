"""Factory for creating cache backends."""

from typing import Dict, Type, Optional, List

from .base import AdapterCacheBackend
from .memory import MemoryCache
from .disk import DiskCache
from .config import CacheConfig, MemoryCacheConfig, DiskCacheConfig

class CacheFactory:
    """Factory for creating cache backend instances."""
    
    # Registry of available cache backends
    _backend_registry: Dict[str, Type[AdapterCacheBackend]] = {
        "memory": MemoryCache,
        "disk": DiskCache,
    }
    
    @classmethod
    def register_backend(
        cls,
        backend_type: str,
        backend_class: Type[AdapterCacheBackend]
    ) -> None:
        """Register a new cache backend type.
        
        Args:
            backend_type: Unique identifier for the backend type
            backend_class: Cache backend class implementing AdapterCacheBackend
        """
        if not issubclass(backend_class, AdapterCacheBackend):
            raise ValueError(
                f"Backend class must implement AdapterCacheBackend interface: {backend_class}"
            )
        cls._backend_registry[backend_type] = backend_class
    
    @classmethod
    def create_backend(cls, config: CacheConfig) -> AdapterCacheBackend:
        """Create a cache backend instance from configuration.
        
        Args:
            config: Cache configuration object
            
        Returns:
            Configured cache backend instance
            
        Raises:
            ValueError: If backend type is not registered
        """
        backend_type = config.backend_type
        if backend_type not in cls._backend_registry:
            raise ValueError(
                f"Unknown cache backend type: {backend_type}. "
                f"Available types: {list(cls._backend_registry.keys())}"
            )
            
        backend_class = cls._backend_registry[backend_type]
        
        # Validate config type matches backend
        if backend_type == "memory" and not isinstance(config, MemoryCacheConfig):
            raise ValueError(
                f"Memory cache backend requires MemoryCacheConfig, got: {type(config)}"
            )
        elif backend_type == "disk" and not isinstance(config, DiskCacheConfig):
            raise ValueError(
                f"Disk cache backend requires DiskCacheConfig, got: {type(config)}"
            )
            
        return backend_class(config)
    
    @classmethod
    def get_registered_backends(cls) -> List[str]:
        """Get list of registered backend types.
        
        Returns:
            List of registered backend type identifiers
        """
        return list(cls._backend_registry.keys()) 