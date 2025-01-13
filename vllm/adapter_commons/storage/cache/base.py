"""Base interface for adapter cache backends."""

import abc
from typing import Optional, Dict, Any, Protocol, runtime_checkable, AsyncIterator
from pathlib import Path
import asyncio

@runtime_checkable
class AdapterCacheBackend(Protocol):
    """Protocol defining the interface for adapter cache backends.
    
    This interface defines the required methods that any cache backend
    (memory or disk) must implement to be used with vLLM's adapter system.
    """
    
    async def get(
        self,
        key: str,
        default: Any = None
    ) -> Optional[Path]:
        """Get an adapter from cache.
        
        Args:
            key: Cache key for the adapter
            default: Value to return if key not found
            
        Returns:
            Path to cached adapter or default if not found
        """
        ...

    async def put(
        self,
        key: str,
        adapter_path: Path,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Store an adapter in cache.
        
        Args:
            key: Cache key for the adapter
            adapter_path: Path to the adapter file
            metadata: Optional metadata about the adapter
            
        Raises:
            AdapterCacheError: If storage fails
        """
        ...

    async def remove(self, key: str) -> bool:
        """Remove an adapter from cache.
        
        Args:
            key: Cache key to remove
            
        Returns:
            True if key was found and removed
        """
        ...

    async def clear(self) -> None:
        """Clear all entries from cache."""
        ...

    async def contains(self, key: str) -> bool:
        """Check if key exists in cache.
        
        Args:
            key: Cache key to check
            
        Returns:
            True if key exists in cache
        """
        ...

    async def get_size(self) -> int:
        """Get total size of cached items in bytes.
        
        Returns:
            Total cache size in bytes
        """
        ...

    async def get_metadata(self, key: str) -> Optional[Dict[str, Any]]:
        """Get metadata for cached adapter.
        
        Args:
            key: Cache key
            
        Returns:
            Metadata dict if found, None otherwise
        """
        ...

    async def iter_keys(self) -> AsyncIterator[str]:
        """Iterate over cache keys.
        
        Yields:
            Cache keys
        """
        ...

    async def cleanup(self) -> None:
        """Clean up any temporary resources."""
        ...

    @property
    def max_size(self) -> int:
        """Maximum cache size in bytes."""
        ...

    @property
    def current_size(self) -> int:
        """Current cache size in bytes."""
        ...

class AdapterCacheError(Exception):
    """Base exception for adapter cache errors."""
    pass

class CacheFullError(AdapterCacheError):
    """Raised when cache is full and cannot store more items."""
    pass

class CacheKeyError(AdapterCacheError):
    """Raised when a cache key operation fails."""
    pass

class CacheIOError(AdapterCacheError):
    """Raised when cache I/O operations fail."""
    pass 