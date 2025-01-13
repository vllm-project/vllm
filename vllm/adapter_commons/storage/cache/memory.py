"""In-memory cache backend for adapters."""

import os
import threading
import tempfile
import shutil
import time
from typing import Optional, Dict, Any, AsyncIterator, Set
from pathlib import Path

from .base import (
    AdapterCacheBackend,
    AdapterCacheError,
    CacheFullError,
    CacheKeyError,
    CacheIOError,
)
from .config import MemoryCacheConfig

class MemoryCache(AdapterCacheBackend):
    """Thread-safe in-memory cache for adapters with LRU eviction."""
    
    def __init__(self, config: MemoryCacheConfig):
        self.config = config
        self._lock = threading.RLock()
        
        # Initialize storage
        self._data: Dict[str, bytes] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._access_times: Dict[str, float] = {}
        self._current_size = 0
        
        # Create temp directory for temporary files
        self._temp_dir = Path(tempfile.mkdtemp(prefix="vllm_memory_cache_"))
    
    @property
    def max_size(self) -> int:
        """Maximum cache size in bytes."""
        return self.config.max_size_bytes
    
    @property
    def current_size(self) -> int:
        """Current cache size in bytes."""
        with self._lock:
            return self._current_size
    
    def _update_access_time(self, key: str) -> None:
        """Update access time for LRU tracking."""
        self._access_times[key] = time.time()
    
    def _get_temp_path(self) -> Path:
        """Get temporary file path."""
        return Path(tempfile.mktemp(dir=self._temp_dir))
    
    async def get(
        self,
        key: str,
        default: Any = None
    ) -> Optional[Path]:
        """Get adapter from cache."""
        with self._lock:
            if key not in self._data:
                return default
            
            # Create temporary file with cached data
            temp_path = self._get_temp_path()
            try:
                with open(temp_path, 'wb') as f:
                    f.write(self._data[key])
                self._update_access_time(key)
                return temp_path
            except OSError as e:
                raise CacheIOError(f"Failed to write temporary file: {e}")
    
    async def put(
        self,
        key: str,
        adapter_path: Path,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Store adapter in cache."""
        try:
            with open(adapter_path, 'rb') as f:
                data = f.read()
        except OSError as e:
            raise AdapterCacheError(f"Failed to read adapter file: {e}")
        
        size = len(data)
        with self._lock:
            # Check if we need to evict
            while self._current_size + size > self.max_size:
                if not self._access_times:
                    raise CacheFullError(
                        f"Cache full and no items to evict. "
                        f"Required: {size}, Available: {self.max_size - self._current_size}"
                    )
                # Evict least recently used
                lru_key = min(
                    self._access_times.items(),
                    key=lambda x: x[1]
                )[0]
                await self._remove_item(lru_key)
            
            # Store data and metadata
            self._data[key] = data
            self._metadata[key] = {
                "size": size,
                "metadata": metadata.copy() if metadata else {},
            }
            self._access_times[key] = time.time()
            self._current_size += size
    
    async def remove(self, key: str) -> bool:
        """Remove adapter from cache."""
        with self._lock:
            if key not in self._data:
                return False
            await self._remove_item(key)
            return True
    
    async def clear(self) -> None:
        """Clear all entries from cache."""
        with self._lock:
            keys = list(self._data.keys())
            for key in keys:
                await self._remove_item(key)
    
    async def contains(self, key: str) -> bool:
        """Check if key exists in cache."""
        with self._lock:
            return key in self._data
    
    async def get_size(self) -> int:
        """Get total size of cached items in bytes."""
        return self.current_size
    
    async def get_metadata(self, key: str) -> Optional[Dict[str, Any]]:
        """Get metadata for cached adapter."""
        with self._lock:
            item = self._metadata.get(key)
            if not item:
                return None
            return item.get("metadata", {}).copy()
    
    async def iter_keys(self) -> AsyncIterator[str]:
        """Iterate over cache keys."""
        with self._lock:
            keys = list(self._data.keys())
        
        for key in keys:
            yield key
    
    async def cleanup(self) -> None:
        """Clean up cache resources."""
        await self.clear()
        try:
            shutil.rmtree(self._temp_dir)
        except OSError:
            pass
    
    async def _remove_item(self, key: str) -> None:
        """Remove item and cleanup resources."""
        data = self._data.pop(key, None)
        if data:
            self._current_size -= len(data)
            self._metadata.pop(key, None)
            self._access_times.pop(key, None) 