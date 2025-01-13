"""Disk-based cache backend for adapters."""

import os
import json
import asyncio
import threading
import hashlib
from typing import Optional, Dict, Any, AsyncIterator, Set
from pathlib import Path
import shutil
import time
import fcntl
from contextlib import contextmanager

from .base import (
    AdapterCacheBackend,
    AdapterCacheError,
    CacheFullError,
    CacheKeyError,
    CacheIOError,
)
from .config import DiskCacheConfig

class DiskCache(AdapterCacheBackend):
    """Thread-safe disk-based cache for adapters with LRU eviction."""
    
    # File names for cache metadata
    METADATA_FILE = "metadata.json"
    INDEX_FILE = "index.json"
    
    def __init__(self, config: DiskCacheConfig):
        self.config = config
        self._lock = threading.RLock()
        
        # Ensure cache directory exists
        self.cache_dir = config.cache_dir.resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metadata storage
        self._metadata_path = self.cache_dir / self.METADATA_FILE
        self._index_path = self.cache_dir / self.INDEX_FILE
        
        # Load or initialize cache state
        self._load_cache_state()
        
        # Track current size
        self._current_size = self._calculate_current_size()
    
    @property
    def max_size(self) -> int:
        """Maximum cache size in bytes."""
        return self.config.max_size_bytes
    
    @property
    def current_size(self) -> int:
        """Current cache size in bytes."""
        with self._lock:
            return self._current_size
    
    def _load_cache_state(self) -> None:
        """Load cache state from disk or initialize if not exists."""
        try:
            if self._metadata_path.exists():
                with open(self._metadata_path, 'r') as f:
                    self._metadata = json.load(f)
            else:
                self._metadata = {}
                
            if self._index_path.exists():
                with open(self._index_path, 'r') as f:
                    self._access_index = json.load(f)
            else:
                self._access_index = {}
                
        except Exception as e:
            raise AdapterCacheError(f"Failed to load cache state: {e}")
    
    def _save_cache_state(self) -> None:
        """Save cache state to disk."""
        if self.config.fsync:
            try:
                with open(self._metadata_path, 'w') as f:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    json.dump(self._metadata, f)
                    f.flush()
                    os.fsync(f.fileno())
                    
                with open(self._index_path, 'w') as f:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    json.dump(self._access_index, f)
                    f.flush()
                    os.fsync(f.fileno())
                    
            except Exception as e:
                raise CacheIOError(f"Failed to save cache state: {e}")
    
    def _get_cache_path(self, key: str) -> Path:
        """Get path for cached adapter file."""
        # Use hash of key as filename to avoid path issues
        filename = hashlib.sha256(key.encode()).hexdigest()
        return self.cache_dir / filename
    
    def _calculate_current_size(self) -> int:
        """Calculate current cache size from metadata."""
        return sum(
            item.get("size", 0)
            for item in self._metadata.values()
        )
    
    @contextmanager
    def _update_access_time(self, key: str):
        """Update access time for LRU tracking."""
        try:
            yield
        finally:
            self._access_index[key] = time.time()
            if self.config.fsync:
                self._save_cache_state()
    
    async def get(
        self,
        key: str,
        default: Any = None
    ) -> Optional[Path]:
        """Get adapter from cache."""
        with self._lock:
            if key not in self._metadata:
                return default
                
            path = self._get_cache_path(key)
            if not path.exists():
                # Clean up inconsistent state
                self._metadata.pop(key, None)
                self._access_index.pop(key, None)
                self._save_cache_state()
                return default
                
            with self._update_access_time(key):
                return path
    
    async def put(
        self,
        key: str,
        adapter_path: Path,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Store adapter in cache."""
        try:
            size = adapter_path.stat().st_size
        except OSError as e:
            raise AdapterCacheError(f"Failed to get adapter size: {e}")
            
        with self._lock:
            # Check if we need to evict
            while self._current_size + size > self.max_size:
                if not self._access_index:
                    raise CacheFullError(
                        f"Cache full and no items to evict. "
                        f"Required: {size}, Available: {self.max_size - self._current_size}"
                    )
                # Evict least recently used
                lru_key = min(
                    self._access_index.items(),
                    key=lambda x: x[1]
                )[0]
                await self._remove_item(lru_key)
            
            # Copy file to cache
            cache_path = self._get_cache_path(key)
            try:
                if self.config.preserve_permissions:
                    shutil.copy2(adapter_path, cache_path)
                else:
                    shutil.copy(adapter_path, cache_path)
            except OSError as e:
                raise CacheIOError(f"Failed to cache adapter: {e}")
            
            # Update metadata
            self._metadata[key] = {
                "size": size,
                "path": str(cache_path),
                "metadata": metadata.copy() if metadata else {},
            }
            self._access_index[key] = time.time()
            self._current_size += size
            
            if self.config.fsync:
                self._save_cache_state()
    
    async def remove(self, key: str) -> bool:
        """Remove adapter from cache."""
        with self._lock:
            if key not in self._metadata:
                return False
            await self._remove_item(key)
            return True
    
    async def clear(self) -> None:
        """Clear all entries from cache."""
        with self._lock:
            keys = list(self._metadata.keys())
            for key in keys:
                await self._remove_item(key)
    
    async def contains(self, key: str) -> bool:
        """Check if key exists in cache."""
        with self._lock:
            return key in self._metadata
    
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
            keys = list(self._metadata.keys())
        
        for key in keys:
            yield key
    
    async def cleanup(self) -> None:
        """Clean up cache resources."""
        await self.clear()
    
    async def _remove_item(self, key: str) -> None:
        """Remove item and cleanup resources."""
        item = self._metadata.pop(key, None)
        if item:
            path = Path(item["path"])
            try:
                path.unlink()
            except OSError:
                pass
            self._current_size -= item.get("size", 0)
            self._access_index.pop(key, None)
            
            if self.config.fsync:
                self._save_cache_state() 