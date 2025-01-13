"""Configuration models for adapter cache backends."""

from typing import Optional, Dict, Any, Literal
from pathlib import Path
from pydantic import BaseModel, Field, validator
import os

class CacheConfig(BaseModel):
    """Base configuration for cache backends."""
    
    backend_type: Literal["memory", "disk"] = Field(
        default="disk",
        description="Type of cache backend to use"
    )
    max_size_bytes: int = Field(
        default=10 * 1024 * 1024 * 1024,  # 10GB
        description="Maximum cache size in bytes"
    )
    eviction_policy: Literal["lru", "fifo"] = Field(
        default="lru",
        description="Cache eviction policy"
    )
    cleanup_interval: int = Field(
        default=300,  # 5 minutes
        description="Interval in seconds between cache cleanup runs"
    )

    @validator("max_size_bytes")
    def validate_max_size(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("max_size_bytes must be positive")
        return v


class MemoryCacheConfig(CacheConfig):
    """Configuration specific to in-memory cache."""
    
    backend_type: Literal["memory"] = "memory"
    max_items: Optional[int] = Field(
        default=1000,
        description="Maximum number of items in cache"
    )
    pre_allocate: bool = Field(
        default=False,
        description="Whether to pre-allocate memory"
    )


class DiskCacheConfig(CacheConfig):
    """Configuration specific to disk cache."""
    
    backend_type: Literal["disk"] = "disk"
    cache_dir: Path = Field(
        default=Path(os.path.expanduser("~/.cache/vllm/adapters")),
        description="Directory to store cached files"
    )
    create_dirs: bool = Field(
        default=True,
        description="Whether to create cache directory if it doesn't exist"
    )
    preserve_permissions: bool = Field(
        default=True,
        description="Whether to preserve file permissions when caching"
    )
    fsync: bool = Field(
        default=True,
        description="Whether to force write to disk on cache operations"
    )
    
    @validator("cache_dir")
    def validate_cache_dir(cls, v: Path) -> Path:
        return Path(os.path.expanduser(str(v))) 