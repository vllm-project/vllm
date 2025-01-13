# Storage and Cache API

This document describes the API for vLLM's adapter storage and caching system.

## Storage Providers

### Base Storage Provider

```python
from vllm.adapter_commons.storage import BaseStorageProvider

class BaseStorageProvider(Protocol):
    """Base protocol for adapter storage providers."""
    
    async def download_adapter(self, uri: str, target_path: str) -> None:
        """Download an adapter from storage to the local filesystem.
        
        Args:
            uri: The URI of the adapter to download.
            target_path: The local path to save the adapter to.
            
        Raises:
            AdapterNotFoundError: If the adapter does not exist.
            AdapterDownloadError: If the download fails.
            AdapterValidationError: If the URI is invalid.
        """
        ...

    async def validate_uri(self, uri: str) -> bool:
        """Check if a URI is valid for this provider.
        
        Args:
            uri: The URI to validate.
            
        Returns:
            bool: True if the URI is valid, False otherwise.
        """
        ...

    async def get_adapter_size(self, uri: str) -> int:
        """Get the size of an adapter in bytes.
        
        Args:
            uri: The URI of the adapter.
            
        Returns:
            int: The size in bytes.
            
        Raises:
            AdapterNotFoundError: If the adapter does not exist.
        """
        ...
```

### Local Storage Provider

```python
from vllm.adapter_commons.storage import LocalStorageProvider, LocalStorageConfig

# Configuration
config = LocalStorageConfig(
    allowed_paths=["/path/to/adapters"],  # List of allowed base directories
    create_dirs=True,                     # Create directories if they don't exist
    verify_permissions=True,              # Check file permissions
)

# Usage
provider = LocalStorageProvider(config)
await provider.download_adapter("file:///path/to/adapters/my-adapter", "./cache/my-adapter")
```

### S3 Storage Provider

```python
from vllm.adapter_commons.storage import S3StorageProvider, S3StorageConfig

# Configuration
config = S3StorageConfig(
    region_name="us-east-1",           # AWS region
    max_concurrent_downloads=4,         # Maximum concurrent downloads
    endpoint_url="http://...",         # Optional custom endpoint
    aws_access_key_id="...",          # Optional credentials
    aws_secret_access_key="...",
)

# Usage
provider = S3StorageProvider(config)
await provider.download_adapter("s3://my-bucket/adapters/my-adapter", "./cache/my-adapter")
```

## Cache Backends

### Base Cache Backend

```python
from vllm.adapter_commons.storage.cache import AdapterCacheBackend

class AdapterCacheBackend(Protocol):
    """Base protocol for adapter cache backends."""
    
    async def get(self, key: str) -> Optional[str]:
        """Get an adapter from the cache.
        
        Args:
            key: The cache key.
            
        Returns:
            Optional[str]: The path to the cached adapter, or None if not found.
            
        Raises:
            CacheIOError: If reading from cache fails.
        """
        ...

    async def put(self, key: str, adapter_path: str) -> None:
        """Store an adapter in the cache.
        
        Args:
            key: The cache key.
            adapter_path: The path to the adapter file.
            
        Raises:
            CacheFullError: If the cache is full.
            CacheIOError: If writing to cache fails.
        """
        ...

    @property
    def max_size(self) -> int:
        """Maximum size of the cache in bytes."""
        ...

    @property
    def current_size(self) -> int:
        """Current size of the cache in bytes."""
        ...
```

### Memory Cache Backend

```python
from vllm.adapter_commons.storage.cache import MemoryCacheBackend, MemoryCacheConfig

# Configuration
config = MemoryCacheConfig(
    max_size_bytes=512 * 1024 * 1024,  # 512MB
    max_items=10,                      # Maximum number of items
)

# Usage
cache = MemoryCacheBackend(config)
await cache.put("my-adapter", "/path/to/adapter")
path = await cache.get("my-adapter")
```

### Disk Cache Backend

```python
from vllm.adapter_commons.storage.cache import DiskCacheBackend, DiskCacheConfig

# Configuration
config = DiskCacheConfig(
    cache_dir="./cache",              # Cache directory
    max_size_bytes=1024 * 1024 * 1024,  # 1GB
    create_dirs=True,                 # Create directory if it doesn't exist
)

# Usage
cache = DiskCacheBackend(config)
await cache.put("my-adapter", "/path/to/adapter")
path = await cache.get("my-adapter")
```

## Factory Classes

### Storage Provider Factory

```python
from vllm.adapter_commons.storage import StorageProviderFactory

# Create factory
factory = StorageProviderFactory()

# Create provider from config
provider = factory.create_provider(config)

# Create provider from URI
provider = factory.create_provider_for_uri("s3://my-bucket/adapter")
```

### Cache Factory

```python
from vllm.adapter_commons.storage.cache import CacheFactory

# Create factory
factory = CacheFactory()

# Create cache backend
cache = factory.create_backend(config)
```

## Error Handling

The storage and cache system defines several exception types:

```python
from vllm.adapter_commons.storage import (
    AdapterStorageError,      # Base class for storage errors
    AdapterNotFoundError,     # Adapter not found in storage
    AdapterDownloadError,     # Download failed
    AdapterValidationError,   # Invalid URI or configuration
)

from vllm.adapter_commons.storage.cache import (
    AdapterCacheError,        # Base class for cache errors
    CacheFullError,          # Cache is full
    CacheKeyError,           # Invalid cache key
    CacheIOError,           # IO operation failed
)
```

## Integration with vLLM

To use storage providers and cache backends with vLLM:

```python
from vllm import LLM
from vllm.lora.request import LoRARequest

# Initialize LLM with storage and cache
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    enable_lora=True,
    storage_provider=storage_provider,
    cache_backend=cache_backend,
)

# Use with LoRA request
outputs = llm.generate(
    prompts,
    sampling_params,
    lora_request=LoRARequest("adapter_name", 0, "s3://my-bucket/adapter"),
)
```

## Environment Configuration

Both storage providers and cache backends support configuration through environment variables:

```bash
# Storage configuration
export VLLM_STORAGE_ALLOWED_PATHS="/path/to/adapters,/another/path"
export VLLM_STORAGE_S3_REGION="us-east-1"
export VLLM_STORAGE_S3_ENDPOINT="http://..."

# Cache configuration
export VLLM_CACHE_TYPE="disk"  # or "memory"
export VLLM_CACHE_DIR="./cache"
export VLLM_CACHE_MAX_SIZE="1GB"
```

## Best Practices

1. **Storage Selection**
   - Use local storage for development and testing
   - Use S3 storage for production deployments
   - Configure appropriate timeouts and retries

2. **Cache Configuration**
   - Use memory cache for small adapters and high performance
   - Use disk cache for large adapters and persistence
   - Set appropriate size limits based on available resources

3. **Resource Management**
   - Always clean up resources using `await provider.cleanup()`
   - Monitor cache size and eviction patterns
   - Use appropriate error handling

4. **Security**
   - Validate and sanitize URIs
   - Use secure S3 endpoints
   - Configure appropriate permissions 