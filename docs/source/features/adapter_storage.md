# Adapter Storage and Caching

This document explains how to use the adapter storage and caching system in vLLM, which provides efficient management of LoRA adapters with support for both local and S3 storage, as well as memory and disk-based caching.

## Storage Providers

vLLM supports multiple storage providers for LoRA adapters:

### Local Storage

Local storage provider allows you to load adapters from your local filesystem:

```python
from vllm.adapter_commons.storage import LocalStorageConfig, StorageProviderFactory

config = LocalStorageConfig(
    allowed_paths=["/path/to/adapters"],
    create_dirs=True,
    verify_permissions=True
)

factory = StorageProviderFactory()
provider = factory.create_provider(config)

# Load adapter using local path
adapter_path = await provider.download_adapter("file:///path/to/adapters/my-adapter")
```

### S3 Storage

S3 storage provider enables loading adapters directly from S3:

```python
from vllm.adapter_commons.storage import S3StorageConfig, StorageProviderFactory

config = S3StorageConfig(
    region_name="us-east-1",
    max_concurrent_downloads=4
)

factory = StorageProviderFactory()
provider = factory.create_provider(config)

# Load adapter from S3
adapter_path = await provider.download_adapter("s3://my-bucket/adapters/my-adapter")
```

## Cache Backends

vLLM provides two cache backends for efficient adapter management:

### Memory Cache

In-memory cache for fastest access:

```python
from vllm.adapter_commons.storage.cache import MemoryCacheConfig, CacheFactory

config = MemoryCacheConfig(
    max_size_bytes=1024 * 1024 * 1024,  # 1GB
    max_items=10
)

factory = CacheFactory()
cache = factory.create_backend(config)

# Cache operations
await cache.put("adapter1", adapter_path, metadata={"version": "1.0"})
cached_path = await cache.get("adapter1")
```

### Disk Cache

Persistent disk-based cache:

```python
from vllm.adapter_commons.storage.cache import DiskCacheConfig, CacheFactory

config = DiskCacheConfig(
    cache_dir="/path/to/cache",
    max_size_bytes=10 * 1024 * 1024 * 1024,  # 10GB
    create_dirs=True,
    preserve_permissions=True
)

factory = CacheFactory()
cache = factory.create_backend(config)

# Cache operations
await cache.put("adapter2", adapter_path, metadata={"version": "2.0"})
cached_path = await cache.get("adapter2")
```

## Environment Configuration

Storage and cache components can be configured using environment variables:

```bash
# Storage configuration
export VLLM_STORAGE_PROVIDER=s3
export VLLM_S3_REGION=us-east-1
export VLLM_S3_MAX_CONCURRENT=4

# Cache configuration
export VLLM_CACHE_TYPE=disk
export VLLM_CACHE_DIR=/path/to/cache
export VLLM_CACHE_SIZE=10GB
```

## Integration with LoRA

The storage and cache system integrates seamlessly with vLLM's LoRA support:

```python
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

# Configure storage and cache
storage_config = S3StorageConfig(region_name="us-east-1")
cache_config = DiskCacheConfig(cache_dir="/path/to/cache")

# Initialize LLM with storage and cache
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    enable_lora=True,
    storage_config=storage_config,
    cache_config=cache_config
)

# Use adapter from S3
outputs = llm.generate(
    prompts,
    sampling_params,
    lora_request=LoRARequest(
        "sql_adapter",
        1,
        "s3://my-bucket/adapters/sql-lora"
    )
)
```

## Best Practices

1. **Storage Selection**:
   - Use local storage for development and testing
   - Use S3 storage for production deployments
   - Configure allowed paths and permissions appropriately

2. **Cache Configuration**:
   - Use memory cache for highest performance
   - Use disk cache for larger adapter sets
   - Set appropriate size limits based on available resources

3. **Resource Management**:
   - Monitor cache usage and eviction rates
   - Configure concurrent downloads based on available bandwidth
   - Clean up temporary files regularly

4. **Security Considerations**:
   - Validate and sanitize adapter paths
   - Use proper AWS credentials management
   - Implement appropriate access controls 