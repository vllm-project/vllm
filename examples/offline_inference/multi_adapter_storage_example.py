"""
This example demonstrates how to use multiple adapters with different storage providers
and caching strategies for efficient adapter management.

Example usage:
    # Run with default settings
    python multi_adapter_storage_example.py
    
    # Run with custom cache sizes
    python multi_adapter_storage_example.py --memory-cache-size 1GB --disk-cache-size 2GB
"""

import argparse
import asyncio
import os
from pathlib import Path
from typing import Dict, List, Optional

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from vllm.adapter_commons.storage import (
    LocalStorageConfig,
    S3StorageConfig,
    StorageProviderFactory,
)
from vllm.adapter_commons.storage.cache import (
    MemoryCacheConfig,
    DiskCacheConfig,
    CacheFactory,
)

# Example prompts for different tasks
PROMPTS = {
    "sql": [
        "[user] Write a SQL query to list all employees in the IT department [/user] [assistant]",
        "[user] Write a SQL query to calculate total revenue by product for January 2024 [/user] [assistant]",
    ],
    "python": [
        "[user] Write a Python function to sort a list of dictionaries by a given key [/user] [assistant]",
        "[user] Write a Python function to find the most common elements in a list [/user] [assistant]",
    ],
}

def parse_size(size_str: str) -> int:
    """Convert a size string (e.g., '1GB') to bytes."""
    units = {
        'B': 1,
        'KB': 1024,
        'MB': 1024 * 1024,
        'GB': 1024 * 1024 * 1024,
    }
    
    size = size_str.strip().upper()
    for unit, multiplier in units.items():
        if size.endswith(unit):
            try:
                value = float(size[:-len(unit)])
                return int(value * multiplier)
            except ValueError:
                raise ValueError(f"Invalid size format: {size_str}")
    
    raise ValueError(f"Unknown size unit in: {size_str}")

def setup_storage_providers():
    """Create and configure storage providers for different adapters."""
    factory = StorageProviderFactory()
    
    providers = {
        "local": factory.create_provider(LocalStorageConfig(
            allowed_paths=["./adapters"],
            create_dirs=True,
            verify_permissions=True,
        )),
        "s3": factory.create_provider(S3StorageConfig(
            region_name="us-east-1",
            max_concurrent_downloads=4,
        )),
    }
    
    return providers

def setup_cache_backends(args):
    """Create and configure cache backends with different strategies."""
    factory = CacheFactory()
    
    backends = {
        "memory": factory.create_backend(MemoryCacheConfig(
            max_size_bytes=args.memory_cache_size,
            max_items=10,
        )),
        "disk": factory.create_backend(DiskCacheConfig(
            cache_dir=args.cache_dir,
            max_size_bytes=args.disk_cache_size,
            create_dirs=True,
        )),
    }
    
    return backends

async def main():
    parser = argparse.ArgumentParser(
        description="Multi-adapter storage example with different providers"
    )
    parser.add_argument(
        "--memory-cache-size",
        type=str,
        default="512MB",
        help="Maximum size for memory cache (e.g., '512MB', '1GB')",
    )
    parser.add_argument(
        "--disk-cache-size",
        type=str,
        default="1GB",
        help="Maximum size for disk cache (e.g., '1GB', '2GB')",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./cache",
        help="Directory for disk cache",
    )
    args = parser.parse_args()
    
    # Convert cache sizes to bytes
    args.memory_cache_size = parse_size(args.memory_cache_size)
    args.disk_cache_size = parse_size(args.disk_cache_size)

    # Create storage providers and cache backends
    storage_providers = setup_storage_providers()
    cache_backends = setup_cache_backends(args)

    # Initialize LLM with storage and cache
    llm = LLM(
        model="meta-llama/Llama-2-7b-hf",
        enable_lora=True,
        storage_provider=storage_providers["local"],  # Default provider
        cache_backend=cache_backends["memory"],      # Default cache
    )

    # Configure sampling parameters
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=256,
        stop=["[/assistant]"],
    )

    # Example adapter configurations
    adapters = [
        {
            "name": "sql_adapter",
            "path": "file:///path/to/sql-adapter",
            "provider": "local",
            "cache": "memory",
            "prompts": PROMPTS["sql"],
        },
        {
            "name": "python_adapter",
            "path": "s3://my-bucket/adapters/python-adapter",
            "provider": "s3",
            "cache": "disk",
            "prompts": PROMPTS["python"],
        },
    ]

    # Generate completions with different adapters
    for adapter in adapters:
        print(f"\nUsing adapter: {adapter['name']}")
        
        # Update storage provider and cache backend
        llm.update_storage_provider(storage_providers[adapter["provider"]])
        llm.update_cache_backend(cache_backends[adapter["cache"]])
        
        # Generate completions
        outputs = llm.generate(
            adapter["prompts"],
            sampling_params,
            lora_request=LoRARequest(adapter["name"], 0, adapter["path"]),
        )
        
        for output in outputs:
            print(f"\nPrompt: {output.prompt}")
            print(f"Generated: {output.outputs[0].text}")

    # Clean up resources
    for provider in storage_providers.values():
        await provider.cleanup()
    for backend in cache_backends.values():
        await backend.cleanup()

if __name__ == "__main__":
    asyncio.run(main()) 