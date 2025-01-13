"""
This example demonstrates how to use the adapter storage and cache features
for efficient LoRA adapter management.

Example usage:
    # Run with local storage and memory cache
    python adapter_storage_example.py --storage-type local
    
    # Run with S3 storage and disk cache
    python adapter_storage_example.py --storage-type s3 --cache-type disk
"""

import argparse
import asyncio
import os
from pathlib import Path

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

# Example prompts for SQL generation
PROMPTS = [
    "[user] Write a SQL query to answer the question based on the table schema.\n\n"
    "context: CREATE TABLE employees (id INT, name VARCHAR, department VARCHAR)\n\n"
    "question: List all employees in the IT department [/user] [assistant]",
    
    "[user] Write a SQL query to answer the question based on the table schema.\n\n"
    "context: CREATE TABLE sales (date DATE, product_id INT, quantity INT, price DECIMAL)\n\n"
    "question: Calculate total revenue by product for January 2024 [/user] [assistant]",
]

def setup_storage_provider(args):
    """Create and configure the storage provider based on arguments."""
    factory = StorageProviderFactory()
    
    if args.storage_type == "s3":
        config = S3StorageConfig(
            region_name=args.s3_region,
            max_concurrent_downloads=4,
        )
    else:
        config = LocalStorageConfig(
            allowed_paths=[args.adapter_dir],
            create_dirs=True,
            verify_permissions=True,
        )
    
    return factory.create_provider(config)

def setup_cache_backend(args):
    """Create and configure the cache backend based on arguments."""
    factory = CacheFactory()
    
    if args.cache_type == "disk":
        config = DiskCacheConfig(
            cache_dir=args.cache_dir,
            max_size_bytes=1024 * 1024 * 1024,  # 1GB
            create_dirs=True,
        )
    else:
        config = MemoryCacheConfig(
            max_size_bytes=512 * 1024 * 1024,  # 512MB
            max_items=10,
        )
    
    return factory.create_backend(config)

async def main():
    parser = argparse.ArgumentParser(description="LoRA adapter storage example")
    parser.add_argument(
        "--storage-type",
        choices=["local", "s3"],
        default="local",
        help="Storage provider type",
    )
    parser.add_argument(
        "--cache-type",
        choices=["memory", "disk"],
        default="memory",
        help="Cache backend type",
    )
    parser.add_argument(
        "--adapter-dir",
        type=str,
        default="./adapters",
        help="Local directory for adapters",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./cache",
        help="Directory for disk cache",
    )
    parser.add_argument(
        "--s3-region",
        type=str,
        default="us-east-1",
        help="AWS region for S3",
    )
    args = parser.parse_args()

    # Create storage provider and cache backend
    storage_provider = setup_storage_provider(args)
    cache_backend = setup_cache_backend(args)

    # Initialize LLM with storage and cache
    llm = LLM(
        model="meta-llama/Llama-2-7b-hf",
        enable_lora=True,
        storage_provider=storage_provider,
        cache_backend=cache_backend,
    )

    # Configure sampling parameters
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=256,
        stop=["[/assistant]"],
    )

    # Example adapter paths
    adapter_paths = [
        "file:///path/to/sql-adapter" if args.storage_type == "local"
        else "s3://my-bucket/adapters/sql-adapter"
    ]

    # Generate completions with different adapters
    for i, adapter_path in enumerate(adapter_paths):
        print(f"\nUsing adapter {i + 1}:")
        outputs = llm.generate(
            PROMPTS,
            sampling_params,
            lora_request=LoRARequest(f"adapter_{i}", i, adapter_path),
        )
        
        for output in outputs:
            print(f"\nPrompt: {output.prompt}")
            print(f"Generated: {output.outputs[0].text}")

    # Clean up resources
    await storage_provider.cleanup()
    await cache_backend.cleanup()

if __name__ == "__main__":
    asyncio.run(main()) 