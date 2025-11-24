#!/usr/bin/env python3
"""
Benchmark comparing RoaringBitmap vs Dense Tensor vs COO for sparse attention masks.
This demonstrates the potential memory savings for long-context models (100k+ tokens).
"""

import time
import gc
import tracemalloc
import numpy as np
import torch
import pyroaring
from typing import List, Tuple, Set
from dataclasses import dataclass
import argparse


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    memory_bytes: int
    creation_time: float
    query_time: float
    intersection_time: float
    num_active_connections: int


class DenseTensorMask:
    """Dense boolean tensor for attention mask (baseline)."""
    
    def __init__(self, num_blocks: int):
        self.num_blocks = num_blocks
        self.mask = torch.zeros((num_blocks, num_blocks), dtype=torch.bool)
    
    def add_attention(self, query_block: int, key_blocks: List[int]):
        """Add attention from query_block to key_blocks."""
        self.mask[query_block, key_blocks] = True
    
    def get_active_blocks(self, query_block: int) -> np.ndarray:
        """Get blocks that query_block attends to."""
        return torch.where(self.mask[query_block])[0].numpy()
    
    def intersection(self, block1: int, block2: int) -> np.ndarray:
        """Get blocks that both block1 and block2 attend to."""
        mask1 = self.mask[block1]
        mask2 = self.mask[block2]
        return torch.where(mask1 & mask2)[0].numpy()


class COOSparseMask:
    """Coordinate list (COO) format for sparse attention."""
    
    def __init__(self, num_blocks: int):
        self.num_blocks = num_blocks
        self.attention_map = {i: set() for i in range(num_blocks)}
    
    def add_attention(self, query_block: int, key_blocks: List[int]):
        """Add attention from query_block to key_blocks."""
        self.attention_map[query_block].update(key_blocks)
    
    def get_active_blocks(self, query_block: int) -> np.ndarray:
        """Get blocks that query_block attends to."""
        return np.array(sorted(self.attention_map[query_block]))
    
    def intersection(self, block1: int, block2: int) -> np.ndarray:
        """Get blocks that both block1 and block2 attend to."""
        set1 = self.attention_map[block1]
        set2 = self.attention_map[block2]
        return np.array(sorted(set1 & set2))


class RoaringAttentionMask:
    """RoaringBitmap-based sparse attention mask."""
    
    def __init__(self, num_blocks: int):
        self.num_blocks = num_blocks
        self.masks = [pyroaring.BitMap() for _ in range(num_blocks)]
    
    def add_attention(self, query_block: int, key_blocks: List[int]):
        """Add attention from query_block to key_blocks."""
        self.masks[query_block].update(key_blocks)
    
    def get_active_blocks(self, query_block: int) -> np.ndarray:
        """Get blocks that query_block attends to."""
        return np.array(self.masks[query_block].to_array())
    
    def intersection(self, block1: int, block2: int) -> np.ndarray:
        """Get blocks that both block1 and block2 attend to."""
        intersection = self.masks[block1] & self.masks[block2]
        return np.array(intersection.to_array())


def create_attention_pattern(num_blocks: int, window_size: int = 128, 
                           global_tokens: int = 16, sparsity: float = 0.99) -> List[Tuple[int, List[int]]]:
    """
    Create a realistic attention pattern for long-context models.
    
    Pattern includes:
    - Sliding window attention (local context)
    - Global attention tokens (attend to/from all)
    - Random sparse connections (retrieval-based attention)
    """
    attention_patterns = []
    
    for query_block in range(num_blocks):
        key_blocks = set()
        
        # 1. Sliding window (local attention)
        start = max(0, query_block - window_size // 2)
        end = min(num_blocks, query_block + window_size // 2)
        key_blocks.update(range(start, end))
        
        # 2. Global tokens (first N blocks attend to/from all)
        if query_block < global_tokens:
            # Global tokens attend to all
            key_blocks.update(range(num_blocks))
        else:
            # All blocks attend to global tokens
            key_blocks.update(range(global_tokens))
        
        # 3. Random sparse connections (simulate retrieval)
        num_sparse = int((1 - sparsity) * num_blocks / 10)  # Very sparse
        sparse_blocks = np.random.choice(num_blocks, size=num_sparse, replace=False)
        key_blocks.update(sparse_blocks)
        
        attention_patterns.append((query_block, list(key_blocks)))
    
    return attention_patterns


def benchmark_mask_implementation(mask_class, attention_patterns: List[Tuple[int, List[int]]], 
                                 num_blocks: int, num_queries: int = 100):
    """Benchmark a specific mask implementation."""
    
    # Creation benchmark
    start_time = time.time()
    mask = mask_class(num_blocks)
    for query_block, key_blocks in attention_patterns:
        mask.add_attention(query_block, key_blocks)
    creation_time = time.time() - start_time
    
    # Memory measurement using actual object size
    gc.collect()
    
    if isinstance(mask, DenseTensorMask):
        # For dense tensor, calculate actual tensor size
        memory_usage = mask.mask.element_size() * mask.mask.nelement()
    elif isinstance(mask, COOSparseMask):
        # For COO, estimate based on stored sets
        memory_usage = sum(len(s) * 8 for s in mask.attention_map.values())  # 8 bytes per int64
    elif isinstance(mask, RoaringAttentionMask):
        # For RoaringBitmap, sum up all bitmap sizes
        tracemalloc.start()
        gc.collect()
        start_memory = tracemalloc.get_traced_memory()[0]
        # Force serialization to measure actual memory
        serialized = [bitmap.serialize() for bitmap in mask.masks]
        memory_usage = sum(len(s) for s in serialized)
        tracemalloc.stop()
    else:
        # Fallback to tracemalloc
        tracemalloc.start()
        gc.collect()
        start_memory = tracemalloc.get_traced_memory()[0]
        _ = mask  # Reference to prevent GC
        gc.collect()
        end_memory = tracemalloc.get_traced_memory()[0]
        memory_usage = end_memory - start_memory
        tracemalloc.stop()
    
    # Query benchmark
    query_blocks = np.random.randint(0, num_blocks, size=num_queries)
    start_time = time.time()
    for query_block in query_blocks:
        _ = mask.get_active_blocks(query_block)
    query_time = time.time() - start_time
    
    # Intersection benchmark
    pairs = [(np.random.randint(0, num_blocks), np.random.randint(0, num_blocks)) 
             for _ in range(num_queries)]
    start_time = time.time()
    for block1, block2 in pairs:
        _ = mask.intersection(block1, block2)
    intersection_time = time.time() - start_time
    
    # Count total active connections
    total_connections = sum(len(mask.get_active_blocks(i)) for i in range(min(100, num_blocks)))
    avg_connections = total_connections / min(100, num_blocks)
    
    return BenchmarkResult(
        memory_bytes=memory_usage,
        creation_time=creation_time,
        query_time=query_time,
        intersection_time=intersection_time,
        num_active_connections=int(avg_connections)
    )


def format_bytes(bytes_val: int) -> str:
    """Format bytes to human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} TB"


def main():
    parser = argparse.ArgumentParser(description='Benchmark sparse attention mask implementations')
    parser.add_argument('--num-blocks', type=int, default=10000,
                        help='Number of blocks (for 100k tokens with block_size=16, use ~6250)')
    parser.add_argument('--window-size', type=int, default=128,
                        help='Sliding window size for local attention')
    parser.add_argument('--global-tokens', type=int, default=16,
                        help='Number of global attention tokens')
    parser.add_argument('--sparsity', type=float, default=0.99,
                        help='Sparsity level (0.99 = 99% sparse)')
    args = parser.parse_args()
    
    print(f"Benchmarking Sparse Attention Masks")
    print(f"=" * 60)
    print(f"Configuration:")
    print(f"  Num blocks: {args.num_blocks:,}")
    print(f"  Approx tokens: {args.num_blocks * 16:,} (assuming block_size=16)")
    print(f"  Window size: {args.window_size}")
    print(f"  Global tokens: {args.global_tokens}")
    print(f"  Sparsity: {args.sparsity:.1%}")
    print(f"=" * 60)
    
    # Create attention pattern
    print("\nGenerating attention pattern...")
    attention_patterns = create_attention_pattern(
        args.num_blocks, args.window_size, args.global_tokens, args.sparsity
    )
    
    # Benchmark implementations
    implementations = [
        ("Dense Tensor", DenseTensorMask),
        ("COO Sparse", COOSparseMask),
        ("RoaringBitmap", RoaringAttentionMask),
    ]
    
    results = {}
    for name, mask_class in implementations:
        print(f"\nBenchmarking {name}...")
        try:
            result = benchmark_mask_implementation(
                mask_class, attention_patterns, args.num_blocks
            )
            results[name] = result
            
            print(f"  Memory usage: {format_bytes(result.memory_bytes)}")
            print(f"  Creation time: {result.creation_time:.4f}s")
            print(f"  Query time (100 queries): {result.query_time:.4f}s")
            print(f"  Intersection time (100 ops): {result.intersection_time:.4f}s")
            print(f"  Avg connections per block: {result.num_active_connections}")
        except Exception as e:
            print(f"  Failed: {e}")
            results[name] = None
    
    # Comparison
    print(f"\n{'=' * 60}")
    print("COMPARISON (relative to Dense Tensor):")
    print(f"{'=' * 60}")
    
    if results.get("Dense Tensor") and results.get("RoaringBitmap"):
        dense = results["Dense Tensor"]
        roaring = results["RoaringBitmap"]
        
        memory_ratio = roaring.memory_bytes / dense.memory_bytes
        query_ratio = roaring.query_time / dense.query_time
        intersection_ratio = roaring.intersection_time / dense.intersection_time
        
        print(f"\nRoaringBitmap vs Dense Tensor:")
        print(f"  Memory: {memory_ratio:.1%} ({(1-memory_ratio)*100:.1f}% reduction)")
        print(f"  Query speed: {1/query_ratio:.1f}x faster" if query_ratio < 1 else f"  Query speed: {query_ratio:.1f}x slower")
        print(f"  Intersection: {1/intersection_ratio:.1f}x faster" if intersection_ratio < 1 else f"  Intersection: {intersection_ratio:.1f}x slower")
        
        # Absolute savings
        memory_saved_mb = (dense.memory_bytes - roaring.memory_bytes) / (1024 * 1024)
        print(f"\n  Absolute memory saved: {memory_saved_mb:.1f} MB")
        
        # Projection for larger models
        print(f"\n  Projected for 1M tokens (~62,500 blocks):")
        scale_factor = 62500 / args.num_blocks
        projected_dense_gb = (dense.memory_bytes * scale_factor * scale_factor) / (1024**3)
        projected_roaring_mb = (roaring.memory_bytes * scale_factor) / (1024**2)
        print(f"    Dense Tensor: {projected_dense_gb:.1f} GB")
        print(f"    RoaringBitmap: {projected_roaring_mb:.1f} MB")
    
    if results.get("COO Sparse") and results.get("RoaringBitmap"):
        coo = results["COO Sparse"]
        roaring = results["RoaringBitmap"]
        
        memory_ratio = roaring.memory_bytes / coo.memory_bytes
        
        print(f"\nRoaringBitmap vs COO Sparse:")
        print(f"  Memory: {memory_ratio:.1%} ({abs(1-memory_ratio)*100:.1f}% {'reduction' if memory_ratio < 1 else 'increase'})")


if __name__ == "__main__":
    main()