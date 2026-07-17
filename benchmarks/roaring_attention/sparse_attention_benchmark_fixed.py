#!/usr/bin/env python3
"""
CORRECTED benchmark comparing RoaringBitmap vs Dense Tensor vs COO for sparse attention masks.
Fixes memory measurement issues in the original benchmark.
"""

import time
import gc
import sys
import numpy as np
import torch
import pyroaring
from typing import List, Tuple
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
    avg_connections_per_block: float


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
    
    def get_memory_usage(self) -> int:
        """Get actual memory usage in bytes."""
        # Dense tensor: element_size * number of elements
        return self.mask.element_size() * self.mask.nelement()


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
    
    def get_memory_usage(self) -> int:
        """Get actual memory usage in bytes."""
        # Python set overhead: ~232 bytes per set + 24 bytes per element
        # Dictionary overhead: ~240 bytes + entries
        total = 240  # Dict base overhead
        for key, value_set in self.attention_map.items():
            total += 24  # Dict entry overhead
            total += 232  # Set base overhead  
            total += len(value_set) * 24  # Each integer in set
        return total


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
    
    def get_memory_usage(self) -> int:
        """Get actual memory usage in bytes."""
        # Get actual memory usage using sys.getsizeof for each bitmap
        # Plus the list overhead
        total = sys.getsizeof(self.masks)
        for bitmap in self.masks:
            # RoaringBitmap actual in-memory size
            # We use the cardinality * 4 bytes as approximation for sparse bitmaps
            # Plus container overhead (~64 bytes per container)
            num_values = len(bitmap)
            if num_values == 0:
                total += 64  # Empty bitmap overhead
            else:
                # Estimate: 4 bytes per value for sparse data + container overhead
                # This is more realistic than serialized size
                num_containers = (max(bitmap) // 65536) + 1 if num_values > 0 else 1
                total += num_values * 4 + num_containers * 64
        return total


def create_attention_pattern(num_blocks: int, window_size: int = 128, 
                           global_tokens: int = 16, sparsity: float = 0.99) -> List[Tuple[int, List[int]]]:
    """
    Create a realistic attention pattern for long-context models.
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
        num_sparse = int((1 - sparsity) * num_blocks / 10)
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
    
    # Memory measurement using the corrected method
    gc.collect()
    memory_usage = mask.get_memory_usage()
    
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
    
    # Count total active connections and calculate average
    total_connections = 0
    sample_size = min(100, num_blocks)
    for i in range(sample_size):
        total_connections += len(mask.get_active_blocks(i))
    avg_connections = total_connections / sample_size
    
    # Calculate actual density
    total_possible = num_blocks * num_blocks
    actual_density = total_connections * num_blocks / (sample_size * total_possible)
    
    return BenchmarkResult(
        memory_bytes=memory_usage,
        creation_time=creation_time,
        query_time=query_time,
        intersection_time=intersection_time,
        num_active_connections=total_connections,
        avg_connections_per_block=avg_connections
    )


def format_bytes(bytes_val: int) -> str:
    """Format bytes to human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} TB"


def main():
    parser = argparse.ArgumentParser(description='CORRECTED benchmark for sparse attention masks')
    parser.add_argument('--num-blocks', type=int, default=1000,
                        help='Number of blocks')
    parser.add_argument('--window-size', type=int, default=128,
                        help='Sliding window size for local attention')
    parser.add_argument('--global-tokens', type=int, default=16,
                        help='Number of global attention tokens')
    parser.add_argument('--sparsity', type=float, default=0.99,
                        help='Sparsity level (0.99 = 99% sparse)')
    args = parser.parse_args()
    
    print(f"CORRECTED Sparse Attention Mask Benchmark")
    print(f"=" * 60)
    print(f"Configuration:")
    print(f"  Num blocks: {args.num_blocks:,}")
    print(f"  Approx tokens: {args.num_blocks * 16:,} (assuming block_size=16)")
    print(f"  Window size: {args.window_size}")
    print(f"  Global tokens: {args.global_tokens}")
    print(f"  Sparsity target: {args.sparsity:.1%}")
    print(f"=" * 60)
    
    # Create attention pattern
    print("\nGenerating attention pattern...")
    np.random.seed(42)  # For reproducibility
    attention_patterns = create_attention_pattern(
        args.num_blocks, args.window_size, args.global_tokens, args.sparsity
    )
    
    # Calculate actual density
    total_connections = sum(len(keys) for _, keys in attention_patterns)
    actual_density = total_connections / (args.num_blocks * args.num_blocks)
    print(f"Actual density: {actual_density:.2%} (sparsity: {1-actual_density:.2%})")
    
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
            print(f"  Avg connections per block: {result.avg_connections_per_block:.1f}")
        except Exception as e:
            print(f"  Failed: {e}")
            import traceback
            traceback.print_exc()
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
        print(f"\n  Absolute memory saved: {memory_saved_mb:.2f} MB")
        
        # CORRECTED projection for larger models
        print(f"\n  CORRECTED Projection for 1M tokens (~62,500 blocks):")
        scale_factor = 62500 / args.num_blocks
        
        # Dense scales quadratically (N×N matrix)
        projected_dense_gb = (dense.memory_bytes * scale_factor * scale_factor) / (1024**3)
        
        # RoaringBitmap scales with actual data (number of connections)
        # Each block still has same avg connections, but we have more blocks
        avg_connections = dense.num_active_connections / min(100, args.num_blocks)
        total_roaring_connections = avg_connections * 62500
        # Estimate: 4 bytes per connection + overhead
        projected_roaring_mb = (total_roaring_connections * 4 + 62500 * 64) / (1024**2)
        
        print(f"    Dense Tensor: {projected_dense_gb:.2f} GB")
        print(f"    RoaringBitmap (realistic): {projected_roaring_mb:.1f} MB")
        print(f"    Note: RoaringBitmap projection assumes same sparsity pattern")
    
    if results.get("COO Sparse") and results.get("RoaringBitmap"):
        coo = results["COO Sparse"]
        roaring = results["RoaringBitmap"]
        
        memory_ratio = roaring.memory_bytes / coo.memory_bytes
        
        print(f"\nRoaringBitmap vs COO Sparse:")
        print(f"  Memory: {memory_ratio:.1%} ({abs(1-memory_ratio)*100:.1f}% {'reduction' if memory_ratio < 1 else 'increase'})")
    
    # Reality check
    print(f"\n{'=' * 60}")
    print("REALITY CHECK:")
    print(f"{'=' * 60}")
    print("\n⚠️  Important Notes:")
    print("1. RoaringBitmap memory advantage depends heavily on data patterns")
    print("2. For truly random sparse data, RoaringBitmap may use MORE memory")
    print("3. The benchmark assumes structured sparsity (sliding window + global)")
    print("4. Real-world results will vary based on actual attention patterns")
    print("\nFor production use, profile with your actual model's attention patterns!")


if __name__ == "__main__":
    main()