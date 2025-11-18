# Tutorial 08: Memory Management Techniques

## Learning Objectives

1. Understand GPU memory allocation strategies in vLLM
2. Learn about memory pools and their management
3. Master fragmentation detection and mitigation techniques
4. Explore memory monitoring and profiling tools
5. Debug out-of-memory errors and optimize memory usage

## Overview

Efficient memory management is critical for maximizing throughput and serving capacity in vLLM. This tutorial covers advanced memory management techniques, from low-level GPU memory allocation to high-level strategies for preventing fragmentation and optimizing memory utilization.

## GPU Memory Architecture

### Memory Hierarchy

```
┌─────────────────────────────────────────────────┐
│              CPU (Host) Memory                   │
│  - Large (tens/hundreds of GB)                  │
│  - Slow access from GPU (~10 GB/s)              │
└────────────────┬────────────────────────────────┘
                 │ PCIe Bus
                 ▼
┌─────────────────────────────────────────────────┐
│              GPU Memory (VRAM)                   │
│                                                  │
│  ┌──────────────────────────────────────────┐  │
│  │  Global Memory (16-80 GB)                │  │
│  │  - Model weights                         │  │
│  │  - KV cache                              │  │
│  │  - Activation buffers                    │  │
│  │  Bandwidth: ~1-2 TB/s                    │  │
│  └──────────────────────────────────────────┘  │
│                                                  │
│  ┌──────────────────────────────────────────┐  │
│  │  L2 Cache (few MB)                       │  │
│  │  Bandwidth: ~5 TB/s                      │  │
│  └──────────────────────────────────────────┘  │
│                                                  │
│  ┌──────────────────────────────────────────┐  │
│  │  L1 Cache / Shared Memory (SM-local)     │  │
│  │  Very fast, very small                   │  │
│  └──────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

### Memory Breakdown in vLLM

For a typical deployment:

```
Total GPU Memory: 80 GB (A100)

Breakdown:
┌───────────────────────────────────────┐
│ Model Weights (13B model, fp16)      │
│ = 13B params × 2 bytes               │
│ = 26 GB                               │
├───────────────────────────────────────┤
│ Activation Buffers                    │
│ = 2-4 GB (depends on batch size)     │
├───────────────────────────────────────┤
│ KV Cache (PagedAttention blocks)     │
│ = ~45-50 GB (remaining space)        │
├───────────────────────────────────────┤
│ CUDA Context & Overhead              │
│ = 2-3 GB                              │
└───────────────────────────────────────┘

KV Cache Blocks:
  Block size: 16 tokens
  Per-block memory: 512 KB (for 13B model)
  Total blocks: ~90,000 blocks
  Serves: ~180-200 concurrent sequences
```

## Memory Pool Management

### PyTorch Memory Allocator

vLLM uses PyTorch's caching allocator:

```python
import torch

# PyTorch maintains a memory pool
# Allocations are served from pool when possible
# Releases return memory to pool (not OS)

def understand_pytorch_allocator():
    """Understand PyTorch memory allocator behavior"""

    # Check initial state
    print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"Reserved:  {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

    # Allocate tensor
    tensor = torch.zeros(1024, 1024, 1024, dtype=torch.float16, device='cuda')

    print(f"\nAfter allocation:")
    print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"Reserved:  {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

    # Delete tensor
    del tensor

    print(f"\nAfter deletion:")
    print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")  # Decreases
    print(f"Reserved:  {torch.cuda.memory_reserved() / 1024**3:.2f} GB")   # Stays same!

    # Memory still reserved in pool for reuse
    # To actually free:
    torch.cuda.empty_cache()

    print(f"\nAfter empty_cache:")
    print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"Reserved:  {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
```

### Custom Memory Pool for KV Cache

```python
class KVCacheMemoryPool:
    """
    Custom memory pool for KV cache blocks.
    Pre-allocates contiguous memory and manages blocks.
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        num_layers: int,
        num_kv_heads: int,
        head_size: int,
        dtype: torch.dtype = torch.float16,
    ):
        self.num_blocks = num_blocks
        self.block_size = block_size

        # Pre-allocate entire KV cache memory
        # Shape: [num_blocks, 2, block_size, num_kv_heads, head_size]
        self.kv_cache = torch.zeros(
            num_blocks,
            2,  # K and V
            block_size,
            num_kv_heads,
            head_size,
            dtype=dtype,
            device='cuda'
        )

        # Track which blocks are free
        self.free_blocks = set(range(num_blocks))
        self.allocated_blocks = set()

        # Memory stats
        self.total_memory = self.kv_cache.numel() * self.kv_cache.element_size()
        self.allocated_memory = 0

    def allocate_block(self) -> int | None:
        """Allocate a single block from pool"""

        if not self.free_blocks:
            return None  # Pool exhausted

        # Get a free block
        block_id = self.free_blocks.pop()
        self.allocated_blocks.add(block_id)

        # Update stats
        block_memory = (
            2 * self.block_size * self.kv_cache.shape[3] * self.kv_cache.shape[4] *
            self.kv_cache.element_size()
        )
        self.allocated_memory += block_memory

        return block_id

    def free_block(self, block_id: int) -> None:
        """Free a block back to pool"""

        if block_id not in self.allocated_blocks:
            raise ValueError(f"Block {block_id} not allocated")

        # Return to free pool
        self.allocated_blocks.remove(block_id)
        self.free_blocks.add(block_id)

        # Update stats
        block_memory = (
            2 * self.block_size * self.kv_cache.shape[3] * self.kv_cache.shape[4] *
            self.kv_cache.element_size()
        )
        self.allocated_memory -= block_memory

        # Optionally zero out block (for security/debugging)
        # self.kv_cache[block_id].zero_()

    def get_block_memory(self, block_id: int) -> torch.Tensor:
        """Get memory slice for a specific block"""

        if block_id not in self.allocated_blocks:
            raise ValueError(f"Block {block_id} not allocated")

        return self.kv_cache[block_id]

    def get_memory_stats(self) -> dict:
        """Get memory pool statistics"""

        return {
            'total_blocks': self.num_blocks,
            'free_blocks': len(self.free_blocks),
            'allocated_blocks': len(self.allocated_blocks),
            'total_memory_gb': self.total_memory / 1024**3,
            'allocated_memory_gb': self.allocated_memory / 1024**3,
            'utilization': len(self.allocated_blocks) / self.num_blocks,
        }
```

## Memory Allocation Strategies

### Strategy 1: First-Fit Allocation

```python
class FirstFitAllocator:
    """
    First-fit allocation: allocate first available block(s).
    Simple and fast, but can lead to fragmentation.
    """

    def __init__(self, memory_pool: KVCacheMemoryPool):
        self.pool = memory_pool

    def allocate(self, num_blocks: int) -> list[int] | None:
        """Allocate num_blocks consecutive or fragmented blocks"""

        if len(self.pool.free_blocks) < num_blocks:
            return None  # Insufficient blocks

        # Get first available blocks
        allocated = []
        free_list = sorted(self.pool.free_blocks)

        for block_id in free_list:
            allocated.append(self.pool.allocate_block())

            if len(allocated) == num_blocks:
                break

        return allocated

    def free(self, blocks: list[int]) -> None:
        """Free allocated blocks"""

        for block_id in blocks:
            self.pool.free_block(block_id)
```

### Strategy 2: Best-Fit Allocation

```python
class BestFitAllocator:
    """
    Best-fit allocation: find smallest free region that fits request.
    Reduces fragmentation but slower.
    """

    def __init__(self, memory_pool: KVCacheMemoryPool):
        self.pool = memory_pool
        # Track contiguous regions
        self.free_regions = self._build_free_regions()

    def _build_free_regions(self) -> list[tuple[int, int]]:
        """Build list of contiguous free regions (start, length)"""

        free_sorted = sorted(self.pool.free_blocks)
        regions = []

        if not free_sorted:
            return regions

        start = free_sorted[0]
        length = 1

        for i in range(1, len(free_sorted)):
            if free_sorted[i] == free_sorted[i-1] + 1:
                # Contiguous
                length += 1
            else:
                # Gap, save current region
                regions.append((start, length))
                start = free_sorted[i]
                length = 1

        # Save last region
        regions.append((start, length))

        return regions

    def allocate(self, num_blocks: int) -> list[int] | None:
        """Allocate using best-fit strategy"""

        # Rebuild regions (could optimize with incremental updates)
        self.free_regions = self._build_free_regions()

        # Find best fit: smallest region that fits request
        best_region = None
        best_size = float('inf')

        for start, length in self.free_regions:
            if length >= num_blocks and length < best_size:
                best_region = (start, length)
                best_size = length

        if best_region is None:
            return None  # No fit found

        # Allocate from best region
        start, _ = best_region
        allocated = []

        for i in range(num_blocks):
            block_id = start + i
            self.pool.allocate_block()
            allocated.append(block_id)

        return allocated
```

### Strategy 3: Buddy Allocator

```python
class BuddyAllocator:
    """
    Buddy allocator: allocate power-of-2 sized blocks.
    Reduces fragmentation by making merging easy.
    """

    def __init__(self, total_blocks: int):
        # Total blocks must be power of 2
        assert total_blocks & (total_blocks - 1) == 0

        self.total_blocks = total_blocks
        self.max_order = total_blocks.bit_length() - 1

        # Free lists for each order (size = 2^order)
        self.free_lists = [[] for _ in range(self.max_order + 1)]

        # Initially, one block of maximum size
        self.free_lists[self.max_order] = [0]

    def allocate(self, num_blocks: int) -> list[int] | None:
        """Allocate power-of-2 blocks using buddy system"""

        # Find order (round up to power of 2)
        order = (num_blocks - 1).bit_length()

        if order > self.max_order:
            return None  # Request too large

        # Find free block of this order or larger
        for current_order in range(order, self.max_order + 1):
            if self.free_lists[current_order]:
                # Found a block
                block_start = self.free_lists[current_order].pop()

                # Split down to required order
                while current_order > order:
                    current_order -= 1
                    # Split block, add buddy to free list
                    buddy_start = block_start + (1 << current_order)
                    self.free_lists[current_order].append(buddy_start)

                # Return allocated blocks
                return list(range(block_start, block_start + (1 << order)))

        return None  # No free blocks

    def free(self, blocks: list[int]) -> None:
        """Free blocks and merge buddies"""

        # Simplification: assume blocks is contiguous power-of-2
        start = blocks[0]
        order = (len(blocks) - 1).bit_length()

        # Try to merge with buddy
        while order < self.max_order:
            # Calculate buddy address
            buddy_start = start ^ (1 << order)

            # Check if buddy is free
            if buddy_start in self.free_lists[order]:
                # Merge with buddy
                self.free_lists[order].remove(buddy_start)
                start = min(start, buddy_start)
                order += 1
            else:
                break  # Can't merge

        # Add merged block to free list
        self.free_lists[order].append(start)
```

## Fragmentation Management

### Detecting Fragmentation

```python
class FragmentationDetector:
    """
    Detect and quantify memory fragmentation.
    """

    def __init__(self, memory_pool: KVCacheMemoryPool):
        self.pool = memory_pool

    def calculate_fragmentation_ratio(self) -> float:
        """
        Calculate fragmentation ratio.

        fragmentation = 1 - (largest_free_region / total_free_blocks)

        0.0 = no fragmentation (all free blocks contiguous)
        1.0 = maximum fragmentation (all free blocks scattered)
        """

        free_blocks = sorted(self.pool.free_blocks)

        if not free_blocks:
            return 0.0  # No free blocks, no fragmentation

        # Find largest contiguous region
        max_region_size = 1
        current_region_size = 1

        for i in range(1, len(free_blocks)):
            if free_blocks[i] == free_blocks[i-1] + 1:
                current_region_size += 1
                max_region_size = max(max_region_size, current_region_size)
            else:
                current_region_size = 1

        fragmentation = 1.0 - (max_region_size / len(free_blocks))

        return fragmentation

    def get_fragmentation_report(self) -> dict:
        """Generate detailed fragmentation report"""

        free_blocks = sorted(self.pool.free_blocks)

        # Find all contiguous regions
        regions = []
        if free_blocks:
            start = free_blocks[0]
            length = 1

            for i in range(1, len(free_blocks)):
                if free_blocks[i] == free_blocks[i-1] + 1:
                    length += 1
                else:
                    regions.append(length)
                    start = free_blocks[i]
                    length = 1

            regions.append(length)

        return {
            'fragmentation_ratio': self.calculate_fragmentation_ratio(),
            'total_free_blocks': len(free_blocks),
            'num_regions': len(regions),
            'largest_region': max(regions) if regions else 0,
            'average_region_size': sum(regions) / len(regions) if regions else 0,
            'region_size_distribution': sorted(regions, reverse=True)[:10],
        }
```

### Defragmentation

```python
class MemoryDefragmenter:
    """
    Defragment KV cache memory by moving blocks.
    """

    def __init__(self, memory_pool: KVCacheMemoryPool):
        self.pool = memory_pool

    def defragment(self, block_mapping: dict[int, Request]) -> dict[int, int]:
        """
        Defragment memory by compacting allocated blocks.

        Args:
            block_mapping: Map of block_id -> owning request

        Returns:
            Relocation map: old_block_id -> new_block_id
        """

        allocated_blocks = sorted(self.pool.allocated_blocks)

        if not allocated_blocks:
            return {}  # Nothing to defragment

        relocation_map = {}
        target_block_id = 0

        for source_block_id in allocated_blocks:
            # Find next free position
            while target_block_id in self.pool.allocated_blocks:
                target_block_id += 1

            if source_block_id != target_block_id:
                # Need to relocate
                self._relocate_block(source_block_id, target_block_id)
                relocation_map[source_block_id] = target_block_id

                # Update mappings
                request = block_mapping.get(source_block_id)
                if request:
                    block_mapping[target_block_id] = request
                    del block_mapping[source_block_id]

            target_block_id += 1

        return relocation_map

    def _relocate_block(self, source: int, target: int) -> None:
        """Move block data from source to target"""

        # Copy block data
        self.pool.kv_cache[target] = self.pool.kv_cache[source].clone()

        # Update allocation tracking
        self.pool.allocated_blocks.remove(source)
        self.pool.allocated_blocks.add(target)
        self.pool.free_blocks.add(source)
        self.pool.free_blocks.remove(target)

    def should_defragment(self) -> bool:
        """Determine if defragmentation is beneficial"""

        detector = FragmentationDetector(self.pool)
        fragmentation = detector.calculate_fragmentation_ratio()

        # Defragment if highly fragmented and low utilization
        utilization = len(self.pool.allocated_blocks) / self.pool.num_blocks

        return fragmentation > 0.5 and utilization < 0.7
```

## Memory Monitoring

### Real-Time Memory Tracker

```python
class MemoryMonitor:
    """
    Monitor GPU memory usage in real-time.
    """

    def __init__(self, device: int = 0):
        self.device = device
        self.history = []

    def snapshot(self) -> dict:
        """Take a memory snapshot"""

        snapshot = {
            'timestamp': time.time(),
            'allocated': torch.cuda.memory_allocated(self.device),
            'reserved': torch.cuda.memory_reserved(self.device),
            'max_allocated': torch.cuda.max_memory_allocated(self.device),
            'max_reserved': torch.cuda.max_memory_reserved(self.device),
        }

        self.history.append(snapshot)

        return snapshot

    def get_current_usage(self) -> dict:
        """Get current memory usage"""

        total = torch.cuda.get_device_properties(self.device).total_memory

        allocated = torch.cuda.memory_allocated(self.device)
        reserved = torch.cuda.memory_reserved(self.device)

        return {
            'total_gb': total / 1024**3,
            'allocated_gb': allocated / 1024**3,
            'reserved_gb': reserved / 1024**3,
            'free_gb': (total - reserved) / 1024**3,
            'utilization': reserved / total,
        }

    def print_usage(self) -> None:
        """Print current memory usage"""

        usage = self.get_current_usage()

        print("GPU Memory Usage:")
        print(f"  Total:     {usage['total_gb']:.2f} GB")
        print(f"  Allocated: {usage['allocated_gb']:.2f} GB")
        print(f"  Reserved:  {usage['reserved_gb']:.2f} GB")
        print(f"  Free:      {usage['free_gb']:.2f} GB")
        print(f"  Utilization: {usage['utilization']*100:.1f}%")

    def plot_history(self):
        """Plot memory usage over time"""

        import matplotlib.pyplot as plt

        if not self.history:
            print("No history to plot")
            return

        times = [(s['timestamp'] - self.history[0]['timestamp']) for s in self.history]
        allocated = [s['allocated'] / 1024**3 for s in self.history]
        reserved = [s['reserved'] / 1024**3 for s in self.history]

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(times, allocated, label='Allocated', linewidth=2)
        ax.plot(times, reserved, label='Reserved', linewidth=2, linestyle='--')

        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Memory (GB)')
        ax.set_title('GPU Memory Usage Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
```

### Memory Profiling

```python
class MemoryProfiler:
    """
    Profile memory allocations to find leaks and inefficiencies.
    """

    def __init__(self):
        self.allocations = []

    def record_allocation(
        self,
        name: str,
        size_bytes: int,
        location: str = ""
    ) -> None:
        """Record a memory allocation"""

        self.allocations.append({
            'timestamp': time.time(),
            'name': name,
            'size_bytes': size_bytes,
            'size_gb': size_bytes / 1024**3,
            'location': location,
            'action': 'allocate',
        })

    def record_free(
        self,
        name: str,
        size_bytes: int
    ) -> None:
        """Record a memory free"""

        self.allocations.append({
            'timestamp': time.time(),
            'name': name,
            'size_bytes': -size_bytes,
            'size_gb': -size_bytes / 1024**3,
            'location': '',
            'action': 'free',
        })

    def analyze(self) -> dict:
        """Analyze allocation patterns"""

        # Group by name
        by_name = {}
        for alloc in self.allocations:
            name = alloc['name']
            if name not in by_name:
                by_name[name] = {
                    'total_allocated': 0,
                    'total_freed': 0,
                    'count': 0,
                }

            if alloc['action'] == 'allocate':
                by_name[name]['total_allocated'] += alloc['size_bytes']
                by_name[name]['count'] += 1
            else:
                by_name[name]['total_freed'] += abs(alloc['size_bytes'])

        # Calculate leaks
        leaks = {}
        for name, stats in by_name.items():
            leaked = stats['total_allocated'] - stats['total_freed']
            if leaked > 0:
                leaks[name] = {
                    'leaked_bytes': leaked,
                    'leaked_gb': leaked / 1024**3,
                    'allocations': stats['count'],
                }

        return {
            'total_allocations': len([a for a in self.allocations if a['action'] == 'allocate']),
            'total_frees': len([a for a in self.allocations if a['action'] == 'free']),
            'leaks': leaks,
            'by_name': by_name,
        }

    def print_report(self) -> None:
        """Print profiling report"""

        analysis = self.analyze()

        print("Memory Profiling Report:")
        print(f"  Total allocations: {analysis['total_allocations']}")
        print(f"  Total frees: {analysis['total_frees']}")

        if analysis['leaks']:
            print("\n  Memory Leaks Detected:")
            for name, leak in sorted(
                analysis['leaks'].items(),
                key=lambda x: x[1]['leaked_bytes'],
                reverse=True
            ):
                print(f"    {name}:")
                print(f"      Leaked: {leak['leaked_gb']:.3f} GB")
                print(f"      Allocations: {leak['allocations']}")
        else:
            print("\n  No memory leaks detected ✓")
```

## Common Memory Issues

### Issue 1: Out of Memory (OOM)

```python
class OOMHandler:
    """Handle out-of-memory situations gracefully"""

    def __init__(self, memory_pool: KVCacheMemoryPool):
        self.pool = memory_pool

    def handle_oom(
        self,
        request: Request,
        kv_cache_manager
    ) -> bool:
        """
        Try to recover from OOM situation.

        Returns:
            True if recovery successful, False otherwise
        """

        print("OOM detected, attempting recovery...")

        # Step 1: Free cached blocks
        num_freed = self._free_cached_blocks(kv_cache_manager)
        print(f"  Freed {num_freed} cached blocks")

        # Step 2: Check if request can now be allocated
        if kv_cache_manager.can_allocate(request):
            print("  Recovery successful ✓")
            return True

        # Step 3: Try defragmentation
        defragmenter = MemoryDefragmenter(self.pool)
        if defragmenter.should_defragment():
            print("  Attempting defragmentation...")
            defragmenter.defragment(kv_cache_manager.request_blocks)

            if kv_cache_manager.can_allocate(request):
                print("  Recovery successful after defragmentation ✓")
                return True

        # Step 4: Try preempting low-priority requests
        print("  Attempting preemption...")
        num_preempted = self._preempt_low_priority(kv_cache_manager, request)
        print(f"  Preempted {num_preempted} requests")

        if kv_cache_manager.can_allocate(request):
            print("  Recovery successful after preemption ✓")
            return True

        print("  Recovery failed ✗")
        return False

    def _free_cached_blocks(self, kv_cache_manager) -> int:
        """Free all cached blocks"""

        num_freed = 0
        cached_blocks = kv_cache_manager.block_pool.get_cached_blocks()

        for block in cached_blocks:
            if block.ref_cnt == 0:
                kv_cache_manager.block_pool.remove_from_cache(block)
                num_freed += 1

        return num_freed

    def _preempt_low_priority(
        self,
        kv_cache_manager,
        new_request: Request
    ) -> int:
        """Preempt low-priority running requests"""

        # Find requests with lower priority
        running_requests = kv_cache_manager.get_running_requests()
        preempt_candidates = [
            req for req in running_requests
            if req.priority < new_request.priority
        ]

        # Sort by priority (lowest first)
        preempt_candidates.sort(key=lambda r: r.priority)

        num_preempted = 0
        for req in preempt_candidates:
            kv_cache_manager.free(req.request_id)
            num_preempted += 1

            # Check if enough freed
            if kv_cache_manager.can_allocate(new_request):
                break

        return num_preempted
```

### Issue 2: Memory Leaks

```python
def detect_memory_leak():
    """Detect memory leaks using PyTorch profiler"""

    import gc

    # Initial state
    torch.cuda.reset_peak_memory_stats()
    initial_allocated = torch.cuda.memory_allocated()

    # Run some operations
    for i in range(100):
        # Your code here
        tensor = torch.randn(1000, 1000, device='cuda')
        # ... operations ...
        del tensor

        # Force garbage collection
        gc.collect()
        torch.cuda.empty_cache()

    # Check if memory increased
    final_allocated = torch.cuda.memory_allocated()
    leaked = final_allocated - initial_allocated

    if leaked > 1024**2:  # More than 1 MB leaked
        print(f"⚠️  Memory leak detected: {leaked / 1024**2:.2f} MB")

        # Find what's holding references
        import sys
        for obj in gc.get_objects():
            if torch.is_tensor(obj) and obj.is_cuda:
                print(f"  Tensor: {obj.shape}, {sys.getrefcount(obj)} refs")
    else:
        print("✓ No memory leak detected")
```

### Issue 3: Fragmentation

```python
def monitor_and_mitigate_fragmentation():
    """Continuously monitor and mitigate fragmentation"""

    detector = FragmentationDetector(memory_pool)
    defragmenter = MemoryDefragmenter(memory_pool)

    while True:
        # Check fragmentation
        report = detector.get_fragmentation_report()

        print(f"Fragmentation: {report['fragmentation_ratio']:.2%}")
        print(f"Largest region: {report['largest_region']} blocks")

        # Defragment if needed
        if report['fragmentation_ratio'] > 0.5:
            print("High fragmentation detected, defragmenting...")
            relocation_map = defragmenter.defragment(block_mapping)
            print(f"Relocated {len(relocation_map)} blocks")

        time.sleep(10)  # Check every 10 seconds
```

## Optimization Techniques

### 1. Memory-Aware Scheduling

```python
def memory_aware_scheduling(
    waiting_queue: list[Request],
    memory_pool: KVCacheMemoryPool,
    memory_threshold: float = 0.9
) -> list[Request]:
    """
    Schedule requests based on memory availability.
    Reject large requests when memory is tight.
    """

    utilization = len(memory_pool.allocated_blocks) / memory_pool.num_blocks

    scheduled = []

    for request in waiting_queue:
        # Estimate memory needed
        blocks_needed = estimate_blocks(request)

        # Check if scheduling would exceed threshold
        new_utilization = (
            (len(memory_pool.allocated_blocks) + blocks_needed) /
            memory_pool.num_blocks
        )

        if new_utilization > memory_threshold:
            # Don't schedule, would exceed threshold
            continue

        scheduled.append(request)

    return scheduled
```

### 2. Lazy Deallocation

```python
class LazyDeallocator:
    """
    Defer deallocation to batch operations.
    Reduces overhead of frequent alloc/dealloc.
    """

    def __init__(self, memory_pool: KVCacheMemoryPool):
        self.pool = memory_pool
        self.pending_frees = []
        self.batch_size = 10

    def free(self, block_id: int) -> None:
        """Queue block for deallocation"""

        self.pending_frees.append(block_id)

        # Batch free when threshold reached
        if len(self.pending_frees) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        """Actually free all pending blocks"""

        for block_id in self.pending_frees:
            self.pool.free_block(block_id)

        self.pending_frees.clear()
```

### 3. Memory Prefetching

```python
async def prefetch_memory_async(
    upcoming_requests: list[Request],
    memory_pool: KVCacheMemoryPool
):
    """
    Prefetch memory asynchronously for upcoming requests.
    """

    for request in upcoming_requests:
        # Estimate blocks needed
        blocks_needed = estimate_blocks(request)

        # Check if we'll need to evict
        if memory_pool.num_free_blocks() < blocks_needed:
            # Proactively evict cached blocks
            await evict_cached_blocks_async(blocks_needed)

        # Prefetch can happen in background
        await asyncio.sleep(0)  # Yield control
```

## Hands-On Exercises

### Exercise 1: Memory Leak Detection

**Objective**: Find and fix a memory leak

```python
def exercise_memory_leak():
    """Exercise: Find the memory leak in this code"""

    memory_monitor = MemoryMonitor()

    # Take initial snapshot
    memory_monitor.snapshot()

    for i in range(100):
        # Allocate tensor
        tensor = torch.randn(1000, 1000, device='cuda')

        # Process tensor
        result = tensor * 2

        # Oops, forgot to clean up result!
        # This is the leak

        memory_monitor.snapshot()

    # Plot to visualize leak
    memory_monitor.plot_history()

# TODO: Fix the leak by adding: del result
```

**Task**: Identify and fix the leak.

### Exercise 2: Implement Defragmentation

**Objective**: Implement a defragmentation algorithm

```python
def exercise_defragmentation():
    """Exercise: Implement defragmentation"""

    # Create fragmented memory state
    pool = KVCacheMemoryPool(num_blocks=100, ...)

    # Allocate blocks in fragmented pattern
    # Blocks: [0, 2, 4, 6, 8, ...] (even only)
    for i in range(0, 50, 2):
        pool.allocate_block()

    # Now odd blocks are free but fragmented
    # Implement defragmentation to compact

    # TODO: Implement defragmenter.defragment()
```

**Task**: Complete the defragmentation implementation.

### Exercise 3: Optimize Allocation Strategy

**Objective**: Find optimal allocation strategy for your workload

```python
def exercise_allocation_strategy():
    """Benchmark different allocation strategies"""

    strategies = {
        'First-Fit': FirstFitAllocator,
        'Best-Fit': BestFitAllocator,
        'Buddy': BuddyAllocator,
    }

    workload = generate_realistic_workload()

    for name, strategy_class in strategies.items():
        allocator = strategy_class(memory_pool)

        # Run workload
        fragmentation_scores = []
        allocation_times = []

        for request in workload:
            start = time.time()
            blocks = allocator.allocate(request.num_blocks)
            allocation_times.append(time.time() - start)

            detector = FragmentationDetector(memory_pool)
            fragmentation_scores.append(detector.calculate_fragmentation_ratio())

        print(f"{name}:")
        print(f"  Avg allocation time: {np.mean(allocation_times)*1000:.3f} ms")
        print(f"  Final fragmentation: {fragmentation_scores[-1]:.2%}")
```

**Task**: Run and determine best strategy.

## References

### Source Code Files

- **Memory Pool**: `/vllm/v1/core/block_pool.py`
- **KV Cache Manager**: `/vllm/v1/core/kv_cache_manager.py`
- **Memory Utils**: `/vllm/v1/core/kv_cache_utils.py`

### CUDA Memory Management

```python
# Useful PyTorch memory functions
torch.cuda.memory_allocated()      # Currently allocated
torch.cuda.memory_reserved()       # Reserved in pool
torch.cuda.max_memory_allocated()  # Peak allocation
torch.cuda.reset_peak_memory_stats()
torch.cuda.empty_cache()           # Release cached memory
torch.cuda.memory_summary()        # Detailed report
```

### Configuration

```python
@dataclass
class MemoryConfig:
    gpu_memory_utilization: float = 0.9  # Use 90% of GPU memory
    max_num_blocks: int | None = None    # Auto-calculate if None
    block_size: int = 16
    enable_prefix_caching: bool = False
```

## Summary

In this tutorial, you learned:

- GPU memory architecture and allocation strategies
- Memory pool management and custom allocators
- Fragmentation detection and mitigation techniques
- Real-time memory monitoring and profiling
- Common memory issues and their solutions
- Optimization techniques for memory-constrained scenarios

Effective memory management is essential for maximizing vLLM's serving capacity. Understanding these techniques helps you optimize memory utilization and handle memory pressure gracefully.

## Next Steps

- **Module 5**: Production Deployment Patterns
- **Module 6**: Performance Optimization Advanced
- Apply these techniques to your production deployments

---

## Module 4 Complete!

Congratulations on completing Module 4: System Components! You now have a deep understanding of:

1. **Scheduler** - Request orchestration and resource allocation
2. **Block Manager** - Memory block management and CoW optimization
3. **Model Executor** - Distributed execution across GPUs
4. **Attention Layer** - PagedAttention and attention backends
5. **Sampler** - Token generation strategies and penalties
6. **KV Cache Management** - Cache lifecycle and prefix caching
7. **Request Batching** - Continuous batching for high throughput
8. **Memory Management** - Advanced memory optimization techniques

You're now equipped to optimize vLLM deployments, debug complex issues, and contribute to the vLLM codebase!
