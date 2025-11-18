# Tutorial 01: Scheduler Deep Dive

## Learning Objectives

1. Understand the core architecture and responsibilities of vLLM's Scheduler
2. Learn how request scheduling algorithms work (FCFS, priority-based)
3. Explore request lifecycle management and state transitions
4. Analyze the interaction between Scheduler, KVCacheManager, and BlockPool
5. Master debugging and performance tuning of the scheduler

## Overview

The Scheduler is the central orchestration component in vLLM that manages the lifecycle of inference requests. It decides which requests to execute, when to execute them, and how to allocate resources efficiently.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      vLLM Scheduler                          │
│                                                              │
│  ┌────────────┐      ┌──────────────┐    ┌──────────────┐  │
│  │  Waiting   │─────▶│   Running    │───▶│  Finished    │  │
│  │   Queue    │      │   Requests   │    │  Requests    │  │
│  └────────────┘      └──────────────┘    └──────────────┘  │
│        │                     │                    │         │
│        │                     │                    │         │
│        ▼                     ▼                    ▼         │
│  ┌────────────────────────────────────────────────────────┐ │
│  │          KV Cache Manager & Block Pool                 │ │
│  └────────────────────────────────────────────────────────┘ │
│                              │                              │
└──────────────────────────────┼──────────────────────────────┘
                               │
                               ▼
                     ┌─────────────────┐
                     │ Model Executor  │
                     └─────────────────┘
```

## Core Components

### 1. Scheduler Class

**File**: `/vllm/v1/core/sched/scheduler.py`

The main `Scheduler` class manages request queues and coordinates resource allocation:

```python
class Scheduler(SchedulerInterface):
    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        structured_output_manager: StructuredOutputManager,
        block_size: int,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        # Key configuration
        self.vllm_config = vllm_config
        self.scheduler_config = vllm_config.scheduler_config

        # Scheduling constraints
        self.max_num_running_reqs = self.scheduler_config.max_num_seqs
        self.max_num_scheduled_tokens = self.scheduler_config.max_num_batched_tokens

        # Request storage
        self.requests: dict[str, Request] = {}

        # Priority queues for requests
        self.waiting = create_request_queue(self.policy)
        self.running: list[Request] = []
        self.finished_req_ids: set[str] = set()
```

**Key Lines**: Lines 52-142

### 2. Request Queues

The scheduler maintains three primary queues:

1. **Waiting Queue**: Newly arrived requests waiting for resources
2. **Running Queue**: Requests currently being processed
3. **Finished Queue**: Completed requests

```python
# From scheduler.py (lines 128-142)
try:
    self.policy = SchedulingPolicy(self.scheduler_config.policy)
except ValueError as e:
    raise ValueError(
        f"Unknown scheduling policy: {self.scheduler_config.policy}"
    ) from e

# Priority queues for requests
self.waiting = create_request_queue(self.policy)
self.running: list[Request] = []

# Finished request tracking
self.finished_req_ids: set[str] = set()
```

## Scheduling Policies

### Available Policies

**File**: `/vllm/v1/core/sched/request_queue.py`

```python
class SchedulingPolicy(str, Enum):
    """Scheduling policy for request queue"""
    FCFS = "fcfs"      # First-Come-First-Serve
    PRIORITY = "priority"  # Priority-based scheduling
```

### FCFS (First-Come-First-Serve)

Simple queue-based scheduling where requests are processed in arrival order:

```
Time ───▶
Request A ─────────────▶ [Processing] ─────▶ Done
         Request B ──────────────────────────▶ [Waiting]
                  Request C ─────────────────────────▶ [Waiting]
```

### Priority-Based Scheduling

Requests with higher priority are processed first:

```
Priority Queue:
┌──────────────────────┐
│ Request C (Priority 1)│ ◀── Processed first
├──────────────────────┤
│ Request A (Priority 5)│ ◀── Processed second
├──────────────────────┤
│ Request B (Priority 9)│ ◀── Processed last
└──────────────────────┘
```

## Request Lifecycle

### State Transition Diagram

```
                  ┌──────────────┐
                  │   WAITING    │
                  └──────┬───────┘
                         │
                         │ schedule()
                         ▼
                  ┌──────────────┐
                  │   RUNNING    │
                  └──────┬───────┘
                         │
                         │ update_from_outputs()
                         ▼
        ┌────────────────┴────────────────┐
        │                                  │
        ▼                                  ▼
┌──────────────┐                  ┌──────────────┐
│  FINISHED_   │                  │  FINISHED_   │
│  STOPPED     │                  │  ABORTED     │
└──────────────┘                  └──────────────┘
```

### Request Status Enum

```python
class RequestStatus(str, Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED_STOPPED = "finished_stopped"
    FINISHED_ABORTED = "finished_aborted"
    FINISHED_LENGTH_CAPPED = "finished_length_capped"
    FINISHED_IGNORED = "finished_ignored"
```

## Scheduling Algorithm Deep Dive

### Main Scheduling Loop

The scheduler's core decision-making happens in the `schedule()` method:

```python
def schedule(self) -> SchedulerOutput:
    """
    Main scheduling logic that decides:
    1. Which waiting requests can start (become running)
    2. Which running requests should continue
    3. Which requests need to be preempted (if any)
    """

    # Step 1: Check running requests for completion
    finished = []
    for req in self.running:
        if req.is_finished():
            finished.append(req)
            self.finished_req_ids.add(req.request_id)

    # Remove finished requests from running
    for req in finished:
        self.running.remove(req)

    # Step 2: Try to schedule new requests from waiting queue
    num_scheduled_tokens = sum(req.num_tokens for req in self.running)

    while (self.waiting and
           len(self.running) < self.max_num_running_reqs):
        req = self.waiting.peek()

        # Check if we have enough resources
        if num_scheduled_tokens + req.num_tokens > self.max_num_scheduled_tokens:
            break  # Cannot schedule more requests

        # Allocate KV cache blocks
        if not self.kv_cache_manager.can_allocate(req):
            break  # Not enough memory

        # Move request from waiting to running
        req = self.waiting.pop()
        self.kv_cache_manager.allocate(req)
        self.running.append(req)
        num_scheduled_tokens += req.num_tokens

    return self._create_scheduler_output()
```

### Resource Constraint Checking

The scheduler enforces two main constraints:

**1. Token Budget Constraint**

```python
# Maximum tokens that can be processed in a single iteration
max_num_scheduled_tokens = scheduler_config.max_num_batched_tokens

# Running total
current_tokens = sum(req.num_computed_tokens for req in running_requests)

# Can we schedule this request?
can_schedule = (current_tokens + new_req.num_tokens) <= max_num_scheduled_tokens
```

**2. Request Count Constraint**

```python
# Maximum concurrent requests
max_num_running_reqs = scheduler_config.max_num_seqs

# Can we add another request?
can_schedule = len(self.running) < max_num_running_reqs
```

## Integration with KV Cache Manager

### Memory Allocation Flow

```
Scheduler.schedule()
    │
    ├──▶ Check if request can be allocated
    │    └──▶ KVCacheManager.can_allocate(request)
    │         └──▶ BlockPool.get_num_free_blocks()
    │
    └──▶ Allocate blocks for request
         └──▶ KVCacheManager.allocate(request)
              └──▶ BlockPool.allocate(num_blocks)
```

### Code Example

```python
# From scheduler implementation
class Scheduler:
    def _try_schedule_request(self, req: Request) -> bool:
        """Try to schedule a single request"""

        # Calculate required blocks
        num_required_blocks = (req.num_tokens + self.block_size - 1) // self.block_size

        # Check if blocks are available
        if not self.kv_cache_manager.can_allocate(num_required_blocks):
            return False

        # Allocate the blocks
        allocated_blocks = self.kv_cache_manager.allocate(req.request_id, num_required_blocks)
        req.kv_cache_blocks = allocated_blocks

        return True
```

## Scheduler Output

The scheduler produces a `SchedulerOutput` object that contains:

```python
@dataclass
class SchedulerOutput:
    """Output from the scheduler containing scheduling decisions"""

    # Requests to execute in this iteration
    scheduled_new_reqs: list[NewRequestData]
    scheduled_cached_reqs: list[CachedRequestData]

    # Requests that completed
    finished_req_ids: set[str]

    # Total number of scheduled tokens
    num_scheduled_tokens: int

    # KV cache block information
    kv_cache_blocks: KVCacheBlocks

    # Grammar/structured output info (if applicable)
    grammar_output: GrammarOutput | None = None
```

## Performance Optimization Strategies

### 1. Continuous Batching

Instead of waiting for all requests in a batch to complete, vLLM continuously adds/removes requests:

```
Traditional Batching:
Batch 1: [A, B, C] ──▶ Wait for all to finish ──▶ Batch 2: [D, E, F]

Continuous Batching:
Time 0: [A, B, C]
Time 1: [A, B, C, D]    (D added)
Time 2: [B, C, D, E]    (A finished, E added)
Time 3: [C, D, E, F]    (B finished, F added)
```

### 2. Preemption

When memory is tight, the scheduler can preempt lower-priority requests:

```python
def _preempt_requests(self) -> list[Request]:
    """Preempt requests to free up memory"""

    preempted = []

    # Sort running requests by priority (lowest first)
    sorted_running = sorted(self.running,
                           key=lambda r: r.priority,
                           reverse=True)

    for req in sorted_running:
        if self.kv_cache_manager.has_enough_free_blocks():
            break

        # Free the KV cache blocks
        self.kv_cache_manager.free(req.request_id)

        # Move back to waiting queue
        self.running.remove(req)
        self.waiting.push(req)
        preempted.append(req)

    return preempted
```

### 3. Prefix Caching

The scheduler works with the KV cache manager to reuse common prefixes:

```
Request A: "Translate to French: Hello"
Request B: "Translate to French: Goodbye"

Common prefix: "Translate to French:"
└──▶ Cache this prefix's KV states
     └──▶ Reuse for Request B (saves computation)
```

## Common Scheduling Patterns

### Pattern 1: Batch Size Optimization

```python
# Dynamically adjust batch size based on request characteristics
def calculate_optimal_batch_size(requests: list[Request]) -> int:
    avg_seq_len = sum(r.num_tokens for r in requests) / len(requests)

    if avg_seq_len < 512:
        # Short sequences: larger batch
        return min(256, len(requests))
    elif avg_seq_len < 2048:
        # Medium sequences: moderate batch
        return min(128, len(requests))
    else:
        # Long sequences: smaller batch
        return min(32, len(requests))
```

### Pattern 2: Priority Handling

```python
# Assign priorities based on request characteristics
def assign_priority(request: Request) -> int:
    priority = 0

    # Shorter requests get higher priority
    if request.max_tokens < 100:
        priority += 10

    # Cached requests get higher priority
    if request.has_prefix_cache:
        priority += 5

    # User-specified priority
    priority += request.user_priority

    return priority
```

## Hands-On Exercises

### Exercise 1: Trace Request Lifecycle

**Objective**: Follow a request through the scheduler

```python
# Add logging to track request state changes
import logging

logger = logging.getLogger(__name__)

class TrackedScheduler(Scheduler):
    def schedule(self) -> SchedulerOutput:
        logger.info(f"Scheduling iteration started")
        logger.info(f"  Waiting: {len(self.waiting)} requests")
        logger.info(f"  Running: {len(self.running)} requests")

        output = super().schedule()

        logger.info(f"  Scheduled: {len(output.scheduled_new_reqs)} new requests")
        logger.info(f"  Finished: {len(output.finished_req_ids)} requests")

        return output
```

**Task**: Run vLLM with this tracked scheduler and observe the output for 10 requests.

### Exercise 2: Implement Custom Scheduling Policy

**Objective**: Create a shortest-job-first scheduling policy

```python
from vllm.v1.core.sched.request_queue import RequestQueue

class ShortestJobFirstQueue(RequestQueue):
    """Schedule requests with fewer tokens first"""

    def push(self, req: Request) -> None:
        # Insert in sorted order by num_tokens
        insert_pos = 0
        for i, existing_req in enumerate(self._queue):
            if req.num_tokens < existing_req.num_tokens:
                insert_pos = i
                break
        self._queue.insert(insert_pos, req)

    def pop(self) -> Request:
        return self._queue.pop(0)

    def peek(self) -> Request:
        return self._queue[0] if self._queue else None
```

**Task**: Integrate this policy and measure throughput compared to FCFS.

### Exercise 3: Memory Pressure Simulation

**Objective**: Test scheduler behavior under memory constraints

```python
def simulate_memory_pressure():
    """Simulate low memory scenario"""

    # Create scheduler with limited cache
    scheduler = Scheduler(
        vllm_config=vllm_config,
        kv_cache_config=KVCacheConfig(
            num_gpu_blocks=100,  # Very limited!
        ),
        block_size=16,
    )

    # Create many large requests
    requests = [
        create_request(request_id=f"req_{i}", num_tokens=512)
        for i in range(50)
    ]

    # Add to scheduler
    for req in requests:
        scheduler.add_request(req)

    # Observe scheduling behavior
    for iteration in range(20):
        output = scheduler.schedule()
        print(f"Iteration {iteration}:")
        print(f"  Scheduled: {len(output.scheduled_new_reqs)}")
        print(f"  Waiting: {len(scheduler.waiting)}")
        print(f"  Running: {len(scheduler.running)}")
```

**Task**: Run this simulation and analyze when/how preemption occurs.

## Common Pitfalls and Solutions

### Pitfall 1: Memory Fragmentation

**Problem**: Many small requests can fragment memory, preventing large requests from being scheduled.

**Solution**: Implement block defragmentation or use a buddy allocator:

```python
def defragment_cache():
    """Reorganize KV cache blocks to reduce fragmentation"""

    # Identify fragmented blocks
    free_blocks = kv_cache_manager.get_free_blocks()

    if is_fragmented(free_blocks):
        # Temporarily pause new requests
        # Move active blocks to contiguous regions
        # Resume scheduling
        pass
```

### Pitfall 2: Starvation of Large Requests

**Problem**: If many small requests keep arriving, large requests may never get scheduled.

**Solution**: Implement aging or priority boosting:

```python
def boost_waiting_requests():
    """Increase priority of requests waiting too long"""

    current_time = time.time()

    for req in scheduler.waiting:
        wait_time = current_time - req.arrival_time

        if wait_time > MAX_WAIT_TIME:
            req.priority += PRIORITY_BOOST
```

### Pitfall 3: Inefficient Batch Formation

**Problem**: Batching requests with very different lengths can waste computation.

**Solution**: Group requests by similar lengths:

```python
def create_length_balanced_batch(requests: list[Request]) -> list[Request]:
    """Create batch with similar-length requests"""

    # Sort by length
    sorted_reqs = sorted(requests, key=lambda r: r.num_tokens)

    # Take requests within 20% length variance
    base_length = sorted_reqs[0].num_tokens
    batch = []

    for req in sorted_reqs:
        if req.num_tokens <= base_length * 1.2:
            batch.append(req)
        else:
            break

    return batch
```

## Debugging Scheduler Issues

### Enable Scheduler Logging

```python
import os
os.environ['VLLM_LOGGING_LEVEL'] = 'DEBUG'

# In scheduler.py, add debug logs
logger.debug(f"Attempting to schedule request {req.request_id}")
logger.debug(f"  Current tokens: {num_scheduled_tokens}/{self.max_num_scheduled_tokens}")
logger.debug(f"  Current requests: {len(self.running)}/{self.max_num_running_reqs}")
```

### Monitor Scheduler Metrics

```python
# Collect scheduler statistics
class SchedulerMetrics:
    def __init__(self):
        self.total_scheduled = 0
        self.total_preempted = 0
        self.total_waiting_time = 0.0
        self.batch_sizes = []

    def record_schedule(self, output: SchedulerOutput):
        self.total_scheduled += len(output.scheduled_new_reqs)
        self.batch_sizes.append(len(output.scheduled_new_reqs))

    def print_stats(self):
        print(f"Total scheduled: {self.total_scheduled}")
        print(f"Average batch size: {sum(self.batch_sizes) / len(self.batch_sizes)}")
        print(f"Total preempted: {self.total_preempted}")
```

### Visualize Scheduler Behavior

```python
def visualize_scheduler_timeline(scheduler_logs: list):
    """Create a Gantt chart of request execution"""

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))

    for req_id, start_time, end_time in scheduler_logs:
        ax.barh(req_id, end_time - start_time, left=start_time)

    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Request ID')
    ax.set_title('Request Scheduling Timeline')
    plt.tight_layout()
    plt.show()
```

## Performance Considerations

### 1. Scheduling Overhead

The scheduler should make decisions quickly to avoid bottlenecking:

```python
# Target: < 1ms per scheduling iteration
with record_function_or_nullcontext('scheduler.schedule'):
    output = scheduler.schedule()
```

### 2. Queue Data Structure Choice

- **FCFS**: Use simple deque (O(1) push/pop)
- **Priority**: Use heap queue (O(log n) push/pop)

```python
from collections import deque
import heapq

# FCFS queue
fcfs_queue = deque()
fcfs_queue.append(request)  # O(1)
fcfs_queue.popleft()  # O(1)

# Priority queue
priority_queue = []
heapq.heappush(priority_queue, (priority, request))  # O(log n)
heapq.heappop(priority_queue)  # O(log n)
```

### 3. Cache-Aware Scheduling

Schedule requests with overlapping prefixes together for better cache hit rates:

```python
def cluster_by_prefix_similarity(requests: list[Request]) -> list[list[Request]]:
    """Group requests with similar prefixes"""

    clusters = []

    for req in requests:
        # Find cluster with similar prefix
        for cluster in clusters:
            if has_prefix_overlap(req, cluster[0]):
                cluster.append(req)
                break
        else:
            # No matching cluster, create new one
            clusters.append([req])

    return clusters
```

## Advanced Topics

### Speculative Decoding Integration

The scheduler coordinates with speculative decoding to schedule both draft and target models:

```python
if self.speculative_config:
    # Schedule draft model tokens
    draft_output = self.draft_scheduler.schedule()

    # Schedule target model verification
    target_output = self.target_scheduler.schedule(
        verify_tokens=draft_output.tokens
    )
```

### Multi-LoRA Scheduling

When using multiple LoRA adapters, the scheduler batches requests by adapter:

```python
def group_by_lora_adapter(requests: list[Request]) -> dict[str, list[Request]]:
    """Group requests by LoRA adapter ID"""

    groups = defaultdict(list)

    for req in requests:
        adapter_id = req.lora_request.lora_id if req.lora_request else "base"
        groups[adapter_id].append(req)

    return groups
```

## References

### Source Code Files

- **Main Scheduler**: `/vllm/v1/core/sched/scheduler.py`
- **Request Queue**: `/vllm/v1/core/sched/request_queue.py`
- **Scheduler Interface**: `/vllm/v1/core/sched/interface.py`
- **Scheduler Output**: `/vllm/v1/core/sched/output.py`
- **Request Definition**: `/vllm/v1/request.py`

### Key Configuration Parameters

```python
# In scheduler_config.py
@dataclass
class SchedulerConfig:
    max_num_seqs: int = 256  # Max concurrent requests
    max_num_batched_tokens: int = 8192  # Max tokens per iteration
    policy: str = "fcfs"  # Scheduling policy
    delay_factor: float = 0.0  # Delay factor for preemption
```

### Related Documentation

- vLLM Architecture Overview
- KV Cache Manager Tutorial (Module 4, Tutorial 2)
- Request Batching Strategies (Module 4, Tutorial 7)
- Performance Tuning Guide

## Summary

In this tutorial, you learned:

- The Scheduler's central role in orchestrating request execution
- How scheduling policies (FCFS, priority) work internally
- The complete request lifecycle and state transitions
- Integration between Scheduler, KV Cache Manager, and Block Pool
- Performance optimization techniques like continuous batching and preemption
- Common pitfalls and how to debug scheduler issues

The scheduler is the brain of vLLM's serving system. Understanding its internals helps you optimize throughput, latency, and resource utilization for your specific workload.

## Next Steps

- **Tutorial 02**: Block Manager Walkthrough - Dive into memory block allocation
- **Tutorial 03**: Model Executor Architecture - Understand request execution
- **Tutorial 07**: Request Batching Strategies - Advanced batching techniques
