# Tutorial 07: Request Batching Strategies

## Learning Objectives

1. Understand continuous batching and why it's crucial for throughput
2. Learn batch formation strategies and their impact on performance
3. Master dynamic batch size optimization techniques
4. Explore iteration-level scheduling and preemption
5. Analyze batching trade-offs between latency and throughput

## Overview

Request batching is one of vLLM's key innovations. Unlike traditional static batching that waits for all requests to complete, vLLM uses continuous batching to maximize GPU utilization by dynamically adding and removing requests from batches. This tutorial explores the algorithms and strategies that make this possible.

## Batching Fundamentals

### Traditional Static Batching

```
Traditional Approach:
┌────────────────────────────────────────────┐
│  Batch 1: [Req A, Req B, Req C]           │
│                                            │
│  All start together                        │
│  All must finish before new batch starts   │
│                                            │
│  Timeline:                                 │
│  Req A: ████████████                       │
│  Req B: ████████████████                   │
│  Req C: ████████████████████               │ ← Wait for slowest!
│         └────────────────┘                 │
│         Wasted GPU cycles                  │
└────────────────────────────────────────────┘

Problems:
❌ Wait for all requests to finish
❌ GPU idle while waiting
❌ Head-of-line blocking
❌ Poor latency for short requests
```

### Continuous Batching (vLLM)

```
Continuous Batching:
┌────────────────────────────────────────────┐
│  Dynamic batch composition                 │
│                                            │
│  Timeline:                                 │
│  Req A: ████████████                       │
│  Req B: ████████████████                   │
│  Req C: ████████████████████               │
│  Req D:     ████████                       │ ← Added mid-batch!
│  Req E:         ████████                   │ ← Added mid-batch!
│  Req F:             ████████████           │ ← Added mid-batch!
│                                            │
│  Batch composition over time:              │
│  T0-T3:  [A, B, C]                        │
│  T4-T8:  [B, C, D]     (A finished)       │
│  T9-T12: [C, D, E]     (B finished)       │
│  T13-T16:[C, E, F]     (D finished)       │
│  ...                                       │
└────────────────────────────────────────────┘

Benefits:
✓ No waiting for slow requests
✓ Continuous GPU utilization
✓ Low latency for short requests
✓ High throughput overall
```

## Core Batching Components

### 1. Batch Manager

```python
class BatchManager:
    """
    Manages batching of requests for efficient GPU utilization.
    """

    def __init__(
        self,
        max_batch_size: int,
        max_tokens_per_batch: int,
    ):
        self.max_batch_size = max_batch_size
        self.max_tokens_per_batch = max_tokens_per_batch

        # Current batch
        self.current_batch: list[Request] = []
        self.current_batch_tokens = 0

    def can_add_to_batch(self, request: Request) -> bool:
        """Check if request can be added to current batch"""

        # Check batch size limit
        if len(self.current_batch) >= self.max_batch_size:
            return False

        # Check token limit
        request_tokens = len(request.prompt_tokens) + request.num_generated_tokens
        if self.current_batch_tokens + request_tokens > self.max_tokens_per_batch:
            return False

        return True

    def add_to_batch(self, request: Request) -> None:
        """Add request to current batch"""

        self.current_batch.append(request)
        request_tokens = len(request.prompt_tokens) + request.num_generated_tokens
        self.current_batch_tokens += request_tokens

    def remove_from_batch(self, request: Request) -> None:
        """Remove completed request from batch"""

        self.current_batch.remove(request)
        request_tokens = len(request.prompt_tokens) + request.num_generated_tokens
        self.current_batch_tokens -= request_tokens

    def get_batch(self) -> list[Request]:
        """Get current batch for execution"""
        return self.current_batch.copy()
```

### 2. Iteration-Level Scheduling

The scheduler makes batching decisions at each iteration:

```python
class IterationScheduler:
    """
    Schedules requests for each iteration.
    Core of continuous batching.
    """

    def __init__(
        self,
        max_num_seqs: int,
        max_num_batched_tokens: int,
    ):
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens

        self.waiting_queue: deque[Request] = deque()
        self.running_batch: list[Request] = []

    def schedule_iteration(self) -> SchedulerOutput:
        """
        Schedule one iteration of batch execution.

        Returns:
            SchedulerOutput with requests to execute
        """

        # Step 1: Remove completed requests
        self._remove_finished_requests()

        # Step 2: Add new requests from waiting queue
        self._add_new_requests()

        # Step 3: Create scheduler output
        return self._create_output()

    def _remove_finished_requests(self) -> None:
        """Remove completed requests from running batch"""

        finished = [
            req for req in self.running_batch
            if req.is_finished()
        ]

        for req in finished:
            self.running_batch.remove(req)

    def _add_new_requests(self) -> None:
        """Add new requests to running batch"""

        while self.waiting_queue:
            # Check if we can add more requests
            if not self._can_add_request():
                break

            # Get next request
            request = self.waiting_queue.popleft()

            # Add to running batch
            self.running_batch.append(request)

    def _can_add_request(self) -> bool:
        """Check if we can add another request to batch"""

        # Check sequence limit
        if len(self.running_batch) >= self.max_num_seqs:
            return False

        # Check token limit
        if self.waiting_queue:
            next_request = self.waiting_queue[0]
            current_tokens = sum(
                len(req.prompt_tokens) + req.num_generated_tokens
                for req in self.running_batch
            )
            next_tokens = len(next_request.prompt_tokens)

            if current_tokens + next_tokens > self.max_num_batched_tokens:
                return False

        return True

    def _create_output(self) -> SchedulerOutput:
        """Create scheduler output for this iteration"""

        # Separate prefill and decode requests
        prefill_reqs = [r for r in self.running_batch if r.num_generated_tokens == 0]
        decode_reqs = [r for r in self.running_batch if r.num_generated_tokens > 0]

        return SchedulerOutput(
            prefill_requests=prefill_reqs,
            decode_requests=decode_reqs,
            num_batched_tokens=sum(
                len(r.prompt_tokens) + r.num_generated_tokens
                for r in self.running_batch
            ),
        )
```

## Batch Formation Strategies

### Strategy 1: Greedy Batching

Add requests greedily until limits are reached:

```python
def greedy_batch_formation(
    waiting_queue: deque[Request],
    max_batch_size: int,
    max_tokens: int,
) -> list[Request]:
    """
    Greedy batching: add requests until full.

    Simple and effective for most workloads.
    """

    batch = []
    batch_tokens = 0

    while waiting_queue and len(batch) < max_batch_size:
        request = waiting_queue[0]
        request_tokens = len(request.prompt_tokens)

        # Check if request fits
        if batch_tokens + request_tokens <= max_tokens:
            batch.append(waiting_queue.popleft())
            batch_tokens += request_tokens
        else:
            break  # Request doesn't fit

    return batch
```

**Pros**: Simple, low overhead
**Cons**: May not optimize for specific metrics

### Strategy 2: Length-Balanced Batching

Group requests with similar lengths:

```python
def length_balanced_batching(
    waiting_queue: deque[Request],
    max_batch_size: int,
    max_tokens: int,
    length_variance_threshold: float = 0.2,
) -> list[Request]:
    """
    Length-balanced batching: group similar-length requests.

    Reduces padding waste and variance in batch execution time.
    """

    if not waiting_queue:
        return []

    # Sort by prompt length
    sorted_queue = sorted(
        waiting_queue,
        key=lambda r: len(r.prompt_tokens)
    )

    # Take first request as anchor
    anchor_length = len(sorted_queue[0].prompt_tokens)

    batch = []
    batch_tokens = 0

    for request in sorted_queue:
        request_length = len(request.prompt_tokens)

        # Check length variance
        length_ratio = request_length / anchor_length
        if length_ratio > (1 + length_variance_threshold):
            break  # Too different in length

        # Check limits
        if len(batch) >= max_batch_size:
            break
        if batch_tokens + request_length > max_tokens:
            break

        batch.append(request)
        batch_tokens += request_length

    # Remove batched requests from queue
    for req in batch:
        waiting_queue.remove(req)

    return batch
```

**Pros**: Reduces padding, more uniform execution
**Cons**: May delay longer requests

### Strategy 3: Priority-Based Batching

Prioritize certain requests:

```python
def priority_batching(
    waiting_queue: deque[Request],
    max_batch_size: int,
    max_tokens: int,
) -> list[Request]:
    """
    Priority-based batching: prioritize high-priority requests.

    Useful for SLO-aware serving.
    """

    # Sort by priority (higher first)
    sorted_queue = sorted(
        waiting_queue,
        key=lambda r: r.priority,
        reverse=True
    )

    batch = []
    batch_tokens = 0

    for request in sorted_queue:
        request_tokens = len(request.prompt_tokens)

        # Check limits
        if len(batch) >= max_batch_size:
            break
        if batch_tokens + request_tokens > max_tokens:
            break

        batch.append(request)
        batch_tokens += request_tokens

    # Remove batched requests from queue
    for req in batch:
        waiting_queue.remove(req)

    return batch
```

**Pros**: Respects priorities, good for SLOs
**Cons**: May starve low-priority requests

### Strategy 4: Hybrid Strategy

Combine multiple strategies:

```python
def hybrid_batching(
    waiting_queue: deque[Request],
    max_batch_size: int,
    max_tokens: int,
) -> list[Request]:
    """
    Hybrid batching: combine priority and length balancing.

    1. Group by priority tier
    2. Within tier, balance by length
    """

    # Group by priority tier
    high_priority = [r for r in waiting_queue if r.priority >= 8]
    medium_priority = [r for r in waiting_queue if 4 <= r.priority < 8]
    low_priority = [r for r in waiting_queue if r.priority < 4]

    batch = []
    batch_tokens = 0

    # Fill batch with high priority first
    for priority_group in [high_priority, medium_priority, low_priority]:
        if len(batch) >= max_batch_size or batch_tokens >= max_tokens:
            break

        # Length-balance within priority group
        sorted_group = sorted(priority_group, key=lambda r: len(r.prompt_tokens))

        for request in sorted_group:
            request_tokens = len(request.prompt_tokens)

            if len(batch) >= max_batch_size:
                break
            if batch_tokens + request_tokens > max_tokens:
                break

            batch.append(request)
            batch_tokens += request_tokens
            waiting_queue.remove(request)

    return batch
```

## Dynamic Batch Size Optimization

### Adaptive Batch Sizing

Adjust batch size based on request characteristics:

```python
class AdaptiveBatchSizer:
    """
    Dynamically adjust batch size based on workload.
    """

    def __init__(
        self,
        min_batch_size: int = 8,
        max_batch_size: int = 256,
        target_batch_tokens: int = 8192,
    ):
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.target_batch_tokens = target_batch_tokens

        # Track statistics
        self.avg_prompt_length = 512  # Initial estimate

    def calculate_batch_size(
        self,
        waiting_queue: deque[Request]
    ) -> int:
        """
        Calculate optimal batch size for current workload.

        Returns:
            Optimal batch size
        """

        if not waiting_queue:
            return self.min_batch_size

        # Estimate average prompt length from queue
        queue_lengths = [len(r.prompt_tokens) for r in list(waiting_queue)[:10]]
        current_avg_length = sum(queue_lengths) / len(queue_lengths)

        # Update moving average
        self.avg_prompt_length = (
            0.9 * self.avg_prompt_length + 0.1 * current_avg_length
        )

        # Calculate batch size to hit target tokens
        optimal_batch_size = int(self.target_batch_tokens / self.avg_prompt_length)

        # Clamp to limits
        optimal_batch_size = max(self.min_batch_size, optimal_batch_size)
        optimal_batch_size = min(self.max_batch_size, optimal_batch_size)

        return optimal_batch_size
```

### Throughput-Aware Batching

Maximize throughput by considering generation length:

```python
def throughput_aware_batching(
    waiting_queue: deque[Request],
    max_tokens: int,
) -> list[Request]:
    """
    Optimize batch for throughput.

    Favors requests with higher expected token/second.
    """

    batch = []
    batch_tokens = 0

    # Score each request by expected throughput
    scored_requests = []
    for request in waiting_queue:
        # Estimate total tokens (prompt + generation)
        estimated_total = len(request.prompt_tokens) + request.max_new_tokens

        # Throughput score (inversely proportional to length)
        score = 1.0 / estimated_total

        scored_requests.append((score, request))

    # Sort by score (highest first)
    scored_requests.sort(key=lambda x: x[0], reverse=True)

    # Build batch
    for score, request in scored_requests:
        request_tokens = len(request.prompt_tokens)

        if batch_tokens + request_tokens > max_tokens:
            break

        batch.append(request)
        batch_tokens += request_tokens

    # Remove from queue
    for req in batch:
        waiting_queue.remove(req)

    return batch
```

## Prefill and Decode Separation

### Mixed Batch Challenge

```
Problem: Prefill and decode have different characteristics

Prefill:
  - Many tokens per request (e.g., 512)
  - Compute-bound
  - Benefits from large batches

Decode:
  - One token per request
  - Memory-bound
  - Benefits from many requests

Mixed batch can be inefficient:
  Prefill req: 512 tokens  }
  Decode req:   1 token    } ← Imbalanced!
  Decode req:   1 token    }
```

### Chunked Prefill

Split long prefills into chunks:

```python
def chunked_prefill(
    prefill_request: Request,
    chunk_size: int = 512,
    decode_requests: list[Request] = None,
) -> list[SchedulerOutput]:
    """
    Split prefill into chunks and interleave with decode.

    Allows decode requests to continue making progress
    while processing long prefill.
    """

    outputs = []
    prompt_tokens = prefill_request.prompt_tokens

    # Split prompt into chunks
    num_chunks = (len(prompt_tokens) + chunk_size - 1) // chunk_size

    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, len(prompt_tokens))

        chunk_tokens = prompt_tokens[start_idx:end_idx]

        # Create scheduler output for this chunk
        chunk_output = SchedulerOutput(
            prefill_requests=[
                create_chunk_request(prefill_request, chunk_tokens)
            ],
            decode_requests=decode_requests if decode_requests else [],
        )

        outputs.append(chunk_output)

    return outputs
```

### Separate Prefill/Decode Batches

Process prefill and decode in separate iterations:

```python
class SeparatedBatchScheduler:
    """
    Scheduler with separated prefill and decode batches.
    """

    def __init__(self):
        self.prefill_waiting: deque[Request] = deque()
        self.decode_running: list[Request] = []

        self.prefill_priority = True  # Alternate between prefill and decode

    def schedule_iteration(self) -> SchedulerOutput:
        """Schedule with alternating prefill/decode priority"""

        if self.prefill_priority:
            # Prefill iteration
            output = self._schedule_prefill_batch()
            self.prefill_priority = False
        else:
            # Decode iteration
            output = self._schedule_decode_batch()
            self.prefill_priority = True

        return output

    def _schedule_prefill_batch(self) -> SchedulerOutput:
        """Schedule a prefill-focused batch"""

        # Take prefill requests
        prefill_batch = []
        while self.prefill_waiting and len(prefill_batch) < 32:
            prefill_batch.append(self.prefill_waiting.popleft())

        # Move to decode running
        self.decode_running.extend(prefill_batch)

        return SchedulerOutput(
            prefill_requests=prefill_batch,
            decode_requests=[],
        )

    def _schedule_decode_batch(self) -> SchedulerOutput:
        """Schedule a decode-focused batch"""

        # Remove finished
        self.decode_running = [
            r for r in self.decode_running
            if not r.is_finished()
        ]

        return SchedulerOutput(
            prefill_requests=[],
            decode_requests=self.decode_running,
        )
```

## Preemption

### When to Preempt

```python
def should_preempt(
    running_batch: list[Request],
    waiting_queue: deque[Request],
    kv_cache_manager: KVCacheManager,
) -> bool:
    """
    Decide whether to preempt running requests.

    Preempt when:
    1. New high-priority request arrives
    2. Memory is insufficient for new requests
    """

    # Check memory pressure
    if waiting_queue:
        next_request = waiting_queue[0]

        if not kv_cache_manager.can_allocate(next_request):
            # Need memory, consider preemption
            return True

    # Check priority
    if waiting_queue:
        max_waiting_priority = max(r.priority for r in waiting_queue)
        min_running_priority = min(r.priority for r in running_batch)

        if max_waiting_priority > min_running_priority + 5:
            # Significant priority difference
            return True

    return False
```

### Preemption Strategy

```python
def select_preemption_victims(
    running_batch: list[Request],
    num_blocks_needed: int,
    kv_cache_manager: KVCacheManager,
) -> list[Request]:
    """
    Select which requests to preempt.

    Strategy: Preempt lowest priority requests first.
    """

    # Sort by priority (lowest first)
    sorted_batch = sorted(running_batch, key=lambda r: r.priority)

    victims = []
    blocks_freed = 0

    for request in sorted_batch:
        # Check how many blocks this request uses
        request_blocks = kv_cache_manager.get_num_blocks(request)

        victims.append(request)
        blocks_freed += request_blocks

        if blocks_freed >= num_blocks_needed:
            break  # Freed enough

    return victims


def preempt_requests(
    victims: list[Request],
    running_batch: list[Request],
    waiting_queue: deque[Request],
    kv_cache_manager: KVCacheManager,
) -> None:
    """
    Preempt selected requests.

    1. Save state (if needed)
    2. Free KV cache
    3. Move back to waiting queue
    """

    for request in victims:
        # Free KV cache blocks
        kv_cache_manager.free(request.request_id)

        # Remove from running
        running_batch.remove(request)

        # Add back to waiting (with higher priority)
        request.priority += 1  # Boost priority after preemption
        waiting_queue.appendleft(request)
```

## Performance Analysis

### Latency vs. Throughput Trade-offs

```python
class BatchingMetrics:
    """Metrics for analyzing batching performance"""

    def __init__(self):
        self.latencies: list[float] = []
        self.batch_sizes: list[int] = []
        self.throughputs: list[float] = []

    def record_iteration(
        self,
        batch_size: int,
        iteration_time: float,
        completed_requests: list[Request],
    ):
        """Record metrics for one iteration"""

        self.batch_sizes.append(batch_size)

        # Record latencies
        for req in completed_requests:
            latency = time.time() - req.arrival_time
            self.latencies.append(latency)

        # Calculate throughput (tokens/second)
        total_tokens = sum(
            len(req.prompt_tokens) + req.num_generated_tokens
            for req in completed_requests
        )
        throughput = total_tokens / iteration_time if iteration_time > 0 else 0
        self.throughputs.append(throughput)

    def analyze(self):
        """Analyze metrics"""

        print("Batching Performance Analysis:")
        print(f"  Average batch size: {np.mean(self.batch_sizes):.1f}")
        print(f"  Average latency: {np.mean(self.latencies)*1000:.1f} ms")
        print(f"  P50 latency: {np.percentile(self.latencies, 50)*1000:.1f} ms")
        print(f"  P99 latency: {np.percentile(self.latencies, 99)*1000:.1f} ms")
        print(f"  Average throughput: {np.mean(self.throughputs):.1f} tokens/s")
```

## Hands-On Exercises

### Exercise 1: Compare Batching Strategies

**Objective**: Evaluate different batching strategies

```python
def benchmark_batching_strategies():
    """Compare batching strategies on synthetic workload"""

    # Generate workload
    requests = generate_workload(
        num_requests=1000,
        length_distribution="mixed"  # Mix of short and long
    )

    strategies = {
        "Greedy": greedy_batch_formation,
        "Length-Balanced": length_balanced_batching,
        "Priority-Based": priority_batching,
        "Hybrid": hybrid_batching,
    }

    for name, strategy_fn in strategies.items():
        metrics = run_simulation(requests, strategy_fn)

        print(f"{name}:")
        print(f"  Avg latency: {metrics.avg_latency:.2f} ms")
        print(f"  Throughput: {metrics.throughput:.1f} tok/s")
        print(f"  Batch utilization: {metrics.batch_util:.1f}%")
        print()
```

**Task**: Run and identify best strategy for your workload.

### Exercise 2: Implement Batch Size Auto-Tuning

**Objective**: Automatically find optimal batch size

```python
class BatchSizeAutoTuner:
    """Auto-tune batch size for optimal performance"""

    def __init__(self):
        self.batch_sizes = [16, 32, 64, 128, 256]
        self.best_batch_size = 32
        self.best_throughput = 0

    def tune(self, workload: list[Request]):
        """Find optimal batch size"""

        for batch_size in self.batch_sizes:
            # Run with this batch size
            metrics = run_simulation(
                workload,
                max_batch_size=batch_size
            )

            print(f"Batch size {batch_size}:")
            print(f"  Throughput: {metrics.throughput:.1f} tok/s")

            if metrics.throughput > self.best_throughput:
                self.best_throughput = metrics.throughput
                self.best_batch_size = batch_size

        print(f"\nOptimal batch size: {self.best_batch_size}")
        return self.best_batch_size
```

**Task**: Implement and run auto-tuning.

### Exercise 3: Visualize Batch Composition

**Objective**: Understand batch dynamics over time

```python
import matplotlib.pyplot as plt

def visualize_batch_composition(scheduler_history):
    """Visualize how batch composition changes over time"""

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    times = [h['time'] for h in scheduler_history]
    batch_sizes = [h['batch_size'] for h in scheduler_history]
    batch_tokens = [h['batch_tokens'] for h in scheduler_history]

    # Plot batch size over time
    ax1.plot(times, batch_sizes, label='Batch Size')
    ax1.set_ylabel('Number of Requests')
    ax1.set_title('Batch Size Over Time')
    ax1.legend()
    ax1.grid(True)

    # Plot batch tokens over time
    ax2.plot(times, batch_tokens, label='Batch Tokens', color='orange')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Number of Tokens')
    ax2.set_title('Batch Tokens Over Time')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()
```

**Task**: Run and analyze batch dynamics.

## Common Pitfalls and Solutions

### Pitfall 1: Batch Size Too Large

**Problem**: Large batches cause OOM or high latency.

**Solution**: Set appropriate limits and monitor:

```python
def safe_batch_formation(waiting_queue, max_batch_size, max_tokens):
    """Batch formation with safety checks"""

    batch = []
    batch_tokens = 0

    # Estimate memory usage
    estimated_memory = 0
    memory_limit = 0.9 * get_available_gpu_memory()

    for request in waiting_queue:
        # Estimate memory for this request
        request_memory = estimate_kv_cache_memory(request)

        if estimated_memory + request_memory > memory_limit:
            break  # Would OOM

        # Add to batch
        if len(batch) < max_batch_size and batch_tokens + len(request.prompt_tokens) <= max_tokens:
            batch.append(request)
            batch_tokens += len(request.prompt_tokens)
            estimated_memory += request_memory
        else:
            break

    return batch
```

### Pitfall 2: Ignoring Decode Phase Characteristics

**Problem**: Treating decode same as prefill causes inefficiency.

**Solution**: Separate handling:

```python
def optimize_for_phase(requests):
    """Optimize batching based on phase"""

    prefill_reqs = [r for r in requests if r.num_generated_tokens == 0]
    decode_reqs = [r for r in requests if r.num_generated_tokens > 0]

    # Prefill: Optimize for compute
    # Smaller batches, more tokens per request
    prefill_batch_size = min(32, len(prefill_reqs))

    # Decode: Optimize for throughput
    # Larger batches, one token per request
    decode_batch_size = min(256, len(decode_reqs))

    return prefill_batch_size, decode_batch_size
```

### Pitfall 3: Head-of-Line Blocking

**Problem**: Long requests blocking short ones.

**Solution**: Use preemption and chunking:

```python
def prevent_hol_blocking(waiting_queue, running_batch):
    """Prevent head-of-line blocking"""

    if waiting_queue:
        # Check if waiting request is much shorter
        waiting_req = waiting_queue[0]
        waiting_length = len(waiting_req.prompt_tokens)

        # Find longest running request
        if running_batch:
            longest_running = max(
                running_batch,
                key=lambda r: len(r.prompt_tokens) + r.max_new_tokens
            )
            longest_length = len(longest_running.prompt_tokens) + longest_running.max_new_tokens

            # If waiting request is much shorter, consider preemption
            if waiting_length < longest_length / 10:
                # Preempt longest running request
                preempt_requests([longest_running], running_batch, waiting_queue, kv_cache_manager)
```

## References

### Source Code Files

- **Scheduler**: `/vllm/v1/core/sched/scheduler.py`
- **Request Queue**: `/vllm/v1/core/sched/request_queue.py`
- **Scheduler Output**: `/vllm/v1/core/sched/output.py`

### Key Papers

- "Orca: A Distributed Serving System for Transformer-Based Generative Models" (Yu et al., 2022)
- "vLLM: Efficient Memory Management for Large Language Model Serving with PagedAttention"

### Configuration

```python
@dataclass
class SchedulerConfig:
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 8192
    max_model_len: int = 4096
    policy: str = "fcfs"
```

## Summary

In this tutorial, you learned:

- Continuous batching vs. traditional static batching
- Batch formation strategies and their trade-offs
- Dynamic batch size optimization techniques
- Prefill/decode separation for efficiency
- Preemption mechanisms for resource management
- Performance analysis and optimization

Request batching is fundamental to vLLM's high throughput. Understanding these strategies helps you optimize for your specific workload and SLOs.

## Next Steps

- **Tutorial 08**: Memory Management Techniques
- **Module 5**: Production Deployment Patterns
- **Module 6**: Performance Optimization Advanced
