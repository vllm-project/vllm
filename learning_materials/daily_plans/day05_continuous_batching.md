# Day 5: Continuous Batching - Dynamic Request Scheduling

> **Goal**: Master continuous batching concepts, understand implementation, analyze performance impact
> **Time**: 6-8 hours
> **Prerequisites**: Day 1-4 completed, solid understanding of scheduling basics
> **Deliverables**: Batching analysis report, performance comparison, scheduling policy experiments

---

## 📅 Daily Schedule

### Morning Session (3-4 hours): Continuous Batching Theory

**9:00-9:45** - Traditional Static Batching Problems
**9:45-11:00** - Continuous Batching Algorithm
**11:00-11:30** - Break + Draw Batching Timelines
**11:30-12:30** - Scheduler Implementation Deep Dive

### Afternoon Session (3-4 hours): Performance Analysis

**14:00-15:00** - Throughput vs Latency Trade-offs
**15:00-16:00** - Batching Metrics and Profiling
**16:00-16:30** - Break
**16:30-18:00** - Hands-On: Analyze Different Workloads

### Evening (Optional, 1-2 hours): Experiments

**19:00-21:00** - Custom scheduler experiments, optimization ideas, prepare for Week 1 review

---

## 🎯 Learning Objectives

By end of day, you should be able to:
- [ ] Explain limitations of static batching
- [ ] Describe continuous batching algorithm
- [ ] Understand vLLM's scheduler implementation
- [ ] Analyze throughput and latency trade-offs
- [ ] Profile and measure batching performance
- [ ] Experiment with different scheduling policies
- [ ] Calculate optimal batch sizes for workloads

---

## 📚 Morning: Continuous Batching Deep Dive (9:00-12:30)

### Task 1: Static Batching Problems (45 min)

**Traditional Static Batching**:

```
Static Batch Processing:

Batch 1: [Req A, Req B, Req C, Req D] (size=4)

Process until ALL complete:
  Step 0: Generate token 1 for all
  Step 1: Generate token 2 for all
  ...
  Step 10: A completes (EOS) → but must continue
  Step 11: Generate token 12 for B, C, D (A idle!)
  ...
  Step 20: B completes → still continue
  Step 21: Generate token 22 for C, D (A, B idle!)
  ...
  Step 50: C completes
  Step 51: Generate token 52 for D (A, B, C idle!)
  ...
  Step 100: D completes

Next batch can only start after step 100!
```

**📊 Efficiency Analysis**:

```
Request completion times:
  A: 10 steps  → waits 90 steps idle
  B: 20 steps  → waits 80 steps idle
  C: 50 steps  → waits 50 steps idle
  D: 100 steps → no wait

Average wait: (90 + 80 + 50 + 0) / 4 = 55 steps
Utilization: Only ~45% of "work" is productive

New requests arrive during step 30:
  E, F, G → Must wait until step 100 to start!
  Increased latency for new requests
```

**❌ Key Problems**:

1. **Padding Waste**: Short sequences wait for longest to finish
2. **Poor GPU Utilization**: Idle when only few sequences left
3. **High Latency for New Requests**: Must wait for entire batch
4. **Unpredictable Performance**: Batch blocked by one slow request

**Real-World Impact**:

```
Production scenario:
  - Average request: 50 tokens
  - Outlier request: 500 tokens
  - Batch size: 32

With static batching:
  - 31 requests finish ~50 tokens (average)
  - 1 request needs 500 tokens
  - All 31 must wait for the outlier!
  - 10x increase in latency for majority

Throughput calculation:
  Total tokens: 31 * 50 + 1 * 500 = 2050 tokens
  Time: 500 steps
  Throughput: 4.1 tokens/step
  BUT if we could remove finished requests:
    First 50 steps: 32 sequences = 32 tokens/step
    Last 450 steps: 1 sequence = 1 token/step
    Could have processed new requests in those 450 steps!
```

### Task 2: Continuous Batching Algorithm (75 min)

**💡 Core Idea**: Add and remove requests from batch dynamically!

```
Continuous Batching:

Step 0: Batch = [A, B, C, D] (4 requests)
  Generate tokens for all 4

Step 10: A completes (EOS)
  Remove A from batch
  Batch = [B, C, D] (3 requests)
  Free A's memory
  Check waiting queue → Add E!
  Batch = [B, C, D, E] (4 requests)

Step 20: B completes
  Remove B, add F
  Batch = [C, D, E, F]

Step 30: E completes
  Remove E, add G
  Batch = [C, D, F, G]

... and so on

Result:
  - No idle time waiting for batch to finish
  - New requests start immediately (if space)
  - Better GPU utilization
  - Lower latency for all requests
```

**Algorithm Details**:

```python
# Continuous batching pseudocode

def continuous_batching_engine():
    """Main execution loop with continuous batching."""

    running_requests = []
    waiting_requests = deque()

    while has_work():
        # === PHASE 1: Remove finished requests ===
        running_requests = [r for r in running_requests if not r.finished]

        # === PHASE 2: Add new requests ===
        while len(running_requests) < max_batch_size and waiting_requests:
            new_request = waiting_requests.popleft()

            # Check if we have memory for this request
            if can_allocate_blocks(new_request):
                allocate_blocks(new_request)
                running_requests.append(new_request)
            else:
                # No memory - put back in queue
                waiting_requests.appendleft(new_request)
                break

        # === PHASE 3: Execute batch ===
        if running_requests:
            # Prepare batch input (mixed prefill + decode)
            batch_input = prepare_batch(running_requests)

            # Run model
            outputs = model.forward(batch_input)

            # Update each request with its new token
            for request, output in zip(running_requests, outputs):
                token = sample_token(output)
                request.append_token(token)

                # Check if finished
                if is_finished(request):
                    free_blocks(request)
                    request.finished = True

        # === PHASE 4: Accept new incoming requests ===
        accept_new_requests(waiting_requests)
```

**Key Benefits**:

```
Continuous vs Static Batching:

Metric                  | Static | Continuous | Improvement
------------------------|--------|------------|------------
Avg time to first token | High   | Low        | 3-5x better
GPU utilization         | 60-70% | 90-95%     | 1.4x better
Throughput (tokens/s)   | Good   | Excellent  | 2-3x better
Tail latency (p99)      | Poor   | Good       | 5-10x better
Batch efficiency        | 50-60% | 85-95%     | 1.7x better
```

### Task 3: vLLM Scheduler Implementation (60 min)

**File**: `vllm/core/scheduler.py`

**Detailed Scheduler Analysis**:

```python
# vllm/core/scheduler.py (detailed walkthrough)

class Scheduler:
    """
    Continuous batching scheduler.

    Key features:
    - Three queues: waiting, running, swapped
    - Dynamic batch composition
    - Memory-aware scheduling
    - Preemption when out of memory
    """

    def __init__(self, scheduler_config, cache_config):
        # Request queues
        self.waiting: Deque[SequenceGroup] = deque()
        self.running: List[SequenceGroup] = []
        self.swapped: Deque[SequenceGroup] = deque()

        # Configuration
        self.max_num_seqs = scheduler_config.max_num_seqs  # e.g., 256
        self.max_num_batched_tokens = scheduler_config.max_num_batched_tokens  # e.g., 2048

        # Block manager
        self.block_manager = BlockSpaceManager(...)

        # Scheduling policy
        self.policy = PolicyFactory.get_policy(scheduler_config.policy)  # FCFS, Priority, etc.

    def add_seq_group(self, seq_group: SequenceGroup) -> None:
        """Add new request to waiting queue."""
        self.waiting.append(seq_group)

    def abort_seq_group(self, request_id: str) -> None:
        """Abort request (user cancellation)."""
        # Find and remove from any queue
        # Free allocated blocks
        pass

    def schedule(self) -> SchedulerOutputs:
        """
        Main scheduling method - called every step.

        Returns which requests to run and memory operations.
        """

        # Outputs to build
        scheduled_seq_groups = []
        num_batched_tokens = 0
        blocks_to_swap_in = {}
        blocks_to_swap_out = {}
        blocks_to_copy = {}

        # === STEP 1: Swap in requests from CPU ===
        # Bring back preempted requests if we have GPU memory
        self._schedule_swapped(
            scheduled_seq_groups,
            blocks_to_swap_in,
        )

        # === STEP 2: Schedule running requests (decode) ===
        # Priority to already-running requests
        running_scheduled = []

        for seq_group in list(self.running):
            # Check if finished
            if seq_group.is_finished():
                self._free_seq_group(seq_group)
                self.running.remove(seq_group)
                continue

            # Decode: 1 token per sequence
            num_tokens = seq_group.num_seqs()

            # Check batch size limits
            if len(scheduled_seq_groups) >= self.max_num_seqs:
                break  # Batch full
            if num_batched_tokens + num_tokens > self.max_num_batched_tokens:
                break  # Token limit reached

            # Check if we can append token (might need new block)
            while not self._can_append_slot(seq_group):
                # Out of memory - need to preempt
                if not self._preempt_running():
                    # Cannot preempt - skip this request
                    break

            if self._can_append_slot(seq_group):
                # Allocate slot for new token
                self._append_slot(seq_group, blocks_to_copy)
                running_scheduled.append(seq_group)
                num_batched_tokens += num_tokens

        scheduled_seq_groups.extend(running_scheduled)

        # === STEP 3: Schedule waiting requests (prefill) ===
        # Try to start new requests
        waiting_scheduled = []

        # Sort by policy (e.g., FCFS, priority)
        self.waiting = self.policy.sort_by_priority(self.waiting)

        while self.waiting:
            seq_group = self.waiting[0]

            # Prefill: process all prompt tokens
            num_tokens = seq_group.get_seqs()[0].get_len()

            # Check limits
            if len(scheduled_seq_groups) >= self.max_num_seqs:
                break  # Too many sequences

            # For prefill, we can do "chunked prefill" - split large prompts
            num_tokens = min(num_tokens, self._get_max_prefill_tokens())

            if num_batched_tokens + num_tokens > self.max_num_batched_tokens:
                break  # Would exceed token limit

            # Can we allocate blocks?
            while not self._can_allocate(seq_group):
                # Out of memory - try to preempt
                if not self._preempt_running():
                    break  # Cannot free memory

            if self._can_allocate(seq_group):
                # Allocate blocks
                self._allocate(seq_group)

                # Move to running
                self.waiting.popleft()
                self.running.append(seq_group)

                waiting_scheduled.append(seq_group)
                num_batched_tokens += num_tokens
            else:
                # Cannot fit - stop trying to schedule more
                break

        scheduled_seq_groups.extend(waiting_scheduled)

        # === STEP 4: Create output ===
        return SchedulerOutputs(
            scheduled_seq_groups=scheduled_seq_groups,
            num_batched_tokens=num_batched_tokens,
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_swap_out=blocks_to_swap_out,
            blocks_to_copy=blocks_to_copy,
            ignored_seq_groups=[],  # Sequences we couldn't schedule
            num_prefill_groups=len(waiting_scheduled),
            num_decode_groups=len(running_scheduled),
        )

    def _can_append_slot(self, seq_group: SequenceGroup) -> bool:
        """Check if we can add one more token to running sequence."""
        # Each sequence might need a new block when it grows
        for seq in seq_group.get_seqs():
            if seq.get_len() % self.block_size == 0:
                # Crossing block boundary - need new block
                return self.block_manager.can_allocate(1)
        return True

    def _append_slot(
        self,
        seq_group: SequenceGroup,
        blocks_to_copy: Dict[int, List[int]]
    ) -> None:
        """Allocate slot for new token in running sequence."""
        for seq in seq_group.get_seqs():
            if seq.get_len() % self.block_size == 0:
                # Allocate new block
                self.block_manager.append_slot(seq)

    def _preempt_running(self) -> bool:
        """
        Preempt running sequences to free memory.

        Strategy:
        1. Preempt by recomputation (restart from beginning)
        2. Preempt by swapping (move to CPU memory)

        Returns whether preemption successful.
        """
        # Find victim (lowest priority running request)
        if not self.running:
            return False

        # Choose victim based on policy
        victim = self.policy.get_victim(self.running)

        # Decide: recompute or swap
        if self._can_swap_out(victim):
            # Swap to CPU
            self._swap_out(victim)
            self.running.remove(victim)
            self.swapped.append(victim)
        else:
            # Recompute from scratch
            self._free_seq_group(victim)
            self.running.remove(victim)
            self.waiting.appendleft(victim)  # Add back to front

        return True

    def _free_seq_group(self, seq_group: SequenceGroup) -> None:
        """Free all blocks allocated to sequence group."""
        self.block_manager.free(seq_group)

    def free_finished_seq_groups(self) -> None:
        """
        Called after each step to free finished requests.
        Enables continuous batching!
        """
        self.running = [sg for sg in self.running if not sg.is_finished()]
```

**📝 Exercise: Trace Scheduling Scenario**

```
Initial State:
  waiting: [A (200 tokens), B (150 tokens), C (100 tokens)]
  running: [D (at 50 tokens), E (at 30 tokens)]
  max_num_seqs: 5
  max_num_batched_tokens: 300
  free_blocks: 20

Step N: schedule()

1. Swap in: (none in swapped)

2. Schedule running (decode):
   - D: 1 token, can append? Yes
     num_batched_tokens = 1
   - E: 1 token, can append? Yes
     num_batched_tokens = 2
   - Result: scheduled = [D, E], tokens = 2

3. Schedule waiting (prefill):
   - A: 200 tokens
     - Total tokens would be 2 + 200 = 202 ✓ (< 300)
     - Num seqs would be 2 + 1 = 3 ✓ (< 5)
     - Can allocate blocks? Needs 13 blocks, have 20 ✓
     - Allocate and schedule A
     - num_batched_tokens = 202
     - Result: scheduled = [D, E, A], tokens = 202

   - B: 150 tokens
     - Total tokens would be 202 + 150 = 352 ✗ (> 300)
     - Cannot fit, stop scheduling

4. Output:
   scheduled = [D (decode), E (decode), A (prefill)]
   num_batched_tokens = 202
   num_prefill_groups = 1
   num_decode_groups = 2

Next step:
  - D, E continue decode
  - A continues prefill or switches to decode
  - B, C still waiting
  - But if D or E finishes, space opens for B!
```

---

## 🔬 Afternoon: Performance Analysis (14:00-18:00)

### Task 4: Throughput vs Latency (60 min)

**Understanding the Trade-off**:

```
Throughput: Total tokens processed per second (system metric)
Latency: Time per request (user metric)

These compete:
  - High throughput → Large batches → Higher latency
  - Low latency → Small batches → Lower throughput
```

**Batch Size Impact**:

```
Experiment results (OPT-13B, A100 40GB):

Batch Size | Throughput (tok/s) | Avg Latency (s) | P99 Latency (s)
-----------|--------------------|-----------------|-----------------
    1      |        50          |      0.10       |      0.12
    4      |       180          |      0.15       |      0.20
    16     |       600          |      0.35       |      0.50
    64     |      1800          |      1.20       |      2.00
   256     |      3500          |      4.50       |      8.00

Observation:
  - Throughput scales sub-linearly (diminishing returns)
  - Latency grows linearly (waiting for batch)
  - Sweet spot depends on use case!
```

**Use Case Considerations**:

```
Interactive Chatbot:
  - Priority: Low latency
  - Target: < 100ms time to first token
  - Config: Small batches (1-8), low max_num_batched_tokens

Offline Batch Processing:
  - Priority: High throughput
  - Target: Maximum tokens/second
  - Config: Large batches (128-256), high max_num_batched_tokens

Production API (Mixed):
  - Priority: Balance both
  - Target: Good throughput, reasonable latency
  - Config: Medium batches (32-64), SLA-based scheduling
```

**Configuration Parameters**:

```python
# vllm/engine/arg_utils.py

class EngineArgs:
    # Batch size limits
    max_num_seqs: int = 256
        # Maximum number of sequences in a batch
        # Higher → better throughput, worse latency

    max_num_batched_tokens: int = 2048
        # Maximum tokens in a batch (prefill + decode)
        # Limits GPU memory usage and latency

    max_model_len: int = 2048
        # Maximum sequence length
        # Affects memory allocation

    # Memory management
    gpu_memory_utilization: float = 0.90
        # Fraction of GPU memory for KV cache
        # Higher → more sequences, but less headroom

    # Scheduling
    scheduler_delay_factor: float = 0.0
        # Delay scheduling to batch more prefills
        # Higher → better throughput, higher latency

    # Preemption
    enable_chunked_prefill: bool = False
        # Split large prefills into chunks
        # Better latency for decode, but overhead
```

### Task 5: Batching Metrics (60 min)

**Key Metrics to Track**:

```python
# Metrics to measure batching efficiency

class BatchingMetrics:
    # Throughput metrics
    tokens_per_second: float
        # Total tokens generated / time

    requests_per_second: float
        # Total requests completed / time

    # Latency metrics (per request)
    time_to_first_token: float  # TTFT
        # Time from request arrival to first generated token
        # Key metric for interactive applications

    time_per_output_token: float  # TPOT
        # Time between output tokens
        # Indicates decode performance

    end_to_end_latency: float
        # Total time from request to completion

    # Batch composition
    avg_batch_size: float
        # Average number of sequences in batch

    avg_batched_tokens: float
        # Average tokens processed per step

    prefill_ratio: float
        # Fraction of tokens that are prefill vs decode

    # Utilization
    gpu_utilization: float
        # GPU compute utilization %

    memory_utilization: float
        # KV cache memory used / available

    # Queuing
    avg_queue_time: float
        # Time requests spend in waiting queue

    queue_length: int
        # Current waiting queue size
```

**Profiling with vLLM**:

```python
# Enable built-in profiling

from vllm import LLM, SamplingParams
import time

# Create engine with specific config
llm = LLM(
    model="facebook/opt-125m",
    max_num_seqs=32,
    max_num_batched_tokens=512,
    gpu_memory_utilization=0.8,
)

prompts = [f"Request {i}: Generate text" for i in range(100)]
sampling_params = SamplingParams(temperature=0.8, max_tokens=100)

# Time execution
start = time.time()
outputs = llm.generate(prompts, sampling_params)
elapsed = time.time() - start

# Calculate metrics
total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
throughput = total_tokens / elapsed
avg_latency = elapsed / len(outputs)

print(f"Throughput: {throughput:.2f} tokens/s")
print(f"Avg latency: {avg_latency:.2f} s/request")
print(f"Requests/s: {len(outputs)/elapsed:.2f}")
```

### Task 6: Hands-On Experiments (90 min)

**Exercise 1: Batch Size Sweep**

```python
#!/usr/bin/env python3
"""
Day 5 Exercise 1: Analyze impact of batch size
"""

from vllm import LLM, SamplingParams
import time
import numpy as np

def benchmark_batch_size(max_num_seqs, num_requests=100):
    """Benchmark with specific batch size."""

    print(f"\n{'='*60}")
    print(f"Testing max_num_seqs={max_num_seqs}")
    print(f"{'='*60}")

    llm = LLM(
        model="facebook/opt-125m",
        max_num_seqs=max_num_seqs,
        max_model_len=512,
        gpu_memory_utilization=0.5,
    )

    prompts = [f"Hello world {i}" for i in range(num_requests)]
    sampling_params = SamplingParams(
        temperature=0.8,
        max_tokens=50,
    )

    # Warmup
    _ = llm.generate(prompts[:4], sampling_params)

    # Benchmark
    start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - start

    # Calculate metrics
    total_tokens = sum(
        len(o.prompt_token_ids) + len(o.outputs[0].token_ids)
        for o in outputs
    )
    output_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)

    throughput = output_tokens / elapsed
    avg_latency = elapsed / len(outputs)

    results = {
        'max_num_seqs': max_num_seqs,
        'num_requests': num_requests,
        'elapsed': elapsed,
        'total_tokens': total_tokens,
        'output_tokens': output_tokens,
        'throughput': throughput,
        'avg_latency': avg_latency,
        'requests_per_sec': len(outputs) / elapsed,
    }

    print(f"  Elapsed: {elapsed:.2f} s")
    print(f"  Throughput: {throughput:.2f} tokens/s")
    print(f"  Avg latency: {avg_latency:.3f} s")
    print(f"  Requests/s: {results['requests_per_sec']:.2f}")

    return results

# Run experiments
batch_sizes = [1, 2, 4, 8, 16, 32, 64]
results = []

for bs in batch_sizes:
    result = benchmark_batch_size(bs, num_requests=100)
    results.append(result)

# Summary
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"{'Batch Size':<12} {'Throughput':<15} {'Latency':<15} {'Req/s':<10}")
print("-" * 60)

for r in results:
    print(f"{r['max_num_seqs']:<12} {r['throughput']:<15.2f} {r['avg_latency']:<15.3f} {r['requests_per_sec']:<10.2f}")

# Find sweet spot
throughputs = [r['throughput'] for r in results]
best_idx = np.argmax(throughputs)
print(f"\nBest throughput: batch_size={results[best_idx]['max_num_seqs']}")
```

**Exercise 2: Mixed Workload Analysis**

```python
#!/usr/bin/env python3
"""
Day 5 Exercise 2: Analyze mixed short/long requests
"""

from vllm import LLM, SamplingParams
import time
import random

def mixed_workload_benchmark():
    """Test with mixed request lengths."""

    llm = LLM(
        model="facebook/opt-125m",
        max_num_seqs=32,
        max_model_len=512,
    )

    # Create mixed workload
    # 70% short (10-20 tokens), 30% long (80-100 tokens)
    prompts = []
    expected_lengths = []

    for i in range(100):
        if random.random() < 0.7:
            # Short request
            prompt = "Short: "
            max_tokens = random.randint(10, 20)
            expected_lengths.append(max_tokens)
        else:
            # Long request
            prompt = "Long: "
            max_tokens = random.randint(80, 100)
            expected_lengths.append(max_tokens)

        prompts.append((prompt, max_tokens))

    # Run and track individual latencies
    latencies = []

    for prompt, max_tokens in prompts:
        start = time.time()

        output = llm.generate(
            [prompt],
            SamplingParams(temperature=0.0, max_tokens=max_tokens)
        )

        latency = time.time() - start
        latencies.append(latency)

    # Analyze
    short_latencies = [l for l, e in zip(latencies, expected_lengths) if e < 50]
    long_latencies = [l for l, e in zip(latencies, expected_lengths) if e >= 50]

    print(f"Short requests (avg): {np.mean(short_latencies):.3f} s")
    print(f"Long requests (avg): {np.mean(long_latencies):.3f} s")
    print(f"Short requests (p99): {np.percentile(short_latencies, 99):.3f} s")
    print(f"Long requests (p99): {np.percentile(long_latencies, 99):.3f} s")

    # Check if long requests hurt short ones
    print(f"\nLatency impact of mixing:")
    print(f"  Short request latency increase due to long: {np.mean(short_latencies) / 0.15:.2f}x")

if __name__ == "__main__":
    mixed_workload_benchmark()
```

**Exercise 3: Continuous vs Static Batching**

```python
#!/usr/bin/env python3
"""
Day 5 Exercise 3: Compare continuous vs static batching
"""

import time
from typing import List

def simulate_static_batching(requests: List[int], batch_size: int):
    """
    Simulate static batching.

    Args:
        requests: List of request lengths (in tokens)
        batch_size: Fixed batch size

    Returns:
        Total time, average latency
    """
    total_time = 0
    latencies = []
    request_idx = 0

    while request_idx < len(requests):
        # Form batch
        batch = requests[request_idx:request_idx + batch_size]
        batch_start_time = total_time

        # Process until longest completes
        max_len = max(batch)
        batch_time = max_len  # Each step = 1 time unit

        total_time += batch_time

        # All requests in batch finish at same time
        for req_len in batch:
            latency = total_time - batch_start_time
            latencies.append(latency)

        request_idx += batch_size

    return total_time, latencies

def simulate_continuous_batching(requests: List[int], batch_size: int):
    """
    Simulate continuous batching.

    Returns:
        Total time, average latency
    """
    total_time = 0
    latencies = []
    running = []
    waiting = requests.copy()

    while running or waiting:
        # Remove finished
        running = [r for r in running if r['remaining'] > 0]

        # Add new requests to fill batch
        while len(running) < batch_size and waiting:
            req_len = waiting.pop(0)
            running.append({
                'total': req_len,
                'remaining': req_len,
                'start_time': total_time,
            })

        if running:
            # Process one step
            for req in running:
                req['remaining'] -= 1

            # Check finished
            newly_finished = [r for r in running if r['remaining'] == 0]
            for req in newly_finished:
                latency = total_time + 1 - req['start_time']
                latencies.append(latency)

            total_time += 1

    return total_time, latencies

# Test
requests = [10, 20, 15, 100, 25, 30, 15, 50, 40, 10]  # Various lengths
batch_size = 4

static_time, static_latencies = simulate_static_batching(requests, batch_size)
continuous_time, continuous_latencies = simulate_continuous_batching(requests, batch_size)

print("Static Batching:")
print(f"  Total time: {static_time}")
print(f"  Avg latency: {sum(static_latencies)/len(static_latencies):.2f}")
print(f"  Max latency: {max(static_latencies)}")

print("\nContinuous Batching:")
print(f"  Total time: {continuous_time}")
print(f"  Avg latency: {sum(continuous_latencies)/len(continuous_latencies):.2f}")
print(f"  Max latency: {max(continuous_latencies)}")

print(f"\nImprovement:")
print(f"  Time: {static_time/continuous_time:.2f}x faster")
print(f"  Avg latency: {sum(static_latencies)/sum(continuous_latencies):.2f}x better")
```

---

## 📝 End of Day Summary

### What You Learned Today

✅ **Static Batching Problems**
- Padding waste with variable-length sequences
- Poor GPU utilization when requests finish
- High latency for new requests
- Unpredictable performance

✅ **Continuous Batching**
- Dynamic batch composition
- Add/remove requests on the fly
- Better utilization and throughput
- Lower average latency

✅ **vLLM Scheduler**
- Three queues: waiting, running, swapped
- Memory-aware scheduling
- Preemption strategies
- Mixed prefill/decode batching

✅ **Performance Analysis**
- Throughput vs latency trade-offs
- Batch size impact
- Configuration parameters
- Metrics and profiling

### Knowledge Check (Quiz)

**Question 1**: What is the main advantage of continuous batching over static batching?
<details>
<summary>Answer</summary>
Continuous batching removes finished requests and adds new ones dynamically, avoiding idle GPU time. This improves:
- Throughput (2-3x better)
- Average latency (3-5x better)
- GPU utilization (60% → 90%)
Static batching must wait for entire batch to finish.
</details>

**Question 2**: How does vLLM handle the case when a new request arrives but the batch is full?
<details>
<summary>Answer</summary>
The request goes into the waiting queue. The scheduler checks every step if:
1. Any running requests finished (frees space)
2. Batch size < max_num_seqs
3. Enough GPU memory available
If all conditions met, it moves from waiting → running.
</details>

**Question 3**: What happens when GPU memory is full and a running request needs another block?
<details>
<summary>Answer</summary>
The scheduler preempts (evicts) a running request using one of two strategies:
1. **Swap out**: Move KV cache to CPU memory (can swap back later)
2. **Recompute**: Free all memory, move back to waiting (cheaper if sequence is short)
Victim selection is policy-based (e.g., lowest priority).
</details>

**Question 4**: Why might large batches have worse latency despite better throughput?
<details>
<summary>Answer</summary>
Large batches:
- Take longer to form (waiting for enough requests)
- Process more tokens per step (takes more time)
- New requests wait longer for batch to have space
Example: batch_size=256 might wait to collect 256 requests before starting, adding seconds of queuing delay.
</details>

**Question 5**: What is "chunked prefill" and when is it useful?
<details>
<summary>Answer</summary>
Chunked prefill splits large prompts into smaller chunks processed over multiple steps instead of all at once. Benefits:
- Avoids exceeding max_num_batched_tokens
- Reduces latency spike for decode requests
- Better fairness between prefill and decode
Trade-off: Slightly lower throughput due to overhead.
</details>

### Daily Reflection

**What went well?**
- [ ] Understood continuous batching benefits
- [ ] Analyzed scheduler implementation
- [ ] Ran performance experiments

**What was challenging?**
- [ ] Complex scheduling logic
- [ ] Trade-off analysis
- [ ] Preemption strategies

**Questions for tomorrow**:
1. _______________________________________
2. _______________________________________
3. _______________________________________

---

## 🚀 Preview: Day 6

Tomorrow's deep dive:
- **KV Cache Management**: Block manager internals
- **Memory Optimization**: Techniques to maximize utilization
- **Block Allocation Strategies**: Different approaches
- **Hands-On**: Calculate memory requirements, optimize configurations

**Preparation**:
- Review PagedAttention from Day 4
- Understand block table concept
- Think about memory fragmentation

---

## 📚 Additional Resources

**Papers & Blogs**:
- [ ] [Orca: Continuous Batching](https://arxiv.org/abs/2210.00659)
- [ ] [Continuous Batching for LLM Inference (Anyscale Blog)](https://www.anyscale.com/blog/continuous-batching-llm-inference)
- [ ] vLLM blog posts on scheduling

**Code Reading**:
- [ ] `vllm/core/scheduler.py` (complete file)
- [ ] `vllm/core/policy.py` (scheduling policies)
- [ ] `vllm/engine/metrics.py` (performance metrics)

**Optional**:
- [ ] Compare with TensorRT-LLM's batching strategy
- [ ] Read about NVIDIA Triton's dynamic batching
- [ ] Explore different scheduling policies (Priority, SJF)

---

**Congratulations on mastering continuous batching! 🎉**

**You now understand how vLLM achieves high throughput!**

---

*Completed: ___/___/___*
*Time spent: _____ hours*
*Confidence level (1-10): _____*
