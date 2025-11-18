# Tutorial 03: Model Executor Architecture

## Learning Objectives

1. Understand the role of the Model Executor in vLLM's architecture
2. Learn how Executors manage workers across single and multiple GPUs
3. Explore the execution flow from scheduler to model inference
4. Master distributed execution strategies (Ray, multiprocess, uniprocess)
5. Debug and optimize executor performance

## Overview

The Model Executor is the execution engine in vLLM that coordinates model inference across one or more GPUs. It bridges the high-level scheduling decisions with low-level GPU operations, managing workers, distributing work, and collecting results.

## Executor Architecture

### High-Level Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    vLLM Engine                            │
│                                                           │
│  ┌─────────────┐      ┌─────────────────────────────┐   │
│  │  Scheduler  │─────▶│  Model Executor (Abstract)  │   │
│  └─────────────┘      └────────────┬────────────────┘   │
│                                    │                     │
│                     ┌──────────────┼──────────────┐      │
│                     │              │              │      │
│                     ▼              ▼              ▼      │
│         ┌──────────────┐ ┌─────────────┐ ┌─────────┐    │
│         │   Ray        │ │ Multiproc   │ │Uniproc  │    │
│         │  Executor    │ │  Executor   │ │Executor │    │
│         └──────┬───────┘ └──────┬──────┘ └────┬────┘    │
└────────────────┼────────────────┼─────────────┼─────────┘
                 │                │             │
                 ▼                ▼             ▼
         ┌───────────────┐┌──────────┐  ┌──────────┐
         │  Worker 0     ││ Worker 0 │  │ Worker 0 │
         │  Worker 1     ││ Worker 1 │  │          │
         │  Worker 2     ││          │  │  (Single)│
         │  Worker N     ││(2 GPUs)  │  │          │
         │  (Ray Cluster)││          │  └──────────┘
         └───────────────┘└──────────┘
```

### Executor Hierarchy

```
Executor (Abstract Base Class)
    │
    ├── UniProcExecutor (Single GPU, in-process)
    │
    ├── MultiprocExecutor (Multi-GPU, multiprocessing)
    │
    └── RayDistributedExecutor (Multi-GPU/Multi-node, Ray)
```

## Core Components

### 1. Abstract Executor Base Class

**File**: `/vllm/v1/executor/abstract.py` (lines 35-100)

```python
class Executor(ABC):
    """
    Abstract base class for vLLM executors.

    An executor is responsible for executing the model on one device,
    or distributed across multiple devices.
    """

    uses_ray: bool = False
    supports_pp: bool = False  # Pipeline parallelism support

    def __init__(self, vllm_config: VllmConfig) -> None:
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.device_config = vllm_config.device_config

    @abstractmethod
    def initialize(self, num_gpu_blocks: int) -> None:
        """Initialize the executor with KV cache blocks"""
        pass

    @abstractmethod
    def execute_model(
        self,
        scheduler_output: SchedulerOutput
    ) -> ModelRunnerOutput:
        """Execute model inference for scheduled requests"""
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Clean up resources"""
        pass

    @staticmethod
    def get_class(vllm_config: VllmConfig) -> type["Executor"]:
        """Factory method to get appropriate executor class"""

        parallel_config = vllm_config.parallel_config
        backend = parallel_config.distributed_executor_backend

        if backend == "ray":
            from vllm.v1.executor.ray_executor import RayDistributedExecutor
            return RayDistributedExecutor
        elif backend == "mp":
            from vllm.v1.executor.multiproc_executor import MultiprocExecutor
            return MultiprocExecutor
        elif backend == "uni":
            from vllm.v1.executor.uniproc_executor import UniProcExecutor
            return UniProcExecutor
        else:
            raise ValueError(f"Unknown backend: {backend}")
```

### 2. Worker Base Class

**File**: `/vllm/v1/worker/worker_base.py`

```python
class WorkerBase:
    """
    Base class for workers that run on individual GPUs.
    Workers execute the actual model inference.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str | None = None
    ):
        self.vllm_config = vllm_config
        self.local_rank = local_rank
        self.rank = rank
        self.device = f"cuda:{local_rank}"

        # Initialize model and cache
        self.model_runner = None
        self.kv_cache = None

    def initialize(self, num_gpu_blocks: int) -> None:
        """Initialize worker: load model, allocate cache"""

        # Set device
        torch.cuda.set_device(self.device)

        # Load model
        self.model_runner = ModelRunner(self.vllm_config)
        self.model_runner.load_model()

        # Allocate KV cache
        self.kv_cache = self._allocate_kv_cache(num_gpu_blocks)

    def execute_model(
        self,
        scheduler_output: SchedulerOutput
    ) -> ModelRunnerOutput:
        """Execute model forward pass"""

        return self.model_runner.execute_model(
            scheduler_output=scheduler_output,
            kv_cache=self.kv_cache
        )
```

## Executor Types

### 1. UniProcExecutor (Single GPU)

**File**: `/vllm/v1/executor/uniproc_executor.py`

Simplest executor for single-GPU scenarios:

```python
class UniProcExecutor(Executor):
    """
    Single-process, single-GPU executor.
    Worker runs in the same process as the engine.
    """

    def __init__(self, vllm_config: VllmConfig) -> None:
        super().__init__(vllm_config)

        # Create worker directly
        self.worker = WorkerBase(
            vllm_config=vllm_config,
            local_rank=0,
            rank=0
        )

    def initialize(self, num_gpu_blocks: int) -> None:
        """Initialize the single worker"""
        self.worker.initialize(num_gpu_blocks)

    def execute_model(
        self,
        scheduler_output: SchedulerOutput
    ) -> ModelRunnerOutput:
        """Execute directly on the worker"""
        return self.worker.execute_model(scheduler_output)

    def shutdown(self) -> None:
        """Cleanup worker resources"""
        del self.worker
        torch.cuda.empty_cache()
```

**Use Case**: Development, testing, or single-GPU deployment

**Pros**:
- Simplest setup
- No communication overhead
- Easy debugging

**Cons**:
- Limited to single GPU
- No scalability

### 2. MultiprocExecutor (Multi-GPU, Single Node)

**File**: `/vllm/v1/executor/multiproc_executor.py`

Uses Python multiprocessing for multiple GPUs:

```python
class MultiprocExecutor(Executor):
    """
    Multi-process executor for multiple GPUs on single node.
    Uses torch.multiprocessing for worker processes.
    """

    def __init__(self, vllm_config: VllmConfig) -> None:
        super().__init__(vllm_config)

        self.num_workers = parallel_config.tensor_parallel_size
        self.workers = []
        self.result_queue = mp.Queue()

    def _init_workers(self) -> None:
        """Spawn worker processes"""

        mp.set_start_method('spawn', force=True)

        for rank in range(self.num_workers):
            worker_process = mp.Process(
                target=self._worker_main,
                args=(rank,)
            )
            worker_process.start()
            self.workers.append(worker_process)

    def _worker_main(self, rank: int) -> None:
        """Main function for worker process"""

        # Create worker
        worker = WorkerBase(
            vllm_config=self.vllm_config,
            local_rank=rank,
            rank=rank
        )

        # Initialize
        worker.initialize(self.num_gpu_blocks)

        # Worker loop
        while True:
            # Receive work from main process
            scheduler_output = self._receive_work(rank)

            if scheduler_output is None:
                break  # Shutdown signal

            # Execute
            result = worker.execute_model(scheduler_output)

            # Send result back
            self._send_result(rank, result)

    def execute_model(
        self,
        scheduler_output: SchedulerOutput
    ) -> ModelRunnerOutput:
        """Distribute work to workers and collect results"""

        # Broadcast scheduler output to all workers
        for rank in range(self.num_workers):
            self._send_work(rank, scheduler_output)

        # Collect results from all workers
        results = []
        for rank in range(self.num_workers):
            result = self._receive_result(rank)
            results.append(result)

        # Aggregate results (for tensor parallel)
        return self._aggregate_results(results)
```

**Use Case**: Multi-GPU on single node (2-8 GPUs typically)

**Pros**:
- Good performance for single node
- Simple deployment
- Low latency (shared memory)

**Cons**:
- Limited to single node
- Process startup overhead

### 3. RayDistributedExecutor (Multi-GPU, Multi-Node)

**File**: `/vllm/v1/executor/ray_executor.py`

Uses Ray for distributed execution across nodes:

```python
class RayDistributedExecutor(Executor):
    """
    Distributed executor using Ray for multi-node setups.
    Can scale across many GPUs and nodes.
    """

    uses_ray = True

    def __init__(self, vllm_config: VllmConfig) -> None:
        super().__init__(vllm_config)

        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()

        # Create remote workers
        self.workers = self._create_ray_workers()

    def _create_ray_workers(self) -> list:
        """Create Ray actor workers"""

        worker_class = ray.remote(
            num_gpus=1,
            num_cpus=1
        )(WorkerBase)

        workers = []
        for rank in range(self.parallel_config.world_size):
            worker = worker_class.remote(
                vllm_config=self.vllm_config,
                local_rank=rank % self.parallel_config.tensor_parallel_size,
                rank=rank
            )
            workers.append(worker)

        return workers

    def execute_model(
        self,
        scheduler_output: SchedulerOutput
    ) -> ModelRunnerOutput:
        """Execute across Ray workers"""

        # Broadcast to all workers (async)
        futures = [
            worker.execute_model.remote(scheduler_output)
            for worker in self.workers
        ]

        # Wait for all to complete
        results = ray.get(futures)

        # Aggregate results
        return self._aggregate_results(results)
```

**Use Case**: Large-scale deployments, multi-node clusters

**Pros**:
- Scales to many nodes
- Fault tolerance
- Resource management
- Flexible scheduling

**Cons**:
- Higher latency (network)
- More complex setup
- Ray dependency

## Execution Flow

### Detailed Execution Sequence

```
┌──────────────────────────────────────────────────────────────┐
│                     Execution Flow                            │
└──────────────────────────────────────────────────────────────┘

1. Scheduler produces SchedulerOutput
   │
   │  ┌─────────────────────────────────┐
   │  │ SchedulerOutput:                │
   │  │  - scheduled_requests           │
   │  │  - kv_cache_blocks             │
   │  │  - num_tokens                   │
   │  └─────────────────────────────────┘
   │
   ▼
2. Engine calls executor.execute_model(scheduler_output)
   │
   ▼
3. Executor distributes to workers
   │
   ├─────────┬─────────┬─────────┐
   ▼         ▼         ▼         ▼
Worker 0  Worker 1  Worker 2  Worker 3
   │         │         │         │
   │  [Parallel Model Execution] │
   │         │         │         │
   ├─────────┴─────────┴─────────┤
   │                             │
   ▼                             ▼
4. Each worker executes model_runner.execute_model()
   │
   ├─── Load KV cache blocks
   ├─── Prepare input tensors
   ├─── Run forward pass
   ├─── Sample next tokens
   └─── Return outputs
   │
   ▼
5. Executor aggregates results
   │
   │  [Tensor Parallel: All-reduce]
   │  [Pipeline Parallel: Collect]
   │
   ▼
6. Return ModelRunnerOutput to engine
   │
   └─────▶ Engine processes outputs
```

### Code Walkthrough: Full Execution

```python
# In LLMEngine
def step(self) -> list[RequestOutput]:
    """Main execution step"""

    # 1. Scheduler decides what to run
    scheduler_output = self.scheduler.schedule()

    # 2. Execute via executor
    model_output = self.executor.execute_model(scheduler_output)

    # 3. Process outputs
    request_outputs = self._process_model_outputs(
        scheduler_output,
        model_output
    )

    return request_outputs

# In Executor
def execute_model(
    self,
    scheduler_output: SchedulerOutput
) -> ModelRunnerOutput:
    """Execute model across workers"""

    # Distribute scheduler output to workers
    outputs = self._run_workers(
        "execute_model",
        scheduler_output=scheduler_output
    )

    # Aggregate outputs (for tensor parallelism)
    if self.parallel_config.tensor_parallel_size > 1:
        return self._aggregate_tensor_parallel(outputs)
    else:
        return outputs[0]

# In Worker
def execute_model(
    self,
    scheduler_output: SchedulerOutput
) -> ModelRunnerOutput:
    """Execute on this worker's GPU"""

    # Prepare inputs from scheduler output
    model_input = self._prepare_model_input(scheduler_output)

    # Run model forward pass
    hidden_states = self.model_runner.execute_model(
        input_ids=model_input.input_ids,
        positions=model_input.positions,
        kv_caches=self.kv_cache,
        attn_metadata=model_input.attn_metadata
    )

    # Sample next tokens
    sampler_output = self.sampler(
        hidden_states,
        sampling_metadata=scheduler_output.sampling_metadata
    )

    return ModelRunnerOutput(
        sampled_tokens=sampler_output.sampled_tokens,
        logprobs=sampler_output.logprobs
    )
```

## Parallelism Strategies

### Tensor Parallelism

Splits model weights across GPUs:

```
Model Layer (e.g., Linear projection):
┌──────────────────────────────────┐
│    Weight Matrix (4096 x 4096)   │
└──────────────────────────────────┘
            │
            │ Split along hidden dim
            ▼
┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
│ GPU 0   │ │ GPU 1   │ │ GPU 2   │ │ GPU 3   │
│(1024 x  │ │(1024 x  │ │(1024 x  │ │(1024 x  │
│ 4096)   │ │ 4096)   │ │ 4096)   │ │ 4096)   │
└─────────┘ └─────────┘ └─────────┘ └─────────┘
     │           │           │           │
     └───────────┴───────────┴───────────┘
                  │
            All-Reduce/Gather
                  │
                  ▼
           Combined Output
```

**Implementation**:

```python
class TensorParallelLinear(nn.Module):
    """Linear layer split across tensor parallel GPUs"""

    def __init__(self, in_features, out_features, tp_size):
        super().__init__()

        # Each GPU gets a slice
        self.out_features_per_gpu = out_features // tp_size

        self.weight = nn.Parameter(
            torch.empty(self.out_features_per_gpu, in_features)
        )

    def forward(self, x):
        # Local computation
        local_output = F.linear(x, self.weight)

        # All-gather across GPUs
        output = tensor_parallel.all_gather(local_output)

        return output
```

### Pipeline Parallelism

Splits model layers across GPUs:

```
Input
  │
  ▼
┌─────────────┐
│   GPU 0     │  Layers 0-7
│ (Embedding  │
│  + Layer 0-7)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   GPU 1     │  Layers 8-15
│ (Layer 8-15)│
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   GPU 2     │  Layers 16-23
│(Layer 16-23)│
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   GPU 3     │  Layers 24-31 + LM Head
│(Layer 24-31)│
└──────┬──────┘
       │
       ▼
    Output
```

**Pros**: Better memory efficiency for very large models
**Cons**: Bubbles in pipeline, more complex

## Hands-On Exercises

### Exercise 1: Trace Execution Flow

**Objective**: Follow a request through the executor

```python
class TracingExecutor(UniProcExecutor):
    """Executor with detailed tracing"""

    def execute_model(self, scheduler_output):
        print("\n=== Executor.execute_model() ===")
        print(f"Requests: {len(scheduler_output.scheduled_new_reqs)}")
        print(f"Tokens: {scheduler_output.num_scheduled_tokens}")

        start = time.time()

        # Call worker
        print("\nCalling worker.execute_model()...")
        result = self.worker.execute_model(scheduler_output)

        elapsed = time.time() - start
        print(f"\nExecution completed in {elapsed*1000:.2f}ms")
        print(f"Sampled tokens shape: {result.sampled_tokens.shape}")

        return result
```

**Task**: Run with this tracing executor and observe the flow.

### Exercise 2: Benchmark Executor Backends

**Objective**: Compare performance of different executors

```python
import time

def benchmark_executor(executor_type: str, num_requests: int):
    """Benchmark executor performance"""

    # Create executor
    if executor_type == "uniproc":
        executor = UniProcExecutor(vllm_config)
    elif executor_type == "multiproc":
        executor = MultiprocExecutor(vllm_config)
    elif executor_type == "ray":
        executor = RayDistributedExecutor(vllm_config)

    # Initialize
    executor.initialize(num_gpu_blocks=1000)

    # Create dummy scheduler outputs
    scheduler_outputs = [
        create_dummy_scheduler_output(num_requests=num_requests)
        for _ in range(100)
    ]

    # Warmup
    for _ in range(10):
        executor.execute_model(scheduler_outputs[0])

    # Benchmark
    start = time.time()
    for scheduler_output in scheduler_outputs:
        executor.execute_model(scheduler_output)
    elapsed = time.time() - start

    throughput = (100 * num_requests) / elapsed
    latency = (elapsed / 100) * 1000

    print(f"{executor_type}:")
    print(f"  Throughput: {throughput:.1f} req/s")
    print(f"  Latency: {latency:.2f} ms/step")

# Run benchmark
for executor_type in ["uniproc", "multiproc", "ray"]:
    benchmark_executor(executor_type, num_requests=32)
```

**Task**: Run on different GPU configurations and compare.

### Exercise 3: Implement Custom Executor

**Objective**: Create a custom executor with load balancing

```python
class LoadBalancedExecutor(Executor):
    """
    Custom executor that balances load across workers
    based on their current utilization.
    """

    def __init__(self, vllm_config: VllmConfig):
        super().__init__(vllm_config)

        self.workers = self._create_workers()
        self.worker_loads = [0.0] * len(self.workers)

    def execute_model(self, scheduler_output):
        """Execute with load balancing"""

        # Split requests by worker load
        assignments = self._assign_requests_to_workers(
            scheduler_output.scheduled_new_reqs
        )

        # Execute on each worker
        futures = []
        for worker_id, requests in assignments.items():
            # Create scheduler output for this worker
            worker_output = self._create_scheduler_output(requests)

            # Submit to worker
            future = self.workers[worker_id].execute_model.remote(worker_output)
            futures.append((worker_id, future))

        # Collect results
        results = []
        for worker_id, future in futures:
            result = ray.get(future)
            results.append(result)

            # Update load estimate
            self.worker_loads[worker_id] = result.execution_time

        return self._merge_results(results)

    def _assign_requests_to_workers(self, requests):
        """Assign requests to workers based on load"""

        assignments = {i: [] for i in range(len(self.workers))}

        # Sort requests by estimated compute (num_tokens)
        sorted_requests = sorted(
            requests,
            key=lambda r: r.num_tokens,
            reverse=True
        )

        # Greedy assignment to least loaded worker
        for req in sorted_requests:
            # Find least loaded worker
            min_load_worker = min(
                range(len(self.workers)),
                key=lambda i: self.worker_loads[i]
            )

            # Assign request
            assignments[min_load_worker].append(req)

            # Update estimated load
            self.worker_loads[min_load_worker] += req.num_tokens

        return assignments
```

**Task**: Implement and test this custom executor.

## Common Pitfalls and Solutions

### Pitfall 1: Worker Initialization Failures

**Problem**: Workers fail to initialize due to CUDA context issues.

```python
# BAD: CUDA context created before fork
torch.cuda.init()  # ❌ Don't do this before spawning workers
executor = MultiprocExecutor(vllm_config)
```

**Solution**: Initialize CUDA in worker processes only:

```python
# GOOD: Initialize CUDA in worker
class WorkerBase:
    def initialize(self, num_gpu_blocks):
        # Set device FIRST
        torch.cuda.set_device(self.device)

        # Then initialize CUDA context
        torch.cuda.init()

        # Load model
        self.model_runner.load_model()
```

### Pitfall 2: Inefficient Result Aggregation

**Problem**: Naive aggregation of large tensors is slow.

```python
# BAD: Concatenate large tensors
results = []
for worker in workers:
    results.append(worker.get_output())

# Slow for large outputs
combined = torch.cat(results, dim=0)
```

**Solution**: Use in-place operations and pre-allocated buffers:

```python
# GOOD: Pre-allocate output buffer
def aggregate_results(workers):
    # Determine output shape
    total_size = sum(w.output_size for w in workers)

    # Pre-allocate
    output = torch.empty(
        total_size,
        dtype=torch.float16,
        device='cuda'
    )

    # Fill in-place
    offset = 0
    for worker in workers:
        size = worker.output_size
        output[offset:offset+size] = worker.get_output()
        offset += size

    return output
```

### Pitfall 3: Deadlocks in Distributed Execution

**Problem**: Workers waiting for each other in collective operations.

```python
# BAD: Not all workers participate in all-reduce
if rank == 0:
    result = torch.distributed.all_reduce(tensor)  # ❌ Deadlock!
```

**Solution**: Ensure all workers participate:

```python
# GOOD: All workers participate
result = torch.distributed.all_reduce(tensor)  # ✓ Works
```

## Debugging Executor Issues

### Enable Detailed Logging

```python
import logging

# Configure executor logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("vllm.executor")

class DebuggableExecutor(Executor):
    def execute_model(self, scheduler_output):
        logger.debug(f"Executing {len(scheduler_output.scheduled_new_reqs)} requests")

        start = time.time()
        result = super().execute_model(scheduler_output)
        elapsed = time.time() - start

        logger.debug(f"Execution took {elapsed*1000:.2f}ms")
        return result
```

### Monitor Worker Health

```python
class HealthMonitoredExecutor(RayDistributedExecutor):
    """Executor with worker health monitoring"""

    def __init__(self, vllm_config):
        super().__init__(vllm_config)
        self.worker_health = [True] * len(self.workers)

    def execute_model(self, scheduler_output):
        # Check worker health before execution
        self._check_worker_health()

        try:
            return super().execute_model(scheduler_output)
        except Exception as e:
            # Mark failed workers
            self._handle_worker_failure(e)
            raise

    def _check_worker_health(self):
        """Ping all workers"""
        for i, worker in enumerate(self.workers):
            try:
                ray.get(worker.ping.remote(), timeout=1.0)
                self.worker_health[i] = True
            except:
                self.worker_health[i] = False
                logger.warning(f"Worker {i} is unhealthy")

    def _handle_worker_failure(self, error):
        """Handle worker failures"""
        logger.error(f"Worker failure: {error}")

        # Attempt to restart failed workers
        for i, healthy in enumerate(self.worker_health):
            if not healthy:
                logger.info(f"Restarting worker {i}")
                self.workers[i] = self._create_worker(i)
```

### Visualize Execution Timeline

```python
import matplotlib.pyplot as plt

class ProfilingExecutor(Executor):
    """Executor that profiles execution"""

    def __init__(self, vllm_config):
        super().__init__(vllm_config)
        self.timeline = []

    def execute_model(self, scheduler_output):
        start = time.time()

        # Record start
        self.timeline.append({
            'event': 'execute_start',
            'time': start,
            'num_requests': len(scheduler_output.scheduled_new_reqs)
        })

        result = super().execute_model(scheduler_output)

        # Record end
        end = time.time()
        self.timeline.append({
            'event': 'execute_end',
            'time': end,
            'duration': end - start
        })

        return result

    def plot_timeline(self):
        """Visualize execution timeline"""

        fig, ax = plt.subplots(figsize=(12, 6))

        for i in range(0, len(self.timeline), 2):
            start_event = self.timeline[i]
            end_event = self.timeline[i+1]

            start_time = start_event['time'] - self.timeline[0]['time']
            duration = end_event['duration']

            ax.barh(0, duration, left=start_time, height=0.5)

        ax.set_xlabel('Time (seconds)')
        ax.set_title('Execution Timeline')
        plt.tight_layout()
        plt.show()
```

## Performance Optimization

### 1. Batch Size Optimization

```python
def find_optimal_batch_size(executor, model_config):
    """Find batch size that maximizes throughput"""

    batch_sizes = [8, 16, 32, 64, 128, 256]
    best_throughput = 0
    best_batch_size = None

    for batch_size in batch_sizes:
        try:
            # Create test scheduler output
            scheduler_output = create_test_output(batch_size)

            # Warmup
            for _ in range(5):
                executor.execute_model(scheduler_output)

            # Benchmark
            start = time.time()
            for _ in range(100):
                executor.execute_model(scheduler_output)
            elapsed = time.time() - start

            throughput = (100 * batch_size) / elapsed

            if throughput > best_throughput:
                best_throughput = throughput
                best_batch_size = batch_size

        except torch.cuda.OutOfMemoryError:
            break  # Batch size too large

    return best_batch_size, best_throughput
```

### 2. Pipeline Overlapping

```python
class PipelinedExecutor(Executor):
    """Executor with overlapped communication and computation"""

    def execute_model(self, scheduler_output):
        # Start communication (async)
        comm_future = self._broadcast_inputs_async(scheduler_output)

        # Prepare local data while communication happening
        local_data = self._prepare_local_data(scheduler_output)

        # Wait for communication
        comm_future.wait()

        # Execute model
        result = self._execute_local(local_data)

        # Start result communication (async)
        result_future = self._gather_results_async(result)

        # Do local post-processing while gathering
        processed = self._postprocess_local(result)

        # Wait for full results
        full_results = result_future.wait()

        return full_results
```

### 3. Worker Reuse

```python
class WorkerPoolExecutor(Executor):
    """Executor with worker pooling for efficiency"""

    def __init__(self, vllm_config):
        super().__init__(vllm_config)

        # Pre-create worker pool
        self.worker_pool = self._create_worker_pool()

        # Keep workers warm
        self._warmup_workers()

    def _warmup_workers(self):
        """Warmup workers to avoid cold start"""

        dummy_output = create_dummy_scheduler_output(batch_size=1)

        for worker in self.worker_pool:
            worker.execute_model(dummy_output)

    def shutdown(self):
        """Graceful shutdown"""

        # Send shutdown signal to all workers
        for worker in self.worker_pool:
            worker.shutdown()

        # Wait for workers to finish
        for worker in self.worker_pool:
            worker.join(timeout=5.0)
```

## Advanced Topics

### Fault Tolerance

```python
class FaultTolerantExecutor(RayDistributedExecutor):
    """Executor with fault tolerance and recovery"""

    def __init__(self, vllm_config):
        super().__init__(vllm_config)
        self.backup_workers = self._create_backup_workers()

    def execute_model(self, scheduler_output):
        try:
            return super().execute_model(scheduler_output)
        except Exception as e:
            logger.error(f"Execution failed: {e}")

            # Attempt recovery
            return self._execute_with_recovery(scheduler_output)

    def _execute_with_recovery(self, scheduler_output):
        """Execute with automatic recovery"""

        # Identify failed workers
        failed_workers = self._identify_failed_workers()

        # Replace with backups
        for worker_id in failed_workers:
            self.workers[worker_id] = self.backup_workers.pop()

        # Retry execution
        return super().execute_model(scheduler_output)
```

### Dynamic Scaling

```python
class AutoScalingExecutor(Executor):
    """Executor that auto-scales workers based on load"""

    def __init__(self, vllm_config):
        super().__init__(vllm_config)

        self.min_workers = 1
        self.max_workers = 8
        self.workers = self._create_workers(self.min_workers)

    def execute_model(self, scheduler_output):
        # Check if we need more workers
        if self._should_scale_up(scheduler_output):
            self._add_workers(1)

        # Check if we can reduce workers
        elif self._should_scale_down(scheduler_output):
            self._remove_workers(1)

        return super().execute_model(scheduler_output)

    def _should_scale_up(self, scheduler_output):
        # Scale up if queue is backing up
        queue_size = len(scheduler_output.scheduled_new_reqs)
        return queue_size > self.num_workers * 10

    def _should_scale_down(self, scheduler_output):
        # Scale down if underutilized
        queue_size = len(scheduler_output.scheduled_new_reqs)
        return queue_size < self.num_workers * 2 and self.num_workers > self.min_workers
```

## References

### Source Code Files

- **Abstract Executor**: `/vllm/v1/executor/abstract.py`
- **UniProc Executor**: `/vllm/v1/executor/uniproc_executor.py`
- **Multiproc Executor**: `/vllm/v1/executor/multiproc_executor.py`
- **Ray Executor**: `/vllm/v1/executor/ray_executor.py`
- **Worker Base**: `/vllm/v1/worker/worker_base.py`

### Configuration

```python
@dataclass
class ParallelConfig:
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    distributed_executor_backend: str = "ray"  # or "mp", "uni"
```

### Related Documentation

- Tutorial 01: Scheduler Deep Dive
- Tutorial 04: Attention Layer Internals
- Module 5: Distributed Inference Patterns

## Summary

In this tutorial, you learned:

- The role of Model Executor in coordinating inference
- Different executor types (UniProc, Multiproc, Ray) and their trade-offs
- Execution flow from scheduler to model inference
- Parallelism strategies (tensor parallel, pipeline parallel)
- Debugging techniques and performance optimization
- Advanced topics like fault tolerance and auto-scaling

The Model Executor is the bridge between high-level scheduling and low-level GPU execution. Understanding its architecture helps you choose the right deployment strategy and optimize performance.

## Next Steps

- **Tutorial 04**: Attention Layer Internals - Dive into attention computation
- **Tutorial 05**: Sampler Implementation - Understand token sampling
- **Module 5**: Distributed Inference - Advanced distributed patterns
