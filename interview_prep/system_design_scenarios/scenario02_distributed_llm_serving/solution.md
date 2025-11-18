# Scenario 02: Distributed LLM Serving - Solution

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Load Balancer                            │
│                    (Request Distribution)                        │
└────────┬────────────────────────────────────────┬───────────────┘
         │                                        │
         │                                        │
    ┌────▼─────────┐                        ┌────▼─────────┐
    │   Replica 1  │                        │   Replica 2  │
    │   (8 GPUs)   │                        │   (8 GPUs)   │
    └────┬─────────┘                        └────┬─────────┘
         │                                        │
┌────────▼──────────────────────────────────────────────────────┐
│            GPU Cluster (Tensor + Pipeline Parallel)           │
│                                                                │
│  ┌──────────────────────────────────────────────────────┐    │
│  │              Pipeline Stage 1 (Layers 1-32)          │    │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐ │    │
│  │  │ GPU 0   │  │ GPU 1   │  │ GPU 2   │  │ GPU 3   │ │    │
│  │  │ TP=4    │←─┤ TP=4    │←─┤ TP=4    │←─┤ TP=4    │ │    │
│  │  └────┬────┘  └─────────┘  └─────────┘  └─────────┘ │    │
│  └───────┼──────────────────────────────────────────────┘    │
│          │ (Pipeline Forward)                                │
│  ┌───────▼──────────────────────────────────────────────┐    │
│  │              Pipeline Stage 2 (Layers 33-64)         │    │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐ │    │
│  │  │ GPU 4   │  │ GPU 5   │  │ GPU 6   │  │ GPU 7   │ │    │
│  │  │ TP=4    │←─┤ TP=4    │←─┤ TP=4    │←─┤ TP=4    │ │    │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘ │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                                │
│  NVLink (600 GB/s) within node                                │
│  InfiniBand (400 Gbps) across nodes                           │
└────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│                   Coordination Layer                          │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────┐  │
│  │   Health    │  │   Request    │  │   Distributed      │  │
│  │   Monitor   │  │   Tracker    │  │   Checkpoint       │  │
│  └─────────────┘  └──────────────┘  └────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

## Parallelism Strategy Decision

### Model Analysis (175B Parameters)

```
Model: GPT-3 scale (175B)
Architecture:
- 96 layers
- 12,288 hidden size
- 96 attention heads
- 128 head dimension
- 49,152 FFN intermediate size

Memory Requirements (FP16):
- Weights: 175B × 2 bytes = 350 GB
- Per-layer: 350 GB / 96 = ~3.6 GB

With INT4 quantization:
- Weights: 175B × 0.5 bytes = 87.5 GB
- Per-layer: ~0.9 GB
```

### Parallelism Options Analysis

#### Option 1: Pure Tensor Parallelism (8-way)

**Configuration:**
- Split each layer across 8 GPUs
- All layers on all GPUs
- Highest communication overhead

**Memory per GPU:**
```
Weights: 350 GB / 8 = 43.75 GB
KV Cache (32 seqs, 2K context): ~8 GB
Activations: ~5 GB
Total: ~57 GB (fits in 80GB A100)
```

**Communication:**
- 2 all-reduce operations per layer
- 96 layers → 192 all-reduce ops
- With NVLink: ~10ms per forward pass
- **Latency:** Good (~50ms per token)
- **Throughput:** Moderate (communication bound)

#### Option 2: Pure Pipeline Parallelism (8-stage)

**Configuration:**
- Split 96 layers into 8 stages (12 layers each)
- Each GPU holds complete 12 layers
- Point-to-point communication

**Memory per GPU:**
```
Weights: 12 layers × 3.6 GB = 43.2 GB
KV Cache: ~8 GB (only for local layers)
Activations: ~10 GB (pipeline buffers)
Total: ~61 GB (fits in 80GB A100)
```

**Communication:**
- Activation tensors between stages
- Hidden size: 12,288 × batch_size × 2 bytes
- 8 stages → 7 transfers per token
- **Latency:** Poor (pipeline bubble)
- **Throughput:** Excellent (micro-batching)

#### Option 3: Hybrid (2-way PP × 4-way TP) - RECOMMENDED

**Configuration:**
- 2 pipeline stages (48 layers each)
- Each stage uses 4-way tensor parallelism
- Best balance of latency and throughput

**Memory per GPU:**
```
Weights: (175 GB / 2) / 4 = 21.875 GB per GPU
KV Cache: ~8 GB
Activations: ~5 GB
Total: ~35 GB (comfortable in 80GB A100)
```

**Communication:**
- Within TP group (4 GPUs): 2 all-reduce per layer (NVLink)
- Between PP stages: 1 transfer per token (NVLink/IB)
- **Latency:** Good (~80ms per token)
- **Throughput:** Good (balanced)

**Why This Works:**
- Minimizes pipeline bubble (only 2 stages)
- Reduces all-reduce frequency (4-way vs 8-way)
- Efficient use of NVLink within node
- Allows expansion to multi-node (IB for PP)

## Detailed Component Design

### 1. Distributed Inference Engine

```python
class DistributedLLMEngine:
    def __init__(self, config):
        self.world_size = 8  # Total GPUs
        self.tp_size = 4     # Tensor parallel
        self.pp_size = 2     # Pipeline parallel
        self.rank = get_rank()

        # Determine which TP group and PP stage
        self.tp_rank = self.rank % self.tp_size
        self.pp_rank = self.rank // self.tp_size

        # Initialize communication groups
        self.setup_communication_groups()

        # Load model shards
        self.model = self.load_model_shard(
            pp_rank=self.pp_rank,
            tp_rank=self.tp_rank
        )

    def setup_communication_groups(self):
        """Create process groups for TP and PP"""
        import torch.distributed as dist

        # Tensor parallel groups (within each PP stage)
        # PP stage 0: [0, 1, 2, 3]
        # PP stage 1: [4, 5, 6, 7]
        self.tp_group = dist.new_group(
            ranks=list(range(
                self.pp_rank * self.tp_size,
                (self.pp_rank + 1) * self.tp_size
            ))
        )

        # Pipeline parallel groups (across stages)
        # [0, 4], [1, 5], [2, 6], [3, 7]
        self.pp_group = dist.new_group(
            ranks=[self.tp_rank + i * self.tp_size
                   for i in range(self.pp_size)]
        )

    def load_model_shard(self, pp_rank, tp_rank):
        """Load specific model shard for this GPU"""
        # Calculate layer range for this PP stage
        layers_per_stage = 96 // self.pp_size
        start_layer = pp_rank * layers_per_stage
        end_layer = (pp_rank + 1) * layers_per_stage

        model = LLMModel(
            start_layer=start_layer,
            end_layer=end_layer,
            tp_rank=tp_rank,
            tp_size=self.tp_size
        )

        # Load weights for this shard
        checkpoint_path = (
            f"model_pp{pp_rank}_tp{tp_rank}.pt"
        )
        model.load_state_dict(
            torch.load(checkpoint_path)
        )

        return model.cuda()

    async def forward_pass(self, input_ids, attention_mask):
        """Execute forward pass with PP and TP"""

        if self.pp_rank == 0:
            # First stage: process input
            hidden_states = self.model.embed(input_ids)
        else:
            # Receive from previous stage
            hidden_states = await self.recv_from_prev_stage()

        # Process through local layers with TP
        for layer in self.model.layers:
            hidden_states = layer(
                hidden_states,
                tp_group=self.tp_group  # All-reduce within TP
            )

        if self.pp_rank == self.pp_size - 1:
            # Last stage: generate output
            logits = self.model.lm_head(hidden_states)
            return logits
        else:
            # Send to next stage
            await self.send_to_next_stage(hidden_states)
            return None

    async def recv_from_prev_stage(self):
        """Receive activation from previous pipeline stage"""
        import torch.distributed as dist

        tensor = torch.empty(
            self.activation_shape,
            device='cuda'
        )

        prev_rank = self.rank - self.tp_size
        dist.recv(tensor, src=prev_rank)

        return tensor

    async def send_to_next_stage(self, tensor):
        """Send activation to next pipeline stage"""
        import torch.distributed as dist

        next_rank = self.rank + self.tp_size
        dist.send(tensor, dst=next_rank)
```

### 2. Tensor Parallel Layer Implementation

```python
class TensorParallelLinear(nn.Module):
    """Column-parallel linear layer with all-reduce"""

    def __init__(self, in_features, out_features, tp_rank, tp_size):
        super().__init__()
        self.tp_rank = tp_rank
        self.tp_size = tp_size

        # Each GPU holds 1/tp_size of output features
        self.out_features_per_gpu = out_features // tp_size

        self.weight = nn.Parameter(
            torch.empty(
                self.out_features_per_gpu,
                in_features
            )
        )

    def forward(self, x, tp_group):
        # Local matmul
        output = F.linear(x, self.weight)

        # All-reduce across TP group
        dist.all_reduce(
            output,
            group=tp_group,
            op=dist.ReduceOp.SUM
        )

        return output


class TensorParallelAttention(nn.Module):
    """Multi-head attention with tensor parallelism"""

    def __init__(self, config, tp_rank, tp_size):
        super().__init__()
        self.tp_rank = tp_rank
        self.tp_size = tp_size

        # Distribute attention heads across GPUs
        self.num_heads = config.num_heads // tp_size
        self.head_dim = config.head_dim

        # QKV projection (column parallel)
        self.qkv = TensorParallelLinear(
            config.hidden_size,
            3 * config.hidden_size // tp_size,
            tp_rank,
            tp_size
        )

        # Output projection (row parallel)
        self.out_proj = TensorParallelLinear(
            config.hidden_size // tp_size,
            config.hidden_size,
            tp_rank,
            tp_size
        )

    def forward(self, hidden_states, kv_cache, tp_group):
        batch_size, seq_len, _ = hidden_states.shape

        # QKV projection (local to each GPU)
        qkv = self.qkv(hidden_states, tp_group)
        q, k, v = qkv.split(self.num_heads * self.head_dim, dim=-1)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Attention computation (local)
        attn_output = self.compute_attention(q, k, v, kv_cache)

        # Output projection (requires all-reduce)
        output = self.out_proj(attn_output, tp_group)

        return output
```

### 3. Pipeline Parallelism with Micro-batching

```python
class PipelineScheduler:
    """1F1B (One Forward One Backward) schedule"""

    def __init__(self, num_microbatches, pp_rank, pp_size):
        self.num_microbatches = num_microbatches
        self.pp_rank = pp_rank
        self.pp_size = pp_size

    async def run_inference(self, batch):
        """Execute inference with micro-batching"""

        # Split batch into micro-batches
        microbatches = self.split_batch(
            batch,
            self.num_microbatches
        )

        outputs = []
        in_flight = []

        for i, microbatch in enumerate(microbatches):
            # Forward pass for this micro-batch
            if self.pp_rank == 0:
                # First stage: start processing
                hidden = await self.process_microbatch(microbatch)
                await self.send_to_next_stage(hidden)

            elif self.pp_rank == self.pp_size - 1:
                # Last stage: wait for input, process, collect output
                hidden = await self.recv_from_prev_stage()
                output = await self.process_microbatch(hidden)
                outputs.append(output)

            else:
                # Middle stage: relay
                hidden = await self.recv_from_prev_stage()
                hidden = await self.process_microbatch(hidden)
                await self.send_to_next_stage(hidden)

        # Combine micro-batch outputs
        if self.pp_rank == self.pp_size - 1:
            return torch.cat(outputs, dim=0)

        return None
```

### 4. Distributed KV Cache Management

```python
class DistributedKVCache:
    """KV cache distributed across TP group"""

    def __init__(self, config, tp_rank, tp_size):
        self.tp_rank = tp_rank
        self.tp_size = tp_size

        # Each GPU stores KV for its attention heads
        self.num_heads = config.num_heads // tp_size
        self.head_dim = config.head_dim

        # Cache storage (paged)
        self.block_size = 16  # tokens per block
        self.num_blocks = 1000

        self.k_cache = torch.empty(
            self.num_blocks,
            self.block_size,
            self.num_heads,
            self.head_dim,
            device='cuda'
        )

        self.v_cache = torch.empty(
            self.num_blocks,
            self.block_size,
            self.num_heads,
            self.head_dim,
            device='cuda'
        )

        # Block allocation table (shared across TP group)
        self.block_allocator = BlockAllocator(self.num_blocks)

    def allocate(self, seq_id, num_tokens):
        """Allocate KV cache blocks for sequence"""
        num_blocks = (num_tokens + self.block_size - 1) // self.block_size
        block_ids = self.block_allocator.allocate(seq_id, num_blocks)
        return block_ids

    def store(self, seq_id, k, v, position):
        """Store K, V at position for sequence"""
        block_ids = self.block_allocator.get_blocks(seq_id)
        block_idx = position // self.block_size
        offset = position % self.block_size

        block_id = block_ids[block_idx]

        self.k_cache[block_id, offset] = k
        self.v_cache[block_id, offset] = v

    def retrieve(self, seq_id, positions):
        """Retrieve K, V for positions"""
        block_ids = self.block_allocator.get_blocks(seq_id)

        # Gather from blocks
        k_list = []
        v_list = []

        for pos in positions:
            block_idx = pos // self.block_size
            offset = pos % self.block_size
            block_id = block_ids[block_idx]

            k_list.append(self.k_cache[block_id, offset])
            v_list.append(self.v_cache[block_id, offset])

        k = torch.stack(k_list)
        v = torch.stack(v_list)

        return k, v
```

## Communication Analysis

### Bandwidth Requirements

**For 2-way PP × 4-way TP with batch_size=16:**

#### Tensor Parallel Communication (NVLink):

```
Per layer, per token:
- QKV all-reduce: 3 × hidden_size × batch_size × 2 bytes
  = 3 × 12,288 × 16 × 2 = 1.18 MB

- Output all-reduce: hidden_size × batch_size × 2 bytes
  = 12,288 × 16 × 2 = 0.39 MB

Total per layer: 1.57 MB
Total per token (48 layers): 75.4 MB

With NVLink (600 GB/s):
Time per token: 75.4 MB / 600 GB/s = 0.126 ms

This is negligible!
```

#### Pipeline Parallel Communication (NVLink/IB):

```
Per token transfer between stages:
- Hidden states: hidden_size × batch_size × 2 bytes
  = 12,288 × 16 × 2 = 0.39 MB

With NVLink (600 GB/s):
Time per token: 0.39 MB / 600 GB/s = 0.65 μs

With InfiniBand (400 Gbps = 50 GB/s):
Time per token: 0.39 MB / 50 GB/s = 7.8 μs

Still negligible!
```

**Conclusion:** Communication is NOT a bottleneck with proper interconnect.

### Latency Breakdown

```
Per token generation (batch_size=16):

1. Computation:
   - 48 layers per stage × 2 stages
   - Each layer: ~0.8 ms (FP16 on A100)
   - Total: 48 × 2 × 0.8 ms = 76.8 ms

2. TP Communication:
   - All-reduce per layer: ~0.1 ms
   - Total: 96 × 0.1 ms = 9.6 ms

3. PP Communication:
   - 1 transfer between stages: ~0.01 ms
   - Negligible

4. Memory Access:
   - KV cache read/write: ~2 ms
   - Activation loading: ~1 ms

Total per token: 76.8 + 9.6 + 2 + 1 = 89.4 ms

For 256 tokens: 89.4 × 256 = 22.9 seconds (just computation)

With continuous batching and parallelism: ~5-8 seconds total
```

## Fault Tolerance Design

### 1. GPU Failure Detection

```python
class HealthMonitor:
    def __init__(self, check_interval=10):
        self.check_interval = check_interval
        self.gpu_health = {}

    async def monitor_gpus(self):
        while True:
            for rank in range(self.world_size):
                health = await self.check_gpu_health(rank)
                self.gpu_health[rank] = health

                if not health.is_healthy:
                    await self.handle_gpu_failure(rank)

            await asyncio.sleep(self.check_interval)

    async def check_gpu_health(self, rank):
        """Check GPU health via heartbeat and memory"""
        try:
            # Heartbeat check
            heartbeat = await self.get_heartbeat(rank)
            if time.time() - heartbeat > 30:
                return HealthStatus(is_healthy=False, reason="heartbeat_timeout")

            # Memory check
            mem_used, mem_total = await self.get_gpu_memory(rank)
            if mem_used > 0.95 * mem_total:
                return HealthStatus(is_healthy=False, reason="oom_risk")

            # CUDA error check
            cuda_errors = await self.get_cuda_errors(rank)
            if cuda_errors:
                return HealthStatus(is_healthy=False, reason="cuda_error")

            return HealthStatus(is_healthy=True)

        except Exception as e:
            return HealthStatus(is_healthy=False, reason=str(e))
```

### 2. Graceful Degradation

**Strategy: Fallback to smaller TP degree**

```python
class FaultTolerantEngine:
    def __init__(self):
        self.current_config = {
            'tp_size': 4,
            'pp_size': 2,
            'num_replicas': 2
        }
        self.fallback_configs = [
            {'tp_size': 4, 'pp_size': 2},  # Normal
            {'tp_size': 2, 'pp_size': 4},  # GPU failure in TP group
            {'tp_size': 1, 'pp_size': 8},  # Multiple failures
        ]

    async def handle_gpu_failure(self, failed_rank):
        """Reconfigure on GPU failure"""

        # Determine which group failed
        tp_group = self.get_tp_group(failed_rank)

        # Can we continue with reduced TP?
        if self.can_reduce_tp(tp_group):
            # Reconfigure to smaller TP group
            new_config = self.get_fallback_config()
            await self.reconfigure(new_config)
        else:
            # Route traffic to healthy replica
            await self.failover_to_replica()

    async def reconfigure(self, new_config):
        """Dynamically reconfigure parallelism"""

        # Save current state
        checkpoint = await self.checkpoint_current_state()

        # Reload model with new parallelism
        await self.reload_model(new_config)

        # Restore state
        await self.restore_from_checkpoint(checkpoint)

        # Resume serving
        self.current_config = new_config
        logger.info(f"Reconfigured to {new_config}")
```

### 3. Request Checkpointing

```python
class RequestCheckpointer:
    """Checkpoint long-running requests"""

    def __init__(self, checkpoint_interval=100):
        self.checkpoint_interval = checkpoint_interval
        self.checkpoints = {}

    async def process_with_checkpointing(self, request):
        """Process request with periodic checkpointing"""

        request_id = request.id
        num_tokens_generated = 0

        # Restore from checkpoint if exists
        if request_id in self.checkpoints:
            state = self.checkpoints[request_id]
            num_tokens_generated = state['num_tokens']
            kv_cache = state['kv_cache']
        else:
            kv_cache = None

        while num_tokens_generated < request.max_tokens:
            try:
                # Generate next token
                token = await self.generate_token(
                    request,
                    kv_cache
                )
                num_tokens_generated += 1

                # Checkpoint periodically
                if num_tokens_generated % self.checkpoint_interval == 0:
                    self.checkpoints[request_id] = {
                        'num_tokens': num_tokens_generated,
                        'kv_cache': kv_cache.clone(),
                        'timestamp': time.time()
                    }

                # Stream to client
                await self.stream_token(request_id, token)

            except GPUFailureException:
                # Checkpoint and retry on another replica
                logger.warning(f"GPU failure, checkpointing {request_id}")
                self.checkpoints[request_id] = {
                    'num_tokens': num_tokens_generated,
                    'kv_cache': kv_cache.clone(),
                    'timestamp': time.time()
                }
                raise

        # Clean up checkpoint
        del self.checkpoints[request_id]
```

## Load Balancing Strategy

### 1. Request Router

```python
class ReplicaLoadBalancer:
    """Route requests to least loaded replica"""

    def __init__(self, replicas):
        self.replicas = replicas
        self.metrics = {
            replica_id: ReplicaMetrics()
            for replica_id in replicas
        }

    def select_replica(self, request):
        """Select replica using multiple factors"""

        scores = {}
        for replica_id in self.replicas:
            metrics = self.metrics[replica_id]

            # Skip unhealthy replicas
            if not metrics.is_healthy:
                continue

            # Scoring factors:
            # 1. Current queue depth (40%)
            queue_score = 1.0 - (metrics.queue_depth / 100)

            # 2. GPU utilization (30%)
            gpu_score = 1.0 - metrics.gpu_utilization

            # 3. P99 latency (30%)
            latency_score = 1.0 - min(metrics.p99_latency / 500, 1.0)

            scores[replica_id] = (
                0.4 * queue_score +
                0.3 * gpu_score +
                0.3 * latency_score
            )

        # Select replica with highest score
        best_replica = max(scores.items(), key=lambda x: x[1])[0]
        return best_replica

    async def route_request(self, request):
        """Route request to selected replica"""

        replica_id = self.select_replica(request)

        # Add to replica's queue
        await self.replicas[replica_id].enqueue(request)

        # Update metrics
        self.metrics[replica_id].queue_depth += 1
```

### 2. Auto-scaling

```python
class ReplicaAutoscaler:
    """Auto-scale number of replicas"""

    def __init__(self, min_replicas=2, max_replicas=8):
        self.min_replicas = min_replicas
        self.max_replicas = max_replicas
        self.current_replicas = min_replicas

    async def scale_decision(self, metrics):
        """Decide whether to scale up or down"""

        avg_queue_depth = np.mean([m.queue_depth for m in metrics])
        avg_gpu_util = np.mean([m.gpu_utilization for m in metrics])
        p99_latency = np.percentile([m.p99_latency for m in metrics], 99)

        # Scale up conditions
        if (avg_queue_depth > 50 or
            avg_gpu_util > 0.85 or
            p99_latency > 400):

            if self.current_replicas < self.max_replicas:
                await self.scale_up()

        # Scale down conditions
        elif (avg_queue_depth < 10 and
              avg_gpu_util < 0.40 and
              p99_latency < 200):

            if self.current_replicas > self.min_replicas:
                await self.scale_down()

    async def scale_up(self):
        """Add new replica"""
        logger.info("Scaling up replicas")

        # Provision new GPU cluster
        new_replica_id = await self.provision_gpu_cluster(
            num_gpus=8,
            tp_size=4,
            pp_size=2
        )

        # Load model shards
        await self.load_model_on_replica(new_replica_id)

        # Add to load balancer
        self.load_balancer.add_replica(new_replica_id)

        self.current_replicas += 1

    async def scale_down(self):
        """Remove replica gracefully"""
        logger.info("Scaling down replicas")

        # Select replica to remove (least loaded)
        replica_to_remove = self.select_replica_to_remove()

        # Drain requests
        await self.drain_replica(replica_to_remove)

        # Remove from load balancer
        self.load_balancer.remove_replica(replica_to_remove)

        # Deallocate resources
        await self.deallocate_gpu_cluster(replica_to_remove)

        self.current_replicas -= 1
```

## Deployment & Operations

### 1. Model Sharding and Distribution

```bash
# Shard model checkpoint for distributed loading
python shard_model.py \
    --model-path models/gpt-175b \
    --tp-size 4 \
    --pp-size 2 \
    --output-dir sharded_models/

# Output structure:
# sharded_models/
# ├── pp0_tp0.pt  (layers 0-47, partition 0)
# ├── pp0_tp1.pt  (layers 0-47, partition 1)
# ├── pp0_tp2.pt  (layers 0-47, partition 2)
# ├── pp0_tp3.pt  (layers 0-47, partition 3)
# ├── pp1_tp0.pt  (layers 48-95, partition 0)
# ├── pp1_tp1.pt  (layers 48-95, partition 1)
# ├── pp1_tp2.pt  (layers 48-95, partition 2)
# └── pp1_tp3.pt  (layers 48-95, partition 3)
```

### 2. Launch Script

```python
# launch_distributed.py
import torch.distributed as dist
import torch.multiprocessing as mp

def init_process(rank, world_size, fn, backend='nccl'):
    """Initialize distributed process"""
    dist.init_process_group(
        backend=backend,
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    fn(rank, world_size)
    dist.destroy_process_group()

def run_worker(rank, world_size):
    """Worker process"""
    # Set CUDA device
    torch.cuda.set_device(rank)

    # Initialize engine
    engine = DistributedLLMEngine(
        model_path=f"sharded_models/pp{rank//4}_tp{rank%4}.pt",
        tp_size=4,
        pp_size=2
    )

    # Start serving
    engine.serve()

if __name__ == '__main__':
    world_size = 8  # 8 GPUs
    mp.spawn(
        init_process,
        args=(world_size, run_worker),
        nprocs=world_size,
        join=True
    )
```

### 3. Monitoring

```yaml
# Prometheus metrics
- name: distributed_llm_metrics
  metrics:
    # Per-GPU metrics
    - gpu_utilization{rank, tp_group, pp_stage}
    - gpu_memory_used{rank}
    - gpu_memory_total{rank}

    # Communication metrics
    - tp_allreduce_latency_ms{layer}
    - pp_transfer_latency_ms{stage}
    - communication_volume_gb{rank}

    # Request metrics
    - request_queue_depth{replica}
    - tokens_per_second{replica}
    - requests_per_second{replica}

    # Fault tolerance
    - gpu_failures_total{rank}
    - replica_failovers_total{replica}
    - checkpoint_operations_total
```

## Cost Analysis

```
Infrastructure (per replica):
- 1× p4d.24xlarge (8× A100 80GB): $32.77/hr
- Reserved instance (1-year): $21.18/hr
- Spot instance: $9.83/hr

For 2 replicas (baseline) + 1 spot (burst):
- 2 reserved: $21.18 × 2 × 730 = $30,923/month
- 1 spot (50% time): $9.83 × 365 = $3,588/month
- Total: $34,511/month

Additional costs:
- Networking (InfiniBand): $1,000/month
- Storage: $500/month
- Monitoring: $500/month
- Total: $36,511/month

Capacity:
- 2 replicas × 50 QPS = 100 QPS baseline
- 3 replicas × 50 QPS = 150 QPS with spot

Cost per 1M tokens:
- 100 QPS × 300 tokens/req × 86400 sec/day × 30 days = 77.8B tokens/month
- Cost: $36,511 / 77,800 = $0.47 per 1M tokens
```

## Trade-offs Summary

| Aspect | TP Only | PP Only | Hybrid (TP+PP) |
|--------|---------|---------|----------------|
| Latency | Best (50ms) | Worst (200ms) | Good (80ms) |
| Throughput | Moderate | Best | Good |
| Communication | High (all-reduce) | Low (p2p) | Balanced |
| Memory | Balanced | Balanced | Flexible |
| Fault Tolerance | Hard (all GPUs needed) | Easier (per stage) | Moderate |
| Scalability | Limited (comm overhead) | Good | Best |
| **Recommendation** | Small models | Batch processing | **General purpose** |

## Alternative Approaches

### Alternative 1: ZeRO-Inference (DeepSpeed)

- Partition model parameters, gradients, optimizer states
- Dynamic parameter all-gather during inference
- Lower memory per GPU, higher communication
- **Use case:** Extreme model sizes (500B+)

### Alternative 2: Mixture of Experts (MoE)

- Only activate subset of parameters per token
- Sparse architecture reduces compute
- Specialized routing and load balancing needed
- **Use case:** Serving MoE models (GPT-4 rumored architecture)

### Alternative 3: Speculative Decoding

- Use small draft model to predict multiple tokens
- Verify with large model
- 2-3x speedup potential
- **Use case:** Latency-critical applications

## Key Takeaways

1. **Hybrid parallelism (TP+PP) is optimal** for 100B+ models
2. **Communication is manageable** with proper interconnect (NVLink/IB)
3. **Fault tolerance requires** checkpointing and replica failover
4. **Cost scales linearly** with number of replicas (~$35K/month per replica)
5. **Monitoring is critical** for distributed system health
