# Disaggregated Inference Integration Plan for Token Parallelism

## Overview

This document outlines the integration plan for token parallelism into disaggregated inference systems, where prefill and decode operations are handled by separate clusters. The goal is to optimize decode performance using token parallelism while maintaining compatibility with existing disaggregated prefill systems.

## Disaggregated Architecture Analysis

### Current vLLM Disaggregated System
vLLM already supports experimental disaggregated prefill/decode with:

1. **Prefill Instance (KV Producer)**:
   - Handles prompt processing and initial KV cache generation
   - Transfers KV cache to decode instances via connectors
   - Optimized for high throughput prompt processing

2. **Decode Instance (KV Consumer)**:
   - Receives KV cache from prefill instances
   - Performs autoregressive token generation
   - Currently limited by attention computation scaling

3. **KV Transfer Infrastructure**:
   - PyNcclConnector for KV cache transfer
   - LookupBuffer for KV cache management
   - Support for different transfer protocols

### Token Parallelism Integration Points
Token parallelism will enhance the decode stage by:
- Distributing attention computation across multiple GPUs
- Partitioning KV cache across token parallel ranks
- Maintaining prefill compatibility with minimal changes

## Integration Architecture

### System Components

#### 1. Enhanced Decode Cluster
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Decode Node 1 │    │   Decode Node 2 │    │   Decode Node 3 │
│  (Root Rank)    │    │  (Worker Rank)  │    │  (Worker Rank)  │
│                 │    │                 │    │                 │
│ ┌─────────────┐ │    │ ┌─────────────┐ │    │ ┌─────────────┐ │
│ │Model Weights│ │    │ │  KV Cache   │ │    │ │  KV Cache   │ │
│ │QKV/O Proj   │ │    │ │ Partition   │ │    │ │ Partition   │ │
│ │MLP Layers   │ │    │ │             │ │    │ │             │ │
│ └─────────────┘ │    │ └─────────────┘ │    │ └─────────────┘ │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    Token Parallel Communication
```

#### 2. KV Transfer Integration
```
Prefill Cluster                    Decode Cluster (Token Parallel)
┌──────────────┐                  ┌──────────────┐──────────────┐──────────────┐
│              │  KV Transfer     │ Root Rank    │ Rank 1       │ Rank 2       │
│ Prefill Node │─────────────────▶│ (Model +     │ (KV Cache    │ (KV Cache    │
│              │  (Full KV Cache) │  KV Cache)   │  Partition)  │  Partition)  │
└──────────────┘                  └──────────────┘──────────────┘──────────────┘
```

## Implementation Plan

### Phase 1: Core Integration (Week 1)

#### Task 1.1: Enhance KV Transfer for Token Parallelism
**File**: `vllm/distributed/kv_transfer/`

Create `TokenParallelKVConnector`:
```python
class TokenParallelKVConnector(BaseKVConnector):
    """KV connector that distributes cache across token parallel ranks"""
    
    def __init__(self, kv_role, kv_rank, kv_parallel_size, 
                 token_parallel_size):
        super().__init__(kv_role, kv_rank, kv_parallel_size)
        self.token_parallel_size = token_parallel_size
        self.tknp_rank = get_tknp_rank()
        
    def send_kv_cache(self, kv_cache_dict):
        """Send KV cache from prefill to decode cluster"""
        if self.kv_role == "kv_producer":
            # Send to decode root rank, which will distribute
            return super().send_kv_cache(kv_cache_dict)
        else:
            raise ValueError("Only producers can send KV cache")
    
    def receive_kv_cache(self):
        """Receive and partition KV cache for token parallel ranks"""
        if self.kv_role == "kv_consumer":
            if self.tknp_rank == 0:
                # Root rank receives full KV cache
                full_kv_cache = super().receive_kv_cache()
                # Partition and distribute to other ranks
                return self._partition_and_distribute_kv(full_kv_cache)
            else:
                # Worker ranks receive their partition
                return self._receive_kv_partition()
        else:
            raise ValueError("Only consumers can receive KV cache")
    
    def _partition_and_distribute_kv(self, kv_cache_dict):
        """Partition KV cache across token parallel ranks"""
        partitioned_cache = {}
        tknp_group = get_tknp_group()
        
        for layer_name, kv_tensors in kv_cache_dict.items():
            # Partition along batch dimension
            batch_size = kv_tensors[0].shape[0]  # Key tensor batch size
            partition_size = batch_size // self.token_parallel_size
            
            # Keep local partition
            start_idx = 0
            end_idx = partition_size
            local_kv = [tensor[start_idx:end_idx] for tensor in kv_tensors]
            partitioned_cache[layer_name] = local_kv
            
            # Send partitions to other ranks
            for rank in range(1, self.token_parallel_size):
                start_idx = rank * partition_size
                end_idx = (rank + 1) * partition_size
                rank_kv = [tensor[start_idx:end_idx] for tensor in kv_tensors]
                
                # Send to rank (implement async communication)
                self._send_kv_partition_to_rank(rank, layer_name, rank_kv)
        
        return partitioned_cache
```

#### Task 1.2: Token Parallel Decode Engine
**File**: `vllm/v1/engine/token_parallel_decode_core.py`

```python
class TokenParallelDecodeCore(EngineCore):
    """Decode engine core with token parallelism support"""
    
    def __init__(self, vllm_config, executor_class, kv_transfer_config):
        super().__init__(vllm_config, executor_class)
        self.kv_transfer_config = kv_transfer_config
        self.token_parallel_size = vllm_config.parallel_config.token_parallel_size
        self.tknp_rank = get_tknp_rank()
        
        # Initialize KV connector with token parallel support
        self.kv_connector = TokenParallelKVConnector(
            kv_role=kv_transfer_config.kv_role,
            kv_rank=kv_transfer_config.kv_rank,
            kv_parallel_size=kv_transfer_config.kv_parallel_size,
            token_parallel_size=self.token_parallel_size
        )
    
    def receive_and_process_prefill_cache(self):
        """Receive KV cache from prefill and prepare for token parallel decode"""
        received_cache = self.kv_connector.receive_kv_cache()
        
        # Initialize local KV cache with received data
        for layer_name, kv_tensors in received_cache.items():
            layer_idx = self._extract_layer_index(layer_name)
            self.model_executor.cache_engine.update_kv_cache(
                layer_idx, kv_tensors, token_parallel_rank=self.tknp_rank
            )
    
    def step_with_token_parallel(self, requests):
        """Execute decode step with token parallel coordination"""
        if self.tknp_rank == 0:
            # Root rank handles full scheduling and model forward
            scheduler_output = self.scheduler.schedule(requests)
            
            # Distribute batch to token parallel ranks
            distributed_output = self._distribute_batch_to_ranks(scheduler_output)
            
            # Execute model with token parallel attention
            model_output = self.model_executor.execute_model(distributed_output)
            
            # Process and return results
            return self.scheduler.update_from_output(scheduler_output, model_output)
        else:
            # Worker ranks participate in attention computation only
            self._participate_in_token_parallel_attention()
            return None
```

#### Task 1.3: Enhanced Decode Worker
**File**: `vllm/worker/token_parallel_decode_worker.py`

```python
class TokenParallelDecodeWorker(Worker):
    """Worker specialized for token parallel decode operations"""
    
    def __init__(self, vllm_config):
        super().__init__(vllm_config)
        self.token_parallel_size = vllm_config.parallel_config.token_parallel_size
        self.tknp_rank = get_tknp_rank()
        
        # Only root rank loads full model weights
        self.load_model_weights = (self.tknp_rank == 0)
        
    def load_model(self):
        """Load model with token parallel considerations"""
        if self.load_model_weights:
            # Root rank loads full model
            super().load_model()
        else:
            # Worker ranks load minimal model (attention only)
            self._load_minimal_model_for_attention()
    
    def _load_minimal_model_for_attention(self):
        """Load only components needed for attention computation"""
        # Load only attention-related parameters
        # Skip MLP layers and embedding layers
        pass
    
    def execute_model(self, execute_model_req):
        """Execute model with token parallel attention"""
        if self.tknp_rank == 0:
            # Root rank handles full forward pass
            return super().execute_model(execute_model_req)
        else:
            # Worker ranks participate in attention only
            return self._execute_attention_only(execute_model_req)
```

### Phase 2: KV Cache Management (Week 1-2)

#### Task 2.1: Partitioned KV Cache Engine
**File**: `vllm/worker/token_parallel_cache_engine.py`

```python
class TokenParallelCacheEngine(CacheEngine):
    """Cache engine that manages partitioned KV cache for token parallelism"""
    
    def __init__(self, cache_config, model_config, parallel_config, device_config):
        super().__init__(cache_config, model_config, parallel_config, device_config)
        self.token_parallel_size = parallel_config.token_parallel_size
        self.tknp_rank = get_tknp_rank()
        
        # Adjust cache size for token parallel partitioning
        self.partition_cache_blocks()
    
    def partition_cache_blocks(self):
        """Partition cache blocks across token parallel ranks"""
        total_blocks = self.cache_config.num_gpu_blocks
        
        # Each rank gets a partition of the cache blocks
        blocks_per_rank = total_blocks // self.token_parallel_size
        remainder_blocks = total_blocks % self.token_parallel_size
        
        # Distribute remainder blocks to first few ranks
        if self.tknp_rank < remainder_blocks:
            self.local_cache_blocks = blocks_per_rank + 1
            self.cache_block_offset = self.tknp_rank * (blocks_per_rank + 1)
        else:
            self.local_cache_blocks = blocks_per_rank
            self.cache_block_offset = (remainder_blocks * (blocks_per_rank + 1) + 
                                     (self.tknp_rank - remainder_blocks) * blocks_per_rank)
    
    def allocate_kv_cache(self, num_blocks):
        """Allocate KV cache considering token parallel partitioning"""
        # Allocate cache for local partition only
        return super().allocate_kv_cache(self.local_cache_blocks)
    
    def update_kv_cache_from_transfer(self, layer_idx, transferred_kv, 
                                    sequence_ids, block_mappings):
        """Update cache with KV data from disaggregated prefill"""
        # Map transferred KV to local cache blocks
        local_kv = self._map_transferred_kv_to_local_cache(
            transferred_kv, sequence_ids, block_mappings
        )
        
        # Update local KV cache
        self.gpu_cache[layer_idx].copy_(local_kv)
```

#### Task 2.2: Batch Distribution Strategy
**File**: `vllm/core/token_parallel_scheduler.py`

```python
class TokenParallelScheduler(Scheduler):
    """Scheduler that handles batch distribution for token parallelism"""
    
    def __init__(self, scheduler_config, cache_config, lora_config, 
                 parallel_config):
        super().__init__(scheduler_config, cache_config, lora_config)
        self.token_parallel_size = parallel_config.token_parallel_size
        self.tknp_rank = get_tknp_rank()
    
    def schedule(self):
        """Schedule requests with token parallel batch distribution"""
        if self.tknp_rank == 0:
            # Root rank performs full scheduling
            scheduler_output = super().schedule()
            
            # Partition batch across token parallel ranks
            return self._partition_batch_for_token_parallel(scheduler_output)
        else:
            # Worker ranks wait for batch partition
            return self._receive_batch_partition()
    
    def _partition_batch_for_token_parallel(self, scheduler_output):
        """Partition scheduled batch across token parallel ranks"""
        partitioned_outputs = []
        
        for seq_group in scheduler_output.scheduled_seq_groups:
            # Determine which rank should handle this sequence group
            assigned_rank = hash(seq_group.request_id) % self.token_parallel_size
            
            if assigned_rank == 0:
                # Keep locally
                partitioned_outputs.append(seq_group)
            else:
                # Send to assigned rank
                self._send_seq_group_to_rank(assigned_rank, seq_group)
        
        # Update scheduler output with local partition
        local_scheduler_output = SchedulerOutputs(
            scheduled_seq_groups=partitioned_outputs,
            num_batched_tokens=sum(sg.num_seqs() for sg in partitioned_outputs),
            # ... other fields
        )
        
        return local_scheduler_output
```

### Phase 3: Service Integration (Week 2)

#### Task 3.1: Disaggregated Service Orchestration
**File**: `vllm/entrypoints/disaggregated_service.py`

```python
class DisaggregatedTokenParallelService:
    """Service orchestrator for disaggregated inference with token parallelism"""
    
    def __init__(self, service_config):
        self.service_config = service_config
        self.prefill_cluster = None
        self.decode_cluster = None
        
    async def start_services(self):
        """Start prefill and token parallel decode services"""
        # Start prefill cluster (existing implementation)
        await self._start_prefill_cluster()
        
        # Start token parallel decode cluster
        await self._start_token_parallel_decode_cluster()
        
        # Setup communication channels
        await self._setup_kv_transfer_channels()
    
    async def _start_prefill_cluster(self):
        """Start traditional prefill service"""
        prefill_config = self.service_config.prefill_config
        
        self.prefill_cluster = await AsyncLLMEngine.from_engine_args(
            EngineArgs(
                model=prefill_config.model,
                tensor_parallel_size=prefill_config.tensor_parallel_size,
                kv_transfer_config=KVTransferConfig(
                    kv_connector="TokenParallelKVConnector",
                    kv_role="kv_producer",
                    kv_rank=0,
                    kv_parallel_size=2
                )
            )
        )
    
    async def _start_token_parallel_decode_cluster(self):
        """Start token parallel decode service"""
        decode_config = self.service_config.decode_config
        
        # Start decode cluster with token parallelism
        self.decode_cluster = await AsyncLLMEngine.from_engine_args(
            EngineArgs(
                model=decode_config.model,
                tensor_parallel_size=decode_config.tensor_parallel_size,
                token_parallel_size=decode_config.token_parallel_size,
                kv_transfer_config=KVTransferConfig(
                    kv_connector="TokenParallelKVConnector",
                    kv_role="kv_consumer",
                    kv_rank=1,
                    kv_parallel_size=2
                ),
                # Decode-specific optimizations
                max_num_seqs=decode_config.max_num_seqs,
                max_model_len=decode_config.max_model_len
            )
        )
    
    async def process_request(self, request):
        """Process request through disaggregated pipeline"""
        # Phase 1: Prefill
        prefill_result = await self.prefill_cluster.generate(
            request.prompt, 
            SamplingParams(max_tokens=1)  # Just prefill
        )
        
        # Phase 2: Transfer KV cache and decode
        decode_result = await self.decode_cluster.generate(
            request.prompt,
            request.sampling_params,
            kv_cache_from_prefill=prefill_result.kv_cache
        )
        
        return decode_result
```

#### Task 3.2: API Integration
**File**: `vllm/entrypoints/openai/api_server_disaggregated.py`

```python
@app.post("/v1/completions")
async def create_completion(request: CompletionRequest):
    """Handle completion request with disaggregated token parallel processing"""
    
    # Route to appropriate service
    if service_type == "prefill":
        # Handle prefill-only requests
        return await handle_prefill_request(request)
    elif service_type == "decode":
        # Handle decode with token parallelism
        return await handle_token_parallel_decode_request(request)
    elif service_type == "unified":
        # Handle full disaggregated pipeline
        return await handle_disaggregated_request(request)

async def handle_disaggregated_request(request: CompletionRequest):
    """Process request through disaggregated pipeline"""
    disaggregated_service = get_disaggregated_service()
    
    # Convert OpenAI request to internal format
    internal_request = convert_openai_to_internal(request)
    
    # Process through disaggregated pipeline
    result = await disaggregated_service.process_request(internal_request)
    
    # Convert back to OpenAI format
    return convert_internal_to_openai(result)
```

## Deployment Configurations

### Configuration Examples

#### 1. Basic Disaggregated Setup
```yaml
# disaggregated_config.yaml
prefill_service:
  model: "meta-llama/Llama-2-70b-hf"
  tensor_parallel_size: 8
  gpu_memory_utilization: 0.9
  max_model_len: 4096
  kv_transfer:
    connector: "TokenParallelKVConnector"
    role: "kv_producer"
    rank: 0
    parallel_size: 2

decode_service:
  model: "meta-llama/Llama-2-70b-hf"  
  tensor_parallel_size: 4
  token_parallel_size: 4
  gpu_memory_utilization: 0.8
  max_num_seqs: 256
  kv_transfer:
    connector: "TokenParallelKVConnector"
    role: "kv_consumer"
    rank: 1
    parallel_size: 2
```

#### 2. Multi-Node Disaggregated Setup
```bash
# Prefill cluster (Node 1-2)
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve meta-llama/Llama-2-70b-hf \
    --tensor-parallel-size 4 \
    --pipeline-parallel-size 2 \
    --port 8100 \
    --kv-transfer-config '{"kv_connector":"TokenParallelKVConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":2}' \
    --service-type prefill

# Decode cluster (Node 3-6) 
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve meta-llama/Llama-2-70b-hf \
    --tensor-parallel-size 2 \
    --token-parallel-size 4 \
    --port 8200 \
    --kv-transfer-config '{"kv_connector":"TokenParallelKVConnector","kv_role":"kv_consumer","kv_rank":1,"kv_parallel_size":2}' \
    --service-type decode
```

#### 3. Kubernetes Disaggregated Deployment
```yaml
# prefill-service.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-prefill
spec:
  replicas: 2
  selector:
    matchLabels:
      app: vllm-prefill
  template:
    spec:
      containers:
      - name: vllm-prefill
        image: vllm/vllm-openai:latest
        resources:
          limits:
            nvidia.com/gpu: 4
        command:
        - vllm
        - serve
        - meta-llama/Llama-2-70b-hf
        - --tensor-parallel-size=4
        - --service-type=prefill
        - --kv-transfer-config={"kv_connector":"TokenParallelKVConnector","kv_role":"kv_producer"}

---
# decode-service.yaml  
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-decode
spec:
  replicas: 4
  selector:
    matchLabels:
      app: vllm-decode
  template:
    spec:
      containers:
      - name: vllm-decode
        image: vllm/vllm-openai:latest
        resources:
          limits:
            nvidia.com/gpu: 2
        command:
        - vllm
        - serve
        - meta-llama/Llama-2-70b-hf
        - --tensor-parallel-size=2
        - --token-parallel-size=4
        - --service-type=decode
        - --kv-transfer-config={"kv_connector":"TokenParallelKVConnector","kv_role":"kv_consumer"}
```

## Performance Optimization

### KV Transfer Optimization
```python
# Optimized KV transfer for token parallelism
class OptimizedTokenParallelKVTransfer:
    def __init__(self, compression_ratio=0.5):
        self.compression_ratio = compression_ratio
        
    def compress_kv_cache(self, kv_tensors):
        """Compress KV cache before transfer"""
        # Use techniques like:
        # - FP16/BF16 precision reduction
        # - Quantization for older tokens
        # - Delta compression for similar sequences
        compressed_kv = []
        for tensor in kv_tensors:
            if tensor.dtype == torch.float32:
                tensor = tensor.half()  # FP16 conversion
            compressed_kv.append(tensor)
        return compressed_kv
    
    async def transfer_kv_async(self, kv_cache, target_ranks):
        """Asynchronous KV cache transfer"""
        transfer_tasks = []
        
        for rank, kv_partition in zip(target_ranks, kv_cache):
            task = asyncio.create_task(
                self._send_kv_to_rank(rank, kv_partition)
            )
            transfer_tasks.append(task)
        
        await asyncio.gather(*transfer_tasks)
```

### Load Balancing Strategies
```python
class TokenParallelLoadBalancer:
    def __init__(self, token_parallel_size):
        self.token_parallel_size = token_parallel_size
        self.rank_loads = [0] * token_parallel_size
        
    def assign_sequence_to_rank(self, sequence_group):
        """Assign sequence group to least loaded rank"""
        # Find rank with minimum load
        min_load_rank = min(range(self.token_parallel_size), 
                           key=lambda r: self.rank_loads[r])
        
        # Update load tracking
        estimated_tokens = self._estimate_sequence_tokens(sequence_group)
        self.rank_loads[min_load_rank] += estimated_tokens
        
        return min_load_rank
    
    def rebalance_if_needed(self):
        """Rebalance load across ranks if imbalance detected"""
        max_load = max(self.rank_loads)
        min_load = min(self.rank_loads)
        
        # Rebalance if load imbalance > 20%
        if (max_load - min_load) / max_load > 0.2:
            return self._rebalance_sequences()
        
        return False
```

## Monitoring and Observability

### Metrics Collection
```python
class DisaggregatedTokenParallelMetrics:
    def __init__(self):
        self.prefill_metrics = PrefillMetrics()
        self.decode_metrics = TokenParallelDecodeMetrics()
        self.transfer_metrics = KVTransferMetrics()
        
    def collect_end_to_end_metrics(self):
        return {
            'prefill_latency': self.prefill_metrics.get_avg_latency(),
            'kv_transfer_latency': self.transfer_metrics.get_transfer_latency(),
            'decode_latency': self.decode_metrics.get_avg_latency(),
            'token_parallel_efficiency': self.decode_metrics.get_parallel_efficiency(),
            'memory_utilization': self.get_memory_utilization(),
            'throughput': self.get_overall_throughput()
        }
    
    def get_parallel_efficiency(self):
        """Calculate token parallel efficiency"""
        ideal_speedup = self.decode_metrics.token_parallel_size
        actual_speedup = self.decode_metrics.get_actual_speedup()
        return actual_speedup / ideal_speedup
```

### Dashboard Integration
```python
# Prometheus metrics for monitoring
from prometheus_client import Counter, Histogram, Gauge

# Define metrics
prefill_requests = Counter('prefill_requests_total', 'Total prefill requests')
decode_requests = Counter('decode_requests_total', 'Total decode requests')
kv_transfer_duration = Histogram('kv_transfer_duration_seconds', 'KV transfer duration')
token_parallel_efficiency = Gauge('token_parallel_efficiency', 'Token parallel efficiency')

# In service code
def track_disaggregated_request(request_id, start_time):
    prefill_requests.inc()
    # ... track through pipeline
    decode_requests.inc()
    token_parallel_efficiency.set(calculate_efficiency())
```

## Testing and Validation

### Integration Tests
```python
class TestDisaggregatedTokenParallel:
    def test_kv_transfer_accuracy(self):
        """Test KV cache transfer maintains accuracy"""
        # Compare outputs with and without disaggregation
        pass
    
    def test_load_balancing(self):
        """Test token parallel load balancing"""
        # Verify even distribution across ranks
        pass
    
    def test_fault_tolerance(self):
        """Test system behavior with node failures"""
        # Test prefill/decode cluster fault handling
        pass
    
    def test_performance_scaling(self):
        """Test performance scaling with token parallelism"""
        # Measure throughput improvements
        pass
```

### Load Testing
```bash
#!/bin/bash
# load_test_disaggregated.sh

# Test disaggregated system under load
python -m pytest tests/test_disaggregated_load.py \
    --num-requests=1000 \
    --concurrent-requests=50 \
    --token-parallel-size=4 \
    --model=meta-llama/Llama-2-70b-hf
```

## Future Enhancements

### Advanced KV Cache Sharing
- Cross-request KV cache sharing for similar prefixes
- Intelligent cache eviction policies
- Dynamic cache size adjustment

### Adaptive Token Parallelism
- Dynamic adjustment of token parallel size based on load
- Auto-scaling decode clusters
- Cost-based optimization

### Multi-Tier Disaggregation
- Separate clusters for different model sizes
- Routing based on request complexity
- Hierarchical KV cache management

## Conclusion

This disaggregated integration plan provides a comprehensive approach to implementing token parallelism in disaggregated inference systems. The key benefits include:

1. **Improved Decode Performance**: Token parallelism specifically targets the decode bottleneck
2. **Resource Efficiency**: Separate optimization of prefill and decode clusters
3. **Scalability**: Independent scaling of prefill and decode capacity
4. **Compatibility**: Maintains compatibility with existing disaggregated systems

The implementation leverages vLLM's existing disaggregated infrastructure while adding token parallelism capabilities to significantly improve decode performance for large-scale inference workloads. 