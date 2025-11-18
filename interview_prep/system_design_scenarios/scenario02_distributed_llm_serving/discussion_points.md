# Scenario 02: Discussion Points - Distributed LLM Serving

## Critical Discussion Topics

### 1. Parallelism Strategy Selection (15 minutes)

**Opening Question:**
"For a 175B parameter model, how would you distribute it across 8 GPUs? Walk me through your reasoning."

**Key Discussion Points:**
- Comparison of TP, PP, DP approaches
- Memory calculation per GPU
- Communication overhead analysis
- Hybrid approach justification

**Follow-up Questions:**

1. **"Why not use pure 8-way tensor parallelism?"**
   - Expected: Communication overhead increases with TP degree
   - All-reduce cost scales with number of GPUs
   - Bandwidth limitations beyond 4-8 way TP
   - Hybrid approach reduces all-reduce frequency

2. **"How would you change the strategy for a 500B parameter model?"**
   - Expected: Need more aggressive parallelism
   - Might need multi-node deployment
   - Consider ZeRO-style parameter partitioning
   - 8-way TP × 4-way PP across 32 GPUs

3. **"What's the impact of pipeline depth on latency?"**
   - Expected: Pipeline bubble reduces efficiency
   - Micro-batching helps hide bubble
   - Deeper pipeline = higher latency, better throughput
   - Trade-off between latency and throughput

**Red Flags:**
- Can't explain difference between TP and PP
- Proposes single-GPU solution
- Doesn't calculate memory requirements
- Ignores communication costs

**Green Flags:**
- Calculates memory per GPU accurately
- Discusses communication patterns
- Proposes hybrid approach with justification
- Mentions specific tools (Megatron-LM, DeepSpeed)

### 2. Communication Optimization (12 minutes)

**Opening Question:**
"How much network bandwidth do you need for your design? Show me the calculation."

**Key Points to Cover:**

**Bandwidth Calculation:**
```
For 4-way TP, per layer:
- All-reduce size: hidden_size × batch_size × 2 bytes
- Ring all-reduce: 2(n-1)/n of data volume
- 4-way: 2(4-1)/4 = 1.5x data volume

For hidden_size=12,288, batch=16:
- Per all-reduce: 12,288 × 16 × 2 = 393 KB
- With overhead: 590 KB
- 2 all-reduces per layer: 1.18 MB
- 96 layers: 113 MB per forward pass

NVLink bandwidth: 600 GB/s
Time: 113 MB / 600 GB/s = 0.19 ms
```

**Follow-ups:**

1. **"What happens if you only have PCIe instead of NVLink?"**
   - Expected: PCIe Gen4 x16 = 64 GB/s (10x slower)
   - Communication becomes bottleneck
   - Would need to reduce TP degree or use faster interconnect
   - Might switch to pure pipeline parallelism

2. **"How does batch size affect communication overhead?"**
   - Expected: Larger batches increase communication volume
   - But computation also increases proportionally
   - Communication-to-computation ratio stays constant
   - Sweet spot: balance memory and efficiency

### 3. Fault Tolerance Design (12 minutes)

**Opening Question:**
"What happens if GPU 3 crashes in the middle of processing a request? How do you handle it?"

**Expected Answer Structure:**

1. **Detection:**
   - Heartbeat mechanism (10s intervals)
   - CUDA error monitoring
   - Memory leak detection
   - Process crash detection

2. **Impact Assessment:**
   - Which TP group affected?
   - Which pipeline stage affected?
   - How many in-flight requests?

3. **Recovery Options:**
   - **Option A:** Failover to replica (preferred)
   - **Option B:** Reconfigure remaining GPUs
   - **Option C:** Checkpoint and restart

4. **State Preservation:**
   - KV cache checkpointing
   - Request queue state
   - Position in generation

**Follow-ups:**

1. **"How do you checkpoint the KV cache efficiently?"**
   - Expected: Distributed KV cache across TP group
   - Need to gather and save all shards
   - Copy-on-write for efficiency
   - Asynchronous checkpointing to avoid blocking

2. **"What if the entire node fails (all 8 GPUs)?"**
   - Expected: Route to another replica
   - Load balancer detects node failure
   - Retry in-flight requests
   - May lose some requests if no checkpointing

### 4. Multi-Node Scaling (10 minutes)

**Opening Question:**
"How would you scale this to 4 nodes (32 GPUs) for even larger models?"

**Discussion Points:**

**Option 1: Larger TP Groups**
```
4 nodes × 8 GPUs = 32 GPUs
Could do: 8-way TP × 4-way PP
- Each TP group spans 2 nodes
- Requires InfiniBand for inter-node all-reduce
- Higher communication latency (50μs vs 5μs)
```

**Option 2: Deeper Pipeline**
```
4-way TP × 8-way PP
- Each TP group within single node (NVLink)
- Pipeline stages across nodes (InfiniBand)
- Lower all-reduce overhead
- Higher pipeline bubble
```

**Trade-offs:**
- TP prefers low-latency interconnect (NVLink)
- PP tolerates higher latency (batch transfers)
- Hybrid: TP within node, PP across nodes

**Follow-up:**
"How does cross-datacenter deployment work?"
- Expected: Not practical for real-time inference
- 50-100ms cross-DC latency breaks TP
- Could use PP with very deep pipeline
- Better: replicate entire model in each DC

## Advanced Topics

### 5. Memory-Efficient Attention

**Question:** "How do you handle 32K context length with limited GPU memory?"

**Techniques to Discuss:**
- Flash Attention (reduce memory from O(N²) to O(N))
- PagedAttention for KV cache
- Sparse attention patterns
- KV cache quantization (8-bit, 4-bit)

**Memory Calculation:**
```
Standard attention (32K context):
- KV cache per layer: 2 × seq_len × hidden_size × 2 bytes
- For 96 layers: 2 × 32768 × 12288 × 96 × 2 = 147 GB

With Flash Attention:
- No need to materialize attention matrix
- Memory: O(seq_len × hidden_size) instead of O(seq_len²)
- Reduces peak memory by ~30-40%

With KV cache quantization (INT8):
- 147 GB → 73.5 GB (2x reduction)
- Minimal accuracy loss (<1%)
```

### 6. Dynamic Batching in Distributed Setting

**Question:** "How do you implement continuous batching with pipeline parallelism?"

**Challenges:**
- Pipeline stages process different micro-batches
- Can't easily add/remove sequences mid-pipeline
- Need synchronized batch updates

**Solutions:**

1. **Synchronized Batching:**
   - Wait for pipeline flush before updating batch
   - Higher latency for new requests
   - Simpler implementation

2. **Asynchronous Batching:**
   - Different batch composition per stage
   - Complex state tracking
   - Better latency

3. **Hybrid Approach:**
   - Continuous batching within TP groups
   - Synchronized micro-batches for PP
   - Balance complexity and performance

## Scenario Extensions

### Extension 1: Mixture of Experts (MoE)

**Question:** "How would your design change for a MoE model where only 2 out of 16 experts are activated per token?"

**Changes Needed:**
- Expert placement strategy (which GPUs)
- Dynamic routing and load balancing
- Expert-level parallelism
- Communication patterns differ (sparse all-to-all)

**Architecture:**
```
Option 1: Expert Parallelism
- Distribute 16 experts across 8 GPUs (2 each)
- All-to-all communication for routing
- Load balancing critical (expert imbalance)

Option 2: Expert + Tensor Parallelism
- 2-way expert parallel × 4-way TP
- Each expert shard across 4 GPUs
- More balanced, higher communication
```

### Extension 2: Multi-Query Attention

**Question:** "How does Multi-Query Attention (MQA) or Grouped-Query Attention (GQA) affect your parallelism strategy?"

**Impact:**
- KV cache size reduced (fewer KV heads)
- Memory savings: 5-10x for KV cache
- Can fit longer contexts or larger batches
- TP communication slightly reduced

**Example:**
```
Standard: 96 attention heads, 96 KV heads
MQA: 96 attention heads, 1 KV head
GQA: 96 attention heads, 8 KV heads

KV cache with GQA:
- 12x reduction vs standard
- 32K context becomes feasible
```

## Common Mistakes

### Mistake 1: Ignoring Topology

**Problem:** Not considering GPU topology in parallelism design

**Example:**
```
Bad: TP group = [GPU 0, GPU 2, GPU 4, GPU 6]
- These GPUs may not have direct NVLink
- High communication latency

Good: TP group = [GPU 0, GPU 1, GPU 2, GPU 3]
- NVSwitch provides full bandwidth between all
```

**Fix:** Use `nvidia-smi topo -m` to understand topology

### Mistake 2: Unrealistic Failure Handling

**Problem:** "We'll just restart the failed GPU"

**Issues:**
- GPU restart takes 30-60 seconds
- Model loading takes 2-5 minutes
- Lost in-flight requests
- No graceful degradation

**Better:** Replica failover + eventual GPU replacement

### Mistake 3: Not Planning for Heterogeneous Hardware

**Problem:** Assuming all GPUs are identical

**Reality:**
- Mixed GPU types (A100, H100)
- Different VRAM sizes (40GB, 80GB)
- Performance variations (boost clocks)

**Solution:**
- Detect GPU capabilities at runtime
- Adaptive parallelism configuration
- Place larger model parts on bigger GPUs

## Interviewer Calibration

### Time Allocation

**For 60-minute interview:**
- 0-10 min: Requirements, model size, constraints
- 10-25 min: Parallelism strategy and architecture
- 25-40 min: Deep dive (2 topics from above)
- 40-50 min: Fault tolerance and operations
- 50-60 min: Extensions and edge cases

### Topic Selection by Background

**Systems Engineer:**
- Focus on: Fault tolerance, networking, monitoring
- Light on: Model architecture details
- Deep dive: Communication protocols, failure recovery

**ML Engineer:**
- Focus on: Parallelism strategies, memory optimization
- Light on: Distributed systems fundamentals
- Deep dive: Attention mechanisms, quantization

**ML Systems (ideal):**
- Balanced coverage across all topics
- Can go deep on both systems and ML aspects

### Scoring Signals

**Strong Hire (L6):**
- Calculates memory and bandwidth accurately
- Proposes hybrid parallelism unprompted
- Discusses fault tolerance proactively
- Mentions specific tools (Megatron, DeepSpeed, NCCL)
- Can extend to MoE or other architectures

**Hire (L5):**
- Understands TP vs PP trade-offs
- Reasonable parallelism strategy
- Basic fault tolerance approach
- Needs prompting for calculations

**No Hire:**
- Can't explain parallelism strategies
- No memory calculations
- Ignores distributed systems challenges
- Unrealistic performance expectations

## Key Resources

**Papers:**
- "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism"
- "GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism"
- "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models"
- "DeepSpeed Inference: Enabling Efficient Inference of Transformer Models at Unprecedented Scale"

**Tools:**
- NVIDIA Megatron-LM
- Microsoft DeepSpeed
- NCCL (NVIDIA Collective Communications Library)
- PyTorch FSDP (Fully Sharded Data Parallel)
