# Scenario 02: Evaluation Rubric - Distributed LLM Serving

## Scoring Framework (Total: 100 points)

| Category | Points | Key Focus |
|----------|--------|-----------|
| Requirements & Clarification | 10 | Understanding of distributed constraints |
| Parallelism Strategy | 30 | TP, PP, hybrid approach selection |
| Communication Design | 20 | Bandwidth calculations, optimization |
| Fault Tolerance | 20 | Failure handling, recovery strategies |
| Memory Management | 10 | KV cache, activation distribution |
| Operations & Monitoring | 10 | Deployment, debugging, scaling |

## Detailed Scoring

### 1. Requirements & Clarification (10 points)

**Outstanding (9-10 points):**
- Asks about model architecture specifics (layers, hidden size, attention heads)
- Inquires about available interconnect (NVLink, InfiniBand)
- Clarifies GPU topology and node configuration
- Questions about failure tolerance requirements
- Asks about cost constraints and trade-offs

**Acceptable (5-6 points):**
- Basic questions about model size
- Asks about performance requirements
- Limited depth in follow-ups

**Weak (0-4 points):**
- Doesn't ask about interconnect
- Misses model distribution constraints
- No questions about failure scenarios

### 2. Parallelism Strategy (30 points)

**2a. Understanding of Parallelism Types (10 points)**

**Outstanding (9-10):**
```
Clearly explains:
- Tensor Parallelism: Split each layer across GPUs, all-reduce communication
- Pipeline Parallelism: Split layers across GPUs, point-to-point communication
- Data Parallelism: Replicate model, parallel batch processing
- When to use each approach
```

**Strong (7-8):**
- Understands main approaches
- Can compare TP vs PP
- Basic trade-off analysis

**Weak (0-5):**
- Confuses parallelism types
- Can't explain communication patterns
- No trade-off discussion

**2b. Memory Calculation (10 points)**

**Outstanding (9-10):**
```
Accurate calculation:
175B parameters × 2 bytes (FP16) = 350 GB
Per GPU with 8-way TP: 350/8 = 43.75 GB
KV cache (32 seqs, 2K ctx): ~8 GB
Activations: ~5 GB
Total: ~57 GB < 80 GB ✓

With quantization:
INT4: 175B × 0.5 bytes = 87.5 GB total
Per GPU: 87.5/8 = ~11 GB
```

**Acceptable (5-7):**
- Rough calculation present
- Order of magnitude correct
- May miss KV cache or activations

**Weak (0-4):**
- No calculation
- Wrong order of magnitude
- Doesn't account for non-weight memory

**2c. Strategy Selection & Justification (10 points)**

**Outstanding (9-10):**
- Proposes hybrid approach (e.g., 2-way PP × 4-way TP)
- Justifies with memory, latency, communication analysis
- Discusses alternatives and why hybrid is better
- Considers scalability to more GPUs

**Example justification:**
"Pure 8-way TP has excessive all-reduce overhead. Pure 8-way PP has large pipeline bubble. Hybrid 2-way PP × 4-way TP minimizes both: only 2 pipeline stages (small bubble), 4-way TP keeps all-reduce manageable with NVLink."

**Strong (6-8):**
- Reasonable parallelism strategy
- Basic justification
- Understands main trade-offs

**Weak (0-5):**
- Only considers one approach
- No justification
- Unrealistic strategy (e.g., single GPU)

### 3. Communication Design (20 points)

**3a. Bandwidth Calculation (10 points)**

**Outstanding (9-10):**
```
Detailed calculation:

TP all-reduce per layer:
- Data volume: hidden_size × batch_size × 2 bytes
- 12,288 × 16 × 2 = 393 KB
- Ring all-reduce: 2(n-1)/n overhead = 1.5x for 4-way
- 590 KB per all-reduce
- 2 per layer: 1.18 MB
- 96 layers: 113 MB per token

NVLink bandwidth: 600 GB/s
Time: 113 MB / 600 GB/s = 0.19 ms per token

PP communication:
- Hidden states: 393 KB per transfer
- 1 transfer between stages per token
- Time: 0.65 μs (negligible)
```

**Strong (6-8):**
- Estimates communication volume
- Knows NVLink/IB bandwidth
- Rough latency estimate

**Weak (0-5):**
- No calculation
- Doesn't know interconnect specs
- Can't estimate impact

**3b. Optimization Strategies (10 points)**

**Outstanding (9-10):**
- Mentions overlapping communication and computation
- Discusses gradient compression (if training)
- Proposes topology-aware placement
- Mentions NCCL optimizations

**Strong (6-8):**
- Basic optimization ideas
- Understands need for fast interconnect
- Mentions batching benefits

**Weak (0-5):**
- No optimization discussion
- Doesn't understand bottlenecks

### 4. Fault Tolerance (20 points)

**4a. Failure Detection (7 points)**

**Outstanding (6-7):**
- Heartbeat monitoring (10s intervals)
- CUDA error detection
- Memory leak monitoring
- Network connectivity checks
- Multi-level health checks

**Acceptable (3-5):**
- Basic health monitoring
- Timeout-based detection
- Limited detail

**Weak (0-2):**
- No detection strategy
- "Restart if fails" only

**4b. Recovery Strategy (8 points)**

**Outstanding (7-8):**
- Multi-tier recovery:
  1. Failover to healthy replica (preferred)
  2. Reconfigure remaining GPUs if possible
  3. Checkpoint and restart as last resort
- Request state preservation
- Graceful degradation plan
- Clear RTO/RPO targets

**Strong (5-6):**
- Replica failover strategy
- Basic checkpointing
- Some state preservation

**Weak (0-4):**
- No recovery plan
- Assumes no failures
- Only mentions restart

**4c. Request Checkpointing (5 points)**

**Outstanding (5):**
```python
# Shows understanding of:
- Periodic KV cache snapshots
- Request state preservation
- Checkpoint frequency trade-offs
- Resume after failure
```

**Acceptable (2-3):**
- Mentions checkpointing
- Basic approach
- Limited detail

**Weak (0-1):**
- No checkpointing strategy

### 5. Memory Management (10 points)

**Outstanding (9-10):**
- Distributed KV cache design (sharded across TP group)
- PagedAttention for efficiency
- Discusses activation checkpointing
- Memory pool management
- Quantization strategies (INT8/INT4 KV cache)

**Example:**
"KV cache is sharded across the TP group. Each GPU stores cache for its attention heads. With 4-way TP and 96 total heads, each GPU manages 24 heads. Use PagedAttention with 16-token blocks to reduce fragmentation. Total KV cache: ~8 GB per GPU for batch of 32."

**Strong (6-8):**
- Understanding of distributed cache
- Basic memory management
- Mentions some optimization

**Weak (0-5):**
- Doesn't understand distributed memory
- No optimization strategies
- Can't explain KV cache distribution

### 6. Operations & Monitoring (10 points)

**Outstanding (9-10):**
- Comprehensive monitoring:
  - Per-GPU utilization, memory, temperature
  - Communication latency (TP, PP)
  - Request queue depth per replica
  - End-to-end latency breakdown
- Distributed tracing for debugging
- Deployment strategy (staged rollout)
- Auto-scaling logic for replicas

**Strong (6-8):**
- Good monitoring coverage
- Basic deployment plan
- Some operational considerations

**Weak (0-5):**
- Minimal monitoring
- No deployment strategy
- Missing operational aspects

## Level Expectations

### L6/Senior Staff (85-100 points)

**Must Demonstrate:**
- Expert knowledge of parallelism strategies
- Accurate memory and bandwidth calculations
- Comprehensive fault tolerance design
- Production operations mindset
- Can propose optimizations unprompted

**Distinguishing Factors:**
- References papers (Megatron-LM, GPipe, ZeRO)
- Discusses advanced topics (MoE, sequence parallelism)
- Proposes novel optimizations
- Deep understanding of GPU interconnects

### L5/Staff (70-84 points)

**Must Demonstrate:**
- Solid understanding of TP and PP
- Can design hybrid parallelism strategy
- Basic fault tolerance
- Reasonable memory estimates

**May Need Prompting:**
- Detailed bandwidth calculations
- Advanced optimizations
- Complex failure scenarios
- Multi-node scaling

### L4/Senior (55-69 points)

**Must Demonstrate:**
- Basic parallelism understanding
- Knows model won't fit on single GPU
- Simple distribution strategy
- Awareness of fault tolerance

**Will Need Guidance:**
- Choosing optimal parallelism
- Communication analysis
- Memory calculations
- Production operations

## Calibration Examples

### Example 1: Outstanding (TP Strategy Discussion)

**Candidate:**
"For 175B parameters, I'd use hybrid parallelism: 2-way pipeline × 4-way tensor parallel. Here's why:

Memory: 350GB FP16 / 8 GPUs = 43.75 GB per GPU with pure TP. This fits in A100 80GB, but leaves only ~20GB for KV cache and activations. With 2-way PP × 4-way TP, each GPU holds 175GB/2/4 = 21.875 GB of weights, giving us 50+ GB for KV cache.

Communication: 4-way TP requires all-reduce after each attention and FFN layer. With NVLink at 600 GB/s and ~1 MB per all-reduce, this adds ~10ms per forward pass across all layers. Acceptable overhead.

Pipeline: 2 stages minimizes pipeline bubble. With micro-batching (8 micro-batches), bubble is only ~10% of total time.

This gives us ~80ms per token latency, meeting our <300ms P99 requirement."

**Score: 28/30 (Parallelism)**
- Perfect understanding ✓
- Accurate calculations ✓
- Clear justification ✓
- Considers alternatives ✓

### Example 2: Acceptable (TP Strategy Discussion)

**Candidate:**
"The model is too big for one GPU, so we need to split it across 8 GPUs. I'd use tensor parallelism to split each layer across the GPUs. Each GPU would hold 1/8th of the model weights. We'd need to do all-reduce operations between the GPUs to combine results."

**Score: 18/30 (Parallelism)**
- Understands need for distribution ✓
- Knows about tensor parallelism ✓
- Missing calculations
- No hybrid approach consideration
- Doesn't analyze communication cost
- No discussion of alternatives

### Example 3: Weak (TP Strategy Discussion)

**Candidate:**
"We could split the model into 8 parts and put each part on a different GPU. Then we run inference on all GPUs and combine the results."

**Score: 8/30 (Parallelism)**
- Vague understanding of distribution
- Doesn't distinguish TP vs PP
- No memory analysis
- No communication consideration
- Confuses parallelism concepts

## Red Flags Checklist

Critical Issues:
- [ ] Proposes single-GPU solution for 175B model
- [ ] Doesn't know what tensor parallelism is
- [ ] Can't calculate memory requirements
- [ ] No understanding of NVLink vs PCIe
- [ ] Ignores communication overhead entirely
- [ ] No fault tolerance consideration
- [ ] Claims unrealistic latency (e.g., "10ms per token for 175B model")

**3+ critical issues → No Hire**

## Green Flags Checklist

Strong Signals:
- [ ] Proposes hybrid parallelism unprompted
- [ ] Accurate memory calculations with KV cache
- [ ] Bandwidth calculations for communication
- [ ] Discusses topology-aware placement
- [ ] References Megatron-LM, DeepSpeed, or similar
- [ ] Comprehensive fault tolerance strategy
- [ ] Mentions specific interconnects (NVLink, InfiniBand)
- [ ] Discusses advanced optimizations (Flash Attention, quantization)
- [ ] Production operations mindset

**6+ green flags → Strong Hire**

## Interview Feedback Template

### Strong Performance (80+)

"Excellent work on the distributed serving design. Your hybrid parallelism strategy (2-way PP × 4-way TP) was well-justified with accurate memory and bandwidth calculations. The fault tolerance design with replica failover and checkpointing was comprehensive. You demonstrated deep understanding of distributed systems and LLM serving."

**Strengths:**
- Accurate technical calculations
- Well-reasoned parallelism strategy
- Comprehensive fault tolerance

**Growth areas:**
- [Minor points if any]

**Recommendation:** Strong Hire for L6/Staff

### Acceptable Performance (60-75)

"Good understanding of distributed model serving. You identified the need for model parallelism and proposed a reasonable strategy. To strengthen, focus on quantitative analysis (memory/bandwidth calculations) and more detailed fault tolerance planning."

**Strengths:**
- Understands parallelism concepts
- Reasonable architecture

**Growth areas:**
- More rigorous calculations
- Deeper fault tolerance planning
- Advanced optimization techniques

**Recommendation:** Hire for L4/L5 depending on other signals

## Scenario-Specific Notes

**Key Discriminators for This Scenario:**

1. **Memory Calculation Accuracy** (Make or break)
   - Can they calculate 175B × 2 bytes = 350 GB?
   - Do they account for KV cache?
   - Do they know A100 has 80 GB?

2. **Parallelism Strategy** (Core competency)
   - TP only → Acceptable
   - Hybrid TP+PP → Strong
   - Can justify choice → Outstanding

3. **Communication Analysis** (Depth indicator)
   - No analysis → Weak
   - Mentions need for fast interconnect → Acceptable
   - Calculates bandwidth requirement → Strong
   - Optimizes communication pattern → Outstanding

4. **Fault Tolerance** (Production readiness)
   - No consideration → Red flag
   - Basic detection → Acceptable
   - Replica failover → Strong
   - Checkpointing + graceful degradation → Outstanding
