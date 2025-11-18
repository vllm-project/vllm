# Scenario 01: Discussion Points & Follow-up Questions

## Deep Dive Discussion Topics

### 1. Batching Strategies (15-20 minutes)

**Initial Question:**
"You mentioned continuous batching. Can you explain how this differs from traditional static batching, and why it's beneficial for LLM inference?"

**Expected Discussion Points:**
- **Static Batching:** All sequences start together, batch completes when longest sequence finishes
- **Continuous Batching:** New sequences join as slots free up, better GPU utilization
- **Trade-offs:** Complexity vs. performance, memory management challenges
- **Implementation:** How to handle variable sequence lengths, memory allocation

**Follow-up Questions:**
1. "What happens when you have requests with very different output lengths (e.g., 10 tokens vs. 2000 tokens)?"
   - Expected: Discussion of sequence length prediction, timeout mechanisms, fairness

2. "How would you prioritize urgent requests in the batching queue?"
   - Expected: Priority queues, preemption strategies, SLA-based scheduling

3. "Can you batch requests for different models together?"
   - Expected: No (different weights), but can batch different prompts for same model

**Red Flags:**
- Doesn't understand the difference between static and continuous batching
- Can't explain memory implications
- Doesn't consider fairness issues

**Green Flags:**
- Mentions iteration-level batching
- Discusses memory fragmentation
- Proposes multi-level batching strategies
- References vLLM or Orca paper

### 2. Memory Management & KV Cache (10-15 minutes)

**Initial Question:**
"The KV cache can be a significant memory bottleneck. How would you optimize memory usage for the KV cache?"

**Expected Discussion Points:**
- **Memory Requirements:** O(batch_size × sequence_length × hidden_size × num_layers)
- **PagedAttention:** Fixed-size blocks, reduce fragmentation
- **Sharing:** Prefix sharing for common prompts
- **Eviction:** LRU or priority-based eviction when memory is tight

**Follow-up Questions:**
1. "How much memory does the KV cache consume for a batch of 32 sequences with 2048 tokens each on a 70B parameter model?"
   - Expected calculation:
     ```
     # Llama-70B: 80 layers, 8192 hidden size
     # KV cache per token: 2 (K,V) × 8192 × 80 × 2 bytes (FP16) = 2.6 MB
     # Total: 32 seqs × 2048 tokens × 2.6 MB = 171 GB
     # With paging (16-token blocks): More efficient allocation
     ```

2. "Can you share KV cache across requests? When would this be beneficial?"
   - Expected: System prompts, common prefixes, RAG contexts
   - Implementation: Copy-on-write, reference counting

3. "What happens when you run out of KV cache memory?"
   - Expected: Eviction policies, request rejection, swap to CPU memory

**Red Flags:**
- Doesn't know what KV cache is
- Can't calculate memory requirements
- Doesn't mention any optimization techniques

**Green Flags:**
- Mentions PagedAttention or similar techniques
- Discusses prefix sharing
- Can calculate memory requirements
- Proposes hybrid CPU/GPU memory strategies

### 3. Model Parallelism (10-15 minutes)

**Initial Question:**
"For a 70B parameter model, you mentioned using tensor parallelism. Can you explain when you'd use tensor parallelism vs. pipeline parallelism?"

**Expected Discussion Points:**
- **Tensor Parallelism:** Split each layer across GPUs, lower latency
- **Pipeline Parallelism:** Split layers across GPUs, higher throughput
- **Data Parallelism:** Replicate model, scale with more requests
- **Hybrid Approaches:** Combine strategies for different needs

**Follow-up Questions:**
1. "How does tensor parallelism affect latency vs. throughput?"
   - Expected: Adds communication overhead (all-reduce), but maintains low latency
   - Good for: Interactive applications
   - Communication cost: ~10-20% overhead with NVLink

2. "What are the communication requirements for 4-way tensor parallelism?"
   - Expected: All-reduce after each layer, requires high-bandwidth interconnect
   - NVLink (600 GB/s) vs. PCIe (64 GB/s)
   - Impact on placement: Need GPUs in same node for TP

3. "When would you use pipeline parallelism instead?"
   - Expected: When model doesn't fit even with TP, or for batch processing
   - Higher latency (pipeline bubble), but better throughput
   - Micro-batching to hide bubble

**Red Flags:**
- Confuses different parallelism strategies
- Doesn't understand communication costs
- Can't explain when to use each approach

**Green Flags:**
- Clearly distinguishes TP, PP, DP
- Discusses communication patterns
- Mentions specific hardware requirements (NVLink)
- Proposes hybrid strategies (e.g., TP within node, PP across nodes)

### 4. Autoscaling & Resource Management (10-15 minutes)

**Initial Question:**
"How would you design the autoscaling system to handle traffic spikes while minimizing cost?"

**Expected Discussion Points:**
- **Metrics:** GPU utilization, queue depth, request latency
- **Scaling Policies:** Threshold-based, predictive, scheduled
- **Cold Start:** Model loading time (2-5 minutes for large models)
- **Cost Optimization:** Spot instances, reserved capacity, over-provisioning

**Follow-up Questions:**
1. "What's the challenge with autoscaling GPU-based services compared to CPU services?"
   - Expected: Expensive resources, long cold start, model loading time
   - Need to be more conservative with scale-down
   - Pre-warming strategies

2. "How would you handle a sudden 10x traffic spike?"
   - Expected:
     - Immediate: Queue requests, graceful degradation
     - Short-term: Scale up existing pools
     - Medium-term: Provision new instances
     - Discuss queue management and timeout strategies

3. "How do you balance using spot instances (cheaper) with reliability requirements?"
   - Expected:
     - Mix of on-demand (baseline) and spot (burst)
     - Spot instance interruption handling
     - Graceful draining of instances
     - Diversify across instance types

**Red Flags:**
- Proposes aggressive scale-down without considering cold start
- Doesn't consider cost implications
- No mention of graceful degradation

**Green Flags:**
- Discusses predictive autoscaling
- Mentions warm pools or pre-warming
- Proposes multi-tier scaling strategy
- Considers spot instance interruption handling

## Advanced Topics

### 5. Multi-Tenancy & Isolation (If time permits)

**Question:**
"If you needed to support multiple customers with different SLA requirements, how would you modify this architecture?"

**Discussion Points:**
- **Dedicated vs. Shared:** Trade-offs between isolation and utilization
- **Resource Quotas:** GPU time, memory, request rate limits
- **Priority Scheduling:** VIP customers get faster processing
- **Cost Attribution:** Track usage per tenant

**Follow-up:**
1. "How would you prevent one tenant from affecting another's performance?"
   - Expected: Separate worker pools, resource limits, priority queues

2. "How do you charge customers fairly?"
   - Expected: Track tokens (input + output), GPU time, request count

### 6. Failure Scenarios & Recovery (If time permits)

**Question:**
"What are the key failure modes in this system, and how do you handle them?"

**Expected Failure Modes:**
1. **GPU Out of Memory:**
   - Cause: Too many concurrent requests, memory leak
   - Detection: OOM errors, memory monitoring
   - Recovery: Reduce batch size, restart worker, reject requests

2. **Model Loading Failure:**
   - Cause: Corrupted weights, insufficient memory
   - Detection: Initialization errors
   - Recovery: Retry from backup, rollback to previous version

3. **Worker Crash:**
   - Cause: CUDA errors, segfaults
   - Detection: Health checks, heartbeat
   - Recovery: Auto-restart, drain requests, route to healthy workers

4. **Network Partition:**
   - Cause: Network issues between services
   - Detection: Timeout, connection errors
   - Recovery: Retry with backoff, circuit breaker, failover

**Follow-up:**
1. "How do you implement graceful degradation when GPUs are at capacity?"
   - Expected: Request queuing, timeout, priority shedding, fallback to smaller model

2. "What's your strategy for rolling updates without downtime?"
   - Expected: Blue-green deployment, canary releases, gradual rollout

### 7. Performance Debugging (If time permits)

**Question:**
"If you notice that P99 latency has increased from 80ms to 150ms, how would you debug this?"

**Debugging Approach:**
1. **Check Recent Changes:**
   - Model updates, config changes, traffic patterns
   - Deployment logs, git history

2. **Analyze Metrics:**
   - Breakdown latency: queuing, inference, post-processing
   - Check GPU utilization, memory usage
   - Look for correlations (time of day, specific models)

3. **Distributed Tracing:**
   - Trace slow requests end-to-end
   - Identify bottleneck component
   - Check for outliers

4. **Hypothesis Testing:**
   - Is it batch size related? (check batch size distribution)
   - Is it model-specific? (compare across models)
   - Is it load-related? (compare at different QPS)

**Expected Tools:**
- Prometheus queries for metrics
- Jaeger/Zipkin for distributed tracing
- GPU profiler (nsys, nvprof)
- Application logs (ELK)

## Scenario Variations & Extensions

### Extension 1: Structured Output Requirements

**Question:**
"What if you need to support structured outputs (e.g., JSON, XML) with guaranteed format compliance?"

**Discussion Points:**
- **Constrained Decoding:** Grammar-based sampling
- **Post-processing Validation:** Verify and retry
- **Function Calling:** Specialized model fine-tuning
- **Performance Impact:** Slower generation, more retries

**Tools:**
- Outlines library for constrained generation
- Grammar-based samplers
- Validation schemas (JSON Schema, Pydantic)

### Extension 2: Safety & Content Moderation

**Question:**
"How would you add safety guardrails to prevent harmful outputs?"

**Discussion Points:**
- **Input Filtering:** Detect harmful prompts
- **Output Filtering:** Scan generated text
- **Moderation Models:** Separate classifier models
- **Performance Impact:** Additional latency (10-50ms)

**Architecture:**
```
Request -> Input Moderator -> LLM -> Output Moderator -> Response
```

**Techniques:**
- OpenAI Moderation API
- Custom classifiers
- Keyword filtering
- Perspective API

### Extension 3: Fine-tuning & Personalization

**Question:**
"How would you support customer-specific fine-tuned models?"

**Discussion Points:**
- **LoRA Adapters:** Lightweight fine-tuning, multiple adapters per base model
- **Model Registry:** Version control for custom models
- **Loading Strategy:** Dynamic loading, adapter swapping
- **Cost Model:** Charge premium for custom models

**Architecture:**
```
Base Model (Llama-70B)
├── Customer A LoRA Adapter
├── Customer B LoRA Adapter
└── Customer C LoRA Adapter
```

**Benefits:**
- Share base model weights (memory efficient)
- Quick adapter swapping (<1s)
- Maintain single infrastructure

## Common Pitfalls to Avoid

### Pitfall 1: Over-engineering Early

**Symptom:** Proposing complex distributed architecture for modest scale

**Example:**
"For 1000 QPS, I'd use a microservices architecture with 20+ services, service mesh, event-driven architecture..."

**Issue:** Too complex for requirements, maintenance burden

**Better Approach:** Start simple, scale as needed
- Begin with monolithic inference service
- Add components only when needed
- Justify each architectural decision

### Pitfall 2: Ignoring Cost

**Symptom:** Designing without considering cost implications

**Example:**
"We'll use 100 A100 GPUs for high availability..."

**Issue:** $300K+/month without justification

**Better Approach:**
- Calculate cost based on requirements
- Optimize for cost efficiency
- Discuss trade-offs (cost vs. performance)

### Pitfall 3: Neglecting Operational Concerns

**Symptom:** Pure technical design without operational considerations

**Missing:**
- Monitoring and alerting
- Deployment strategy
- Debugging tools
- Documentation

**Better Approach:**
- Include observability from the start
- Discuss deployment and rollback strategies
- Plan for troubleshooting

### Pitfall 4: Not Understanding LLM Specifics

**Symptom:** Treating LLM inference like any other ML inference

**Example:**
"We'll batch requests every 100ms and process together..."

**Issue:** Doesn't understand autoregressive generation, variable output length

**Better Approach:**
- Understand LLM inference characteristics
- Discuss continuous batching
- Consider KV cache implications

### Pitfall 5: Unrealistic Performance Expectations

**Symptom:** Proposing numbers without calculations

**Example:**
"We can handle 10,000 QPS with one A100..."

**Issue:** Physically impossible given model size and latency requirements

**Better Approach:**
- Calculate throughput based on model size and latency
- Use realistic numbers from benchmarks
- Show your math

## Interviewer Calibration Notes

### Scoring Guidelines

**Outstanding (L6/Senior Staff):**
- Comprehensive architecture with LLM-specific optimizations
- Accurate performance calculations
- Discusses multiple alternatives with trade-offs
- Proactively mentions operational concerns
- Deep understanding of continuous batching, memory management
- Can extend design to handle new requirements

**Strong (L5/Staff):**
- Solid architecture covering main components
- Understands batching and basic optimizations
- Can explain trade-offs
- Mentions monitoring and deployment
- May need prompting for advanced topics

**Good (L4/Senior):**
- Reasonable high-level design
- Basic understanding of LLM inference
- Can discuss main trade-offs
- Needs guidance on advanced topics
- May have gaps in performance calculation

**Needs Improvement (L3/Mid-level):**
- Generic ML serving architecture without LLM specifics
- Doesn't understand batching or memory management
- Can't calculate performance
- Misses operational concerns

### Time Allocation Recommendations

**For 60-minute interview:**
- 0-10 min: Requirements clarification
- 10-25 min: High-level architecture + components
- 25-45 min: Deep dive (2-3 topics from above)
- 45-55 min: Trade-offs, alternatives, extensions
- 55-60 min: Wrap-up, candidate questions

**Deep Dive Topic Selection:**
Based on candidate's background:
- **Systems background:** Focus on batching, autoscaling, distributed systems
- **ML background:** Focus on model optimization, quantization, memory management
- **Mixed:** Balanced discussion across topics

### Red Flags to Watch For

1. **Lack of clarifying questions** - Jumps to solution without understanding requirements
2. **No calculations** - Proposes numbers without justification
3. **Can't explain trade-offs** - Every decision is "obvious"
4. **Doesn't know fundamentals** - What is KV cache? What is tensor parallelism?
5. **Over-confidence** - Claims everything is "simple" or "easy"
6. **Analysis paralysis** - Spends entire time discussing options, never decides

### Green Flags to Look For

1. **Structured thinking** - Clearly organizes thoughts, uses frameworks
2. **Back-of-envelope calculations** - Validates proposals with math
3. **Trade-off analysis** - Weighs pros and cons
4. **Operational mindset** - Considers monitoring, debugging, deployment
5. **Iterative design** - Starts simple, adds complexity as needed
6. **Domain knowledge** - References papers, tools, best practices
7. **Communication** - Explains clearly, checks understanding

## Example Interview Flow

### Opening (2 minutes)

**Interviewer:**
"We're building a production LLM inference service. The goal is to serve multiple large language models with high throughput and low latency. Take a minute to think about what questions you'd like to ask."

### Clarification Phase (8 minutes)

**Good Candidate:**
1. "What's the expected traffic pattern? QPS? Peak vs average?"
2. "What models are we serving? Size? Architecture?"
3. "What are the latency requirements? P50? P99?"
4. "Are there different tiers of service? Different SLAs?"
5. "What's the budget constraint? Infrastructure cost ceiling?"
6. "Do we need to support streaming responses?"
7. "Any compliance or data residency requirements?"

**Interviewer provides:** Requirements from problem statement

### Design Phase (35 minutes)

**Candidate presents:**
1. High-level architecture diagram (5 min)
2. Component breakdown (10 min)
3. Deep dive into 2-3 components (20 min)

**Interviewer probes:**
- "Why did you choose vLLM over TensorRT-LLM?"
- "How does continuous batching work exactly?"
- "Walk me through the memory layout for a batch"
- "What happens when GPU runs out of memory?"

### Trade-offs Discussion (10 minutes)

**Interviewer:**
"What are the main trade-offs in your design?"

**Good Candidate discusses:**
- Latency vs throughput (batching strategy)
- Cost vs performance (GPU type, quantization)
- Complexity vs flexibility (custom vs off-the-shelf)
- Isolation vs efficiency (multi-tenancy)

### Extensions (5 minutes)

**Interviewer:**
"How would you extend this to support fine-tuned models per customer?"

**Candidate:** Discusses LoRA adapters, model registry, etc.

## Resources for Candidates

### Papers to Reference
- "Efficient Memory Management for Large Language Model Serving with PagedAttention" (vLLM)
- "Orca: A Distributed Serving System for Transformer-Based Generative Models"
- "Fast Inference from Transformers via Speculative Decoding"
- "FlashAttention: Fast and Memory-Efficient Exact Attention"

### Tools to Know
- vLLM, TensorRT-LLM, Text Generation Inference
- Ray Serve, Triton Inference Server
- Quantization: AWQ, GPTQ, SmoothQuant
- Monitoring: Prometheus, Grafana, Datadog

### Benchmarks to Reference
- vLLM benchmark numbers
- NVIDIA TensorRT-LLM performance data
- Model inference latency calculators
