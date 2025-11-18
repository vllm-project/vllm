# Scenario 02: Distributed LLM Serving at Scale

## Problem Statement

Design a distributed serving system for extremely large language models (100B+ parameters) that spans multiple GPUs and potentially multiple nodes. The system must handle models too large to fit on a single GPU while maintaining acceptable latency and high availability.

## Interview Context

**Time Allocation:** 45-60 minutes
- Clarifying questions: 5-10 minutes
- High-level design: 10-15 minutes
- Deep dive: 20-25 minutes
- Discussion & trade-offs: 10-15 minutes

**Complexity Level:** L5/L6 (Senior/Staff Engineer)

## Requirements

### Functional Requirements

1. **Model Support**
   - Serve models with 100B - 175B parameters
   - Support for dense and MoE (Mixture of Experts) models
   - Handle models that don't fit in single GPU memory (80GB A100)

2. **Distribution Strategy**
   - Distribute model across 8-16 GPUs
   - Support both single-node (8 GPUs) and multi-node (2-4 nodes)
   - Efficient tensor placement and communication

3. **Inference Serving**
   - Streaming and batch inference
   - Support for long context (up to 32K tokens)
   - Dynamic request routing

### Non-Functional Requirements

1. **Performance**
   - **Latency:** P99 < 300ms for first token (acceptable for large models)
   - **Throughput:** 50+ queries per second per model instance
   - **Context Length:** Support up to 32K tokens

2. **Scalability**
   - Scale horizontally by adding more GPU clusters
   - Support 2-4 model replicas for high availability
   - Handle 200-500 concurrent requests

3. **Reliability**
   - **Availability:** 99.95% uptime
   - Handle GPU failures gracefully
   - Support rolling updates without downtime
   - Checkpoint and recovery for long-running requests

4. **Resource Efficiency**
   - Minimize inter-GPU communication overhead
   - Efficient memory usage across GPUs
   - Load balancing across replicas

### Constraints and Assumptions

1. **Infrastructure**
   - Cloud deployment with GPU instances
   - Access to p4d (8x A100 80GB with NVLink)
   - High-bandwidth networking (400Gbps InfiniBand or equivalent)

2. **Scale**
   - Initial: 50 QPS average
   - 6 months: 200 QPS average
   - Budget: $100K-150K/month for compute

3. **Model Characteristics**
   - 175B parameters (GPT-3 scale)
   - FP16 weights: ~350GB
   - INT4 quantized: ~90GB
   - 96 layers, 12,288 hidden size

4. **Network**
   - Single region deployment initially
   - May expand to multi-region later
   - Cross-datacenter latency: 50-100ms

## Clarifying Questions (Expected)

Candidates should ask questions like:

1. **Model Architecture**
   - Is this a dense model or MoE?
   - What's the model architecture (layers, hidden size, attention heads)?
   - Do we need to support multiple model architectures?

2. **Workload Patterns**
   - What's the typical context length?
   - What's the ratio of long vs short contexts?
   - Are there batching opportunities?

3. **Fault Tolerance**
   - What happens if one GPU fails in the cluster?
   - Do we need checkpointing for long requests?
   - Recovery time objectives (RTO)?

4. **Performance Trade-offs**
   - Is latency or throughput more important?
   - Can we sacrifice some latency for better throughput?
   - Memory vs speed trade-offs?

5. **Cost Considerations**
   - What's the target cost per token?
   - Can we use spot instances?
   - Trade-off between replication and cost?

## Technical Challenges to Address

1. **Model Parallelism**
   - How to split 175B model across GPUs?
   - Tensor parallelism vs pipeline parallelism trade-offs
   - Optimal parallelism degree (4-way, 8-way, 16-way)

2. **Communication Overhead**
   - All-reduce operations in tensor parallelism
   - Point-to-point communication in pipeline parallelism
   - Network bandwidth requirements

3. **Memory Management**
   - KV cache distribution across GPUs
   - Activation memory for large contexts
   - Memory-efficient attention mechanisms

4. **Fault Tolerance**
   - GPU failure detection and recovery
   - Request state preservation
   - Graceful degradation strategies

5. **Load Balancing**
   - Routing requests to healthy replicas
   - Avoiding hot spots
   - Handling heterogeneous request sizes

## Success Criteria

A strong candidate will:
- Understand different parallelism strategies (TP, PP, DP)
- Calculate communication overhead and bandwidth requirements
- Design fault-tolerant distributed system
- Consider memory constraints across GPU topology
- Propose monitoring and debugging strategies for distributed system
- Discuss trade-offs between parallelism approaches

## Time Management Hints for Candidate

- **0-5 min:** Clarify model size, parallelism requirements, scale
- **5-20 min:** Present parallelism strategy and distributed architecture
- **20-40 min:** Deep dive into communication, fault tolerance, memory
- **40-50 min:** Discuss trade-offs, alternatives, optimization
- **50-60 min:** Address edge cases and scaling scenarios

## Evaluation Focus Areas

1. **Distributed Systems Knowledge** (30%)
   - Parallelism strategies
   - Fault tolerance
   - Consistency and coordination

2. **LLM-Specific Expertise** (30%)
   - Model parallelism techniques
   - Memory management at scale
   - Communication patterns

3. **Performance Engineering** (20%)
   - Latency analysis
   - Bandwidth calculations
   - Optimization strategies

4. **System Reliability** (20%)
   - Failure scenarios
   - Monitoring and debugging
   - Graceful degradation

## Difficulty Level: ★★★★☆ (Hard)

This scenario requires deep understanding of distributed systems, model parallelism, and production serving. It tests ability to design complex systems with multiple failure modes and performance considerations.

## Key Concepts to Demonstrate

**Must Know:**
- Tensor Parallelism (TP)
- Pipeline Parallelism (PP)
- Data Parallelism (DP)
- NVLink vs InfiniBand communication
- All-reduce operations
- Memory distribution strategies

**Nice to Have:**
- Mixture of Experts (MoE) serving
- Sequence parallelism
- Context parallelism
- Flash Attention for memory efficiency
- ZeRO optimizer strategies

## Common Mistakes to Avoid

1. **Proposing single-GPU solution** - 175B model doesn't fit
2. **Ignoring communication overhead** - Major performance bottleneck
3. **Not considering fault tolerance** - Critical for production
4. **Unrealistic latency expectations** - 300ms P99 is reasonable, not 50ms
5. **Missing memory calculations** - Must account for KV cache, activations
