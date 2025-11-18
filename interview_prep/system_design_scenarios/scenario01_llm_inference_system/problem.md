# Scenario 01: Production LLM Inference System Design

## Problem Statement

Design a production-grade LLM inference serving system for a company that wants to deploy multiple large language models to serve customer requests. The system should handle high throughput while maintaining low latency and supporting multiple model variants.

## Interview Context

**Time Allocation:** 45-60 minutes
- Clarifying questions: 5-10 minutes
- High-level design: 10-15 minutes
- Deep dive: 20-25 minutes
- Discussion & trade-offs: 10-15 minutes

**Complexity Level:** L5/L6 (Senior/Staff Engineer)

## Requirements

### Functional Requirements

1. **Multi-Model Support**
   - Support serving 3-5 different LLM models simultaneously
   - Models range from 7B to 70B parameters
   - Support for different model architectures (GPT, LLaMA, etc.)

2. **Request Handling**
   - RESTful API for synchronous inference
   - Streaming support for token-by-token generation
   - Support for batch inference requests

3. **Model Management**
   - Hot-swapping models without service interruption
   - Model versioning and rollback capabilities
   - Health checks and readiness probes

### Non-Functional Requirements

1. **Performance**
   - **Throughput:** Handle 1000+ queries per second (QPS)
   - **Latency:** P99 latency < 100ms for first token
   - **Latency:** P99 latency < 500ms for complete response (256 tokens)

2. **Scalability**
   - Horizontal scaling to support growing traffic
   - Handle traffic spikes (2x normal load)
   - Support for auto-scaling based on load

3. **Reliability**
   - 99.9% availability (3 nines)
   - Graceful degradation under load
   - Proper error handling and retry mechanisms

4. **Resource Efficiency**
   - Maximize GPU utilization (>70%)
   - Minimize memory overhead
   - Efficient request batching

### Constraints and Assumptions

1. **Infrastructure**
   - Cloud deployment (AWS/GCP/Azure)
   - Access to GPU instances (A100/H100)
   - Budget: ~$50K/month for compute

2. **Scale**
   - Initial: 100 QPS average, 500 QPS peak
   - 6 months: 500 QPS average, 1000 QPS peak
   - Geographic: Single region initially

3. **Team**
   - 2-3 ML engineers
   - 1-2 infrastructure engineers
   - DevOps support available

## Clarifying Questions (Expected)

Candidates should ask questions like:

1. **Workload Characteristics**
   - What's the typical input/output token length?
   - What's the ratio of different model usage?
   - Are there peak hours or seasonal patterns?

2. **Consistency Requirements**
   - Do we need deterministic outputs?
   - Can we cache responses?
   - How do we handle model updates?

3. **Security & Compliance**
   - Are there data privacy requirements?
   - Do we need request logging/auditing?
   - Authentication/authorization requirements?

4. **Operational**
   - What's the rollout strategy for new models?
   - Monitoring and alerting requirements?
   - SLA expectations from customers?

## Success Criteria

A strong candidate will:
- Ask clarifying questions about requirements
- Propose a clear, scalable architecture
- Discuss trade-offs between different approaches
- Consider operational aspects (monitoring, deployment)
- Identify potential bottlenecks and mitigation strategies
- Demonstrate knowledge of LLM inference optimization techniques

## Time Management Hints for Candidate

- **0-5 min:** Clarify requirements and constraints
- **5-20 min:** Present high-level architecture with components
- **20-40 min:** Deep dive into 2-3 critical components
- **40-50 min:** Discuss trade-offs, alternatives, and optimizations
- **50-60 min:** Address follow-up questions and edge cases

## Evaluation Focus Areas

1. **System Design Fundamentals** (25%)
   - Component decomposition
   - API design
   - Data flow understanding

2. **LLM-Specific Knowledge** (30%)
   - Understanding of inference optimization
   - Batching strategies
   - Memory management

3. **Scalability & Performance** (25%)
   - Horizontal/vertical scaling approaches
   - Bottleneck identification
   - Performance optimization

4. **Operational Excellence** (20%)
   - Monitoring and observability
   - Deployment strategies
   - Failure handling

## Difficulty Level: ★★★☆☆ (Medium)

This scenario tests fundamental system design skills with LLM-specific considerations. It's broad enough to explore multiple areas while being concrete enough to dive deep into specific components.
