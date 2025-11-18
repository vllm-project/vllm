# LLM System Design Interview Scenarios

## Overview

This directory contains 10 comprehensive system design interview scenarios focused on LLM inference systems, targeting Senior/Staff engineer positions at companies like NVIDIA, OpenAI, and Anthropic.

Each scenario is designed for a 45-60 minute interview and includes:
- **problem.md** - Design challenge with requirements
- **solution.md** - Complete architecture and implementation details
- **discussion_points.md** - Key topics and follow-up questions
- **rubric.md** - Evaluation criteria and scoring guidelines

## Scenarios Summary

### Scenario 01: LLM Inference System Design ★★★☆☆ (Medium)
**Focus:** Production LLM serving with multi-model support
**Key Topics:** Batching, API design, model management, monitoring
**Requirements:** 1000 QPS, <100ms P99 latency

**Complexity Rating:** Medium
**Target Level:** L5/L6 (Senior/Staff)
**Key Skills:** System design fundamentals, LLM inference basics, batching strategies

---

### Scenario 02: Distributed LLM Serving ★★★★☆ (Hard)
**Focus:** Serving 100B+ parameter models across multiple GPUs/nodes
**Key Topics:** Tensor parallelism, pipeline parallelism, communication optimization
**Requirements:** 50+ QPS, <300ms P99, 175B parameters

**Complexity Rating:** Hard
**Target Level:** L5/L6 (Senior/Staff)
**Key Skills:** Distributed systems, model parallelism, GPU networking, fault tolerance

---

### Scenario 03: KV Cache Management System ★★★★☆ (Hard)
**Focus:** Efficient memory management for LLM KV cache
**Key Topics:** PagedAttention, prefix caching, copy-on-write, eviction policies
**Requirements:** >90% memory utilization, <5% fragmentation

**Complexity Rating:** Hard
**Target Level:** L5/L6 (Senior/Staff)
**Key Skills:** Memory management, data structures, OS concepts, optimization

---

### Scenario 04: Request Batching & Scheduling ★★★★☆ (Hard)
**Focus:** Intelligent batching and priority-based scheduling
**Key Topics:** Continuous batching, fairness, SLA enforcement, preemption
**Requirements:** P99 <100ms (high priority), >80% GPU utilization

**Complexity Rating:** Hard
**Target Level:** L5/L6 (Senior/Staff)
**Key Skills:** Scheduling algorithms, queue management, fairness, performance optimization

---

### Scenario 05: Multi-Tenant LLM Platform ★★★★★ (Very Hard)
**Focus:** Serve multiple customers with isolation and SLA guarantees
**Key Topics:** Resource isolation, quota enforcement, cost attribution, custom models
**Requirements:** Gold/Silver/Bronze SLAs, 100+ tenants, accurate billing

**Complexity Rating:** Very Hard
**Target Level:** L5/L6 (Senior/Staff)
**Key Skills:** Multi-tenancy, resource management, billing systems, isolation

---

### Scenario 06: LLM Inference Optimization Pipeline ★★★★☆ (Hard)
**Focus:** End-to-end optimization with quantization and kernel optimization
**Key Topics:** AWQ/GPTQ quantization, Flash Attention, TensorRT, A/B testing
**Requirements:** 2-4x speedup, <2% accuracy loss

**Complexity Rating:** Hard
**Target Level:** L5/L6 (Senior/Staff)
**Key Skills:** Model optimization, quantization, kernel optimization, performance tuning

---

### Scenario 07: High-Availability LLM Service ★★★★★ (Very Hard)
**Focus:** Achieve 99.99% uptime with multi-region deployment
**Key Topics:** Failover, health monitoring, zero-downtime deployment, disaster recovery
**Requirements:** 99.99% availability, RTO <5min, RPO <1min

**Complexity Rating:** Very Hard
**Target Level:** L5/L6 (Senior/Staff)
**Key Skills:** High availability, distributed systems, reliability engineering, operations

---

### Scenario 08: Cost Optimization for LLM Serving ★★★★☆ (Hard)
**Focus:** Reduce infrastructure costs by 50% while maintaining performance
**Key Topics:** Spot instances, quantization, model routing, caching, batching
**Requirements:** 50% cost reduction, P99 <200ms maintained

**Complexity Rating:** Hard
**Target Level:** L5/L6 (Senior/Staff)
**Key Skills:** Cost optimization, cloud economics, resource management, trade-off analysis

---

### Scenario 09: A/B Testing Framework for LLMs ★★★★☆ (Hard)
**Focus:** Compare model versions with statistical rigor and safety
**Key Topics:** Traffic routing, quality measurement, statistical analysis, gradual rollout
**Requirements:** Statistical significance, automatic rollback, <5% overhead

**Complexity Rating:** Hard
**Target Level:** L5/L6 (Senior/Staff)
**Key Skills:** Experimentation, statistics, quality measurement, safety mechanisms

---

### Scenario 10: Edge LLM Deployment ★★★★☆ (Hard)
**Focus:** Deploy LLMs on edge devices with strict resource constraints
**Key Topics:** Model compression, TensorRT, edge/cloud orchestration, power management
**Requirements:** <500ms P99, <10GB disk, <30W power

**Complexity Rating:** Hard
**Target Level:** L5/L6 (Senior/Staff)
**Key Skills:** Edge computing, model compression, resource constraints, hybrid systems

---

## Difficulty Distribution

- **Medium (1):** Scenario 01
- **Hard (7):** Scenarios 02, 03, 04, 06, 08, 09, 10
- **Very Hard (2):** Scenarios 05, 07

## Topic Coverage

### Core LLM Inference Concepts
- Continuous batching (01, 04)
- KV cache management (03)
- Model parallelism (02)
- Quantization (06, 08, 10)

### System Design Fundamentals
- Distributed systems (02, 05, 07)
- High availability (07)
- Cost optimization (08)
- Multi-tenancy (05)

### Advanced Topics
- A/B testing & experimentation (09)
- Edge deployment (10)
- Inference optimization (06)
- Request scheduling (04)

## Usage Guidelines

### For Interviewers

1. **Selection:** Choose scenarios based on:
   - Candidate level (L4-L6)
   - Role focus (systems, ML, ML systems)
   - Time available (45-60 min)

2. **Customization:** Adjust requirements based on:
   - Company scale
   - Specific technologies
   - Time constraints

3. **Evaluation:** Use rubrics to:
   - Score consistently
   - Compare candidates
   - Provide feedback

### For Candidates

1. **Preparation:**
   - Study problem statements
   - Review solution architectures
   - Practice explaining trade-offs

2. **During Interview:**
   - Ask clarifying questions
   - Start with high-level design
   - Deep dive into 2-3 components
   - Discuss trade-offs and alternatives

3. **Key Success Factors:**
   - Demonstrate LLM-specific knowledge
   - Show systems thinking
   - Quantify (latency, cost, throughput)
   - Consider operational aspects

## File Structure

```
system_design_scenarios/
├── README.md (this file)
├── scenario01_llm_inference_system/
│   ├── problem.md
│   ├── solution.md
│   ├── discussion_points.md
│   └── rubric.md
├── scenario02_distributed_llm_serving/
│   ├── problem.md
│   ├── solution.md
│   ├── discussion_points.md
│   └── rubric.md
...
└── scenario10_edge_deployment/
    ├── problem.md
    ├── solution.md
    ├── discussion_points.md
    └── rubric.md
```

## Key Concepts Across Scenarios

### Must-Know Topics
1. **Continuous Batching** - Iteration-level batching for LLMs
2. **PagedAttention** - Efficient KV cache memory management
3. **Model Parallelism** - TP, PP, DP for large models
4. **Quantization** - AWQ, GPTQ, INT4/INT8
5. **Flash Attention** - Memory-efficient attention

### Important Tools & Frameworks
- **vLLM** - High-throughput LLM serving
- **TensorRT** - NVIDIA inference optimization
- **Ray Serve** - Distributed serving framework
- **Megatron-LM** - Model parallelism library
- **DeepSpeed** - Training & inference optimization

## Recommended Study Path

### Beginner (Preparing for L4/Senior)
1. Start with Scenario 01 (basics)
2. Study vLLM architecture
3. Practice Scenario 04 (batching)

### Intermediate (Preparing for L5/Staff)
1. Master Scenarios 01, 04
2. Study Scenarios 02, 03, 06
3. Deep dive into model parallelism
4. Practice cost calculations

### Advanced (Preparing for L6/Senior Staff)
1. Master all medium/hard scenarios
2. Study Scenarios 05, 07 (very hard)
3. Read research papers (vLLM, Orca, FlashAttention)
4. Practice explaining trade-offs

## Additional Resources

### Papers
- "Efficient Memory Management for Large Language Model Serving with PagedAttention" (vLLM)
- "Orca: A Distributed Serving System for Transformer-Based Generative Models"
- "FlashAttention: Fast and Memory-Efficient Exact Attention"
- "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism"

### Tools Documentation
- vLLM: https://github.com/vllm-project/vllm
- TensorRT-LLM: https://github.com/NVIDIA/TensorRT-LLM
- Ray Serve: https://docs.ray.io/en/latest/serve/
- DeepSpeed Inference: https://www.deepspeed.ai/inference/

## Updates and Contributions

This collection is maintained as part of the vLLM learning project. Scenarios are based on real interview questions from leading AI companies and reflect production system design challenges.

**Last Updated:** November 2025

---

## Quick Reference: Scenario Selection Guide

| Scenario | Duration | Difficulty | Best For |
|----------|----------|------------|----------|
| 01 | 45-60 min | Medium | Warm-up, fundamentals |
| 02 | 60 min | Hard | Distributed systems focus |
| 03 | 45-60 min | Hard | Memory management focus |
| 04 | 45-60 min | Hard | Algorithms & scheduling |
| 05 | 60 min | Very Hard | Multi-tenancy expertise |
| 06 | 45-60 min | Hard | Optimization focus |
| 07 | 60 min | Very Hard | Reliability engineering |
| 08 | 45-60 min | Hard | Cost consciousness |
| 09 | 45-60 min | Hard | Experimentation platform |
| 10 | 45-60 min | Hard | Edge computing |

---

**Note:** All scenarios assume familiarity with:
- Basic machine learning concepts
- Distributed systems fundamentals
- Cloud infrastructure (AWS/GCP/Azure)
- Production system design principles
