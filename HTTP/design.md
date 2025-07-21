# Design Document  
**Hybrid Tensor and Token Parallelism for Distributed LLM Inference**

## Overview

This document outlines the design and implementation plan for a new parallelism architecture to accelerate LLM inference on distributed multi-node multi-GPU systems. The primary goal is to improve throughput and memory efficiency during large batch, long-sequence decoding.

---

## Motivation

Profiling large-scale LLM inference workloads reveals that **attention becomes the dominant bottleneck during the decode stage**, especially under large batch sizes and long sequence lengths. Unlike the prefill stage, decoding requires heavy memory I/O due to **KV cache management**, where each request maintains its own unique cache. This leads to:

- Increased memory pressure
- Higher I/O requirements for KV cache lookups and writes
- Limited scalability with standard tensor and data parallel setups

**Key insight:**  
MLP layers are relatively compute-bound and scale well under existing tensor parallel systems, but **attention layers require additional memory and I/O scaling**. Our solution is to allocate more GPUs specifically for attention to increase compute, memory, and bandwidth capacity **only for attention**, while keeping the MLP layers lightweight.

---

## Proposed Architecture: Hybrid Tensor and Token Parallelism (HTTP)

We propose a **Hybrid Tensor and Token Parallelism (HTTP)** architecture that combines:

- **Standard tensor parallelism** for feedforward and projection layers  
- **Pipeline parallelism** for model stage splitting  
- **Token parallelism (non-replicated data parallelism)** specifically for attention

### Key Design Principles

1. **Attention is sharded across more GPUs** than MLP layers to handle KV cache bottlenecks.
2. **MLP layers and attention projections (QKV, output projection)** are processed by the **root rank** in each token parallel group.
3. Other ranks in the token parallel group **do not hold model weights**, instead they allocate their memory to cache and process attention computation on their respective KV partitions.

---

## Implementation Details

### Process Group Setup

- The system initializes **tensor parallel** and **pipeline parallel** groups as usual.
- An additional **token parallel process group** (reusing vLLM-style data parallel groups) is created.
- **No model replication** is performed in token parallel groups.  
  Only the root rank holds the model weights.

### Computation Flow

1. **MLP Layers:**  
   The entire batch is processed by the **root rank** of each token parallel group.

2. **Attention Layers:**
   - **QKV projection** is performed by the root rank.
   - The resulting tensors are **scattered** across token parallel group ranks.
   - Each rank computes attention over its batch partition, using the KV cache it holds.
   - The outputs are **gathered** back to the root rank for the final projection.

---

## Assumptions and Scope

1. We assume a **disaggregated prefill and decode system**.
2. **Prefill workers** populate the KV cache, and attention workers in this architecture consume it.
3. This design **only targets the decode stage** of inference.
4. **KV cache management and load balancing** will be handled by a dedicated batch manager (to be implemented later).

---

## Prototype Status

A **proof-of-concept implementation** has been developed in:  
`HTTP/prototype/hamp_attention.py`

This prototype is based on **FlashAttention**, with the following design choices:

- **Root ranks hold the model weights**  
- **Non-root ranks focus exclusively on KV cache and attention computation**, freeing memory for larger batches and longer context lengths

---

## Benefits

- **Improved scalability** of attention layers in the decode stage
- **Memory efficiency** by avoiding weight replication in token parallel groups
- **Increased throughput and concurrency** for LLM serving workloads
- **Modularity**, allowing seamless integration with existing tensor and pipeline parallel systems
