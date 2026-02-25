# Relating the JAX/XLA/TPU PDF to vLLM Extensions (KV Cache Tiering)

This document connects ideas from **“Decoding the JAX AI Stack: From Python Code to Silicon (JAX / XLA / TPU)”** (LLMsys 12) to the **vllm_extensions** tiered KV cache work and to vLLM inference in general.

## 1. Memory tiering (HBM vs VMEM vs our GPU vs CPU)

**In the PDF**  
- TPU Ironwood has **HBM** (high capacity, lower bandwidth) and **VMEM** (on-chip, high bandwidth, small capacity).  
- XLA assigns tensors to **memory spaces** (e.g. S(0)=HBM, S(1)=VMEM) and uses **copy-start / copy-done** to move data between tiers.  
- Keeping hot data in VMEM and streaming through it avoids HBM bottlenecks (“math is cheap; bandwidth is expensive”).

**In vllm_extensions**  
- We implement **two tiers**: GPU (fast, limited) and CPU pinned (slower, larger).  
- “Hot” KV cache blocks stay on GPU; “cold” ones are **evicted** to CPU and **fetched** back when needed.  
- So we reuse the same **tiered memory** idea: capacity vs speed trade-off and explicit movement between tiers.

**Takeaway**  
Same principle (tiered memory + controlled data movement); different hardware (GPU/CPU vs TPU HBM/VMEM) and different compiler (manual/async in Python/CUDA vs XLA-generated copies).

---

## 2. Async data movement and overlapping with compute

**In the PDF**  
- **copy-start** kicks off a DMA transfer (e.g. HBM → VMEM) and returns immediately; **copy-done** is used as a barrier when the data is needed.  
- Computation is scheduled so that while one kernel runs, the next kernel’s data is being copied. This **hides latency** and keeps utilization high.

**In vllm_extensions**  
- We use **CUDA streams** for GPU↔CPU copies: `async_transfer_to_gpu()`, and we **prefetch** blocks from CPU to GPU before they are needed.  
- The **SequentialPrefetcher** issues async transfers for “next” blocks so that by the time the scheduler needs a block, it may already be on GPU.  
- So we aim for the same pattern: **overlap data movement with compute** instead of blocking on every transfer.

**Takeaway**  
PDF: compiler-managed async copies (copy-start/copy-done). Ours: application-managed async transfers and prefetch. Goal is the same: overlap transfer and compute.

---

## 3. Prefetching and “what to bring in next”

**In the PDF**  
- XLA decides **when** to issue copy-start so that data lands in VMEM just in time for the kernel that consumes it.  
- Scheduling and layout (tiling, fusion) are chosen so that memory traffic and compute are balanced.

**In vllm_extensions**  
- We **predict** which blocks will be needed next (e.g. sequential order in the block table) and **prefetch** those from CPU to GPU.  
- Prediction is currently simple (sequential); it could be extended (e.g. attention-based or cost-aware) similar to how XLA uses the graph to decide what to load next.

**Takeaway**  
Both systems decide “what to move next” to reduce stalls: XLA at compile time from the HLO graph; we at runtime from access patterns and (optionally) attention.

---

## 4. Tiling and block granularity

**In the PDF**  
- Tensors are **tiled** (e.g. 8×128) to match hardware (VREG, MXU).  
- Tiling defines how data is laid out and moved in chunks; the compiler optimizes for those chunk sizes.

**In vLLM / vllm_extensions**  
- KV cache is managed in **blocks** (fixed-size token blocks).  
- Our “block” is the unit of allocation, eviction, and transfer (GPU↔CPU).  
- So we also work in **chunks**; the chunk is logical (sequence block) rather than a compiler-chosen tile, but the idea of “move and manage in fixed-size units” is the same.

**Takeaway**  
Both use chunked/tiled data for efficient movement and allocation; we do it at the KV-cache block level.

---

## 5. Attention and fusion (PDF) vs attention-aware eviction (ours)

**In the PDF**  
- Attention is implemented as **fused** kernels (e.g. Q×Kᵀ + mask + softmax + P×V) with careful layout and rematerialization to avoid storing the full 1024×1024 softmax matrix in VMEM.  
- Layout (e.g. column-major, T(8,128)) is chosen so the VPU/MXU can stream data efficiently.

**In vllm_extensions**  
- We don’t change the attention kernel. We use **attention information** to decide **which KV blocks to keep on GPU**: **AttentionWeightedEvictionPolicy** and **HybridEvictionPolicy** evict blocks with lower cumulative attention scores.  
- So “attention” in the PDF is about **how** to compute; for us it’s about **which** cache blocks are more valuable to keep in the fast tier.

**Takeaway**  
PDF optimizes the attention op itself; we optimize which parts of the KV cache stay in the fast tier using attention (and recency/frequency).

---

## 6. Where the PDF goes further (and we don’t)

- **Compilation and fusion**: XLA traces JAX → JAXpr → StableHLO → HLO → LLO → VLIW and fuses ops (e.g. mask+softmax) and does rematerialization. We don’t change the vLLM model or kernels; we only manage where KV blocks live.  
- **Sharding (GSPMD / shard_map)**: The PDF describes distributing tensors and compute across many TPUs. Our extensions are about **single-node** GPU+CPU tiering, not multi-device sharding. vLLM has its own parallelism (tensor/pipe/data) elsewhere.  
- **Precision and layout**: PDF discusses bf16/fp32, layout (minor-to-major, T(8,128)), and how that affects the MXU/VPU. We work at block level and don’t change tensor layout or dtype.

---

## 7. Summary table

| Concept in PDF              | In vllm_extensions / vLLM                          |
|----------------------------|----------------------------------------------------|
| Memory tiering (HBM/VMEM)  | GPU (fast) vs CPU (capacity) tiering               |
| copy-start / copy-done     | Async CUDA transfers + prefetch                  |
| Overlap transfer & compute| Prefetch next blocks while current step runs      |
| Tiling / chunks            | KV cache blocks as unit of move/evict             |
| “What to keep hot”         | Eviction policies (LRU, attention, hybrid)         |
| Compiler-driven layout     | We don’t change kernel layout; we manage placement|

---

## 8. Possible future directions inspired by the PDF

- **Roofline-style reasoning**: For our GPU↔CPU transfers, estimate “arithmetic intensity” of a decode step vs transfer cost to decide how aggressive prefetch or eviction should be.  
- **Cost-aware prefetch**: Like XLA’s scheduling, use a simple cost model (e.g. transfer time vs compute time) to decide how many blocks to prefetch and in what order.  
- **Structured tracing**: Export block access/eviction/fetch traces (we already have **AccessTracer**) and analyze them similarly to how the PDF uses XProf/Pacchetto for TPU, to see if we’re transfer-bound or compute-bound.

These markdown files (README.md and this doc) summarize the vllm_extensions project and how it relates to the JAX/XLA/TPU material in the PDF.
