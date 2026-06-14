# 4x 3080 Ti PCIe Architecture Notes

This document defines the architecture and optimization goals for the `vllm-3080ti` fork.

## Target hardware

- 4x NVIDIA RTX 3080 Ti 12GB
- Ampere / SM86
- PCIe interconnect (no NVLink assumption)
- Linux host with CUDA-capable driver stack

## Design premise

The main difference from dual-2080Ti-NVLink projects is that this fork must treat **inter-GPU communication as a first-class bottleneck**.

On 4x 3080 Ti PCIe systems, increasing tensor parallelism can reduce memory pressure while simultaneously increasing per-token communication cost. The best route therefore depends on:

- model size
- weight format
- KV cache format
- context length
- batch and concurrency shape
- actual PCIe topology

## Primary optimization axes

### 1. Parallelism selection

This fork will treat parallelism as a tunable route parameter, not a fixed default.

Initial intended routes:

- `TP=1`: small models, lowest communication overhead
- `TP=2`: likely default sweet spot for many 14B-class models
- `TP=4`: capacity-first route for larger models or heavier contexts

### 2. Communication-aware serving

The fork should prefer configurations that reduce avoidable collectives during decode.

Areas to evaluate:

- rank mapping based on PCIe topology
- TP=2 versus TP=4 on identical weights
- quantization routes that avoid a jump from TP=2 to TP=4
- custom all-reduce on/off behavior
- NCCL environment tuning per topology class

### 3. Memory-efficiency routes

Because each GPU has only 12GB VRAM, memory pressure is a major design constraint.

The fork will emphasize:

- INT4 / GPTQ / AWQ routes
- FP8 routes where stable and performant
- INT8 KV routes for longer practical context
- explicit capacity/speed tradeoff documentation

### 4. Decode-first tuning

For local interactive inference, decode throughput and tail latency matter more than peak synthetic prefill numbers alone.

This fork will evaluate routes by:

- single-request decode speed
- long-context stability
- communication overhead sensitivity
- practical API serving usability

## Non-goals for phase 1

Phase 1 is not a kernel-rewrite branch.

It does **not** initially promise:

- a full SM86-specific kernel stack rewrite
- immediate replacement of upstream attention/GEMM kernels
- universal speedups for every model architecture

Instead, phase 1 builds the operational framework to discover which routes deserve deeper low-level work.

## Planned outputs

- topology probe script
- deployment guide for 4x 3080 Ti hosts
- initial profile taxonomy
- benchmark recipes for TP/KV/quantization sweeps
- branch-local documentation of Ampere-specific tuning assumptions
