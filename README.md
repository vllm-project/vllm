# vLLM 3080 Ti PCIe Definitive Edition

This fork is an opinionated vLLM runtime project targeting **4x RTX 3080 Ti 12GB on PCIe**.

It is inspired by the hardware-focused approach of `weicj/vLLM-2080Ti-Definitive`, but the optimization target here is different:

- **Ampere / SM86**, not Turing / SM75
- **4x 3080 Ti 12GB over PCIe**, not dual 2080 Ti 22GB + NVLink
- **communication-aware parallelism**, not NVLink-first tensor-parallel assumptions
- **practical route presets for local inference**, not one-size-fits-all upstream defaults

## Project goals

This fork is being shaped around a few concrete goals:

1. Improve real-world inference speed on 4x 3080 Ti PCIe hosts.
2. Reduce avoidable multi-GPU communication overhead in tensor-parallel serving.
3. Provide reproducible launch profiles for common model/quantization/KV-cache routes.
4. Make topology-aware deployment easier with helper scripts and documentation.
5. Preserve compatibility with upstream vLLM wherever possible while documenting local tuning choices.

## Initial scope

The first phase of this fork focuses on project scaffolding and reproducible operations rather than immediately rewriting low-level kernels.

Initial deliverables include:

- 3080 Ti PCIe project documentation
- topology probe tooling
- communication-aware launch/profile design
- benchmark workflow for TP=2 and TP=4 routes
- profile templates for common Qwen/Llama-class deployments

## Optimization philosophy

For this hardware, the central bottleneck is often **PCIe communication**, not only raw compute.

That means the main tuning questions are:

- when `tensor_parallel_size=2` beats `tensor_parallel_size=4`
- when weight quantization reduces enough memory pressure to avoid expensive communication
- when INT8 KV or more aggressive KV compression increases useful context without collapsing decode speed
- how GPU rank mapping should follow actual PCIe topology
- which routes are best for single-request decode, long-context text, or multiple isolated workspaces

## Planned project structure

This fork will introduce and/or standardize the following areas:

- `docs/design/3080ti_pcie_architecture.md`
- `docs/deployment/3080ti_pcie_topology.md`
- `scripts/topo_probe.sh`
- `profiles/` style route presets for 3080 Ti PCIe hosts
- benchmark guidance for comparing TP, KV, and quantization routes

## Status

Current branch: `sm86-4x3080ti-pcie-init`

This branch is the initial bootstrap for a dedicated 3080 Ti PCIe serving project.

## Upstream and inspiration

- Upstream base: `vllm-project/vllm`
- Hardware-focused inspiration: `weicj/vLLM-2080Ti-Definitive`

The purpose of this fork is **not** to mechanically transplant SM75-specific optimizations into Ampere.
Instead, it is to reuse the same engineering discipline — route-based tuning, launch reproducibility, and hardware-aware benchmarking — for a very different multi-GPU topology.
