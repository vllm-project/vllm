# Changelog

## `--moe-cpu-offload`

Added Case 1 MoE CPU offload support.

### Feature

- Adds one public flag: `--moe-cpu-offload`.
- Dense models ignore the flag and run normally.
- MoE models enable CPU-backed expert staging.
- Large MoE LLMs can reside on smaller GPUs by keeping expert weights in CPU
  memory and using GPU memory only for routing, KV cache, and active expert
  compute.
- Multiple large MoE LLM instances can reside on the same GPU more easily
  because inactive expert weights do not need to occupy GPU memory.
- Full expert weights stay in CPU memory as the source of truth.
- Router path and KV cache stay on GPU.
- Active experts are transferred passively after router computation identifies
  the routed expert set.
- Compute uses passive expert model loading: route first, copy only the needed
  active expert weights to GPU, compute, then release those GPU expert copies.
- GPU expert copies are retired/freed after active expert computation.
- If active experts cannot fit as one group, execution can be split into smaller
  passive waves.
- Before GPU expert transfer, free GPU memory is checked. If memory is
  insufficient, transfer waits 5 seconds and retries up to 10 times.

### Example Test Layout

- Endpoint 1: one `gemma-4-26B-A4B-it` MoE model on one 40GB GPU.
- Endpoint 2 and 3: two `gemma-4-26B-A4B-it` MoE model instances sharing one
  40GB GPU, each served from a separate vLLM endpoint.
- This validates the intended use case: passive expert loading lets large MoE
  models fit on smaller GPU memory budgets and allows multiple large MoE
  endpoints to colocate on the same GPU.

### Known Non-Fatal Warnings

- Performance can degrade versus all-GPU expert residency because active expert
  weights are copied from CPU memory to GPU memory during inference.
- vLLM may warn that model `generation_config.json` overrides default sampling
  parameters.
- vLLM may warn about missing optimized fused MoE config files and fall back to
  default MoE configs.
