# MoE Expert Pool

An opt-in way to serve a MoE model whose expert weights do not fit in GPU
memory: the expert weights stay in pinned host memory, and one GPU bank
shared by all MoE layers holds a subset of expert rows that is managed on
the device.

```bash
vllm serve <model> --moe-expert-pool-rows 258
```

`--moe-expert-pool-rows N` (config field `OffloadConfig.moe_expert_pool_rows`,
default `0` = off) sets how many expert rows per layer are resident at
startup. The bank size is `N x number of MoE layers x row bytes`; rows can
move between layers at run time.

## How it works

- Loading: the NVFP4 expert tensors are allocated in pinned host memory
  instead of the GPU. The loader still moves one layer at a time to the GPU
  for the Marlin conversion and restores the converted tensors to pinned
  memory, which becomes the pool's source.
- Installation: after every layer has been processed, the pool allocates the
  bank, fills each layer's initial rows and binds a Marlin consumer to the
  bank. The placement is frozen (gate closed) during profiling and CUDA
  graph capture and opened at the end of warm-up.
- Decode: a device-side step program looks up each routed expert in the
  bank, promotes misses (device LRU over all layers), copies the needed rows
  from the host source with fixed-grid kernels, and runs Marlin on the bank
  with logical expert alignment and a physical-row remap. No host-side
  routing readback or cache planning sits between routing and the GEMM, so
  the MoE layers stay inside the captured CUDA graph (`FULL_DECODE_ONLY`).
- Wider batches (prefill): resident experts are read from the bank and the
  rest directly from the pinned host source through its accelerator view.

## Requirements and limits

- ModelOpt NVFP4 checkpoints with the Marlin MoE backend
  (`--moe-backend marlin`). Other quantization methods or NVFP4 backends
  are rejected when the layer is built.
- No expert, data or sequence parallelism.
- `N` must be at least `top_k` and every MoE layer must have the same expert
  count and top-k.
- Prefill of long inputs reads the non-resident experts from host memory on
  every chunk; expect prefill to be bounded by host-to-device bandwidth.

## Example

Qwen3.8-Flash-Next NVFP4 on a 48 GiB GPU budget with 258 rows per layer
(about a 32 GiB bank): decode about 63 tok/s at 4096 context with
`--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'`, versus
about 7 tok/s with `--offload-backend uva --cpu-offload-gb 40`. A
reproducible client and the exact launch flags are in
`benchmarks/expert_pool/`.
