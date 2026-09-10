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
startup; the effective count is `min(N, E - 1)` for `E` experts per layer.
The bank holds `layers x min(N, E - 1)` rows plus `top_k x decode tokens`
staging rows, so its storage is `(layers x min(N, E - 1) + staging) x row
bytes`; the planner's tables and scratch buffers are separate, small
allocations. Rows can move between layers at run time.

## How it works

- Loading: the four per-expert tensors (gate/up and down weights and their
  block scales) are allocated in pinned host memory instead of the GPU. The
  loader still moves one layer at a time to the GPU for the Marlin
  conversion and restores those tensors to pinned memory, which becomes the
  pool's source. The small per-expert global scales stay on the device and
  are copied into a pinned, contiguous host buffer when the pool is
  installed.
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
  every chunk, which adds host-read transfer cost per chunk; the dominant
  term of long-input prefill time has not been profiled.

## Example

Qwen3.8-Flash-Next NVFP4 with 258 rows per layer (about a 32 GiB bank),
`--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'`, 4096
context, one request: decode about 63 tok/s (median of three fresh server
launches). That number was measured on an RTX PRO 6000 Blackwell Max-Q
limited to 48 GiB, on a combination of this feature with the deferred PLE
rows change (01554/vllm#46 on top of #54129), which this model needs to
load its PLE table within that budget; it is not a measurement of this
branch alone. For reference, the existing `--offload-backend uva
--cpu-offload-gb 40` path gave about 7 tok/s in a single run on a
different base commit; see `benchmarks/expert_pool/README.md` for the exact
heads and conditions rather than reading the two as a controlled
comparison.
