# MoE Expert Pool

An opt-in way to serve a MoE model whose expert weights do not fit in GPU
memory: the expert weights stay in pinned host memory, and one GPU bank
shared by all MoE layers holds a subset of expert rows that is managed on
the device.

```bash
vllm serve <model> --moe-backend marlin --moe-expert-pool-rows 258
```

`--moe-expert-pool-rows N` (config field `OffloadConfig.moe_expert_pool_rows`,
default `0` = off) sets how many expert rows per layer are resident at
startup; the effective count is `min(N, E - 1)` for `E` experts per layer.
The bank holds `layers x min(N, E - 1)` rows plus `top_k x decode tokens`
staging rows, so its storage is `(layers x min(N, E - 1) + staging) x row
bytes`; the planner's tables and scratch buffers are separate, small
allocations. Rows can move between layers at run time.

Prefill staging is enabled by default when the pool is enabled. It uses
two full layers' worth of extra GPU memory, leaving less room for the KV
cache. Add `--no-moe-expert-pool-prefill-staging` to disable it.

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
- Wider batches (prefill): resident experts are read from the bank. Marlin
  reads its weights once per M-block, so a non-resident expert read through
  the host view would cross PCIe several times per step. Uncaptured prefill
  steps of at least 512 tokens copy each full layer into a device staging
  buffer by DMA, one layer ahead of the compute, and read non-resident rows
  from there. Copying resident rows too avoids reading the device-managed
  placement back to the host. Smaller wide batches read their routed rows
  through the host view. `--no-moe-expert-pool-prefill-staging` turns staging
  off and saves its two full-layer buffers of GPU memory.

## Requirements and limits

- ModelOpt NVFP4 checkpoints with the Marlin MoE backend
  (`--moe-backend marlin`). NVFP4 layers on other backends are rejected
  when the layer is built; MoE layers with another quantization method
  (e.g. an MTP head the checkpoint leaves in BF16) are not pooled and keep
  their experts on the GPU. Specialized ModelOpt MoE subclasses, such as
  MuseSpark's post-expert normalization path, are not supported.
- Speculative decoding: the decode lanes are sized for
  `1 + num_speculative_tokens` tokens per request. A BF16 draft MoE needs
  its own backend, e.g. `--speculative-config '{"method": "mtp",
  "num_speculative_tokens": 3, "moe_backend": "triton"}'`.
- No expert, data or sequence parallelism.
- `N` must be at least `top_k` and every MoE layer must have the same expert
  count and top-k.
- Each bank weight tensor must fit in 32 GiB, the range Marlin MoE indexes
  with 32-bit offsets (e.g. 8192 rows of 4 MiB gate/up weights); a larger
  bank is rejected at installation with the largest `N` that fits.
- Every staged prefill step moves each full layer over PCIe once, including
  resident rows, so staging can transfer more data when most experts are
  resident. Larger `--max-num-batched-tokens` amortizes this transfer over
  more tokens. Staging does not synchronize the host with the GPU.

## Example

Qwen3.8-Flash-Next NVFP4 with 258 rows per layer (about a 32 GiB bank),
`--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'`, 4096
context, one request, and 768 MiB fixed KV cache: decode about 65 tok/s
(median of three fresh server launches per condition). Measurements used
an RTX PRO 6000 Blackwell Max-Q with approximately 48 GiB available GPU
memory and a 100 GiB host-memory limit, at production commit `27e2470a`.
They are not measurements on RTX 6000 Ada.

With 512-token prefill chunks, staging used an additional 2.64 GiB of GPU
memory and had a median measured-request TTFT of 9.3 seconds, versus
6.9 seconds with staging disabled; decode throughput was similar. This
workload does not establish a long-input prefill speedup.

See `benchmarks/expert_pool/README.md` for the exact launch, sampling,
per-run results, and the separately identified historical deferred-PLE
measurements. Generated continuations differed across fresh launches, so
these are response-throughput measurements for fixed input prompts.
