# ZoomKV attention: experimental GPU-only and K+V CPU offload

> For serving parameters and benchmark procedures in Chinese, see
> [SERVING_AND_BENCHMARK.zh-CN.md](./SERVING_AND_BENCHMARK.zh-CN.md).

ZoomKV is an experimental vLLM V1 backend for sparse long-context decode.
Prefill, short-context decode, and unsupported decode shapes use dense
attention. GPU-only mixed batches can use sparse attention for the eligible
decode prefix and dense attention for the prefill suffix; offload mixed
batches restore cold pages and use dense attention. Pure single-token
long-context decode uses a fixed-width sparse path. This tree targets vLLM
v0.20.0 and is under performance testing; it has not been production
validated.

## What changed after `b3cb5f4d1`

- K+V CPU offload is now a directly testable mode. vLLM's physical page remains
  16 tokens, while completed retrieval-zone data is copied asynchronously D2H
  in 64-token logical units.
- Retrieval is no longer hierarchical Quest parent/child selection. It is one
  16-token `q·centroid` chunk-mean pass: keep Top-200 chunks, apply 8-token KIVI
  selection to the first 60 and 4-token selection to the rest, then retain the
  final Top-100 tokens.
- The offload hot path now has a persistent `physical_to_slot` map, direct
  physical retrieval, and one hybrid K+V gather kernel. Head dimensions 128 and
  256 use a 16-byte vectorized UVA path when layout requirements are met.
- Hybrid-cache block-id expansion/reuse invalidation was fixed. CPU pools are
  allocated only for full-attention KV caches, and offload state participates
  in zero/invalidate/free hooks.
- The Qwen3 performance path directly invokes `fused_qk_norm_rope`, keeps the
  long-context RoPE cache resident in FP32, and optionally emits per-layer NVTX
  ranges with `VLLM_ZOOMKV_LAYER_NVTX=1`.

The removed configuration names `zoomkv_quest_chunk`,
`zoomkv_quest_large_chunk`, `zoomkv_quest_large_ratio`,
`zoomkv_quest_small_ratio`, and `zoomkv_dense_ratio` are invalid and must not
be passed.

## Offload lifecycle

Only regular full-attention layers use ZoomKV; local/GDN layers are unchanged.
The CPU pool is created only for full-attention KV caches.

1. A completed retrieval-zone 64-token logical unit is asynchronously copied
   to pinned host memory. Its four 16-token GPU pages remain valid: **warm**.
2. On sparse decode, warm retrieval-zone pages are zeroed after D2H completes:
   **cold**. Sink, local-window, and in-flight write pages remain on GPU.
3. Hybrid sparse gather reads hot pages from GPU and cold pages directly from
   mapped pinned memory. One kernel gathers both K and V.
4. Before any dense reader (for example chunked prefill, mixed decode, or a
   prefix-cache hit), cold K+V pages are restored H2D: **warm** again. The CPU
   copy is retained, so a later cold transition only needs GPU zeroing.
5. Block reuse/free invalidates summaries, slot mappings, masks, and CPU slots.

`zoomkv_cpu_bytes_per_rank` is the total pinned **K+V** budget for each worker
rank, not a per-layer or per-tensor allowance. Pool exhaustion safely leaves
additional pages GPU-resident, reducing memory savings.

## Retrieval and routing

For every eligible decode step and KV group:

1. Mean-reduce its GQA query heads.
2. Score every retrieval-zone 16-token chunk by `q·centroid`.
3. Keep at most 200 chunks.
4. Run KIVI token selection with budget 8 for the leading 60 chunks and 4 for
   the remaining chunks.
5. Select the final Top-100 tokens and attend to
   `sink + local window + retrieved tokens`.

Dense attention is used below `zoomkv_full_attention_threshold`, for prefill,
multi-token/speculative steps, and unsupported mixed shapes. The current
performance template sets the threshold to 3072.

## Current configuration

- `zoomkv_sink_size=64`: always-attended prefix.
- `zoomkv_local_size=256`: always-attended recent window.
- `zoomkv_chunk_size=16`: retrieval chunk; must equal `block_size`.
- `zoomkv_chunk_candidates=200`: chunk-mean candidates.
- `zoomkv_dense_chunks=60`: leading candidates using the larger KIVI budget.
- `zoomkv_dense_topk=8`, `zoomkv_sparse_topk=4`.
- `zoomkv_final_topk=100`.
- `zoomkv_full_attention_threshold=3072` in the performance template (code
  default: 2000).
- `zoomkv_enable_offload=true` for CPU-offload testing.
- `zoomkv_cpu_bytes_per_rank=25769803776` (24 GiB) recommended as a starting
  point for each rank; size for the host and NUMA topology.
- `zoomkv_offload_unit_tokens=64`.
- `zoomkv_strict_kernels=true` for the performance template; this fails fast if
  the required extension kernels are unavailable.

The implementation requires a 16-token KV block and FP16/BF16 KV. The
vectorized UVA gather is specialized for head dimensions 128 and 256.
Speculative/multi-token sparse decode and KV connectors are not supported.
Offload is excluded from full CUDA Graph capture. CPU-offload and
`zoomkv_dense_fallback=true` cannot be enabled together.

## Build and launch

Build the optional CUDA extension used by strict mode:

```bash
cmake -S . -B build -DVLLM_BUILD_ZOOMKV_EXT=ON
cmake --build build --target _zoomkv_C
python -c "import vllm._zoomkv_C"
```

Launch the editable performance-test template (24 GiB pinned K+V per rank by
default):

```bash
bash examples/features/zoomkv/serve_zoomkv_qwen36_example.sh
```

Override host-specific values without editing it:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
MODEL_PATH=/path/to/Qwen3.6-27B \
ZOOMKV_CPU_BYTES_PER_RANK=$((24 * 1024**3)) \
bash examples/features/zoomkv/serve_zoomkv_qwen36_example.sh
```

Do not also pass `--attention-backend`; the backend is supplied in
`--attention-config`.

For a GPU-only comparison, use the same configuration with
`zoomkv_enable_offload=false`. For a dense routing comparison, GPU-only mode
may additionally set `zoomkv_dense_fallback=true`; keep the prompt, TP, batch,
generation length, and warmup identical.

## Test and benchmark commands

Lightweight CPU tests:

```bash
pytest -q tests/v1/attention/zoomkv/test_zoomkv_ops.py -k cpu
```

GPU-only smoke:

```bash
python examples/features/zoomkv/zoomkv_gpu_only.py \
  --model /path/to/model \
  --tensor-parallel-size 2 \
  --threshold 3072 \
  --output-json /tmp/zoomkv-smoke.json
```

Recall measurement (the current CLI has no Quest ratio arguments):

```bash
python examples/features/zoomkv/measure_topk_recall.py \
  --model /path/to/model \
  --tensor-parallel-size 2 \
  --threshold 3072 \
  --chunk-candidates 200 \
  --dense-chunks 60 \
  --dense-topk 8 \
  --sparse-topk 4 \
  --final-topk 100 \
  --output-json /tmp/zoomkv-recall.json
```

Focused 128K retrieval profiling:

```bash
python examples/features/zoomkv/profile_retrieval_128k.py --help
```

End-to-end GPU-only comparison:

```bash
python examples/features/zoomkv/benchmark_zoomkv_gpu_only.py \
  --model /path/to/model \
  --mode sparse \
  --threshold 3072 \
  --output-tokens 1024 \
  --runs 3 \
  --output-json /tmp/zoomkv-gpu-only.json
```

For the K+V CPU-offload path, add:

```bash
  --enable-offload \
  --cpu-bytes-per-rank $((24 * 1024**3))
```

Write generated JSON, logs, traces, and profiler output outside the repository.
Recall instrumentation synchronizes the GPU and is for debugging only.

## Performance status (2026-09-04)

The implementation is experimental and still being optimized. A temporary
same-machine baseline on NUMA node 0, TP=2, batch size 1, and 1024 decode
tokens measured:

- GPU-only TPOT: 14.03 ms at 64K; 14.28 ms at 128K.
- Offload, vectorized UVA TPOT: 27.72 ms at 64K; 33.70 ms at 128K.

CPU offload is currently slower than GPU-only. These numbers are only a
stage baseline, not production results or a general performance claim.

## Implementation map

- Configuration: `vllm/config/attention.py`
- Backend/routing: `vllm/v1/attention/backends/zoomkv_attn.py`
- Retrieval: `vllm/v1/attention/ops/zoomkv/retriever.py`
- Summary/state: `vllm/v1/attention/ops/zoomkv/state.py`
- CPU pool/lifecycle: `vllm/v1/attention/ops/zoomkv/offload.py`
- Paged gather/attention: `vllm/v1/attention/ops/zoomkv/paged.py`
- Dispatch: `vllm/v1/attention/ops/zoomkv/kernels.py`
- CUDA hybrid gather: `vllm/v1/attention/ops/zoomkv/cuda/h2d_gather_tokens.cu`
- Direct physical retrieval:
  `vllm/v1/attention/ops/zoomkv/cuda/physical_retrieval.cu`
- CUDA build: `cmake/zoomkv.cmake`
- Unit tests: `tests/v1/attention/zoomkv/test_zoomkv_ops.py`
- 128K profiler: `examples/features/zoomkv/profile_retrieval_128k.py`

The original standalone ZoomKV project's benchmark results are not results for
this vLLM integration.
