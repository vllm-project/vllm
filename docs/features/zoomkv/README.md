# ZoomKV attention (GPU-only + K+V CPU offload)

ZoomKV is an experimental vLLM V1 attention backend for sparse long-context
decode. Prefill and short-context decode use dense attention. Long-context,
single-token decode retrieves a fixed sparse context using hierarchical Quest
and KIVI reranking.

This implementation targets vLLM v0.20.0. It has passed an end-to-end
Qwen3.6-27B tensor-parallel smoke test on two NVIDIA H20 GPUs and Qwen3-4B
accuracy/latency tests on a single GPU.

## Current integration status

This first PR includes:

- a native V1 `ZOOMKV` attention backend and configuration validation;
- dense prefill/short-context fallback plus hierarchical Quest and
  dense/sparse KIVI reranking for long single-token decode;
- physical-block min/max/centroid/4-bit Key summaries with Triton/CUDA
  kernels and reference fallbacks;
- optional K+V CPU offload with warm/cold/restore block lifecycle;
- block-reuse invalidation and local prefix-cache restoration support;
- recall instrumentation, unit tests, smoke tests, and same-prompt
  dense/sparse latency benchmarks.

Known limitations in this first PR:

- mixed prefill/decode batches and multi-token/speculative decode fall back
  to dense attention;
- CUDA Graphs and KV connectors are not supported;
- sparse decode is currently a per-request Python loop and is slower than
  dense attention in the measured 12K/32K Qwen3-4B tests;
- the non-block-aligned boundary between the retrieval and local windows can
  omit up to 15 tokens. A follow-up will extend the always-attended tail to
  cover that boundary without consuming the retrieved Top-K budget;
- CPU offload capacity is fixed by `zoomkv_cpu_bytes_per_rank`; exhaustion
  degrades safely to GPU-resident blocks but reduces memory savings.

## Modes

- **GPU-only (default):** full Key and Value stay on GPU. Block summaries
  and 4-bit packed Keys live on GPU for retrieval.
- **K+V CPU offload (`zoomkv_enable_offload=True`):** after a 16-token
  child chunk completes, its full-precision Key and Value are async-copied
  to pinned host memory (the block becomes *warm*; the GPU pages stay
  intact for dense readers). Once the block enters the sparse-decode
  retrieval zone its GPU pages are zeroed (*cold*, no extra PCIe traffic).
  Sparse decode gathers cold tokens straight from pinned memory; before any
  dense read (prefill chunks, mixed batches, prefix-cache hits) cold blocks
  are restored to GPU and can later be re-zeroed for free because block
  content is immutable.

## Algorithm

ZoomKV is used only for layers backed by regular full attention. Hybrid models
such as the local Qwen3.6-27B checkpoint contain Gated DeltaNet (GDN) and full
attention layers; the GDN layers are unchanged.

### Prefill / chunked prefill

1. Write Key/Value into the normal paged KV cache.
2. When a 16-token child chunk completes, build min/max/centroid and 4-bit
   packed Key block summaries on GPU.
3. Aggregate every 16 child chunks into a 256-token parent min/max.
4. If offload is enabled, async D2H the completed child Key and Value into
   the pinned CPU pool. GPU pages are NOT zeroed during prefill (the same
   step's dense attention still reads them); zeroing happens lazily when the
   block enters the sparse-decode retrieval zone.

### Sparse decode

1. Reduce GQA query heads to one retrieval query per KV group using the
   group mean (default, matching the original ZoomKV implementation).
   `zoomkv_per_query_head=True` selects the strongest head instead, but
   measured Top-K recall is significantly lower.
2. Run hierarchical Quest over parent then child chunks.
3. Use centroid density scoring and KIVI reranking to select the final Top-K
   tokens.
4. Attend to `sink + local window + retrieved Top-K` tokens (Value from GPU;
   Key from GPU hot pages or CPU for offloaded blocks).

The backend falls back to dense attention for prefill, mixed prefill/decode
batches, multi-token decode, and sequences below
`zoomkv_full_attention_threshold`.

## Requirements and current support

| Item | First-release support |
| --- | --- |
| Accelerator | NVIDIA CUDA; validated on H20 (compute capability 9.0) |
| Model | Qwen3.6-27B dense and sparse smoke test passed |
| Tensor parallelism | TP=2 smoke test passed |
| Head size | 128 or 256 |
| KV dtype | FP16 or BF16 |
| KV block size | Exactly 16 |
| Prefill | Dense |
| Sparse decode | Pure single-token decode batches |
| CUDA Graph | Not supported; use eager mode |
| Prefix caching | Supported (GPU-only and offload modes) |
| Speculative decoding | Not supported; multi-token steps use dense fallback |
| KV connector | Not supported |
| K+V CPU offload | Supported via `zoomkv_enable_offload=True` |

Use `block_size=16` and `enforce_eager=True`. GPU-only mode keeps full
Key/Value on GPU. With offload enabled, cold Key/Value pages move to pinned
host memory and are transparently restored before any dense read (including
prefix-cache hits).

## Python example

```python
from vllm import LLM, SamplingParams
from vllm.config.attention import AttentionConfig
from vllm.v1.attention.backends.registry import AttentionBackendEnum

llm = LLM(
    model="/data/qyl/models/Qwen3.6-27B",
    trust_remote_code=True,
    dtype="bfloat16",
    tensor_parallel_size=2,
    max_model_len=65536,
    block_size=16,
    enforce_eager=True,
    attention_config=AttentionConfig(
        backend=AttentionBackendEnum.ZOOMKV,
        zoomkv_sink_size=64,
        zoomkv_local_size=256,
        zoomkv_final_topk=100,
        zoomkv_full_attention_threshold=2000,
    ),
)

outputs = llm.generate(
    ["A long prompt ..."],
    SamplingParams(max_tokens=32, temperature=0.0),
)
```

## Server example

```bash
vllm serve /data/qyl/models/Qwen3.6-27B \
  --trust-remote-code \
  --dtype bfloat16 \
  --tensor-parallel-size 2 \
  --max-model-len 65536 \
  --block-size 16 \
  --enforce-eager \
  --attention-config '{
    "backend":"ZOOMKV",
    "zoomkv_sink_size":64,
    "zoomkv_local_size":256,
    "zoomkv_final_topk":100,
    "zoomkv_full_attention_threshold":2000
  }'
```

Do not also pass `--attention-backend`; it is mutually exclusive with the
backend in `--attention-config`.

## Configuration

| Option | Default | Description |
| --- | ---: | --- |
| `zoomkv_sink_size` | 64 | Prefix tokens always included in sparse attention |
| `zoomkv_local_size` | 256 | Most recent tokens always included |
| `zoomkv_final_topk` | 100 | Final retrieved-token budget per KV head |
| `zoomkv_quest_chunk` | 16 | Child chunk and required KV block size |
| `zoomkv_quest_large_chunk` | 256 | Parent chunk size |
| `zoomkv_quest_large_ratio` | 0.8 | Fraction of parent chunks retained |
| `zoomkv_quest_small_ratio` | 0.5 | Fraction of child chunks retained |
| `zoomkv_dense_ratio` | 0.4 | Fraction of candidate chunks treated as dense |
| `zoomkv_dense_topk` | 16 | Tokens retained from a dense candidate chunk |
| `zoomkv_sparse_topk` | 8 | Tokens retained from a sparse candidate chunk |
| `zoomkv_full_attention_threshold` | 2000 | Dense attention below this sequence length |
| `zoomkv_dense_fallback` | false | Force dense attention for debugging |
| `zoomkv_strict_kernels` | false | Fail rather than use a kernel fallback |
| `zoomkv_enable_offload` | false | Enable K+V CPU offload of cold blocks |
| `zoomkv_cpu_bytes_per_rank` | 8 GiB | Pinned host Key pool budget per rank |
| `zoomkv_per_query_head` | false | Use strongest query head per KV group instead of the group mean (lowers recall; kept for experiments) |

`sink_size`, `local_size`, and `quest_large_chunk` must be divisible by
`quest_chunk`. The first release requires `quest_chunk=16`.
`zoomkv_enable_offload` cannot be combined with `zoomkv_dense_fallback`.

## Optional CUDA extension

The Python package works with Triton/PyTorch fallbacks. For the CUDA Quest,
Top-K, density, mask, KIVI, and K+V H2D kernels, configure the vLLM build
with:

```bash
cmake -S . -B build -DVLLM_BUILD_ZOOMKV_EXT=ON
cmake --build build --target _zoomkv_C
python -c "import vllm._zoomkv_C"
```

Set `zoomkv_strict_kernels=true` only after the production kernels are
available. Without strict mode, the runtime can JIT compile individual CUDA
operators or use a reference implementation.

## Smoke test

The example runs one short dense request and one long sparse request:

```bash
python examples/features/zoomkv/zoomkv_gpu_only.py \
  --model /data/qyl/models/Qwen3.6-27B \
  --tensor-parallel-size 2 \
  --threshold 512 \
  --output-json /tmp/zoomkv-smoke.json
```

The worker log prints `ZoomKV GPU-only sparse decode path is active` when the
long request enters sparse decode.

## Validation status

The v0.20.0 integration passed the following checks:

- ZoomKV unit tests, including CUDA block-summary and sparse-attention paths
- Unified optional CUDA extension compilation and binding checks
- Qwen3.6-27B BF16, TP=2 on two NVIDIA H20 GPUs
- Dense path: 12 prompt tokens, 8 output tokens
- Sparse path: 1,332 prompt tokens, 8 output tokens, threshold 512
- Both smoke requests produced `Paris`
- Qwen3-4B needle retrieval with K+V offload: 3/3 prompts recovered the target
- Qwen3-4B measured retrieval recall: recall@100 0.700 against
  exact-attention Top-K and 0.869 against the retrieval-query oracle
- Prefix-cache smoke tests in GPU-only and K+V offload modes

On Qwen3-4B at approximately 12K tokens (single GPU, BF16, eager, batch 1),
dense decode measured 25.5 ms/token and ZoomKV GPU-only sparse decode measured
45.2 ms/token. At approximately 32K tokens the corresponding measurements
were 26.3 and 44.6 ms/token. These results show that retrieval cost is nearly
context-length independent, but the current unbatched eager implementation
does not yet beat dense attention at these context lengths.

The smoke test proves integration and execution-path correctness, not
representative long-context quality or performance. The measured sparse wall
time includes first-use Triton/JIT compilation and must not be treated as a
benchmark.

### Measuring per-decode-step Top-K recall

`VLLM_ZOOMKV_RECALL_LOG=<dir>` enables a debug probe that, at every sparse
decode step, compares the retrieved Top-K tokens of each layer and KV head
with the exact-attention Top-K (softmax of q.K over the full sequence,
aggregated per GQA group, restricted to the retrieval zone). Each worker
appends JSONL records to `<dir>/recall.<pid>.jsonl`. The probe synchronizes
the GPU per record and only supports GPU-only mode (not offload); never
enable it in production runs.

```bash
python examples/features/zoomkv/measure_topk_recall.py \
  --model <model> --tensor-parallel-size 1 \
  --prompt-sentences 600 --output-tokens 64 \
  --output-json /tmp/zoomkv-recall.json
```

The script reports, per decode step: recall@`final_topk`, retrieved
attention-mass coverage of the retrieval zone versus the oracle Top-K
coverage, and the fraction of total attention mass inside the zone.

### Same-prompt benchmark

After warmup, dense fallback and ZoomKV sparse decode were measured on the
same prompt at each context length using Qwen3.6-27B BF16, TP=2, two NVIDIA
H20 GPUs, eager mode, and three runs per configuration:

| Prompt tokens | Output tokens | Dense median | Sparse median | Result |
| ---: | ---: | ---: | ---: | --- |
| 7,712 | 32 | 3.814 s | 4.036 s | Sparse is 5.8% slower |
| 7,712 | 128 | 10.158 s | 11.108 s | Sparse is 9.4% slower |
| 15,962 | 128 | 12.180 s | 13.386 s | Sparse is 9.9% slower |
| 63,812 | 128 | 32.063 s | 33.166 s | Sparse is 3.4% slower |

These are end-to-end request times and include the identical prefill in both
modes. The current implementation does not yet provide a speedup at this
context length. Sparse output also diverged from dense output, so retrieval
quality requires evaluation and tuning before production use.

Reproduce one side of the comparison with:

```bash
python examples/features/zoomkv/benchmark_zoomkv_gpu_only.py \
  --model /data/qyl/models/Qwen3.6-27B \
  --mode sparse \
  --prompt-repeats 700 \
  --output-tokens 128 \
  --runs 3
```

## Why GPU-only first

For Qwen3.6-27B, only 16 of 64 layers use full attention. With BF16 KV,
TP=2, two KV heads per rank, and head size 256, the approximate KV footprint
per rank is:

| Context length | KV cache per rank |
| ---: | ---: |
| 16K | 0.5 GiB |
| 32K | 1 GiB |
| 64K | 2 GiB |

At 16K-64K, GPU-only integration avoids PCIe transfer and invasive scheduler
changes for a modest KV footprint. Offload becomes more relevant for 128K+
contexts, high concurrency, or deployments where model weights leave little
KV capacity.

## Implementation map

- Backend: `vllm/v1/attention/backends/zoomkv_attn.py`
- Retrieval: `vllm/v1/attention/ops/zoomkv/retriever.py`
- Block summaries: `vllm/v1/attention/ops/zoomkv/state.py`
- K+V CPU pool: `vllm/v1/attention/ops/zoomkv/offload.py`
- Paged gather and sparse attention: `vllm/v1/attention/ops/zoomkv/paged.py`
- CUDA dispatch: `vllm/v1/attention/ops/zoomkv/kernels.py`
- CUDA build: `cmake/zoomkv.cmake`
- Unit tests: `tests/v1/attention/zoomkv/test_zoomkv_ops.py`
- Benchmark: `examples/features/zoomkv/benchmark_zoomkv_gpu_only.py`

The original standalone ZoomKV project and its published benchmark results are
not vLLM integration results and should not be quoted as such.
