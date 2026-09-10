# PCP TP-Indexer measurements

These are bounded diagnostic measurements, not a saturated serving benchmark.
Implementation measured: `a2a11409a` (PCP enablement) and `4f5a0a183` (direct
publication), based on #54951 head `b466281a9`. The packaging commit changes
only scripts/documentation/evidence, not the implementation.

## Engine comparison

GLM-5.2 NVFP4, four GB200 GPUs, TP2PCP2 DCP1 EP4, FP8 KV, V2 eager runner,
FLASH_ATTN MLA prefill, existing sparse consumer, FlashInfer CUTLASS MoE,
prefix caching off. C1; 65536 scheduled-token budget; 8 generated tokens.
Torch 2.13.0+cu130. Synthetic repeated-text prompts, not a quality evaluation.

| Input tokens | Replicated scoring TTFT | Sharding + AllGather | Sharding + direct |
| --- | ---: | ---: | ---: |
| 65536 | 1.900839 s | 1.835469 s | 1.833061 s |
| 131072 | 4.094189 s | 3.814938 s | 3.813031 s |

Each arm has one excluded warmup and one measured request in the same engine,
with identical symmetric-buffer allocation. Baseline disables row sharding;
the other two arms differ in publication. Exact generated tokens match.
Every worker confirms activation. The 128K request has two prefill steps.
Sharding's directional TTFT improvement is 3.44%/6.82%. Direct's additional
0.13%/0.05% does not establish an engine speedup.

The callback consistently selects the existing AllReduce+RMSNorm fallback
because this host rejects FlashInfer multicast workspace initialization.
That workaround is benchmark-only and affects all arms equally. Local callback
serialization is explicitly enabled below; do not use untrusted callbacks.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 VLLM_TP_TOPK_DIRECT=1 \
VLLM_USE_V2_MODEL_RUNNER=1 VLLM_ALLREDUCE_USE_FLASHINFER=0 \
VLLM_ALLREDUCE_USE_SYMM_MEM=0 VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
OMP_NUM_THREADS=1 PYTHONPATH="$PWD" .venv/bin/python \
benchmarks/pcp_tp_topk/engine_bench.py --model /path/to/GLM-5.2-NVFP4 \
  --input 65536 --output /tmp/tp2pcp2-64k.json
```

Repeat with `--input 131072`. Raw results are in `results/`.

## Component comparison

Uses production PCP mirrored row lengths, the cost-balanced partitioner,
native TopK, PyNCCL variable-size broadcasts and direct publication. Native
TopK uses synthetic random logits, not captured model logits; MQA is excluded.

| Step | AllGather + copy-back | Direct including both barriers |
| --- | ---: | ---: |
| First 64K | 618.54 us | 402.74 us |
| Next 64K | 585.12 us | 349.18 us |

Median of maximum rank durations per iteration; four warmup pairs and twelve
measured pairs with alternating order. CUDA-event intervals include waits
and possible host launch gaps. Direct transport wins 34.89%/40.32%, but its
absolute savings are only about 0.22 ms per Indexer publication. Output-byte
imbalance explains long finish waits on smaller producers. Exact transport
outputs are checked. Do not interpret these percentages as engine speedups.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 PYTHONPATH="$PWD" \
.venv/bin/python benchmarks/pcp_tp_topk/component_profile.py \
  --output /tmp/component-profile.json
```

## Validation limits

96 CPU row-sharding tests and a four-GPU uneven-row, lane-isolation, repeated
reuse and fused k-pool expansion test passed. The GPU test also captures direct
publication and fused expansion in CUDA graphs, replays each eight times with
changing inputs, and checks the downstream output plus padding. The expanded
test passes in 18.10 seconds on four GB200 GPUs with Torch 2.13.0+cu130.

Formatting/lint/mypy passed for the implementation. TP4PCP2 requires eight GPUs
and was not run. Graph-enabled engine execution is blocked by
`PCPManager.validate_config`: sparse-MLA PCP explicitly rejects PIECEWISE,
FULL_AND_PIECEWISE and FULL_DECODE_ONLY. The guard is also present on upstream
main when checked on September 10, 2026. It was not bypassed. Component graph
success must not be interpreted as engine graph support. GLM k-pool model-level
performance remains unvalidated.
