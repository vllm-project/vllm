# MiniMax-M3 decode through the ATOM kernel library

This experimental path calls ATOM's fused sparse-layer kernels from native
vLLM. It is disabled by default. vLLM retains scheduling, cache allocation,
EAGLE3, the model loop, and CUDA graph ownership. ATOM supplies kernels, without
loading its engine, runner, or platform plugin.

## Draft status and upstream compatibility

The implementation was measured on vLLM `2eaa3bc5ac` and ATOM `922b35196`
with the accompanying scalar-cache compatibility patch. It requires
`VLLM_CACHE_ABI_VERSION=1`; an unmodified ATOM installation is insufficient.

This snapshot predates ATOM's shared mono refactor and correctness fixes in
[2479](https://github.com/ROCm/ATOM/pull/2479),
[2483](https://github.com/ROCm/ATOM/pull/2483), and
[2485](https://github.com/ROCm/ATOM/pull/2485). Before merge, port the cache ABI
and adapter to the current kernel and runtime interfaces, preserve the
upstream store-publication and batch-invariance fixes, and repeat validation.
The old snapshot can publish completion before global stores finish and can
let padding change live rows' MoE reduction order. Its previous accuracy and
performance results do not exclude these defects. ATOM
[2452](https://github.com/ROCm/ATOM/pull/2452) records the original diagnosis;
it is not an outstanding dependency on current ATOM main.

This is an alternative ownership boundary to vLLM
[59705](https://github.com/vllm-project/vllm/pull/59705), which vendors the
kernels, and [59653](https://github.com/vllm-project/vllm/pull/59653), which
uses their AITER port. This path imports ATOM directly and retains the native
BF16 attention weights for fallback, using separate converted copies for mono.

## Supported experiment

- gfx950 with 256 CUs, TP4/PP1/DP1, V2 GPU model runner.
- MiniMax-M3-MXFP4 with 57 sparse layers (indices 3 through 59).
- EAGLE3 with three speculative tokens; the draft remains native.
- FP8 main and index caches, block size 128, maximum model length 16384,
  at most four sequences, prefix caching disabled.
- Shared-expert fusion enabled; no context/expert parallelism, microbatching,
  LoRA, KV transfer, sleep, or online weight updates. Configured weight transfer
  and weight reload are rejected before changing weights.
- Decode/verification buckets 1, 4, 8, and 16. Prefill, mixed batches, dense
  layers, and unsupported step shapes use native execution.

Initialization errors propagate collectively. A kernel execution failure is
an error, rather than a successful native retry. Graph teardown precedes peer
memory release. KV and index allocation remains entirely native; the adapter
uses static scalar K/V scales and independent index addressing, without a
shadow cache or dynamic scale sidecar.

## Launch the pinned experiment

Install the paired patched ATOM revision in the same environment first. Set
`TARGET_MODEL` and `DRAFT_MODEL` to the local MiniMax-M3-MXFP4 and
MiniMax-M3-EAGLE3-GQA checkpoints. The recorded draft revision is
`Inferact/MiniMax-M3-EAGLE3-GQA@96692486b5fd38ebf8fd2a5f6bb53427d30819a8`.

```bash
export VLLM_PLUGINS=''
export ATOM_DISABLE_VLLM_PLUGIN=1
export AITER_LOG_LEVEL=WARNING
export HIP_VISIBLE_DEVICES=0,1,2,3
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1
export VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ROCM_USE_ATOM_M3_MONO=1

vllm serve "$TARGET_MODEL" \
    --served-model-name minimax-m3 --host 127.0.0.1 --port 8000 \
    --tensor-parallel-size 4 --pipeline-parallel-size 1 \
    --language-model-only --max-model-len 16384 --max-num-seqs 4 \
    --max-num-batched-tokens 8192 --gpu-memory-utilization 0.70 \
    --num-gpu-blocks-override 1024 --block-size 128 --kv-cache-dtype fp8 \
    --attention-config '{"indexer_kv_dtype":"fp8"}' \
    --no-enable-prefix-caching \
    --speculative-config "{\"method\":\"eagle3\",\"model\":\"$DRAFT_MODEL\",\"num_speculative_tokens\":3,\"rejection_sample_method\":\"standard\"}"
```

For the native baseline set `VLLM_ROCM_USE_ATOM_M3_MONO=0` and keep the remaining
settings unchanged. Accuracy requests must use the checkpoint's chat template
with `thinking_mode=enabled`; the recorded evaluation rendered it client-side.
Never use synthetic acceptance for accuracy evaluation.

## Recorded validation and limits

The October 2, 2026 experiment used Python 3.12.14, Torch
`2.13.0+git733fca1`, ROCm 7.2.3, AITER 0.1.23, and FlyDSL 0.3.4.1.
Full GSM8K, fixed five-shot multiturn prompts, thinking enabled, standard
EAGLE3, temperature zero, and an 8192-token generation limit gave native
1277/1319 (96.8158%) and mono 1276/1319 (96.7400%). The earlier raw-completion
protocol gave 1142 versus 1108; that regression remains unresolved.

Performance used fixed random token requests, 256 output tokens, and synthetic
expected acceptance length 2.83. Three paired rounds (A/C, C/A, A/C), each with
16 warmup and 128 measured requests per cell, produced 36 valid cells and 4608
measured requests. Values below are three-round means, not natural-acceptance
production predictions.

| Input | Concurrency | Native TPOT (ms) | Mono TPOT (ms) | E2E change | Output throughput change |
| ----- | ----------- | ---------------- | -------------- | ---------- | ------------------------ |
| 1024 | 1 | 2.647 | 1.763 | -29.38% | +41.59% |
| 1024 | 2 | 3.162 | 2.376 | -21.32% | +27.09% |
| 1024 | 4 | 3.390 | 3.012 | -9.38% | +10.35% |
| 8192 | 1 | 3.563 | 2.716 | -18.58% | +22.81% |
| 8192 | 2 | 4.393 | 3.784 | -12.53% | +14.32% |
| 8192 | 4 | 5.784 | 5.957 | -3.75% | +3.89% |

Mono TPOT at 8K/concurrency 4 was 6.208/6.201/5.462 ms across rounds, so this
case did not establish stable decode acceleration. Extra sampled memory was
approximately 1.64-1.69 GiB per rank. FP8 projection copies, BF16 router, and
MoE arithmetic differ from native; there was no same-precision unfused control.
The gains cannot be attributed exclusively to fusion.

Layer-stage checks, native/mono cache interoperability, graph replay, request
cancellation, preemption/recompute, page reuse, and shutdown were exercised
outside CI. They do not cover the newly identified synchronization and batch
invariance defects. Before readiness, retain the external GPU validation
harness in an appropriate test/benchmark location, exercise repeated identical
inputs and poisoned padding, and rerun graph/lifecycle, GSM8K, and performance
on the ported kernels. The existing worker test suite contains CPU regression
checks for rejecting weight updates before mutation.
