# DeepSeek-V4.1 PCP with decode-only CUDA graphs

The NVIDIA MRV2 path supports `FULL_DECODE_ONLY` for DeepSeek-V4.1 PCP.
Prefill, including decoder SWA bounded replay, remains eager. This builds on
the PCP baseline, fixed DSpark verification and shared Engram storage, and
bounded replay from [#59857](https://github.com/vllm-project/vllm/pull/59857),
[#61553](https://github.com/vllm-project/vllm/pull/61553), and
[#61554](https://github.com/vllm-project/vllm/pull/61554).

## Metadata and dispatch

Reuse the existing MRV2 target and DFlash/DSpark graph managers. Metadata is
prepared before graph execution, so tensors referenced by captured kernels
must retain their storage across requests:

- Copy gathered SWA positions and compressed attention/indexer cache slots
  into persistent buffers.
- Update compressor gather indices, retained ring-write slots and predecessor
  indices in place. Captured dummy inputs must not freeze the ring links used
  after a rejected draft or a change in PCP request ownership.
- Match decode graphs using the largest rank-local request count and the
  existing rank-local token count. Global input staging still reserves space
  for all requests before PCP partitions them.

The computation, weights, quantization and speculative acceptance policy are
unchanged. The existing replicated draft and proposal broadcast are reused.
Generic PCP graph capture from
[#53867](https://github.com/vllm-project/vllm/pull/53867) is not reimplemented.

## Configuration and limits

Remove `--enforce-eager` and select `FULL_DECODE_ONLY`. For example, the
H20 validation used this bounded capture ladder:

```bash
--compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[5,6,10,12,20,24,40,48],"max_cudagraph_capture_size":48}'
--cudagraph-metrics
```

Capture sizes must accommodate both target verification and draft queries.
The example covers the tested DSpark5 target and draft batches through global
concurrency eight; it is not a universal tuning recommendation. Unmatched
runtime sizes retain the existing eager fallback. Capture logs alone do not
prove replay; check the target's runtime `CUDAGraph Stats` as well.

Existing V4.1 PCP restrictions remain: PP=DP=DCP=1, no microbatching, KV
transfer, prefix caching or MegaMoE, and fixed DSpark verification when
speculation is enabled. Speculative graphs require the indexer's existing
flattened or varlen query path. Native speculative query metadata that cannot
refresh empty decode owners remains rejected. Prefill graph modes are not
enabled by this change.

## Validation scope

Focused tests cover metadata address stability, request ownership changes,
rejected-draft ring updates, local dispatch sizing and configuration guards.
On four H20 GPUs, TP1/PCP4/EP4 with DSpark5, Humming FP8 activations, bounded
replay and `VLLM_GPU_SYNC_CHECK=error` passed 32 fixed GSM8K canaries, eight
long/ragged boundary checks, and 27 natural-EOS requests at concurrency
1, 3, 4, 5, 8, 5, 1. Target FULL execution and nonzero real speculative
acceptance were observed. These are targeted regressions, not a full model
accuracy evaluation or a claim of bitwise logit equivalence.

Graphs consume persistent memory and can add padding work. Keep capture,
warmup and performance measurements separate, and compare matching eager and
graph workloads. H20 results do not establish performance on other GPUs.
