# DeepSeek-V4.1 PCP with DSpark: implementation plan

Status: implementation plan only. This draft does not enable the feature.
No GPU execution, model evaluation, or performance result is claimed.

This is a follow-up to [#59857](https://github.com/vllm-project/vllm/pull/59857),
pinned initially to `7908ee8906475842ba0ecd1c09eed6d878718f05`. That PR provides the
NVIDIA text-only DeepSeek-V4.1-Flash PCP baseline. Its MoE padding dependency,
[#60946](https://github.com/vllm-project/vllm/pull/60946), must be resolved before
validating the combined runtime. Historical results in the parent PR do not
validate this branch.

The initial target is MRV2, FlashMLA sparse attention, PP=DP=DCP=1 and one
colocated instance. Each follow-up starts independently from the parent;
combining the three features requires separate integration validation.

## Current boundary

`vllm/models/deepseek_v41/nvidia/model.py` rejects speculative decoding when
PCP is enabled. `CompressorMetadataBuilder` in
`vllm/models/deepseek_v41/compressor.py` also rejects the combination: the
current cross-rank ring publication retains only the newest open ratio-2 row.
Verification and rejection need enough history to resume at the accepted
position, so removing these checks alone is insufficient.

## Proposed implementation

1. Reuse the MRV2 replicated DSpark path from
   [#56107](https://github.com/vllm-project/vllm/pull/56107): restore target
   auxiliary states to global request/token order before draft consumption.
2. Derive the compressor history needed for fixed-length verification and
   publish the necessary predecessor rows across PCP ranks. Preserve ring
   state for the accepted prefix when verification rejects later tokens.
3. Audit V4.1 draft attention, indexer, SWA and MoE inputs for replicated draft
   semantics while target prefill remains sharded. Keep invalid padding from
   writing to cache blocks or entering another request's state.
4. Enable only the validated fixed-verification DSpark configuration. Adaptive
   verification, prefix caching, bounded replay and CUDA graphs remain separate
   follow-ups until their combinations are tested.

## Existing work to reuse

[#54036](https://github.com/vllm-project/vllm/pull/54036) addresses sharded
context-KV precomputation and direct KV publication. It does not change the
V4.1 model/compressor files. Start with replicated drafting for correctness;
do not duplicate its direct-publication optimization.

## Acceptance criteria

- [ ] Extend `tests/kernels/test_compressor_kv_cache.py` with verification
  acceptance/rejection at ratio-2 boundaries, PCP2/4, empty ranks, and mixed
  prefill/decode batches. Check resumed state against an unsharded reference.
- [ ] Verify global auxiliary-state ordering and request isolation, including
  short replicated prefills and changing batch membership.
- [ ] Run the real V4.1 checkpoint at TP8/PCP1, TP4/PCP2 and TP2/PCP4 with fixed
  verification. Confirm nonzero accepted draft tokens and the intended path.
- [ ] Compare quality against the same checkpoint without speculation; record
  accuracy, acceptance length, API errors, TTFT, TPOT and output throughput.
- [ ] Record checkpoint revision, code SHA, hardware, launch flags, seeds and
  repeated-run variability. Run with `VLLM_GPU_SYNC_CHECK=error`.

Replicated drafting repeats draft work on PCP ranks. Its correctness baseline
must precede performance claims about sharded drafting.
