# DeepSeek-V4.1 PCP with SWA bounded replay: implementation plan

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

`vllm/models/deepseek_v41/attention.py` disables bounded replay under PCP
because slot padding only knows the rank-local batch. In
`nvidia/model.py`, prefix caching is rejected with PCP and
`_decoder_replay_supported` excludes per-rank shrinking replay batches.
`nvidia/model_state.py` already owns replay starts, cached-group write masks
and the decoder replay sub-batch for the non-PCP path.

## Proposed implementation

1. Define one mapping from original requests and absolute positions to PCP
   virtual rows. Use it consistently for replay starts, Engram lookback,
   query boundaries and local versus gathered cache slots.
2. Pad writes to prefix-cacheable groups for replayed tokens in both the local
   and gathered views. Rebuild only the required SWA window and retain the
   lower bound that prevents attention from reading stale window KV.
3. Make decoder replay sub-batch construction agree across ranks. Ranks with
   no retained local rows must still participate in required collectives with
   compatible shapes and ordering. Audit cross-rank ratio-2 compressor pairs.
4. Validate eager cold-prefill decoder trimming first, then local prefix-cache
   hits and SWA reconstruction. Lift each restriction only for tested cases.
   Prefix-cache support needed for this replay flow belongs in this track;
   external cache transfer, DSpark and graph composition require later work.

## Existing work to reuse

Reuse [#56227](https://github.com/vllm-project/vllm/pull/56227) and
[#58132](https://github.com/vllm-project/vllm/pull/58132).
[#60046](https://github.com/vllm-project/vllm/pull/60046) concerns P/D replay
endpoints; [#58289](https://github.com/vllm-project/vllm/pull/58289) and
[#57872](https://github.com/vllm-project/vllm/pull/57872) target encoder-only
P/D. Those are related semantics, not a validated colocated PCP implementation.

## Acceptance criteria

- [ ] Extend existing replay-start, replay-batch and decoder-replay tests in
  `tests/models/`, plus `tests/v1/core/test_prefix_replay.py`, at the cheapest
  level that checks the mapping and write-preservation contracts.
- [ ] Cover no-hit, partial-hit and full-hit requests; windows around block
  and PCP boundaries; request reuse; empty local replay rows; chunked prefill;
  prompt logprobs; and mixed prefill/decode. Compare with replay-disabled runs.
- [ ] Verify byte preservation of reused cache entries and correct SWA
  reconstruction, without reads below replay start or duplicate invalid writes.
- [ ] On a real checkpoint, compare PCP1/2/4 with replay off/on. Test cold
  prompts and repeated-prefix requests separately, proving the replay path ran.
- [ ] Report accuracy, input throughput, TTFT, KV memory and replayed token
  counts at fixed prompt lengths and concurrency; include repeated-run spread.
- [ ] Record code/checkpoint SHAs and flags; use `VLLM_GPU_SYNC_CHECK=error`.

Reducing decoder work can lower long-prefill cost, but communication and short
replay windows can limit PCP scaling. A larger benefit than DSpark is a
hypothesis to measure, not a result of this draft.
