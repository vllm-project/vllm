# DeepSeek-V4.1 PCP with CUDA Graph: implementation plan

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

The parent PR documents eager-only support. Existing graph capability markers
on individual attention metadata builders do not establish correctness of
the entire V4.1 PCP execution path. The rank-local compressor links, gathered
cache slots, sparse/SWA metadata, Engram lookback and MoE routing inputs all
need to remain valid when a graph replays with different live requests.

## Proposed implementation

1. Start with PIECEWISE graphs for the non-speculative, replay-disabled V4.1
   PCP baseline. Keep FULL mode and the other two feature combinations outside
   the initial support claim.
2. Audit capture-time versus per-step work in `compressor.py`,
   `sparse_mla.py`, `nvidia/model_state.py` and `nvidia/model.py`. Keep graph
   inputs at stable addresses and refresh values and metadata before replay;
   do not freeze request-dependent boundaries or valid-token counts.
3. Ensure rank-local padding, gathered cache-write views, MoE routing rows and
   collective shapes agree for every supported capture bucket. Padding must
   neither become a cache write nor change request identity.
4. Reuse the MRV2 dispatcher and record its global/local width policy. Only
   introduce model-specific changes needed by V4.1; any generic fix should be
   factored out and checked against ongoing upstream work.

## Existing work to reuse

[#53639](https://github.com/vllm-project/vllm/pull/53639) proposes dispatch by
rank-local token count and is an alternative to
[#53515](https://github.com/vllm-project/vllm/pull/53515). Its files concern the
generic runner/PCP manager, not V4.1 model capture.
[#56107](https://github.com/vllm-project/vllm/pull/56107) already includes
generic PIECEWISE graph support. This track must reuse the applicable upstream
policy; it is not another implementation of local graph dispatch.

## Acceptance criteria

- [ ] Add focused coverage to existing compressor, SWA and PCP suites for
  metadata refresh and repeated replay, including different request identities
  that share the same graph bucket.
- [ ] Test short replicated prefills, empty local decode shards, mixed batches,
  ratio-2 boundaries, padding buckets and changing batch membership at PCP2/4.
- [ ] Compare real-checkpoint eager and PIECEWISE runs at matching topology,
  dtype and workload. Verify graph capture and replay actually occur; an eager
  fallback does not establish support.
- [ ] Check numerical/quality agreement, cache integrity and absence of host
  synchronization using `VLLM_GPU_SYNC_CHECK=error`.
- [ ] Report graph memory, capture/startup cost, TTFT, TPOT and throughput with
  repeated-run variability. Evaluate SM90 and SM100 separately when available.
- [ ] Retain explicit limits until FULL graphs, DSpark and bounded-replay
  combinations each have their own validation evidence.

Graphs can reduce launch overhead but consume persistent memory and may add
padding work. Measure both prefill and decode workloads before claiming a gain.
