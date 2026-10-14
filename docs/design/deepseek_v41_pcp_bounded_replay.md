# DeepSeek-V4.1 PCP with decoder SWA bounded replay

This draft enables cold-prefill decoder trimming with PCP. It builds on
[#59857](https://github.com/vllm-project/vllm/pull/59857) and
[#61553](https://github.com/vllm-project/vllm/pull/61553), which supplies fixed
DSpark verification and optional shared host Engram storage. Prefix-cache hits,
KV transfer and CUDA Graph composition remain outside this change.

## Implementation

Reuse the existing `ReplayBatch`, `InputBatch`, decoder replay gather kernel,
attention builders and PCP process group. Prepare the replay builders during
startup so their constant copies do not first run in a real request. Retain each
existing PCP segment's intersection with the request's global scheduled tail;
do not change token ownership or introduce another PCP partitioner.

- Size per-segment replay metadata for PCP's two segments per original request.
  Original request state remains indexed by the existing global request indices.
- Recover each request's scheduled bounds with a CPU-group reduction indexed by
  request state. Ignore empty segments. Stage the global tail floor separately
  from the prefix layers' replay floor; short chunks preserve their context.
- Agree on whether a replay batch is needed and its largest rank-local size.
  Short or empty peers execute the same collectives with padded inputs.
- Select the local slice of the original gathered cache slots, then gather the
  retained slots in the same rank-major order as the new KV payload. Padding
  carries `PAD_SLOT_ID`; it does not write cache entries.
- Reuse the existing decoder-layer runner to gather retained rows and scatter
  their outputs back. Zero padding is discarded rather than becoming a real row.
- Dispatch local FlashMLA prefill/decode by token counts, since a retained batch
  can still contain request metadata for an empty query partition.

The range and padding agreements add two small host collectives per preparation;
neither reads device tensors. Ownership remains zigzag: a single long request's
tail can reside on one rank while peers execute padded work. This differs from
SGLang's interleave CP layout, which distributes that tail across ranks.

The model's existing guards continue to require PP=DP=DCP=1, no prefix caching,
no KV transfer, no microbatching, and a supported MoE/attention backend. The
existing sparse-PCP runner rejects CUDA graphs. Fixed DSpark remains subject to
the existing guard that its draft reads no hidden states outside the replay
window. Requests with prompt logprobs retain all their rows.

## Validation

Three existing replay suites passed on H20: **27 tests**. Added coverage checks
global-tail intersections, rank-local cache-slot selection, peer padding, empty
query partitions and output scattering. Existing replay-start, prompt-logprobs, adaptive-boundary and graph
regressions also pass. Unit tests mock PCP collectives; they do not by themselves
validate a distributed model. All applicable pre-commit hooks passed.

On four H20 GPUs, TP1/PCP4/EP4 with fixed DSpark5, shared Engram and
`VLLM_GPU_SYNC_CHECK=error` passed 32/32 fixed GSM8K canaries (no HTTP errors or
truncation; 3,965 of 5,645 draft tokens accepted). Eight additional checks passed:
exact 1,023/1,025-token prompts, including full-row prompt-logprobs controls, and
ragged concurrent 8,191/8,193/16,385/65,537-token prompts followed by natural
decode. These use public GSM8K questions with neutral background context; they
are targeted regressions, not a complete model accuracy evaluation.

The first real trimming exposed lazy metadata-builder initialization in the hot
path. Moving that existing initialization to startup fixed the failure while
keeping GPU synchronization checks enabled. Failed-run evidence is retained.
The first global-tail model also exposed a mixed batch with zero decode tokens
but nonzero decode request metadata. Dispatching by token counts fixes the empty
decode reshape without skipping the outer KV publication or PCP collectives.

Initial cold-prefill comparison: four H20-3e GPUs, TP1/PCP4/EP4, fixed DSpark5,
Humming A16, 8,192-token chunks, eager execution, prefix caching off. Each round
uses eight requests with 65,536 input tokens and one output token, continuously
replenished by four workers. The three formal samples exclude a warmup round;
all responses report zero cached tokens.

| Input tokens/s | Replay off | Local-segment tail | Global request tail |
| --- | ---: | ---: | ---: |
| Sample 1 | 5,658.45 | 9,302.84 | 9,729.21 |
| Sample 2 | 5,665.05 | 9,301.57 | 9,729.46 |
| Sample 3 | 5,664.64 | 9,303.91 | 9,729.19 |
| Median | 5,664.64 | 9,302.84 | 9,729.21 |

The current global-tail version measures **4.58%** above the previous
local-segment-tail implementation and **71.75%** above replay off. These are
initial sequential comparisons with three within-arm samples, not independent
interleaved A/B pairs. Launch arguments match except for
the off/on replay flag, and all per-round input hashes match. The baseline predates the replay
patch and a non-executable type annotation in shared Engram; it has not been
rerun on the final source. The runtime reuses the lab's compiled extensions,
rather than a clean full build. Backend autotuning was not explicitly frozen
across engine restarts. No SGLang-relative or decode speedup is claimed.

This follows the existing bounded-replay semantics, including its SWA window
floor. The canaries do not prove logit equivalence to replay-disabled execution
or replace a broader model-quality evaluation.

## Relationship to existing work

[#56227](https://github.com/vllm-project/vllm/pull/56227) and
[#58132](https://github.com/vllm-project/vllm/pull/58132) provide the existing
replay implementation being reused. This change makes its retained batch obey
PCP collective and cache-slot contracts. It does not add P/D endpoints
([#60046](https://github.com/vllm-project/vllm/pull/60046)) or encoder-only
connectors ([#58289](https://github.com/vllm-project/vllm/pull/58289),
[#57872](https://github.com/vllm-project/vllm/pull/57872)).
