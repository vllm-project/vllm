# AuxOutput Logprobs Replay Design

## Status

Implemented initial design. The implementation supports raw, fixed-width
top-k logprobs for explicitly opted-in requests; the broader extensions and
open decisions remain documented below.

## Problem

The AuxOutput connector currently captures and replays R3 (routed-expert
indices) across requests that share prefix-cache blocks. RL rollout and scoring
workloads also need cached token logprobs and prompt logprobs to be replayed
when the corresponding token prefix is served from KV cache. They need this
without changing the regular serving path or forcing ordinary requests to pay
for logprob capture, transfer, storage, or replay.

Here, "replay" means reading an auxiliary artifact published by an earlier
forward for the same KV block hashes and attaching the artifact rows to the
current request's output. It is distinct from trace replay and speculative
decoding.

## Data Flow Before This Change

The scheduler builds `AuxOutputConnectorMetadata` with each request's output
start position and newly available KV block hashes. The worker captures R3
rows, stages complete hash blocks in a bounded store, and materializes rows
from cached blocks when the worker's capture cursor is behind the scheduler's
requested output cursor. `take_output` trims the worker result to the accepted
token range before the scheduler attaches it to the engine output.

Relevant boundaries were:

- `AuxOutputConfig` enabled only `enable_return_routed_experts`.
- `PendingAuxOutput` snapshots R3 and copies it asynchronously alongside
  sampler output.
- `BlockObjectStore` assumes all objects have one fixed byte size, and the R3
  publisher uses one fixed dtype and shape per token.
- Generated logprobs are exposed as `LogprobsLists`; prompt logprobs are
  `LogprobsTensors` collected by `PromptLogprobsWorker` and carried separately
  in `ModelRunnerOutput`.
- Requests with `prompt_logprobs` currently skip local prefix-cache reads in
  `KVCacheManager.get_computed_blocks`, because prompt logprobs otherwise
  cannot be produced for cached prompt tokens.

This means R3's capture/store API is not a suitable direct payload API for
logprobs. In particular, top-k rows have request-dependent width and the
current store cannot hold variable-size records.

## Feasibility

The feature is feasible within the existing connector lifecycle. KV block
hashes already identify token-aligned cache content; the scheduler already
delivers incremental hashes, terminal notifications, and generation changes;
and the worker already handles asynchronous capture, delayed hash arrival,
bounded retention, and replay. These parts should be generalized or reused.

There are three substantive additions:

1. Define a stable, typed logprob artifact representation and compatibility
   identity. A KV block hash alone is not enough: cached scores must not cross
   incompatible model-weight versions, LoRA adapters, tokenizer/model
   identities, logprobs modes, or requested scoring layouts.
2. Capture prompt-score rows before they are discarded or merged by
   `PromptLogprobsWorker`, and allow opted-in requests to use prefix cache while
   fulfilling prompt-logprob output from replayed artifacts.
3. Merge replayed rows with live output using the existing token positions,
   handling async scheduling, speculative rejection, chunked prefill, and
   request-specific top-k/fixed-token requests without altering standard
   `ModelRunnerOutput` behavior for non-opted-in requests.

The main risk is correctness, not basic storage feasibility. Replay must be
fail-closed on missing, malformed, or incompatible artifacts: an RL request
must not silently receive stale or position-shifted scores. The first
implementation should support a deliberately constrained layout and report a
clear error for unsupported combinations.

## Goals and Non-Goals

Goals:

- Replay generated-token logprobs and prompt logprobs for explicitly opted-in
  RL/scoring requests.
- Preserve all existing behavior for requests that do not opt in. In
  particular, the default config remains disabled and normal logprobs continue
  through the existing sampler and prompt-logprob worker.
- Reuse KV-block identity and connector request lifecycle where semantics
  match, while keeping R3 output behavior intact.
- Preserve causal alignment: the score row at token position `p` is the score
  distribution used to score token `p` (computed from the preceding context).

Non-goals for the initial version:

- Changing OpenAI-compatible response schemas or standard API defaults.
- Replaying full-vocabulary logits or arbitrary logits-processor state.
- Replaying across model-weight updates unless a stable weight-version
  fingerprint is explicitly provided.
- Supporting every logprobs mode, speculative-decoding mode, parallel layout,
  or connector combination on day one.

## Proposed API and Opt-In

Add disabled-by-default AuxOutput settings, named consistently with existing
configuration:

- `enable_logprobs_replay`: capture/replay generated-token logprob artifacts.
- `enable_prompt_logprobs_replay`: capture/replay prompt logprob artifacts.
  This currently requires `enable_logprobs_replay`, because the score for a
  token at a KV-block boundary is produced by the preceding token and can be
  delivered by the generated-logprobs path.

The configuration enables the data-plane capability; it does not cause all
requests to capture or expose values. Requests opt in with the internal
`SamplingParams.extra_args["aux_output_replay"]` flag. This is intentionally
not exposed as an OpenAI request field. Normal requests that only ask for standard API
logprobs continue to compute and return them through the current path, not
through the AuxOutput store.

For opted-in requests, requested scoring layout must be part of the artifact
compatibility key. A practical initial contract is:

- Support one fixed, bounded top-k width shared by `logprobs` and
  `prompt_logprobs` when both are requested.
- Reject `-1` (full vocabulary), custom token-ID layouts, processed logprobs,
  and mixed incompatible layouts initially.
- Include the `logprobs_mode` and layout in the versioned artifact key.

## Artifact Model

Keep R3 in its fixed-size store and add a typed causal-logprobs artifact in a
variable-size store. Prompt and generated rows share one position space and,
when their layouts match, one artifact. An artifact block is identified by:

`(schema_version, generation, compatibility_fingerprint, kv_block_hash,
boundary_target_token_id)`

The compatibility fingerprint covers logprobs mode and scoring layout. The KV
block hash supplies token, cache-salt, prompt-embedding, and LoRA identity. The
boundary target token is also required because a block stores causal rows in
`(start, end]`: row `end` includes the selected/target token at `end`, which is
not covered by the KV hash for `[start, end)`. When the trailing boundary token
is not known at scheduling time, the worker keeps the hash pending and binds it
to the sampled token before the first replay lookup. The store is scoped to one
engine/model generation. Online weight updates are rejected because there is
not yet a reliable weight-version identity. Request ID is deliberately absent
so compatible shared-prefix blocks remain reusable.

Prefix caching remains a configuration prerequisite for logprob replay in this
implementation: EngineCore creates the request block hasher only for prefix
cache/KV-cache identities, and those hashes are part of the artifact key. This
requirement is independent from the MoE-only restrictions of routed-expert
capture; logprob-only replay does not require an MoE model.

The stored payload is a versioned NPZ record with explicit absolute positions,
token IDs, score values, and selected-token ranks. Decode validates schema and
shape before reconstructing the existing `LogprobsLists` /
`LogprobsTensors` contract. Integer token IDs/ranks use fixed-width integers,
scores use float32, and loading disables Python pickle.

Two storage approaches are viable:

1. A generalized variable-size object store that records payload length and
   uses per-kind block encoders. This is the preferred direction because
   prompt and sample records have different schemas and configured widths.
2. A separate fixed-shape store per artifact kind and fingerprint. This is a
   smaller first patch if the implementation deliberately restricts each
   namespace to one fixed layout, but it duplicates capacity accounting and
   complicates combined R3/logprob operation.

The chosen design should account capacity in bytes, preserve reference pins
and LRU semantics, and fail closed if an artifact needed for replay was
evicted. The initial implementation applies `max_bytes` independently to the
fixed-size R3 store and variable-size logprob store. A future shared budget or
per-kind quotas can be added after measuring mixed-workload retention.

## Capture and Replay Flow

### Scheduler metadata

Extend per-request scheduler metadata with the requested artifact kinds,
logprob compatibility fingerprint, and the token ranges that need to be
emitted/replayed. Preserve the current R3 start semantics independently:
`routed_experts_prompt_start` must not become the implicit start for logprobs.
Continue sending newly available block hashes and terminal events once, as
today. Logprob block metadata additionally carries one optional boundary target
token ID per hash. A missing trailing ID is resolved from the sampler's selected
token column before replay or publication.

The metadata must distinguish:

- prompt token scores, whose first causal row may be empty/not scoreable and
  whose final prompt input row predicts the first generated token;
- generated-token scores, which align to accepted sampled tokens;
- tokens computed in the current step versus already-computed prefix tokens.

### Worker capture

For generated token logprobs, tap the existing sampler logprob tensors before
conversion/serialization and gather rows by request plus cumulative per-request
generated-token offsets. Store only accepted tokens; speculative rejected
suffixes must never be published as canonical artifacts. Maintain an accepted
generated-row cursor independently from the scheduler's optimistic token start,
so the next async step reconnects after a rejected speculative suffix.

For prompt logprobs, capture token-aligned rows from `PromptLogprobsWorker`
after its chunk aggregation and before the result is discarded. Preserve the
cached-prefix boundary when the GPU runner returns a full-prompt CPU tensor:
its cached prefix is allocated but not populated, so only the live suffix may
be merged with replayed rows. Preserve the current chunked-prefill behavior. To
make prompt-logprob cache hits useful, an
opted-in request may read prefix KV blocks; the connector supplies prompt
logprobs for the cached portion and the worker computes the uncached portion
as usual.

The scheduler consumes prompt replay only on a step that emits an engine
output. Intermediate chunked-prefill steps may carry captured prompt tensors
while `num_sampled == 0`; they must not be treated as missing artifacts.

Publish known rows once the corresponding block hash is available. Logprob
blocks may be sparse at first (notably generated-only replay at the
prompt/decode boundary) and are atomically replaced by a merged artifact as
more causal rows become available. Prompt replay still requires the final
materialized range to contain every position; missing or evicted rows fail
closed. Hashes may arrive later than capture, so pending rows are retained
until their hash update arrives, just as R3 does today.

### Worker replay and scheduler merge

On a cache hit, read artifacts by block hash and materialize only the rows in
the request's needed range. Merge replayed and newly computed rows into the
existing output structures on the scheduler/output path. The merge must use
absolute token positions, not array concatenation assumptions, so it remains
correct under chunking, async output lag, preemption/resume, and stale
speculative output.

For prompt logprobs, replace the current blanket `skip_reading_prefix_cache`
behavior only for opted-in requests when the connector is enabled and the
required artifact fingerprint/layout is known. Non-opted-in requests retain
the current cache bypass and normal prompt-logprob computation. If replay is
not available, the system should either recompute the needed prompt (with an
explicit opt-in fallback policy) or fail clearly; silently omitting rows is
not acceptable for RL consumers.

## Non-RL Path Isolation

The following invariants are required:

- All new config flags default to `False`; `AuxOutputConfig.enabled` only
  becomes true when at least one auxiliary artifact feature is enabled.
- Requests without the RL/replay opt-in do not get extra capture, D2H copies,
  block storage, lookups, or output fields. Existing standard logprobs remain
  sourced from sampler output and `PromptLogprobsWorker`.
- R3-only deployments preserve current capture shape, output semantics,
  compatibility checks, and capacity behavior as far as possible.
- The no-AuxOutput path must not add branches in the hot model forward beyond
  a disabled/`None` check already used by the current connector integration.
- Existing OpenAI API response types and defaults do not change.

## Compatibility and Initial Limitations

The first implementation gates unsupported combinations during config
validation or request admission with actionable errors. It supports only
`raw_logprobs`, fixed top-k layouts, and matching generated/prompt widths;
full-vocabulary `-1`, custom token-ID layouts, prompt-only replay, and online
weight updates are rejected. Prompt replay requires generated logprobs replay
to supply causal block-boundary rows. R3-specific MoE, adaptive verification,
PP, and context-parallel restrictions remain scoped to R3 so they do not
accidentally constrain logprob-only configurations.

The `ModelRunnerOutput` / `EngineCoreOutput` transport currently represents
sample and prompt logprobs differently. Decide whether the connector returns a
typed `AuxRequestOutput` containing optional per-kind ranges or whether
replayed logprobs are merged into the existing `logprobs` and
`prompt_logprobs_dict` fields before scheduler consumption. Prefer the latter
if it can preserve existing ownership and serialization contracts; do not
expose a second conflicting source of truth to API output processing.

## Implementation Phases

1. **Contract and identity:** settle request opt-in, supported logprobs modes,
   fixed-token vs top-k shape, model/LoRA/weight fingerprint source, and
   behavior on cache miss/eviction. Add config validation and unit tests for
   isolation and unsupported layouts.
2. **Typed storage:** generalize block object encoding/storage while retaining
   existing R3 helpers or adapting them without changing R3 output. Test
   round-trip, duplicate hash reuse, late hash arrival, eviction, generation
   reset, and reference release.
3. **Generated logprobs:** capture accepted sampler rows, publish block
   artifacts, replay rows for shared-prefix continuations, and merge with
   `LogprobsLists`. Test variable generated-token counts and speculative
   acceptance/rejection.
4. **Prompt logprobs:** capture rows across chunked prefill, enable prefix hits
   only for opted-in compatible requests, replay cached prompt rows, compute
   the uncached suffix, and reconstruct `LogprobsTensors`. Test first-token
   boundary, chunk boundaries, full hit, partial hit, and preemption/resume.
5. **End-to-end RL validation:** compare replayed outputs against cache-disabled
   reference outputs exactly for token IDs/ranks and within an explicitly
   defined tolerance for score values. Measure throughput, D2H volume, and
   store capacity under mixed R3/logprob load. Confirm non-opted-in requests
   have no behavior change.

## Test Strategy

Keep tests close to the existing
`tests/distributed/aux_output_connector/test_store.py` and extend the focused
model-runner/scheduler suites for output alignment. Required cases include:

- default-off and non-opted-in requests retain the legacy path;
- same block hashes plus same fingerprint replay identical rows;
- different model/LoRA/layout fingerprints do not share artifacts;
- output ranges remain correct with partial prefix hits and async stale output;
- speculative rejected rows are not published;
- prompt logprobs are correct for chunked prefill and cached-prefix replay;
- missing/evicted artifacts fail or follow the explicitly selected recompute
  policy, never return partial silent results;
- generation reset, request finish, and preemption release references without
  leaking entries;
- R3-only and mixed R3/logprob runs preserve R3 values and obey the byte cap.

## Open Decisions

- Whether the internal `SamplingParams.extra_args["aux_output_replay"]` flag
  should later be generalized into a caller-controlled AuxOutput request context.
- Which component owns the authoritative model-weight version when online
  weight transfer is enabled?
- Should store misses trigger recomputation (simpler serving semantics, higher
  compute) or fail fast (stronger RL data integrity)? This should be explicit
  per workload, with fail-fast as the default for replay-required RL requests.
- Future selected-token-ID support must use an independent compatibility
  fingerprint; the initial implementation supports fixed-width top-k only.
- Should per-kind capacity quotas be added after measuring the effect of shared
  R3/logprob retention?

## Source References

- `vllm/distributed/aux_output_connector/connector.py`
- `vllm/distributed/aux_output_connector/worker.py`
- `vllm/distributed/aux_output_connector/store.py`
- `vllm/distributed/aux_output_connector/routed_experts.py`
- `vllm/config/aux_output.py`
- `vllm/v1/worker/gpu/sample/logprob.py`
- `vllm/v1/worker/gpu/sample/prompt_logprob.py`
- `vllm/v1/outputs.py`
- `vllm/v1/core/kv_cache_manager.py`
