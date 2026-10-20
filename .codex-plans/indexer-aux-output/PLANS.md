# Indexer AuxOutput migration

## Scope and baseline

Initial audit baseline: upstream `main` at `8407bc0bb6`, fetched on 2026-10-20.
Before committing, fetched and merged `main` again at `cc8a41f328`.
Existing PR: vllm-project/vllm#49555, head `f4f96b90e9`.
The user's current instruction authorizes implementation and commits directly,
and defers all test execution until interactive code review is complete.
This supersedes the attachment's audit-only/APPROVED gate for this iteration.

Duplicate checks read #47280 and its comments, open PRs mentioning that issue,
and open indexer/replay PRs. #49555 is the existing feature PR to update.
#57968 adds a separate Mooncake backend; this implementation does not depend on it.
The PR conversation contains stale merge-conflict and pre-commit bot notices,
plus historical validation of the old Slot Store implementation. Those results
do not validate this migration.

## Existing architecture

- `config/aux_output.py`: R3 flag, fixed SHM capacity, forward graph hash.
- `connector.py`: one generation and incremental KV hashes; terminal events.
- `worker.py`: one pending-output queue, accepted-token cursors, publication and
  prefix materialization, plus deferred cleanup while output is in flight.
- `routed_experts.py`: dtype/shape-parameterized block/tail pool and store helpers.
- `store.py`: anonymous shared-memory arena, fixed object slots, references and LRU.
- MRV2 `AsyncOutput`: the existing copy stream and event own D2H completion.
- Scheduler clips to accepted logical positions; OutputProcessor accumulates
  chunks and accounts for API-side string stops.

## Model audit and hooks

| Model | Capture location | Selection and mapping |
| --- | --- | --- |
| DeepSeek V3.2 | `DeepseekV32Attention._sparse_indexer_and_attn`; ROCm `_fused_attention` | `index_topk` request-relative token IDs; producer sources explicitly map to IndexCache consumers |
| GLM-5.2 | Registry `GlmMoeDsaForCausalLM` uses DeepSeek V3.2 | Same token semantics; no inferred KPool encoding |
| GLM-5.3/Flash | `glm5next/common/attention.py:Indexer.forward` after its backend returns | Pool expansion and TP row exchange have completed; compact representation retains pool order, per-pool validity and literal causal tail |
| DeepSeek V4 | Common `_sparse_indexer_and_attn` | Only C4A instantiates an indexer; IDs refer to compressed cache rows, ratio 4; HCA/SWA excluded |
| DeepSeek V4.1 | Common `_sparse_indexer_and_attn`, after final candidate filtering | Only configured index sources; raw final compressed IDs and explicit consumer mapping; candidates are not persisted separately |
| Qwen4Exp NVIDIA | `QSAIndexer.forward`, before returning the expanded result | Block IDs plus logical query position and visible blocks; decoder reproduces token expansion, tail and count column |
| Qwen4Exp AMD | `QSAIndexer.forward` after `_select` | Backend's actual token selection, raw int32; no NVIDIA count-column assumption |
| Step5 NVIDIA | `_forward_prefill` after selection; decode adapter immediately after selection | Actual ordered Region IDs per KV group; current Region and original selection width preserved; TP groups gathered in canonical order |
| HY-V4 NVIDIA | `_indexer_and_attn` | Independent full/shared indexer configuration; only actual sources captured |
| Dots3 NOTE | Reference MLA wrapper and its overridden forward | Only full/DSA attention, before physical KV translation; sliding-window layers excluded |
| GLM4-MoE-Lite | Reference DeepSeek MLA wrapper | Sparse configurations use explicit IndexCache sources; dense configurations have no Indexer artifact |
| MiniMax-M3 | Audited `common/indexer.py` and NVIDIA/AMD implementations | Separate per-KV-head Block selection, init/local blocks and fused sparse-table construction; adapter deferred, unsupported source discovery fails explicitly |

Both NVIDIA and AMD DeepSeek V4/V4.1 preparation paths join through the common
capture location. Short-context indexers may fill indices without scoring; the
hook still captures that filled buffer. Dense-MHA scoring elision is disabled
for the V3.2/reference-wrapper Replay producers so they never export stale rows.
MTP sources are excluded by backbone layer bounds; only the target model is bound.

## Design

1. `AuxOutputProfile` identifies model/revision/configuration, codec version,
   actual source layers, consumers, semantics and persistent shape.
2. Model-owned sources register descriptors only when Replay is enabled.
   `IndexerCapturer` binds fixed GPU views; each producer writes before the
   shared source buffer can be overwritten. Capture and bounded snapshot pools
   are allocated during model loading, inside GPU memory profiling, before KV
   planning. Slots are released only when their asynchronous output is consumed.
   A copied validity bitmap detects
   missing producer/token rows after the existing D2H event.
3. A combined fixed byte record contains optional R3 and Indexer data. It uses
   the existing worker cursors, hash stream, tail pool, background writer,
   retain/release operations and request finish events. No second Scheduler.
4. R3-only storage and namespaces remain unchanged. Combined/Indexer keys add
   profile identity to the generation namespace; combined records also name
   the R3 shape/dtype. Independent prompt offsets are clipped at the Scheduler.
5. Packed output stays compact through OutputProcessor. Raw output uses the
   profile's CPU decoder at final assembly. HTTP extensions are terminal-only.
6. Preemption rewinds temporary tails and pending blocks. Non-aligned resume
   recovers the missing boundary prefix from pending rows or a hashed object.

## Files

- Config, EngineArgs, LLM and SamplingParams: independent flags and offsets.
- New `profile.py`: schema, identity, decoder and SHM capacity calculation.
- New `indexer_capture.py`: stable device views, model source binding, snapshots.
- Model files in the table: explicit source descriptors and consumption hooks.
- AuxOutput Worker/Connector: combined payload and independent output cropping.
- Existing block buffer: tail rewind and presence query; Store: slot-size check.
- Scheduler, EngineCoreOutput, OutputProcessor, CompletionOutput: transport,
  accepted-range clipping, accumulation and stop-string trimming.
- OpenAI Chat/Completion models and serving: raw/packed vLLM extensions.
- Rust engine wire model: append fields without shifting existing positions.
- Existing store/config/output tests and new source/codec CPU oracle suite.

## Memory planning

Let `S` be combined persistent bytes/token, `B` the Scheduler block size,
`H` the hash block size, `N` KV blocks. Store need is `N * B * S`, object size
`H * S`. Capacity below this need fails before creating the mmap.

Worker staging includes the existing bounded full-block/tail pool and concurrent
host step copies. Budget diagnostics add an estimate of 512 bytes/object and
host reserve `max(512 MiB, 5% available RAM)`, and multiply store/staging estimates
by local DP output-worker count. Existing allocated host/offload memory is
reflected in available RAM. These are estimates, not allocator guarantees or
atomic reservations across independent processes.

Per-source bytes/token, where `K` is effective topk, `P` pool ratio,
`R` QSA compression ratio and `G` global KV groups:

| Format | Raw | Default persistent |
| --- | --- | --- |
| DeepSeek/GLM-5.2 | `4*K` | `4*K` |
| GLM KPool | `4*(K+P-1)` | `4*(K/P+P-1)` if the pool mask fits int32; otherwise raw |
| QSA NVIDIA | `4*(K+R)` including count | `4*(K/R+2)` |
| QSA AMD | `4*(K+R-1)` | raw |
| Step5 | `4*G*(K+2)` | same; includes current Region and original width |

Sum actual sources, then add R3 `layers * expert_topk * expert_dtype_bytes`.
No CPU-budget ratio is converted into a suggested GPU utilization percentage.

## Test design and matrix

The module's contract is selection rows -> lossless profile record -> accepted
logical rows, with no missing prefix or stale source. Cheapest checks are
CPU source/codec oracles and the real block-store/connector state machine.
Extend existing suites rather than duplicating their helpers.

- Source and codec oracles: token/compressed IDs, unsorted pools, partial causal
  pools, tail/padding, QSA count column, Step5 groups and prefill/decode layout.
- Snapshot isolation and completeness: later producers and steps cannot modify
  pending snapshots; omitted sources remain invalid.
- Store/Worker: fixed slots, SHM budget, full/tail blocks, delayed hashes,
  same-batch publication, shared prefixes, eviction, reset, finish while pending,
  speculative rejection, optimistic cursors, preemption and separate offsets.
- Output: combined R3/Indexer, raw/packed accumulation, terminal publication,
  stop trimming, HTTP models and appended IPC fields.
- Existing R3 regressions: config, store, capture, scheduler/async scheduler and
  OutputProcessor suites listed in the task attachment.
- Future GPU acceptance: compare each backend's selection at the hook against
  decoded Replay, including padding/counts, prefix hits, MTP rejection and real
  preemption. Repeat with `VLLM_GPU_SYNC_CHECK=error` and measure enabled/disabled
  overhead. This iteration deliberately executes no tests or model evaluations.

The new CPU oracle file is collected by existing model-executor Buildkite jobs
on CUDA, AMD and Intel; this does not provide real-model GPU selection coverage.

## Review boundaries and risks

- DeepSeek V4.1 decoder replay retains all capture rows when either AuxOutput
  artifact is enabled, including the Indexer-only mode.
- MRV2 generate runner and prefix caching are required. Existing PP, DCP/PCP,
  adaptive-verification and remote-PD restrictions remain in effect.
- MiniMax-M3 has no adapter in this iteration. CPU/TPU/OOT inference backends,
  external pluggable MLA replacements and Rust HTTP Replay delivery are not
  claimed supported. The Rust raw engine protocol retains the new fields.
- GPU hooks, CUDA graphs, TP collectives, accuracy and performance are untested.
- Pool compaction currently encodes the final expanded selection; moving capture
  before expansion is a later optimization requiring the same post-exchange
  ordering and short-context validity guarantees.
- SHM retains the existing immutable first-writer contract. Recomputed rows may
  reuse a prior same-position stored selection; no new object version protocol.
- Human line-by-line review, CPU regressions, model evaluations and real GPU
  comparisons remain prerequisites to publishing validated support claims.

Suggested review slices: profile/lifecycle; DeepSeek and GLM-5.2; V4/V4.1;
GLM KPool/QSA; Step5 plus additional model audits. They currently share one
implementation commit to keep the existing PR update reviewable as requested.
