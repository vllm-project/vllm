# Steering Runtime Design

This document describes the runtime design of activation steering in vLLM.
It is intended for contributors working on scheduler, worker, model wiring,
prefix caching, or graph execution.

For user-facing setup and examples, see [Activation Steering](../features/steering.md).

## Design Goals

The runtime has to satisfy all of the following at once:

- support global and per-request steering
- distinguish prefill and decode behavior
- preserve prefix-cache correctness
- work under continuous batching
- work with `torch.compile` and CUDA graph replay
- avoid per-step graph recompilation or graph invalidation

That combination is what makes steering more subtle than a simple residual add.

## Core Model

Steering is represented as effective phase-specific configs:

```text
effective_prefill = base + prefill_specific
effective_decode  = base + decode_specific
```

This exists independently for:

- global steering state
- per-request steering state

At runtime, the effective per-token steering seen by the model is:

```text
prefill token -> global_prefill_effective + request_prefill_effective
decode token  -> global_decode_effective  + request_decode_effective
```

## Main Components

### `SamplingParams`

`SamplingParams` owns the user-facing fields:

- `steering_vectors`
- `prefill_steering_vectors`
- `decode_steering_vectors`

It resolves them into phase-specific effective vectors and hashes.

### `Request`

Each request carries:

- `prefill_steering_config_hash`
- `decode_steering_config_hash`

These are phase-specific identities used by:

- scheduler admission
- worker registration
- prefix-cache key generation for prefill

### `Scheduler`

The scheduler admits requests subject to steering-capacity limits and must
reason about phase transitions:

- prefill requests consume prefill steering capacity
- decode requests consume decode steering capacity
- requests near the prefill/decode boundary may require transition-aware
  reservation so decode admission does not fail mid-step

### `SteeringManager`

The worker-side `SteeringManager` owns:

- per-request config registration
- refcounting
- table row assignment
- global vector caches
- population of per-layer steering tables

Rows are phase-aware. A config hash is not enough on its own; the manager
must also know whether that hash is registered as prefill or decode.

### Model Runner

The model runner assembles:

- request-to-row mappings
- token-to-row steering index buffers
- per-layer steering tables

It assumes the scheduler has already reserved a row for any per-request
steering hash that flows through. If the worker ever sees a hash it has not
registered, that is a scheduler accounting bug and the worker raises rather
than silently substituting global rows.

## Table and Index Layout

Each steerable layer owns a steering table per hook point. Conceptually:

| Row | Meaning |
| --- | --- |
| 0 | no steering |
| 1 | global effective prefill |
| 2 | global effective decode |
| 3+ | per-request rows |

All layers share the same token-to-row index for a step. Different hook
points reuse the same row mapping but look up different per-hook tables.

That is why steering supports multiple hook points without multiplying the
per-token bookkeeping cost.

## Phase Semantics

The critical invariant is:

- prefill semantics are tied to prompt-token KV creation
- decode semantics are tied to generated-token continuation

This affects both admission and APC.

Phase detection cannot rely on trivial heuristics like "one token means
decode". It has to use the actual request state, because chunked prefill,
full cache hits, and resumed streaming requests all complicate that boundary.

## Prefix Cache Semantics

Prefix caching must separate requests whose prompt KV differs.

That means:

- prefill steering is part of the cache identity
- decode-only steering is not part of prompt KV identity
- global base/prefill steering changes invalidate cache reuse

The cache key integration happens through extra block-hash components attached
to prefill hashing.

### Why Streaming Continuation Is Tricky

In resumable streaming, prior output tokens can be folded back into the prompt
for the next turn. That means a block that was previously decode-only can
become a prompt block in the continued request.

When that happens, the request must refresh all APC-related state:

- phase-specific steering hashes
- block-hash override fields used by cache hashing
- any block-hash chain whose old phase interpretation is now stale
- prefix-cache read policy derived from current sampling params

If those are not refreshed together, cache hits and misses become incorrect.

## Strict Capacity Contract

Capacity bookkeeping is owned by the scheduler. Before a request that uses
per-request steering can be dispatched, the scheduler must reserve a row for
its phase-specific hash in the running set; if no row is available, the
request stays in the waiting queue.

The worker enforces the contract loudly:

- `SteeringManager.register_config` raises `RuntimeError` when called past
  `max_steering_configs`. The scheduler should never let this fire — when it
  does, that is a scheduler bug and the exception propagates up through the
  runner.
- `SteeringManager.get_row_for_config` raises `RuntimeError` for unregistered
  nonzero hashes for the same reason. There is no silent fallback to row 1/2
  ("global"); a worker miss can no longer mask itself as the request running
  with global-only steering.

The runner has no pending-registration retry loop and no pending-globals
queue. State is initialised eagerly during `load_model`, so by the time the
HTTP API or the scheduler can talk to the runner the manager already exists,
and `set_steering_vectors` writes straight through.

This is the place where scheduler capacity logic and worker state are most
tightly coupled. The scheduler tracks `scheduled_steering_configs` for both
the running set (including transition reservations for requests about to
finish prefill) and the waiting admission check; reaching the worker without
that reservation now crashes loudly rather than degrading silently.

## Continuous Batching

Steering has to work with mixed batches containing:

- unsteered requests
- globally steered requests
- distinct per-request steered requests
- prefill and decode tokens in the same step

The key runtime requirement is that every token in the flattened batch maps
to the correct steering row for its request and phase.

That is why the system uses request-aware row assignment plus a per-step token
index buffer rather than trying to mutate model weights directly.

## `torch.compile` and CUDA Graphs

Steering correctness under compiled execution depends on one core rule:

- graph replay must read live steering buffers, not constants specialized at trace time

The current design achieves that by using persistent GPU buffers and an opaque
custom steering op. Steering data is updated in-place between steps, and graph
replay observes the updated buffer contents.

Important consequences:

- steering changes do not require recompiling the model
- graph replay can serve requests with different steering configs across steps
- correctness depends on buffer updates and row/index population happening
  before the forward pass

## Extending Steering to New Models

To add steering to another model family, contributors need to wire:

- layer indices
- per-hook steering tables
- the shared steering index
- `apply_steering` calls at the intended residual-stream hook points

The extension work is model-specific, but the runtime invariants above do not
change.

## Current Boundaries

This design document reflects the v1 steering runtime. Known boundaries:

- no v2 model runner integration yet (v2 is dev-flag-gated in vllm; steering
  integration is pending)
- see [Activation Steering](../features/steering.md#supported-scope) for the
  current list of wired decoder architectures

## Distributed Execution

Steering runs on every tensor-parallel (TP) and pipeline-parallel (PP) rank
and relies on a *determinism contract* rather than cross-rank collectives in
the hot path.

### Determinism contract

> Steering state is shared-nothing with deterministic replay. Every worker
> executes identical `set_steering_vectors` / `clear_steering_vectors` calls
> (via `collective_rpc`) and sees an identical `SchedulerOutput` stream, so
> every worker's `SteeringManager` derives identical `config_to_row`
> assignments, identical `free_rows` state, and an identical
> `steering_index` tensor each step — even though each worker stores
> vectors only for layers it physically owns. No cross-rank collectives
> are needed in the hot path.

### What each rank stores vs. doesn't

Per rank:

- **Full** `config_to_row`, `free_rows`, `pending` queues on the local
  `SteeringManager`. Row allocation is fully rank-local and runs for every
  config on every rank, even configs whose layers are all owned by other
  PP stages. Row ids flow through `steering_index` into the
  `apply_steering` gather on every rank, so they *must* match across ranks.
- **Only locally-owned tensors** in `global_base_vectors`,
  `global_prefill_vectors`, `global_decode_vectors`, and per-config tensor
  dicts (PR 1 filters these by `locally_owned_layers`).
- A `steering_index` tensor shared across all layers on this worker.

What does *not* happen at the worker layer:

- NCCL all-reduce / broadcast of steering tables.
- Any inter-rank coordination of row ids.
- Any inter-rank coordination of `SchedulerOutput`.

### Collective operations

- `POST /v1/steering/set`, `POST /v1/steering/clear`: one
  `collective_rpc` per endpoint call (two for set — validate then apply).
  All ranks receive identical kwargs.
- `GET /v1/steering`, `GET /v1/steering/layers`: one read-only
  `collective_rpc` each; router aggregates.
- Per-forward-step: none.

### Row-allocation invariants

Because every rank processes the same sequence of `register_config` calls
(driven by the shared `SchedulerOutput`), their `SteeringManager` state
stays in lock-step:

- `config_to_row` is identical on every rank.
- `free_rows` is identical on every rank.
- `steering_index` is identical on every rank.

This holds even for configs whose layers are not locally owned: row
allocation is independent of whether tensors actually materialize on this
rank.

### Failure modes and debugging

- **TP divergence** (TP ranks within the same PP stage reporting different
  valid layer sets from `set_steering_vectors` validate) is a *server-side
  invariant violation*, not user error. The router returns HTTP 500 with a
  message naming the diverging ranks. Typical root cause: model-loading
  asymmetry (one rank loaded different weights).
- **PP disjointness** is expected. PP ranks report disjoint layer sets;
  the router unions them.
- Deep-merge of `GET /v1/steering` payloads asserts per-triple equality.
  If two workers report the same `(layer, hook, norm_key)` with different
  values, the merge raises and the router returns HTTP 500.
- Debug endpoints:
    - `GET /v1/steering/layers` — per-layer hook-point availability
    aggregated across ranks. Useful to confirm which layers are
    steerable on the current model shape.
    - `GET /v1/steering` — per-layer active norms aggregated across
    ranks.

## Named Steering Modules (runtime)

Named steering modules are pre-registered vector configurations that requests
reference by name instead of sending vectors inline. The runtime shape is:

- The registry lives on FastAPI app state
  (`app.state.steering_module_registry`), populated either at startup via
  `--steering-modules` or at runtime via
  `POST /v1/steering/modules/register`. The registry implementation is in
  `vllm/entrypoints/openai/steering/registry.py`.
- Resolution happens in the OpenAI serving handlers
  (`chat_completion/serving.py`, `completion/serving.py`) when a request
  specifies `steering_name` in `extra_body`. The resolver looks up the named
  module, merges it with any inline `steering_vectors` fields via
  `merge_steering_specs`, and writes the merged spec back onto
  `SamplingParams` before the request enters the scheduler.
- The scheduler and worker do not distinguish named from inline vectors once
  the spec is on `SamplingParams` — the rest of the runtime sees only the
  final resolved vectors.
