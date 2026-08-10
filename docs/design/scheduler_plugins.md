# Scheduler Plugin Framework

## Status

This document proposes a V1 scheduler plugin framework. It does not change the
scheduler's ownership of batching, token budgets, KV cache allocation, or
preemption safety.

## Motivation

The V1 scheduler currently supports `fcfs` and `priority` policies directly in
the scheduler core. Extending it with session affinity, prefix-cache locality,
or SLA-aware ordering requires either modifying the core or replacing the
entire scheduler through `scheduler_cls`.

Replacing the scheduler has a large compatibility surface. Most scheduling
extensions only need to influence a few decisions while retaining the core
scheduler's correctness model. The framework therefore provides one plugin
registration system with multiple extension points, following the structure of
the Kubernetes Scheduler Framework.

The initial goals are:

- express the existing `fcfs` and `priority` policies as built-in plugins;
- preserve their observable behavior and existing configuration;
- support out-of-tree filtering, scoring, and preemption ranking;
- enable session affinity and local prefix-cache locality;
- leave room for SLA and QoS policies without exposing KV cache mutation;
- keep the no-extension fast path equivalent to the current scheduler.

This framework schedules requests within one V1 scheduler instance. Selecting
an engine or data-parallel rank is a routing decision and is out of scope.

## Non-goals

- Replacing the V1 scheduler core.
- Allowing plugins to allocate or free KV cache blocks.
- Allowing plugins to construct `SchedulerOutput`.
- Adding an RPC-based or external scheduler.
- Invoking plugins for every generated token.
- Providing cross-engine or cross-rank session affinity.
- Making arbitrary scheduler internals a stable public API.

## Current behavior

`SchedulerConfig.policy` accepts `fcfs` or `priority`. The selected policy
currently controls three behaviors.

### Waiting queues

`fcfs` uses deques. New requests are appended, and preempted requests are
prepended. `priority` uses heaps ordered by:

```python
(request.priority, request.arrival_time, request.request_id, id(request))
```

A lower priority value is scheduled first.

### Waiting and skipped queue selection

The scheduler keeps separate `waiting` and `skipped_waiting` queues. Under
`fcfs`, previously skipped requests are retried before the regular waiting
queue. Under `priority`, the two queue heads are compared and the request with
the higher priority is selected.

### Preemption

When KV allocation fails, `fcfs` preempts the last running request. `priority`
preempts the running request with the largest `(priority, arrival_time)` key.

All three behaviors must remain compatible when the policies become plugins.

## Design overview

There is one `SchedulerPlugin` type and one `SchedulerPluginManager`. A plugin
may implement any subset of the framework's extension points.

```text
request admission
       |
       v
   QueueSort
       |
       v
candidate selection
       |
       v
     Filter
       |
       v
      Score
       |
       v
core feasibility and KV allocation
       |
       +---- allocation failure ----> PreemptionScore
       |
       v
SchedulerOutput
```

The scheduler core invokes extension points and applies their results. Plugins
never receive mutable scheduler queues, block pools, or cache managers.

## Plugin contract

The public plugin object uses a small base class with no-op defaults. This
allows one plugin to implement multiple extension points without requiring
unrelated methods.

```python
class SchedulerPlugin:
    name: str

    def queue_sort_key(self, request: Request) -> tuple:
        raise NotImplementedError

    def filter(
        self,
        request: Request,
        state: SchedulingCycleState,
    ) -> FilterResult:
        return FilterResult.allow()

    def score(
        self,
        request: Request,
        state: SchedulingCycleState,
    ) -> float:
        return 0.0

    def preemption_score(
        self,
        request: Request,
        state: PreemptionCycleState,
    ) -> float:
        return 0.0
```

Extension-point membership is explicit in configuration. The manager does not
infer membership by checking whether a method was overridden.

Lifecycle callbacks may be added after the scheduling extension points are
stable. They are not required for the first implementation. In particular, the
framework must not introduce a callback on every decode step.

## Extension points

### QueueSort

QueueSort defines the base ordering and preemption reinsertion behavior. Exactly
one QueueSort plugin is enabled in a scheduler profile.

Multiple QueueSort plugins are not supported because independently defined
ordering relations cannot be composed reliably.

The framework owns the queues. A QueueSort plugin supplies ordering operations
and cannot add, remove, or mutate requests itself. The initial internal
contract includes:

```python
class QueueSortPlugin(SchedulerPlugin):
    def queue_sort_key(self, request: Request) -> tuple: ...

    def select_queue(
        self,
        waiting_head: Request | None,
        skipped_head: Request | None,
    ) -> WaitingQueue: ...

    def preempted_request_position(self) -> QueuePosition: ...
```

`select_queue` receives only queue heads and returns an enum. The core performs
the actual pop. `preempted_request_position` is also declarative; the core
performs reinsertion.

The concrete queue representation remains an internal optimization. Built-in
FCFS can retain deque operations and built-in priority can retain heap
operations, so migrating to plugins does not require a slower generic queue.

### Filter

Filter determines whether a waiting request may be considered in the current
scheduling cycle.

```python
@dataclass(frozen=True)
class FilterResult:
    allowed: bool
    reason: str | None = None
```

All enabled Filter plugins must allow a request. A rejection means "skip for
this cycle", not "finish the request". The scheduler retains ownership of the
request and ensures that it remains queued.

Filter runs only after core-independent blockers, such as unavailable grammar
or remote KV state, have been handled. Core feasibility checks involving LoRA,
encoder capacity, the token budget, and KV allocation remain authoritative.

### Score

Score ranks candidates that passed Filter. Multiple scores are combined using
configured weights:

```python
total_score = sum(
    plugin.score(request, state) * weight
    for plugin, weight in score_plugins
)
```

Higher scores are preferred. QueueSort is the deterministic tie-breaker. Score
results must be finite numbers; NaN and infinity are configuration or runtime
errors rather than valid priorities.

Score operates on a bounded candidate window. This makes plugin overhead
independent of an unbounded waiting queue and limits how far a request can move
ahead of the base QueueSort policy.

### PreemptionScore

PreemptionScore ranks requests that the scheduler core has determined are safe
preemption candidates. It does not execute preemption.

Higher preemption scores mean that a request is a more desirable victim. The
core remains responsible for:

- constructing the candidate set;
- restoring the token and encoder budgets;
- freeing KV and encoder cache state;
- resetting request state;
- reinserting the request into the waiting queue.

PreemptionScore is separate from Score because admission preference and
preemption cost are not necessarily inverse relationships.

## Cycle state

Plugins receive a read-only view built for one scheduling cycle.

```python
@dataclass(frozen=True)
class CandidateInfo:
    queue_position: int
    waiting_time: float
    local_cached_tokens: int


@dataclass(frozen=True)
class SchedulingCycleState:
    now: float
    token_budget: int
    encoder_budget: int
    num_running_requests: int
    candidates: Mapping[str, CandidateInfo]
```

The initial state intentionally excludes `KVCacheManager`, request queues, and
mutable scheduler collections.

`local_cached_tokens` is computed by a side-effect-free core API. Probing cache
locality must not allocate blocks, update LRU state, publish KV events, update
hit metrics, or initiate remote KV transfers.

State is constructed lazily. If no enabled extension point needs candidate
cache information, the scheduler does not perform cache probes.

## Built-in plugins

### FCFS

The `fcfs` built-in plugin implements QueueSort and PreemptionScore. Its queue
operations preserve the existing deque behavior, including skipped-queue retry
and preempted-request prepend behavior.

Its PreemptionScore preserves selection of the last running request. It must
use the running-list position supplied by the core rather than reconstructing
the position from request timestamps.

### Priority

The `priority` built-in plugin implements QueueSort and PreemptionScore.
QueueSort preserves `Request.__lt__` ordering. PreemptionScore preserves the
current selection of the largest `(priority, arrival_time)` value.

The initial migration deliberately retains the current difference between the
queue tie-break key and preemption key. Changing that behavior is a separate
policy change, not part of pluginization.

## Registration and profiles

In-tree plugins are registered in a built-in registry:

```python
BUILTIN_SCHEDULER_PLUGINS = {
    "fcfs": FCFSSchedulerPlugin,
    "priority": PrioritySchedulerPlugin,
}
```

Out-of-tree plugins use a dedicated entry-point group:

```toml
[project.entry-points."vllm.scheduler_plugins"]
session-affinity = "my_package.scheduler:SessionAffinityPlugin"
```

Scheduler plugins execute in the engine-core process. Discovery must not
instantiate them in API-only or worker-only processes.

Only explicitly configured scheduler plugins are instantiated. Merely
installing a package must not alter scheduling behavior.

A profile selects plugins by extension point:

```yaml
queue_sort:
  enabled:
    - name: fcfs

filter:
  enabled:
    - name: tenant-quota

score:
  enabled:
    - name: session-affinity
      weight: 100
    - name: prefix-locality
      weight: 1

preemption_score:
  enabled:
    - name: fcfs
```

The Python configuration should use typed dataclasses rather than accepting an
unvalidated dictionary. The YAML form above only illustrates the structure.

## Backward compatibility

`SchedulerConfig.policy` and the existing CLI option remain supported.

```python
policy="fcfs"
```

is normalized to the built-in profile:

```yaml
queue_sort:
  enabled:
    - name: fcfs
preemption_score:
  enabled:
    - name: fcfs
```

Likewise, `policy="priority"` enables the priority plugin at both extension
points.

The compatibility rules are:

1. If no explicit scheduler plugin profile is supplied, `policy` selects the
   equivalent built-in profile.
2. An explicit QueueSort plugin and a non-default `policy` cannot both be
   specified. Configuration validation fails with a clear error.
3. `scheduler_cls` remains supported as the whole-scheduler escape hatch.
   Scheduler plugins configure the built-in scheduler and do not wrap an
   arbitrary custom `scheduler_cls` in the first version.
4. The default remains FCFS.
5. Existing priority values and their lower-is-higher meaning do not change.
6. AsyncScheduler inherits the same initialized plugin manager from Scheduler.

## Default fast path

Pluginization must not require candidate scanning on the default path.

When the profile contains only a built-in QueueSort and its matching built-in
PreemptionScore, the manager selects specialized operations equivalent to the
current branches:

```python
request_queue = self._select_waiting_queue_for_scheduling()
request = request_queue.peek_request()
```

Candidate windows, cycle state, cache probes, filter calls, and score fusion are
enabled only when the corresponding extension points contain additional
plugins.

The built-in policy objects are created once at scheduler initialization. No
entry-point discovery or dynamic import occurs in `schedule()`.

## Candidate selection

With Filter or Score plugins enabled, the framework performs these steps:

1. Ask QueueSort for up to `candidate_window` candidates without removing them.
2. Remove candidates blocked by scheduler-owned request state.
3. Lazily build the requested candidate information.
4. Run Filter plugins in configured order.
5. Run and combine Score plugins.
6. Select the highest score, breaking ties with QueueSort order.
7. Ask the owning queue to remove the selected request.
8. Continue through the existing core scheduling path.

If every candidate is filtered, the scheduler leaves them queued and stops
admission for the cycle. Filtered candidates must not be repeatedly moved
between `waiting` and `skipped_waiting` merely because a plugin rejected them.

The first version should require a positive, explicitly configured candidate
window whenever Filter or Score plugins are enabled.

## Failure behavior

Plugin loading and initialization failures abort scheduler startup.

Runtime exceptions, invalid return types, and non-finite scores abort the
scheduling cycle and surface as engine errors. Silently ignoring a broken
policy would make scheduler behavior unpredictable and could violate SLA or
isolation expectations.

Filter reasons and plugin names should be exposed through debug logging and
future scheduling metrics, but high-cardinality request or session identifiers
must not become metric labels.

## Session affinity and prefix locality

A session-affinity plugin is a Score plugin. A request receives affinity only
when it has a typed `session_id` and a real local cache hit:

```python
class SessionAffinityPlugin(SchedulerPlugin):
    name = "session-affinity"

    def score(self, request, state):
        if request.session_id is None:
            return 0.0
        return float(
            state.candidates[request.request_id].local_cached_tokens
        )
```

This avoids treating a session identifier as proof of locality. Candidate
windows bound reordering, and a later aging plugin can reduce starvation risk.

A generic prefix-locality plugin can score `local_cached_tokens` without
requiring `session_id`.

## SLA and QoS

The first framework version supports SLA and QoS ordering through Filter and
Score, for example tenant concurrency limits and deadline urgency.

Resource reservation and token-budget modification are intentionally deferred.
They require a separate extension point whose result is validated and clamped
by the scheduler core. They must not be approximated by exposing mutable budget
objects to plugins.

## Security and compatibility

Scheduler plugins execute trusted Python code in the engine-core process. They
have the same trust implications as a custom scheduler class even though their
API surface is narrower. Plugins are opt-in and should be included in
`VLLM_PLUGINS` allowlisting behavior where applicable.

The documented plugin interfaces should be versioned before being declared
stable. Internal request fields not present in the interface documentation are
not part of the compatibility guarantee.

## Implementation plan

### Phase 1: behavior-preserving policy migration

- Add the plugin interface, manager, extension-point enums, and built-in
  registry.
- Implement FCFS and priority built-in plugins.
- Normalize `SchedulerConfig.policy` into a built-in plugin profile.
- Replace policy conditionals in queue creation, queue-head selection, and
  preemption victim ranking with manager calls.
- Keep the specialized deque and heap fast paths.
- Run the existing scheduler and priority test suites unchanged.

### Phase 2: Filter and Score

- Add typed profile configuration and explicit candidate-window validation.
- Add non-mutating candidate iteration and removal to RequestQueue.
- Add lazy `SchedulingCycleState` construction.
- Implement Filter AND semantics and weighted Score fusion.
- Add deterministic ordering, invalid-result, and starvation-bound tests.

### Phase 3: locality

- Add the side-effect-free local cache probe.
- Add in-tree reference session-affinity and prefix-locality plugins.
- Measure scheduler CPU overhead and token-level prefix-cache hit rate.

### Phase 4: QoS

- Add request metadata required by accepted QoS use cases.
- Evaluate a core-clamped token-budget hint extension point.
- Evaluate preemption scoring with asynchronous and pipeline scheduling.

## Test plan

Behavior compatibility is the first implementation's primary contract.

- Run all existing scheduler tests without changing expected results.
- Verify FCFS waiting, skipped, prepend, and preemption order.
- Verify priority waiting, skipped, tie-break, and preemption order.
- Verify synchronous and asynchronous schedulers use the same profile.
- Verify an empty extension profile takes the specialized fast path.
- Verify one QueueSort plugin is required and multiple QueueSort plugins fail
  validation.
- Verify multiple Filter plugins use AND semantics.
- Verify multiple weighted Score plugins compose deterministically.
- Verify zero scores preserve QueueSort order.
- Verify filtered requests remain queued.
- Verify plugins cannot access mutable queues or KV cache managers.
- Verify cache probing has no allocation, LRU, event, metric, or remote-transfer
  side effects.
- Benchmark scheduler CPU time with no plugins, one Score plugin, and several
  Score plugins across representative queue lengths.

Model evaluations are not required for the behavior-preserving migration. The
locality phase must report workload-specific cache-hit, throughput, TTFT, and
latency results because it changes request ordering and serving behavior.
