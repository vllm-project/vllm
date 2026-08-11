# Scheduler Plugin Framework

## Status

This document defines the target architecture. The current development branch
contains an earlier scalar-callback prototype; it is not the compatibility or
performance contract described here.

The framework is designed around two equal requirements:

- the built-in scheduler policies must retain their specialized performance;
- out-of-tree plugins must be able to implement new queueing, filtering,
  scoring, preemption, and stateful scheduling policies without replacing the
  scheduler core.

Performance is part of the public contract. An abstraction that preserves
behavior but changes a built-in operation from constant to linear time is not
considered compatible.

## Motivation

The V1 scheduler directly supports FCFS and priority policies. Adding session
affinity, prefix-cache locality, tenant fairness, or SLA-aware ordering
currently requires modifying the core or replacing the complete scheduler via
`scheduler_cls`.

Replacing the scheduler exposes a large and unstable compatibility surface.
Most policies need to influence a bounded set of decisions while retaining the
core scheduler's ownership of token budgets, KV allocation, batching, and
request state transitions. The plugin framework therefore provides multiple
extension points behind a compiled execution plan.

The goals are:

- express FCFS and priority as built-in plugins without slowing their hot paths;
- support out-of-tree QueueSort, Filter, Score, and PreemptionScore plugins;
- support stateful policies through batched lifecycle events;
- compute expensive scheduling features only when an enabled plugin requests
  them;
- bound plugin work independently of the total waiting-queue length;
- make algorithmic complexity and scheduler CPU overhead testable contracts.

The framework schedules requests within one V1 scheduler instance. Selecting
an engine or data-parallel rank remains a routing decision.

## Non-goals

- Replacing the V1 scheduler core.
- Allowing plugins to allocate or free KV blocks.
- Allowing plugins to construct `SchedulerOutput`.
- Adding an RPC-based or external scheduler.
- Invoking plugins for every generated token.
- Providing cross-engine or cross-rank session affinity.
- Exposing mutable scheduler collections as public APIs.
- Treating arbitrary `Request` fields as stable plugin APIs.

## Ownership boundary

The scheduler core owns mechanisms and enforces invariants:

- batching and token budgets;
- KV and encoder-cache allocation;
- scheduler-owned blockers and request state transitions;
- LoRA and multimodal constraints;
- validating plugin results;
- applying queue mutations and preemption;
- constructing `SchedulerOutput`.

Plugins own policy:

- the base waiting-queue discipline;
- current-cycle eligibility;
- admission ranking;
- preemption-victim ranking;
- policy-local incremental state.

Plugins operate on stable request handles and read-only feature views. They do
not receive mutable queues, block pools, cache managers, or scheduler-owned
`Request` objects as part of the public interface.

## Architecture

Plugin discovery and composition occur once during EngineCore initialization:

```text
configured profile
       |
       v
plugin registry and descriptor validation
       |
       v
profile compiler -----> required feature set
       |
       v
CompiledSchedulerPlan
       |
       +---- QueueDiscipline
       +---- CandidatePipeline
       +---- PreemptionPipeline
       +---- subscribed lifecycle callbacks
```

The compiled plan contains direct references to initialized plugins, feature
providers, reusable buffers, fused weights, and specialized operations. The
hot path performs no entry-point discovery, dynamic imports, plugin-type
inspection, or profile normalization.

Profiles containing only a matching built-in QueueSort and PreemptionScore are
compiled to dedicated FCFS or priority plans. They do not execute the generic
candidate pipeline.

## Public data model

### Stable request handles

The core assigns an opaque handle while a request is registered:

```python
RequestHandle = NewType("RequestHandle", int)
```

Handles allow queue disciplines to maintain indexes without retaining public
references to mutable `Request` objects. A handle is valid only until the
corresponding request finishes or is removed from the scheduler.

### Columnar candidate batches

Candidate information is exposed as a read-only, columnar view:

```python
class CandidateBatch(Protocol):
    size: int
    handles: ReadOnlyIntBuffer
    queue_ranks: ReadOnlyIntBuffer

    def feature(self, feature: SchedulerFeature) -> ReadOnlyBuffer: ...
```

The core owns and reuses the underlying storage. Implementations must not
allocate one object or dictionary entry per candidate on every scheduling
iteration. String and structured metadata may be exposed by indirect views
when requested.

The initial feature set includes:

```python
class SchedulerFeature(Enum):
    PRIORITY = auto()
    ARRIVAL_TIME = auto()
    WAITING_TIME = auto()
    QUEUE_POSITION = auto()
    NUM_PROMPT_TOKENS = auto()
    NUM_COMPUTED_TOKENS = auto()
    LOCAL_CACHED_TOKENS = auto()
    SESSION_ID = auto()
    TENANT_ID = auto()
    DEADLINE = auto()
    NUM_PREEMPTIONS = auto()
    RUNNING_TENANT_COUNTS = auto()
```

Adding a feature does not add an eagerly populated field to every cycle.

### Cycle state

Cycle state contains scalar, read-only values that are valid for one compiled
candidate plan:

```python
@dataclass(slots=True, frozen=True)
class SchedulingCycleState:
    now: float
    block_size: int
    token_budget: int
    encoder_budget: int
    num_running_requests: int
    cycle_id: int
```

Request-specific values live in `CandidateBatch`. Plugins must treat the
snapshot as immutable. Core resource feasibility remains authoritative even
when resources change after the snapshot is created.

## Plugin descriptor and contract

An entry point returns a descriptor rather than an unversioned class:

```python
@dataclass(frozen=True)
class SchedulerPluginDescriptor:
    name: str
    api_version: int
    plugin_version: str
    extension_points: frozenset[ExtensionPoint]
    required_features: frozenset[SchedulerFeature]
    factory: Callable[..., SchedulerPlugin]
    capabilities: PluginCapabilities
```

The descriptor permits validation without guessing which methods a plugin
overrides. `PluginCapabilities` records batch support, native-buffer support,
and lifecycle subscriptions.

One plugin instance may participate in multiple explicitly configured
extension points. When the same instance is reused, its constructor arguments
must be identical at all extension points.

Scalar adapters may be provided for development convenience, but the stable
high-performance interfaces are batch-oriented. In-tree plugins must implement
the batch interfaces or a built-in intrinsic.

## Extension points

### QueueSort

Exactly one QueueSort plugin is enabled. It creates a queue discipline that
stores request handles and policy-owned index state:

```python
class QueueSortPlugin(SchedulerPlugin):
    def create_discipline(
        self,
        context: PluginInitContext,
    ) -> QueueDiscipline: ...
```

```python
class QueueDiscipline(Protocol):
    def add(
        self,
        handle: RequestHandle,
        position: QueuePosition,
    ) -> None: ...

    def remove(self, handle: RequestHandle) -> None: ...

    def update(
        self,
        handle: RequestHandle,
        changed: RequestFieldMask,
    ) -> None: ...

    def peek_candidates(
        self,
        scan_limit: int,
        output: CandidateHandleBuffer,
    ) -> int: ...
```

The core invokes mutations after validating the handle and request transition.
The discipline owns ordering indexes but never mutates scheduler state.

This interface permits dynamic QueueSort implementations such as deadline
queues, tenant round-robin, deficit scheduling, and hierarchical fair queues.
It also permits specialized internal structures:

- FCFS uses constant-time deque operations and constant-time removal handles;
- priority uses an indexed heap with logarithmic add, update, and removal;
- candidate traversal is bounded and never materializes the full queue.

Waiting and previously skipped requests may use separate discipline instances.
QueueSort defines their merge order through a bounded candidate cursor rather
than exposing either mutable queue.

### Filter

Filter determines current-cycle eligibility:

```python
class FilterPlugin(SchedulerPlugin):
    required_features: frozenset[SchedulerFeature]

    def filter_batch(
        self,
        candidates: CandidateBatch,
        state: SchedulingCycleState,
        allowed: MutableBitMask,
        scratch: PluginScratch,
    ) -> None: ...
```

`allowed` initially contains one bit for every candidate. Filter plugins run in
configured order and may only clear bits, giving deterministic AND semantics.
A rejection means "not eligible in this cycle", not "finish the request".

Reasons are written only when debug tracing is enabled. Normal operation must
not allocate `FilterResult` objects or reason strings per candidate.

The framework may expose a scalar `filter()` adapter for simple external
plugins. The adapter is a compatibility convenience and is not the performance
reference implementation.

### Score

Score ranks candidates that passed Filter:

```python
class ScorePlugin(SchedulerPlugin):
    required_features: frozenset[SchedulerFeature]

    def score_batch(
        self,
        candidates: CandidateBatch,
        state: SchedulingCycleState,
        scores: MutableFloatBuffer,
        scratch: PluginScratch,
    ) -> None: ...
```

The compiled pipeline preallocates score and scratch buffers. Multiple scores
are fused using configured finite weights:

```python
total[i] += weight * plugin_score[i]
```

All outputs must be finite. Higher scores are preferred. QueueSort rank is the
stable deterministic tie-breaker. Fusion and stable ranking happen once per
candidate plan, not once per admitted request.

### PreemptionScore

PreemptionScore ranks the running requests that the core has declared safe to
preempt:

```python
class PreemptionScorePlugin(SchedulerPlugin):
    def score_preemption_batch(
        self,
        candidates: RunningCandidateBatch,
        state: PreemptionCycleState,
        scores: MutableFloatBuffer,
        scratch: PluginScratch,
    ) -> None: ...
```

Multiple PreemptionScore plugins may be combined with finite weights. The core
performs preemption, budget restoration, cache cleanup, state reset, and queue
reinsertion.

Built-in FCFS is compiled to the intrinsic "last running request" operation and
therefore remains constant time. It must not be implemented by scoring and
scanning the complete running list.

## Feature providers

Plugins declare their required candidate and cycle features in their
descriptors. The profile compiler takes the union and activates only the
necessary providers:

```python
class SchedulerFeatureProvider(Protocol):
    provided_features: frozenset[SchedulerFeature]

    def populate(
        self,
        handles: ReadOnlyIntBuffer,
        features: MutableFeatureTable,
        context: FeatureProviderContext,
    ) -> None: ...
```

Providers operate in batches and share results across plugins. Expensive
features may be populated lazily when the first enabled stage needs them.

The local prefix-cache provider must use a side-effect-free API. Probing must
not allocate blocks, change LRU state, emit cache events, update hit metrics,
or initiate remote transfers.

Feature providers are versioned separately from internal scheduler objects.
This prevents the plugin API from growing into a frozen view of all scheduler
internals.

## Candidate collection and selection

Candidate work is bounded by two independent limits:

```python
candidate_window: int
candidate_scan_limit: int
```

`candidate_window` is the maximum number of Filter-approved requests passed to
Score. `candidate_scan_limit` is the maximum number of QueueSort-ordered
requests inspected in one plan.

The collection algorithm is:

1. Traverse at most `candidate_scan_limit` handles in QueueSort order.
2. Remove scheduler-owned blockers from consideration without exposing them to
   plugins.
3. Lazily populate features required by Filter.
4. Run the compiled Filter pipeline.
5. Continue scanning past rejected candidates until `candidate_window`
   candidates are accepted, the scan limit is reached, or the queue is
   exhausted.
6. Lazily populate additional Score features for accepted candidates.
7. Run and fuse Score plugins once.
8. Produce a stable ranked candidate plan.

Filtered requests remain queued, but they do not permanently occupy every
position in the scoring window. This prevents a rejected prefix from starving
eligible requests behind it while keeping work bounded.

The scheduler attempts ranked candidates in order. A candidate-local failure,
such as an encoder or request-specific constraint, advances to the next
candidate. A global exhaustion result, such as no usable KV capacity for any
candidate, ends admission. The core defines and tests this classification.

The plan is valid for one scheduling cycle. Static Filter and Score results are
not recomputed after every admission. Plugins that require incremental cycle
feedback declare that capability; the compiler then selects an incremental
plan with explicit invalidation points.

## Stateful plugins and lifecycle

Stateful policies require scheduler feedback. Lifecycle callbacks are batched,
explicitly subscribed, and limited to coarse events:

```python
class SchedulerPlugin:
    def on_requests_added(self, events: RequestEventBatch) -> None: ...
    def on_requests_finished(self, events: RequestEventBatch) -> None: ...
    def on_requests_preempted(self, events: RequestEventBatch) -> None: ...
    def on_cycle_completed(self, summary: CycleSummary) -> None: ...
```

The compiler creates direct callback lists for each event. Unsubscribed events
have no plugin dispatch cost. There is no per-generated-token callback.

Callbacks may update plugin-local state but cannot mutate scheduler-owned
requests or resources. Their exceptions follow the same fail-closed behavior
as scheduling extension points.

These callbacks enable deficit round robin, tenant concurrency accounting,
aging, and policies using admission or completion history.

## Registration and profiles

In-tree descriptors live in a built-in registry. Out-of-tree plugins use:

```toml
[project.entry-points."vllm.scheduler_plugins"]
session-affinity = "my_package.scheduler:get_scheduler_plugin"
```

Discovery occurs only in the EngineCore process. Only plugins named by the
active profile are loaded and instantiated. Installing a package alone cannot
change scheduling behavior.

The loader validates:

- unique plugin names;
- framework API version;
- declared extension points;
- required feature availability;
- batch/native capabilities;
- constructor configuration;
- lifecycle subscriptions.

An example profile is:

```python
SchedulerPluginProfile(
    queue_sort=SchedulerPluginSpec(name="fcfs"),
    filters=[
        SchedulerPluginSpec(name="tenant-quota"),
    ],
    scores=[
        SchedulerPluginSpec(name="session-affinity", weight=100.0),
        SchedulerPluginSpec(name="prefix-locality", weight=1.0),
        SchedulerPluginSpec(name="aging", weight=0.1),
    ],
    preemption_scores=[
        SchedulerPluginSpec(name="priority", weight=1.0),
        SchedulerPluginSpec(name="recompute-cost", weight=-0.5),
    ],
    candidate_window=32,
    candidate_scan_limit=256,
)
```

Profile execution order is explicit. Duplicate use of a stateful plugin name
refers to the same instance only when its arguments are identical.

## Backward compatibility

`SchedulerConfig.policy` continues to accept `fcfs` and `priority`.

Without an explicit profile:

- `policy="fcfs"` compiles to `BuiltinFCFSPlan`;
- `policy="priority"` compiles to `BuiltinPriorityPlan`.

Compatibility requirements are:

1. Existing queue, skipped-request, reinsertion, and preemption behavior is
   unchanged.
2. The default FCFS admission path continues to use deque head operations.
3. The default FCFS preemption path continues to select the final running
   request in constant time.
4. The default priority path continues to use the existing `Request.__lt__`
   ordering and heap operations.
5. `scheduler_cls` remains the whole-scheduler escape hatch.
6. AsyncScheduler uses the same compiled plan and plugin instances initialized
   by Scheduler.
7. An explicit QueueSort and a conflicting legacy policy fail validation.

## Specialized fast paths

The profile compiler selects a concrete plan once:

```text
BuiltinFCFSPlan
BuiltinPriorityPlan
FilteredPlan
ScoredPlan
FilteredScoredPlan
DynamicQueuePlan
IncrementalPlan
```

Scheduler stores direct bound operations from the compiled plan. It does not
branch on extension-point presence for every candidate.

The two built-in plans perform no candidate-buffer construction, feature
population, Filter dispatch, Score fusion, or generic preemption scan. Plugin
objects may exist for configuration and introspection without appearing in the
hot call graph.

## Complexity contract

Let:

- `N` be the total waiting-queue length;
- `S` be `candidate_scan_limit`;
- `W` be `candidate_window`;
- `F` be the number of Filter plugins;
- `P` be the number of Score plugins;
- `K` be the number of requests admitted in the cycle.

The generic candidate pipeline targets:

```text
candidate traversal       O(S)
Filter execution          O(S * F)
Score execution           O(W * P)
stable ranking            O(W log W)
FCFS selected removal     O(1)
indexed-heap removal      O(log N)
```

The complete cycle target is:

```text
O(S * F + W * P + W log W + K log N)
```

It must not contain `O(K * S * P)` rescoring or `O(K * N)` arbitrary-removal
behavior. Default FCFS admission is `O(K)` and default FCFS victim selection is
`O(1)`.

## Failure behavior

Plugin discovery, API-version mismatch, configuration, and initialization
failures abort scheduler startup.

Runtime exceptions, invalid buffer writes, invalid indexes, and non-finite
scores abort the scheduling cycle and surface as engine errors. Silently
ignoring a failed policy could violate fairness or isolation guarantees.

Plugin names and bounded diagnostic reasons may be exposed in debug logging.
High-cardinality request, tenant, or session identifiers must not become metric
labels.

## Security

Scheduler plugins execute trusted native or Python code in the EngineCore
process and have the same trust implications as a custom scheduler. Plugins are
opt-in and must participate in `VLLM_PLUGINS` allowlisting where applicable.

Read-only views prevent accidental mutation through the supported API; they are
not a sandbox against malicious Python code.

## Performance requirements

Performance is evaluated against the commit immediately preceding the plugin
migration on identical hardware and configuration. Initial acceptance targets
are:

| Scenario | Maximum scheduler CPU regression |
| --- | ---: |
| Built-in FCFS, no extension plugins | 1% |
| Built-in priority, no extension plugins | 2% |
| One batch Score, window 32 | 10 microseconds per candidate plan |
| Three batch Scores, window 64 | 30 microseconds per candidate plan |

The absolute microsecond targets may be revised using published baseline data,
but the no-extension limits and bounded scaling requirements are mandatory.

Increasing the waiting queue from 1,000 to 100,000 requests must not materially
increase generic plugin execution time when `S` and `W` are unchanged. Selected
candidate removal must not scan or rebuild the complete queue.

Benchmarks report at least:

- scheduler wall and CPU time;
- cycles and admitted requests per second;
- candidate collection, feature, Filter, Score, and ranking time;
- request and token throughput;
- TTFT and TPOT percentiles;
- prefix-cache hit rate for locality policies.

## Implementation plan

### Phase 0: baseline and guards

- Add scheduler and queue microbenchmarks.
- Record FCFS and priority baselines.
- Add complexity regression tests that detect full-queue iteration and rebuilds.

### Phase 1: descriptors and compiled built-in plans

- Add versioned plugin descriptors and selected entry-point discovery.
- Compile profiles during Scheduler initialization.
- Implement intrinsic `BuiltinFCFSPlan` and `BuiltinPriorityPlan`.
- Preserve existing tests without changing expected scheduling results.

### Phase 2: queue disciplines

- Add stable request handles.
- Implement an indexed FCFS discipline and indexed priority heap.
- Add bounded, non-materializing candidate cursors.
- Guarantee constant or logarithmic selected removal.

### Phase 3: batch Filter and Score

- Add reusable candidate and scratch buffers.
- Add independent scan and scoring limits.
- Add feature dependency compilation and lazy providers.
- Compile batch Filter, weighted Score fusion, and stable ranking.
- Continue after candidate-local feasibility failures.

### Phase 4: preemption and lifecycle

- Add composable batch PreemptionScore.
- Add batched, subscribed lifecycle events.
- Add explicit incremental-plan invalidation capabilities.

### Phase 5: locality and QoS

- Add the side-effect-free local-cache feature provider.
- Add reference session-affinity, prefix-locality, aging, tenant-fairness, and
  deadline plugins.
- Publish workload-specific throughput, latency, fairness, and cache-hit data.

## Test plan

- Run all existing scheduler tests unchanged for built-in profiles.
- Verify FCFS and priority queue, skipped, reinsertion, and preemption behavior.
- Verify built-in plans do not construct candidate buffers or dispatch plugin
  callbacks.
- Verify arbitrary FCFS and priority removals meet their complexity contracts.
- Verify Filter AND semantics and configured execution order.
- Verify rejected prefixes do not starve eligible candidates behind them.
- Verify scan and scoring limits independently bound work.
- Verify weighted Score and PreemptionScore fusion is deterministic.
- Verify zero scores preserve QueueSort order.
- Verify candidate-local failures advance to the next ranked candidate.
- Verify global exhaustion stops admission.
- Verify only requested features are populated and shared across plugins.
- Verify local-cache probing has no allocation, LRU, event, metric, or remote
  transfer side effects.
- Verify lifecycle events are batched and sent only to subscribers.
- Verify entry points are loaded only in EngineCore and only when configured.
- Verify API-version, capability, duplicate-name, and invalid-result failures.
- Benchmark no-plugin and representative Filter/Score compositions across
  queue lengths and candidate limits.

Model evaluations are not required for the behavior-preserving built-in
migration. Any policy that changes request ordering must report workload-
specific throughput, TTFT, TPOT, fairness, and cache-hit results.
