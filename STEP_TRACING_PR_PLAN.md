# Step-Level Tracing Implementation Plan

## Executive Summary

This plan implements **step-level tracing** as an extension to vLLM's existing request journey tracing system. The implementation consists of **5 independently mergeable PRs** that add:

1. **Timestamp precision upgrade** - Journey tracing dual-writes nanosecond timestamps alongside existing float timestamps (backward compatible)
2. **vLLM-side sampling** - Probabilistic request sampling independent of OTEL SDK sampling
3. **Step batch summary stream** - Sampled per-step batch summary events (no per-request lists)
4. **KV metrics utilities** - Read-only observability helpers for KV cache metrics
5. **Rich snapshot stream** - Subsampled per-running-request detailed progress events with KV/cache metrics

**Key Design Principles**:
- Small, independently safe PRs following vLLM conventions
- OTEL-native event model (no manual epoch timestamp overrides)
- Monotonic timestamps as attributes only (for correlation/duration)
- **Fully backward compatible** - existing journey traces unchanged, dual-write for transitions
- Clear separation: batch-level vs request-level observability
- Deterministic test sampling for CI reliability
- **Tracing payload rule**: Only include quantities that are already available or cheaply computable at emission point (no expensive scans, no per-token iteration, no new bookkeeping)

---

## Implementation Progress

| PR | Status | Branch | Commit | Notes |
|----|--------|--------|--------|-------|
| PR #3 | ✅ **MERGED** | `step-batch-summary-tracing` | 4c1afa8f5 | Step batch summary (CLI args defined, wiring fixed in PR #5) |
| PR #4 | ✅ **MERGED** | `kv-metrics-observability-utils` | 94aab4cb9 | KV cache observability utilities |
| PR #5 | ✅ **COMPLETE** | `pr5ofstepstream` | f951860dc | Rich request snapshots + CLI wiring fix (ready for PR) |
| PR #1 | ⏸️ DEFERRED | - | - | Timestamp precision (orthogonal, low priority) |
| PR #2 | ⏸️ DEFERRED | - | - | Journey sampling (orthogonal, low priority) |

**Current HEAD**: ab555b83e (main branch, PR #5 not yet merged)

**Key Outcomes**:
- Step-level tracing infrastructure complete (PR #3)
- KV observability utilities available (PR #4)
- Per-request rich snapshots implemented (PR #5)
- Step tracing test suite passing (all tests green)
- CLI flags properly wired (PR #5 fix): `--step-tracing-enabled`, `--step-tracing-sample-rate`, `--step-tracing-rich-subsample-rate`

---

## A. PR Dependency Graph

```
PR #1: Timestamp Precision Upgrade (INDEPENDENT) ⏸️ DEFERRED

PR #2: Journey Tracing Sampling Knob (INDEPENDENT) ⏸️ DEFERRED

PR #3: Step Batch Summary Stream (INDEPENDENT) ✅ MERGED
  ↓
  ├─→ PR #4: KV/Cache Metrics Utilities (parallel to PR #3) ✅ MERGED
  │
  └─→ PR #5: Rich Request Snapshot Stream (needs PR #3 + PR #4) ✅ COMPLETE
```

**Dependency Notes**:
- **PR #1** is **INDEPENDENT** - Step tracing can proceed without it (uses own nanosecond timestamps)
- **PR #2** is **INDEPENDENT** - Can reuse sampling patterns in later PRs but no code dependency
- **PR #3** is **INDEPENDENT** - Can reuse sampling patterns from PR #2 if available, but no code dependency
- **PR #4** can be developed in parallel to PR #3 (no dependency)
- **PR #5** depends on both PR #3 (step ID) and PR #4 (KV metrics utilities)

**Implementation Order** (as completed):
1. ✅ PR #3 (step batch summary) - establishes `scheduler_steps` span + core payload (commit 4c1afa8f5)
2. ✅ PR #4 (KV utils) - pure helpers + tests (commit 94aab4cb9)
3. ✅ PR #5 (rich snapshots) - consumes both (commit f951860dc, ready for PR)
4. ⏸️ PR #2 (journey sampling) - deferred (orthogonal, low priority)
5. ⏸️ PR #1 (dual-write) - deferred (orthogonal, low priority)

---

## B. PR Behavioral Contracts

### PR #1: Journey Trace Timestamp Precision Upgrade (Dual-Write)

**Branch**: `journey-ts-dual-write-nanoseconds`

**Depends On**: None (INDEPENDENT)

**Behavioral Contract**:

**CHANGES**:
- Add new nanosecond-precision timestamp attribute to all journey events: `ts.monotonic_ns` (int, via `time.monotonic_ns()`)
- **KEEP existing** `ts.monotonic` attribute (float seconds, via `time.monotonic()`) - dual-write for backward compatibility
- Both attributes emitted on every journey event during transition period
- Update `RequestJourneyEvent` dataclass to include both fields

**PRESERVED**:
- **FULLY BACKWARD COMPATIBLE** - existing `ts.monotonic` attribute unchanged (name, type, value)
- All journey event emission points unchanged (QUEUED, SCHEDULED, FIRST_TOKEN, PREEMPTED, FINISHED)
- Event structure unchanged (same fields, new optional field added)
- OTEL epoch timestamps (`time.time_ns()`) unchanged
- Existing journey tracing consumers continue to work without modification

**COMPATIBILITY GUARANTEES**:
- **Zero breaking changes** - old consumers can ignore `ts.monotonic_ns`
- New consumers can use `ts.monotonic_ns` for higher precision
- Both attributes represent identical times (just different units/precision)
- Future PR can deprecate `ts.monotonic` after migration period

**FAILURE GUARANTEES**:
- Tracing failures remain non-fatal (span.add_event() wrapped in try/except)
- No new exception paths introduced
- Dual-write adds minimal overhead (~1 extra attribute per event)

**SCOPE BOUNDARIES**:
- IN SCOPE: Add `ts.monotonic_ns` alongside existing `ts.monotonic` in journey tracing emission
- IN SCOPE: Update journey event dataclass and emission logic
- IN SCOPE: Document deprecation plan for `ts.monotonic` (comment only, no enforcement)
- OUT OF SCOPE: Removing `ts.monotonic` (future PR after migration)
- OUT OF SCOPE: Non-journey-tracing monotonic timestamps

**VERIFICATION APPROACH**:
- Unit tests verify BOTH `ts.monotonic` and `ts.monotonic_ns` present
- Unit test verifies `ts.monotonic` unchanged (float, seconds, same values as before)
- Unit test verifies `ts.monotonic_ns` is integer, nanoseconds, reasonable value
- Unit test verifies consistency: Either (a) compute `ts.monotonic` from `ts.monotonic_ns` in production code to ensure exact consistency, OR (b) loosen test tolerance significantly (e.g., within 100ms) to avoid CI flakiness from separate time calls
- Integration test: Full request trace has both attributes on all events

**Critical Files**:
- `vllm/v1/core/sched/scheduler.py` - Compute both `time.monotonic()` AND `time.monotonic_ns()` in `_emit_journey_event()`
- `vllm/v1/core/sched/journey_events.py` - Add field: `ts_monotonic_ns: int` (keep existing `ts_monotonic: float`)
- `vllm/tracing.py` - Add constant: `JOURNEY_TS_MONOTONIC_NS = "ts.monotonic_ns"` (keep existing `JOURNEY_TS_MONOTONIC`)
- `tests/v1/core/test_scheduler.py` - Add dual-write timestamp test

---

### PR #2: Journey Tracing vLLM-Side Sampling Knob

**Branch**: `journey-tracing-sample-rate`

**Depends On**: None (INDEPENDENT)

**Behavioral Contract**:

**CHANGES**:
- Add CLI flag: `--journey-tracing-sample-rate` (float, default `1.0`, range `[0.0, 1.0]`)
- Add field to `ObservabilityConfig`: `journey_tracing_sample_rate: float = 1.0`
- When `--enable-journey-tracing` is true, vLLM samples requests probabilistically at request entry
- Unsampled requests: NO journey events emitted, NO core span created (zero overhead)
- Sampled requests: Full journey trace as today (all events)
- Sampling decision persists for request lifetime (no partial traces)
- **Sampling state stored in scheduler dict** `self._journey_sampled_requests: set[str]` keyed by `request.request_id` (NO Request object mutation)
- **Sampling stability**: Sampling is best-effort probabilistic and NOT guaranteed stable across runs (unless a future deterministic mode with seed is added)

**PRESERVED**:
- Existing behavior when `journey_tracing_sample_rate=1.0` (default): All requests traced
- OTEL SDK/collector sampling still applies independently (can further drop traces)
- Journey event structure unchanged
- Non-journey observability (scheduler stats, metrics) unchanged
- Request object unchanged (no new fields added)

**COMPATIBILITY GUARANTEES**:
- Default `1.0` → zero behavior change unless flag explicitly set
- Sampling is additive overhead reduction, not a breaking change
- Existing traces unaffected (just fewer of them if sampled)

**FAILURE GUARANTEES**:
- Invalid sample rate (e.g., `-1`, `2.0`) → validation error at startup (fail fast)
- Sampling logic wrapped in try/except → no crashes on edge cases
- If sampling fails, request proceeds normally (worst case: no trace, not blocked)

**SCOPE BOUNDARIES**:
- IN SCOPE: Per-request sampling decision at request entry (`add_request()`)
- IN SCOPE: CLI flag, config field, validation
- IN SCOPE: Deterministic sampling option for tests (injectable sampler function)
- OUT OF SCOPE: Per-event sampling (all-or-nothing per request)
- OUT OF SCOPE: Dynamic sampling rate adjustment (static per server instance)
- OUT OF SCOPE: Step tracing sampling (that's PR #3)

**VERIFICATION APPROACH**:
- Unit test with `sample_rate=0.0` → zero spans created
- Unit test with `sample_rate=1.0` → all requests traced
- **Deterministic test** with stable hash-based sampler → exact expected set sampled
- Test unsampled request completes successfully with no journey events
- Test sampled request produces full trace (all events present)
- Test sampling state cleaned up on request completion (memory leak check)

**Critical Files**:
- `vllm/config/observability.py` - Add `journey_tracing_sample_rate: float = 1.0`
- `vllm/engine/arg_utils.py` - Add CLI flag `--journey-tracing-sample-rate`
- `vllm/v1/core/sched/scheduler.py` - Add `self._journey_sampled_requests: set[str]` in `__init__`
- `vllm/v1/core/sched/scheduler.py` - Add sampling decision in `add_request()`
- `vllm/v1/core/sched/scheduler.py` - Guard `_emit_journey_event()` on sampling set membership
- `vllm/v1/core/sched/scheduler.py` - Clean up sampling state in `_end_core_span_and_cleanup()`
- `tests/v1/core/test_scheduler.py` - Add deterministic sampling tests

**Implementation Sketch**:
```python
# In Scheduler.__init__():
self._journey_sampled_requests: set[str] = set()

# Optional: Injectable sampler for deterministic tests
self._journey_sampler: Callable[[str, float], bool] = (
    lambda req_id, rate: random.random() < rate
)

# In add_request():
if self._enable_journey_tracing:
    if self._journey_sampler(request.request_id, self.journey_tracing_sample_rate):
        self._journey_sampled_requests.add(request.request_id)
        core_span = self._create_core_span(request)
        # ... emit QUEUED event

# In _emit_journey_event():
if request.request_id not in self._journey_sampled_requests:
    return  # Not sampled, skip emission
# ... proceed with emission

# In _end_core_span_and_cleanup():
self._journey_sampled_requests.discard(request.request_id)

# For tests: Deterministic sampler using stable hash
def _make_deterministic_sampler(seed: int = 0) -> Callable[[str, float], bool]:
    """Create deterministic sampler for tests using stable hash.

    Uses hashlib.sha1 (NOT Python's hash() which is salted per process).
    """
    import hashlib
    def sampler(key: str, rate: float) -> bool:
        hash_bytes = hashlib.sha1(f"{seed}:{key}".encode()).digest()
        hash_value = int.from_bytes(hash_bytes[:8], 'big') / (2**64)
        return hash_value < rate
    return sampler
```

---

### PR #3: Step Batch Summary Stream

**Branch**: `step-batch-summary-tracing`

**Depends On**: None (INDEPENDENT)

**Behavioral Contract**:

**CHANGES**:
- Add CLI flags:
  - `--step-tracing-enabled` (bool, default `False`) - Master switch for step-level tracing
  - `--step-tracing-sample-rate` (float, default `0.01` = 1%) - Probabilistic step sampling
- Add fields to `ObservabilityConfig`: `step_tracing_enabled: bool`, `step_tracing_sample_rate: float`
- When enabled, scheduler emits **one batch summary event per sampled step** on a long-lived OTEL span
- Span name: `scheduler_steps` (SpanKind.INTERNAL)
- Event name: `step.BATCH_SUMMARY`
- Emission point: **End of `schedule()` method**, just before return (after SchedulerOutput construction)
- Sampling decision: Per-step probabilistic (independent of journey tracing)
- **Span lifecycle policy**:
  - End span ONLY if a deterministic shutdown/close hook exists (e.g., `EngineCore.shutdown()` or `Scheduler.close()`)
  - If no reliable hook exists, leave span open indefinitely (acceptable for most OTEL backends, document in PR)
  - Do NOT use `__del__` (unreliable in Python due to GC timing, ref cycles, interpreter shutdown)

**PRESERVED**:
- Journey tracing unchanged (orthogonal streams)
- SchedulerOutput unchanged (no new fields)
- Scheduler logic unchanged (batch summary is observability-only, no side effects)
- Zero overhead when disabled (single boolean check)

**COMPATIBILITY GUARANTEES**:
- Default `False` → zero behavior change, zero overhead
- Independent from journey tracing (can enable one, both, or neither)
- No changes to existing trace structure

**FAILURE GUARANTEES**:
- Batch summary emission wrapped in try/except → never crashes scheduler
- Invalid sample rate → validation error at startup
- OTEL unavailable → graceful degradation (no batch summary, scheduler proceeds)
- Span creation failure → graceful degradation (log warning, disable step tracing)

**SCOPE BOUNDARIES**:
- IN SCOPE: Step-level summary from SchedulerOutput and scheduler state (cheaply available)
- IN SCOPE: Lightweight KV cache pressure signals via existing BlockPool methods (no expensive iteration)
- IN SCOPE: Deterministic sampling for tests (injectable sampler)
- OUT OF SCOPE: Per-request lists (that's PR #5)
- OUT OF SCOPE: Expensive scans or per-token iteration
- OUT OF SCOPE: Span rotation (unless proven necessary by backend limits)

**VERIFICATION APPROACH**:
- Unit test with `step_tracing_enabled=False` → no span created
- Unit test with `step_tracing_enabled=True, sample_rate=1.0` → every step emits batch summary
- **Deterministic test** with stable hash-based step sampler → exact expected steps traced
- Verify batch summary attributes match SchedulerOutput fields
- Verify batch composition sums correctly (expected invariant)
- Verify batch summary emitted even for empty schedules (zero requests)

**Critical Files**:
- `vllm/config/observability.py` - Add step tracing config fields
- `vllm/engine/arg_utils.py` - Add CLI flag definitions (NOTE: Wiring to EngineArgs → ObservabilityConfig was missing in PR #3, fixed later in PR #5)
- `vllm/v1/core/sched/scheduler.py` - Add step span creation and batch summary emission logic
- `vllm/tracing.py` - Add span attribute constants for step batch summary
- `tests/v1/core/test_scheduler.py` - Add deterministic step batch summary tests

---

### PR #4: KV Cache Metrics Utilities for Observability

**Branch**: `kv-metrics-observability-utils`

**Depends On**: None (can be developed in parallel to PR #3)

**Behavioral Contract**:

**CHANGES**:
- Add utility module: `vllm/v1/core/kv_cache_observability.py` (new file)
- Provides read-only functions to extract per-request and per-step KV metrics for tracing/observability
- **CRITICAL**: Uses ONLY existing exposed interfaces from `KVCacheManager` and `BlockPool` (read-only queries)
- **CRITICAL**: KV metrics set is intentionally minimal and may be finalized during implementation based on what is cheaply accessible
- Functions:
  - `get_per_request_kv_metrics(request, manager) -> PerRequestKVMetrics`
  - `get_step_kv_summary(block_pool) -> StepKVSummary`
- Dataclasses define **guaranteed** vs **optional** fields:
  - **Guaranteed**: GPU metrics (always available from `req_to_blocks`, `num_cached_block`, `block_pool`)
  - **Optional**: Offload metrics, effective prompt length (best-effort, may be `None`)

**PRESERVED**:
- All existing KV cache allocation/eviction logic unchanged
- Scheduler logic unchanged
- KVCacheManager interface unchanged (utilities call existing methods)
- No new public APIs, no KV subsystem changes

**COMPATIBILITY GUARANTEES**:
- New utility module, zero impact on existing code paths
- Can be imported and used by future PRs without modifying KV subsystem

**FAILURE GUARANTEES**:
- Utilities handle missing data gracefully (return `None` for unavailable metrics)
- No exceptions raised to caller (defensive programming with try/except inside utilities)
- If KV manager unavailable → return empty/minimal metrics

**SCOPE BOUNDARIES**:
- IN SCOPE: Read-only queries on existing exposed interfaces (`req_to_blocks`, `num_cached_block`, `block_pool`)
- IN SCOPE: Distinguish GPU-resident (guaranteed) vs offloaded KV (optional, best-effort)
- IN SCOPE: **Minimum guaranteed signals**:
  - Per-request: `kv.blocks_allocated_gpu` (from `len(req_to_blocks[req_id])`)
  - Per-request: `kv.blocks_cached_gpu` (from `num_cached_block[req_id]`)
  - Per-step: `kv.blocks_total_gpu`, `kv.blocks_free_gpu`, `kv.usage_gpu_ratio` (from `block_pool`)
- IN SCOPE: **Optional signals** (best-effort, from existing fields only):
  - Per-request/step: CPU/disk offload blocks (if offload manager available and per-request tracking exists)
  - Per-request: `effective_prompt_len` (only if `request.num_cached_tokens` already exists; do NOT add new Request fields)
- **CRITICAL**: If any desired KV metric is not cheaply accessible via existing read-only fields/interfaces in the current codebase, it MUST be omitted or marked optional-best-effort
- OUT OF SCOPE: Modifying KV cache allocation logic
- OUT OF SCOPE: Adding new fields to Request object (read-only contract)
- OUT OF SCOPE: New bookkeeping or data structures solely for observability
- OUT OF SCOPE: New KV subsystem APIs or methods
- OUT OF SCOPE: Expensive scans or per-block iteration beyond existing methods

**VERIFICATION APPROACH**:
- Unit test verifies functions return expected structure
- Unit test with mock request → check allocated blocks count
- Unit test with prefix caching enabled → check cached blocks count
- Unit test with missing offload manager → optional fields are `None`
- Integration test with real scheduler → KV metrics match expected values

**Critical Files**:
- `vllm/v1/core/kv_cache_observability.py` (NEW FILE) - Utility functions and dataclasses
- `vllm/v1/core/kv_cache_manager.py` - Read existing methods (no changes)
- `vllm/v1/core/block_pool.py` - Read existing methods (no changes)
- `vllm/v1/request.py` - Read existing fields (no changes)
- `tests/v1/core/test_kv_cache_observability.py` (NEW FILE) - Unit tests

**Implementation Sketch**:
```python
@dataclass
class PerRequestKVMetrics:
    """Per-request KV cache metrics for observability.

    GUARANTEED fields (from existing KV cache manager interfaces):
    - blocks_allocated_gpu: GPU KV blocks allocated to this request
    - blocks_cached_gpu: GPU blocks with prefix cache hits

    OPTIONAL fields (best-effort, from existing fields only):
    - blocks_cpu_offload, blocks_disk_offload: Offload metrics (None if unavailable)
    - effective_prompt_len: Prompt tokens after prefix cache (None if unavailable)

    Contract: Uses ONLY existing exposed interfaces. If not accessible, omit.
    """
    blocks_allocated_gpu: int
    blocks_cached_gpu: int
    blocks_cpu_offload: int | None = None
    blocks_disk_offload: int | None = None
    effective_prompt_len: int | None = None

@dataclass
class StepKVSummary:
    """Step-level KV cache summary for batch summary events.

    GUARANTEED fields (from BlockPool methods):
    - blocks_total_gpu, blocks_free_gpu, usage_gpu_ratio

    OPTIONAL fields (best-effort):
    - blocks_cpu_offload, blocks_disk_offload
    """
    blocks_total_gpu: int
    blocks_free_gpu: int
    usage_gpu_ratio: float
    blocks_cpu_offload: int | None = None
    blocks_disk_offload: int | None = None

def get_per_request_kv_metrics(request: Request, manager: KVCacheManager) -> PerRequestKVMetrics:
    """Extract per-request KV metrics (read-only, existing interfaces only)."""
    try:
        # GUARANTEED: GPU blocks
        allocated = len(manager.req_to_blocks.get(request.request_id, []))
        cached = manager.num_cached_block.get(request.request_id, 0)

        # OPTIONAL: Offload (best-effort)
        cpu_offload = None
        disk_offload = None
        if hasattr(manager, 'offload_manager') and manager.offload_manager:
            pass  # Query if available

        # OPTIONAL: Effective prompt (only if request.num_cached_tokens exists)
        effective_prompt = None
        if hasattr(request, 'num_cached_tokens') and isinstance(getattr(request, 'num_cached_tokens', None), int):
            cached_tokens = request.num_cached_tokens
            if 0 < cached_tokens < request.num_prompt_tokens:
                effective_prompt = request.num_prompt_tokens - cached_tokens

        return PerRequestKVMetrics(
            blocks_allocated_gpu=allocated,
            blocks_cached_gpu=cached,
            blocks_cpu_offload=cpu_offload,
            blocks_disk_offload=disk_offload,
            effective_prompt_len=effective_prompt,
        )
    except Exception as e:
        logger.debug("Failed to get KV metrics for request %s: %s", request.request_id, e)
        return PerRequestKVMetrics(blocks_allocated_gpu=0, blocks_cached_gpu=0)
```

---

### PR #5: Rich Request Snapshot Stream

**Branch**: `rich-request-snapshot-stream`

**Depends On**: PR #3 (step batch summary for step ID), PR #4 (KV metrics utilities)

**Behavioral Contract**:

**CHANGES**:
- Add CLI flag: `--step-tracing-rich-subsample-rate` (float, default `0.001` = 0.1%)
- When enabled AND step is batch-summary-sampled, perform second probabilistic decision
- If subsampled: Emit **one event per running request** in the executed batch
- Event model: Multiple OTEL events with same name on `scheduler_steps` span, grouped by `step.id`
- Event name: `step.REQUEST_SNAPSHOT`
- Emission point: End of `schedule()` method (after batch summary, if subsampled)
- **Source of truth**: `scheduler.running` at SchedulerOutput construction point (same set used to build SchedulerOutput)
- Payload: Per-request progress, token counts (from SchedulerOutput), KV metrics (via PR #4 utilities)
- **Payload rule**: Only include cheaply available quantities (Request fields, SchedulerOutput data, PR #4 utilities - no expensive iteration)
- **Also fixes PR #3 CLI wiring bug**: Wire all three step tracing flags through `EngineArgs` to `ObservabilityConfig` (regression test added)

**PRESERVED**:
- Journey tracing unchanged (orthogonal)
- Batch summary stream unchanged (same span, separate event)
- Scheduler logic unchanged (observability-only)
- Zero overhead when disabled or unsampled (two boolean checks)

**COMPATIBILITY GUARANTEES**:
- Default `0.001` → extremely sparse (low overhead)
- Independent subsample → further reduces batch summary overhead
- Can adjust rate independently of batch summary sample rate

**FAILURE GUARANTEES**:
- Event emission wrapped in try/except per request → one failure doesn't skip others
- Invalid sample rate → validation error at startup
- Missing KV metrics → emit event with partial data (mark fields as unavailable)

**SCOPE BOUNDARIES**:
- IN SCOPE: **ONLY running requests in executed batch** (from `scheduler.running` at SchedulerOutput construction point)
- IN SCOPE: **Explicit contract**: Snapshot uses same request set that SchedulerOutput is built from
- IN SCOPE: Per-request progress and token counts (cheaply available from Request and SchedulerOutput)
- IN SCOPE: Per-request KV metrics via PR #4 utilities (cheaply queryable)
- OUT OF SCOPE: Waiting queue snapshots (would blow up cardinality)
- OUT OF SCOPE: Preempted requests (not in `scheduler.running` at emission time)
- OUT OF SCOPE: Expensive per-token iteration or detailed block traversal
- OUT OF SCOPE: Historical request state (only current step snapshot)

**VERIFICATION APPROACH**:
- Unit test with `rich_subsample_rate=0.0` → no rich events
- Unit test with `rich_subsample_rate=1.0` + `step_sample_rate=1.0` → all running requests traced
- **Deterministic test** with stable hash-based subsampler → exact expected steps get rich snapshots
- Verify event count matches `len(scheduler.running)` at SchedulerOutput construction point
- Verify each event has correct `step.id` (matches batch summary step)
- Verify per-request KV metrics present (allocated blocks > 0 for running requests)
- Verify phase correctly identified (using scheduler's `num_output_tokens == 0` logic)
- Verify snapshot uses same request set as SchedulerOutput (no drift)
- Test with zero running requests → no rich events (only batch summary if step sampled)

**Critical Files**:
- `vllm/config/observability.py` - Add `step_tracing_rich_subsample_rate: float`
- `vllm/engine/arg_utils.py` - Add CLI flag + wire all step tracing flags to EngineArgs/ObservabilityConfig (fixes PR #3 bug)
- `vllm/v1/core/sched/scheduler.py` - Add rich snapshot emission (after batch summary, iterate `self.running`)
- `vllm/v1/core/kv_cache_observability.py` - Use utilities from PR #4
- `vllm/tracing.py` - Add span attribute constants for rich snapshot
- `tests/v1/core/test_step_tracing.py` - Add rich snapshot tests + CLI wiring regression test

---

## C. Locked Payload Specifications

### C.1. PR #1: Timestamp Attribute Addition (Dual-Write)

**Change Summary**:
| Attribute | Type | Unit | Status |
|-----------|------|------|--------|
| `ts.monotonic` | `float` | seconds | **KEPT** (unchanged, backward compatible) |
| `ts.monotonic_ns` | `int` | nanoseconds | **ADDED** (new, higher precision) |

**Impact**: All journey events (QUEUED, SCHEDULED, FIRST_TOKEN, PREEMPTED, FINISHED)

**Dual-Write Example**:
```json
{
  "ts.monotonic": 1234567.890123,
  "ts.monotonic_ns": 1234567890123456
}
```

**Migration Path**: Future PR can deprecate `ts.monotonic` after consumers adopt `ts.monotonic_ns`

---

### C.2. PR #3: Step Batch Summary Event Payload

**Event Name**: `step.BATCH_SUMMARY`

**Span Name**: `scheduler_steps` (SpanKind.INTERNAL, one long-lived span)

**Emission Point**: End of `schedule()` method (just before return, after SchedulerOutput construction)

**Emission Logic**:
```python
# At end of schedule(), before return
if self.step_tracing_enabled and self._step_sampler(curr_step, self.step_tracing_sample_rate):
    self._emit_step_batch_summary(output, curr_step, step_start_ns)
```

**Source of Truth**: SchedulerOutput and scheduler state at end of `schedule()`

**Payload Rule**: Only cheaply available quantities from SchedulerOutput, scheduler state, and BlockPool methods

**Required Attributes** (always present):

| Attribute | Type | Unit | Definition | Computation |
|-----------|------|------|------------|-------------|
| `step.id` | `int` | count | Monotonic scheduler step number | `output.scheduler_step` |
| `step.ts_start_ns` | `int` | nanoseconds | Monotonic timestamp at step start | Captured at `schedule()` entry via `time.monotonic_ns()` |
| `step.ts_end_ns` | `int` | nanoseconds | Monotonic timestamp at step end | Captured before emission via `time.monotonic_ns()` |
| `step.duration_us` | `int` | microseconds | Step duration | `(ts_end_ns - ts_start_ns) // 1000` |
| `queue.running_depth` | `int` | count | Requests in RUNNING state | `len(scheduler.running)` at SchedulerOutput construction |
| `queue.waiting_depth` | `int` | count | Requests in WAITING state | `len(scheduler.waiting)` at SchedulerOutput construction |
| `batch.num_prefill_reqs` | `int` | count | Running requests in prefill phase | Count where `req.num_output_tokens == 0` |
| `batch.num_decode_reqs` | `int` | count | Running requests in decode phase | Count where `req.num_output_tokens > 0` |
| `batch.scheduled_tokens` | `int` | count | Total tokens scheduled this step | `output.total_num_scheduled_tokens` |
| `batch.prefill_tokens` | `int` | count | Prefill tokens scheduled | Sum `output.num_scheduled_tokens[req_id]` where `req.num_output_tokens == 0` |
| `batch.decode_tokens` | `int` | count | Decode tokens scheduled | Sum `output.num_scheduled_tokens[req_id]` where `req.num_output_tokens > 0` |
| `batch.num_finished` | `int` | count | Requests finished in this step | `len(output.finished_req_ids)` |
| `batch.num_preempted` | `int` | count | Requests preempted in this step | `len(output.preempted_req_ids)` if present else 0 |
| `kv.usage_gpu_ratio` | `float` | ratio | KV cache usage (0.0-1.0) | `block_pool.get_usage()` |
| `kv.blocks_total_gpu` | `int` | count | Total GPU KV blocks in pool | `block_pool.num_gpu_blocks - 1` |
| `kv.blocks_free_gpu` | `int` | count | Free GPU KV blocks | `block_pool.get_num_free_blocks()` |

**Optional Attributes** (availability depends on config):

| Attribute | Type | Unit | Definition | Availability |
|-----------|------|------|------------|--------------|
| `kv.blocks_cpu_offload` | `int` | count | CPU-offloaded KV blocks | Present only if offload enabled and accessible |
| `kv.blocks_disk_offload` | `int` | count | Disk-offloaded KV blocks | Present only if offload enabled and accessible |

**Invariants**:
- `step.id` monotonically increasing (never resets)
- `step.duration_us >= 0`
- `batch.num_prefill_reqs + batch.num_decode_reqs <= queue.running_depth` (equality expected under standard settings)
- `batch.prefill_tokens + batch.decode_tokens == batch.scheduled_tokens` (expected, may differ with speculative decode)
- `kv.usage_gpu_ratio` ∈ [0.0, 1.0]
- `kv.blocks_free_gpu >= 0`, `kv.blocks_total_gpu >= kv.blocks_free_gpu`
- `kv.blocks_free_gpu ≈ kv.blocks_total_gpu * (1 - kv.usage_gpu_ratio)` (sanity check, not strict - may differ due to reserved/null blocks)
- `queue.running_depth >= 0`, `queue.waiting_depth >= 0`

**Example Event**:
```json
{
  "name": "step.BATCH_SUMMARY",
  "attributes": {
    "step.id": 42,
    "step.ts_start_ns": 1234567890123456,
    "step.ts_end_ns": 1234567892345678,
    "step.duration_us": 2222,
    "queue.running_depth": 8,
    "queue.waiting_depth": 12,
    "batch.num_prefill_reqs": 2,
    "batch.num_decode_reqs": 6,
    "batch.scheduled_tokens": 512,
    "batch.prefill_tokens": 128,
    "batch.decode_tokens": 384,
    "batch.num_finished": 1,
    "batch.num_preempted": 0,
    "kv.usage_gpu_ratio": 0.75,
    "kv.blocks_total_gpu": 1024,
    "kv.blocks_free_gpu": 256
  }
}
```

---

### C.3. PR #5: Rich Request Snapshot Event Payload

**Event Name**: `step.REQUEST_SNAPSHOT`

**Span Name**: Same span as batch summary (`scheduler_steps`)

**Emission Point**: After batch summary emission, if subsampled (end of `schedule()`)

**Emission Logic**:
```python
# After emitting batch summary
if self.step_tracing_enabled and batch_summary_was_sampled:
    if self._rich_sampler(curr_step, self.step_tracing_rich_subsample_rate):
        for req in self.running:  # At SchedulerOutput construction point
            self._emit_request_snapshot(req, curr_step, output)
```

**Source of Truth**: `scheduler.running` at SchedulerOutput construction point (same set used to build SchedulerOutput)

**Payload Rule**: Only cheaply available Request fields, SchedulerOutput data, and PR #4 utilities (no expensive iteration)

**Cardinality**: One event per running request in executed batch (excludes waiting/preempted)

**Grouping**: All events share same `step.id`

**Required Attributes** (always present):

| Attribute | Type | Unit | Definition | Computation |
|-----------|------|------|------------|-------------|
| `step.id` | `int` | count | Step number | `output.scheduler_step` |
| `request.id` | `str` | N/A | Request identifier | `request.request_id` |
| `request.phase` | `str` | N/A | "PREFILL" or "DECODE" | `"PREFILL"` if `req.num_output_tokens == 0` else `"DECODE"` |
| `request.num_prompt_tokens` | `int` | count | Total prompt tokens | `request.num_prompt_tokens` |
| `request.num_computed_tokens` | `int` | count | Tokens computed so far | `request.num_computed_tokens` |
| `request.num_output_tokens` | `int` | count | Output tokens generated | `request.num_output_tokens` (property) |
| `request.num_preemptions` | `int` | count | Preemption count | `request.num_preemptions` |
| `request.scheduled_tokens_this_step` | `int` | count | Tokens scheduled in this step | `output.num_scheduled_tokens[request_id]` |
| `kv.blocks_allocated_gpu` | `int` | count | GPU KV blocks allocated | Via `get_per_request_kv_metrics()` from PR #4 |
| `kv.blocks_cached_gpu` | `int` | count | GPU blocks with prefix cache hits | Via `get_per_request_kv_metrics()` from PR #4 |

**Optional Attributes** (best-effort):

| Attribute | Type | Unit | Definition | Availability |
|-----------|------|------|------------|--------------|
| `kv.blocks_cpu_offload` | `int` | count | CPU-offloaded KV blocks | If offload enabled and tracked per-request |
| `kv.blocks_disk_offload` | `int` | count | Disk-offloaded KV blocks | If offload enabled and tracked per-request |
| `request.max_tokens` | `int` | count | Max output tokens | `request.sampling_params.max_tokens` if available |
| `request.effective_prompt_len` | `int` | count | Prompt tokens after prefix cache | If `request.num_cached_tokens` exists and trackable |

**Timing Correlation**:
- No per-request monotonic timestamp needed (use step timing from batch summary)
- OTEL event timestamp (epoch) auto-set by span.add_event()
- Correlation via `step.id` is sufficient

**Invariants**:
- `request.phase` ∈ {"PREFILL", "DECODE"}
- `request.num_output_tokens >= 0`
- `request.num_computed_tokens >= request.num_prompt_tokens + request.num_output_tokens` (resets on preempt)
- `request.num_preemptions >= 0`
- `kv.blocks_allocated_gpu >= kv.blocks_cached_gpu`
- `kv.blocks_allocated_gpu >= 0`

**Example Event**:
```json
{
  "name": "step.REQUEST_SNAPSHOT",
  "attributes": {
    "step.id": 42,
    "request.id": "req-abc123",
    "request.phase": "DECODE",
    "request.num_prompt_tokens": 100,
    "request.num_computed_tokens": 115,
    "request.num_output_tokens": 15,
    "request.num_preemptions": 1,
    "request.scheduled_tokens_this_step": 8,
    "kv.blocks_allocated_gpu": 12,
    "kv.blocks_cached_gpu": 5
  }
}
```

**Contract**: Snapshot taken from `scheduler.running` at SchedulerOutput construction point, ensuring consistency with executed batch.

---

## D. Implementation Notes

### D.1. Sampling Rate Defaults

| Flag | Default | Rationale |
|------|---------|-----------|
| `--journey-tracing-sample-rate` | `1.0` | Backward compatible (journey tracing already opt-in) |
| `--step-tracing-sample-rate` | `0.01` | 1% gives good coverage with low overhead |
| `--step-tracing-rich-subsample-rate` | `0.001` | 0.1% extreme sparsity for high-cardinality events |

**Combined Overhead Example**:
- 1000 steps/sec → 10 batch summaries/sec → 0.08 rich events/sec (with 8 running requests)

### D.2. KV Metrics Availability

| Metric | Guaranteed? | Source |
|--------|-------------|--------|
| GPU allocated/cached blocks | ✅ Yes | `req_to_blocks`, `num_cached_block` |
| GPU total/free blocks, usage | ✅ Yes | `block_pool` methods |
| CPU/disk offload blocks | ❌ Optional | Best-effort if offload manager exists |
| Effective prompt length | ❌ Optional | Best-effort if `request.num_cached_tokens` exists |

**Note**: KV metrics set may be finalized during PR #4 implementation based on what is cheaply accessible via existing interfaces.

### D.3. Span Architecture

**Journey Tracing Spans** (existing):
- `llm_request` (API layer, per-request)
- `llm_core` (Scheduler, per-request)

**Step Tracing Span** (new):
- `scheduler_steps` (Scheduler, one long-lived span)
- Contains multiple `step.BATCH_SUMMARY` and `step.REQUEST_SNAPSHOT` events

**Span Lifecycle Policy**:
- Created at scheduler init or lazily on first trace
- Ended ONLY if deterministic shutdown hook exists (`EngineCore.shutdown()` or `Scheduler.close()`)
- If no reliable hook, leave open (acceptable for OTEL backends)
- Do NOT use `__del__` (unreliable)

### D.4. Monotonic Timestamp Correlation

**Correlation via step.id**:
```python
# Journey event
span.add_event("journey.SCHEDULED", {"ts.monotonic_ns": T})

# Step batch summary
span.add_event("step.BATCH_SUMMARY", {
    "step.id": 42,
    "step.ts_start_ns": T1,
    "step.ts_end_ns": T2,
})

# Rich snapshot
span.add_event("step.REQUEST_SNAPSHOT", {
    "step.id": 42,
    "request.id": "req-abc",
})
```

**Query Pattern**: Find journey event with `ts.monotonic_ns=T`, find step with `step.id=N` where `T1 <= T <= T2`, correlate via `step.id`.

### D.5. Testing Strategy

**Deterministic Sampling** (PR #2, #3, #5):
```python
def _make_deterministic_sampler(seed: int = 0) -> Callable[[str, float], bool]:
    """Stable hash-based sampler using hashlib.sha1 (NOT hash())."""
    import hashlib
    def sampler(key: str, rate: float) -> bool:
        hash_bytes = hashlib.sha1(f"{seed}:{key}".encode()).digest()
        hash_value = int.from_bytes(hash_bytes[:8], 'big') / (2**64)
        return hash_value < rate
    return sampler
```

**Per-PR Tests**:
- **PR #1**: Dual-write verification (both attributes present, values consistent)
- **PR #2**: Deterministic sampling tests (exact sample sets, no statistical tests)
- **PR #3**: Deterministic step sampling, payload validation, KV consistency checks
- **PR #4**: Interface validation, defensive error handling, optional field handling
- **PR #5**: Deterministic rich subsampling, cardinality checks, payload validation

**Regression Tests**: All existing scheduler tests must pass, no performance regression when disabled

---

## E. Verification Checklist

### PR #1: Timestamp Precision (DEFERRED)
- [ ] Both `ts.monotonic` and `ts.monotonic_ns` present on all journey events
- [ ] `ts.monotonic` unchanged (backward compatible)
- [ ] New constant `JOURNEY_TS_MONOTONIC_NS` added
- [ ] Values consistent (computed from same source OR loose tolerance to avoid CI flakiness)

### PR #2: Sampling (DEFERRED)
- [ ] CLI flag accepted, invalid rates rejected
- [ ] Sampling works (0.0 → no spans, 1.0 → all spans)
- [ ] Deterministic test validates exact sample set
- [ ] Sampling state cleaned up (no memory leak)

### PR #3: Batch Summary ✅ COMPLETE
- [x] CLI flags accepted, disabled by default
- [x] Step span lifecycle policy implemented (long-lived span)
- [x] All required attributes present and correct
- [x] Batch composition sums validate
- [x] Empty schedule emits batch summary
- [x] CLI wiring regression test added

### PR #4: KV Utilities ✅ COMPLETE
- [x] New module exists, uses only existing interfaces
- [x] GPU metrics always present (from existing exposed fields)
- [x] Optional fields handled gracefully (None if unavailable)
- [x] No new KV subsystem APIs, methods, or Request fields added
- [x] All metrics cheaply accessible via existing read-only interfaces

### PR #5: Rich Snapshot ✅ COMPLETE
- [x] CLI flag accepted, subsampling works correctly
- [x] Event count matches running request count at SchedulerOutput construction
- [x] All required attributes present
- [x] Only cheaply available quantities included
- [x] Snapshot uses same request set as SchedulerOutput
- [x] Two-stage sampling implemented (batch → rich subsample)
- [x] CLI wiring tested (regression protection)

---

## F. Design Decisions (Locked)

1. **Span lifecycle**: Deterministic hook or open span (no `__del__`, no rotation)
2. **KV metrics**: Minimal set from existing interfaces, finalized during PR #4 based on cheap accessibility
3. **Tracing payload**: Only cheaply available quantities (no expensive scans, no new bookkeeping)
4. **Empty schedules**: Emit batch summary with zero counts (liveness monitoring)
5. **Rich snapshot scope**: Only RUNNING requests at SchedulerOutput construction point
6. **Phase detection**: Use scheduler's `num_output_tokens == 0` logic
7. **Timestamp upgrade**: Dual-write (backward compatible)
8. **Sampling determinism**: `hashlib.sha1` for tests, `random.random()` for production (not stable across runs)
9. **KV field naming**: `kv.blocks_<type>_<scope>` pattern (consistent, explicit)

---

**END OF PLAN**

Ready for implementation.
