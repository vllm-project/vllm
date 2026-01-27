# Journey Tracing Dual-Stream Architecture - Implementation Plan

## Overview

This document outlines the implementation plan for dual-stream journey tracing using OpenTelemetry spans. The implementation is split into **9 independently safe PRs**, each building on the previous ones while maintaining system stability and correctness.

**Architecture**: Dual-stream design with parent-child span linkage
- **API Layer**: Creates parent spans (`llm_request`) in OpenAI serving layer
- **Core Layer**: Creates child spans (`llm_core`) in scheduler
- **Parent-Child Linkage**: Via W3C Trace Context propagation (traceparent headers)
- **Real-Time Emission**: Events emitted directly to spans (no buffering)

**Timeline**: ~2 weeks for complete implementation
**Total Changes**: ~560 lines added, ~150 lines removed, 41 tests

---

## ✅ Completed Prerequisites

### PR #0: Remove EngineCoreEvent System (COMPLETED)

**Status**: ✅ Merged (commit 717f90eb5)

**What was removed**:
- `EngineCoreEvent` and `EngineCoreEventType` classes (v0.0.1 legacy metrics system)
- `events` field and `record_event()`/`take_events()` methods from Request
- `update_from_events()` logic from IterationStats
- Redundant time delta calculations from `do_tracing()`

**What was preserved**:
- ✅ `RequestJourneyEvent` system (current v0.1.x journey tracing)
- ✅ `OutputProcessor.do_tracing()` method (current OTEL export mechanism)
- ✅ Journey event buffering (still needed for do_tracing())

**What was added**:
- Timestamp extraction from `RequestJourneyEvent` to restore Prometheus metrics
- `queued_ts` from first QUEUED event
- `scheduled_ts` from first SCHEDULED event

**Key clarification**: EngineCoreEvent was the "legacy" system (now removed). The current tracing system is `RequestJourneyEvent` + `do_tracing()` (kept and functional).

---

## Core Principles (Applied to ALL PRs)

### 🔒 The Iron Rule: No Incomplete Resources

**If a PR creates any resource that needs cleanup (span, dict entry, set membership, per-request state), that SAME PR must:**

1. ✅ Terminate it on all exits
2. ✅ Clean it on all termination paths
3. ✅ Have tests proving it

**NO EXCEPTIONS:**
- ❌ "We'll add cleanup later"
- ❌ "We'll add ABORTED in the next PR"
- ❌ "We'll fix the leak in PR #X"

### 1. Current Tracing System Preserved

- ✅ **Current system**: `RequestJourneyEvent` + buffering + `OutputProcessor.do_tracing()`
- ✅ **Already removed**: `EngineCoreEvent` (v0.0.1 legacy, see PR #0 above)
- ✅ New dual-stream code runs **in parallel** with current buffering system
- ✅ `OutputProcessor.do_tracing()` remains functional throughout (not being removed)
- ✅ `RequestJourneyEvent` dataclass preserved (used for Prometheus metrics and do_tracing())
- ✅ Journey event buffering preserved until PR #9 (then removed, but do_tracing() stays)

### 2. Performance First

- ✅ Zero overhead when `enable_journey_tracing=False`
- ✅ Minimal overhead when enabled but tracer not configured
- ✅ Early returns with single boolean checks
- ✅ No unnecessary allocations
- ✅ Cached checks where possible

### 3. Memory Safety

- ✅ No memory leaks in any PR
- ✅ All dictionaries properly cleaned up
- ✅ Spans always closed on all paths
- ✅ Tests verify no leaks
- ✅ Centralized cleanup functions

### 4. No Breaking Changes

- ✅ Backward compatible at every step
- ✅ No API changes to public interfaces
- ✅ No config changes (use existing flags)
- ✅ Existing functionality preserved
- ✅ Safe to merge each PR independently

### 5. Minimal Code Changes

- ✅ Only necessary changes for the feature
- ✅ No refactoring unrelated code
- ✅ No style/formatting-only changes
- ✅ Keep diffs focused and reviewable

### 6. Test Coverage

- ✅ Each PR includes relevant tests
- ✅ Tests verify memory safety (no leaks)
- ✅ Tests verify performance (disabled mode)
- ✅ Tests verify graceful degradation
- ✅ Tests verify all termination paths

### 7. Initialization Safety

- ✅ API server tracing properly initialized
- ✅ API spans emit correctly
- ✅ Parent-child linkage works (API → Core)
- ✅ Graceful degradation if OTLP endpoint not configured
- ✅ No race conditions in initialization order

### 8. Defensive Programming

- ✅ Tracing failures never break request processing
- ✅ All span operations wrapped in try/except
- ✅ Idempotent operations (safe to call multiple times)
- ✅ Safe when span is None or not recording

### 9. Cleanup Decoupling (CRITICAL)

**Core spans and journey state are INDEPENDENT concerns:**

- ✅ **Core span cleanup**: Runs whenever `_core_spans` dict has an entry (regardless of flags)
- ✅ **Journey state cleanup**: Only runs when `_enable_journey_tracing=True`
- ✅ **NEVER gate span cleanup behind feature flags** - spans created = spans cleaned
- ✅ Cleanup methods handle both concerns independently in same function

**Bad (creates leaks)**:
```python
def _end_core_span_and_cleanup(self, request: Request) -> None:
    if not self._enable_journey_tracing:
        return  # ❌ SKIPS SPAN CLEANUP if flag is off!
```

**Good (decoupled)**:
```python
def _end_core_span_and_cleanup(self, request: Request) -> None:
    request_id = request.request_id

    # Cleanup #1: Core spans (independent of journey flag)
    core_span = self._core_spans.pop(request_id, None)
    if core_span is not None:
        core_span.end(end_time=time.time_ns())

    # Cleanup #2: Journey state (only if enabled)
    if self._enable_journey_tracing:
        self._first_token_emitted.discard(request_id)
        self._journey_prefill_hiwater.pop(request_id, None)
```

---

## Scheduler Step Counter Contract

**Purpose**: Track scheduler invocations for event correlation and tracing.

**Rules**:
1. **Incremented**: At START of `schedule()` call (before any work)
2. **Frequency**: Increments on EVERY call, even empty schedules
3. **First value**: First schedule produces step `1` (initialized to `0`)
4. **Lifetime**: NEVER reset - monotonic for lifetime of scheduler
5. **Thread safety**: Scheduler is single-threaded, no locking needed

**Event Correlation**:
- `QUEUED` (in `add_request`): Uses current `scheduler_step_counter` (may be 0 if no schedule yet)
- `SCHEDULED` (in `schedule`): Uses current step (after increment)
- `FIRST_TOKEN` (in `update_from_output`): Uses `scheduler_output.scheduler_step`
- `FINISHED` (in `update_from_output` or `finish_requests`): Uses output step or current step

**Test Requirements**:
- All tests follow this semantic consistently
- Tests verify step is monotonic (never decreases)
- Tests verify step increments even on empty schedules
- Tests verify correlation across events for same request

---

## PR Sequence Summary

| PR # | Branch | Goal | Size | Tests | Status |
|------|--------|------|------|-------|--------|
| #0 | `removelegacy` | Remove EngineCoreEvent | ~130 removed | 1 new + existing | ✅ **COMPLETED** |
| #1 | `pr1ofjourney` | Init tracer in scheduler | ~25 lines | 4 | ✅ **COMPLETED** |
| #2 | `journey-tracing-02-core-spans-lifecycle` | Create & cleanup core spans | ~100 lines | 6 | ✅ **COMPLETED** |
| #3 | `pr3ofjourney` | Add journey state & cleanup | ~26 lines | 4 | ✅ **COMPLETED** |
| #4 | `pr4ofjourney` | Emit events to core spans | ~113 lines | 9 | ✅ **COMPLETED** |
| #5 | `pr5ofjourney` | Add API span tracking dict | ~67 lines | 8 | ✅ **COMPLETED** |
| #6 | `journey-tracing-06-api-spans-full-lifecycle` | Create & close API spans | ~150 lines | 9 | **All closure paths in same PR** ✅ |
| #7 | `journey-tracing-07-context-propagation` | Link parent-child spans | ~25 lines | 4 | No new resources |
| #8 | `journey-tracing-08-api-additional-events` | Emit API lifecycle events | ~80 lines | 5 | No new resources |
| #9 | `journey-tracing-09-remove-buffering` | Remove journey event buffering | ~150 removed | 4 | Clean break |

**Total**: ~560 lines added, ~150 lines removed, 45 tests

---

## PR Dependencies

```
PR #0 (Remove EngineCoreEvent) ✅ COMPLETED
    ↓
PR #1 (Scheduler Tracer Init) ✅ COMPLETED
    ↓
PR #2 (Core Span + Cleanup) ✅ COMPLETED
    ↓
PR #3 (Journey State + Cleanup) ✅ COMPLETED
    ↓
PR #4 (Core Event Emit) ✅ COMPLETED
    ↓
PR #5 (API Metadata) ← independent, can be parallel
    ↓
PR #6 (API Span + DEPARTED/ABORTED) ← MUST include all closure paths
    ↓
PR #7 (Context Propagation) ← depends on #2 and #6
    ↓
PR #8 (API Additional Events) ← no new resources, safe
    ↓
PR #9 (Remove Buffering) ← depends on all above working
```

**Critical PRs**: #2 and #6 must include complete resource lifecycle (create + cleanup).

---

## Detailed PR Breakdown

---

### PR #1: Engine - Scheduler Tracer Init ✅ COMPLETED

**Branch**: `pr1ofjourney`

**Status**: ✅ **COMPLETED** 

**Goal**: Initialize tracer in scheduler without creating any per-request state.

**Why Safe**: No per-request state introduced, no spans created, no cleanup needed.

#### Changes

```python
# vllm/v1/core/sched/scheduler.py

# Add at top of file (after existing imports)
try:
    from vllm.tracing import SpanAttributes
except Exception:
    SpanAttributes = None  # type: ignore

class Scheduler:
    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        structured_output_manager: StructuredOutputManager,
        block_size: int,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        # ... existing initialization code ...

        # Existing journey tracing flag
        self._enable_journey_tracing = (
            self.observability_config.enable_journey_tracing
        )

        # ... existing journey tracing state initialization ...

        # NEW: Initialize tracer for OTEL span creation
        self.tracer: Any | None = None
        if self._enable_journey_tracing:
            endpoint = self.observability_config.otlp_traces_endpoint
            if endpoint is not None:
                try:
                    from vllm.tracing import init_tracer
                    self.tracer = init_tracer("vllm.scheduler", endpoint)
                except Exception as e:
                    logger.warning(
                        "Failed to initialize tracer for journey tracing: %s", e
                    )
```

**Also Updated**:
- `tests/v1/core/utils.py`: Added `otlp_traces_endpoint` parameter to `create_scheduler()`
- `tests/v1/core/test_scheduler.py`: Added `patch` import for mocking

#### Safety Checklist

- ✅ No per-request state introduced (only class-level tracer instance)
- ✅ No spans created (just initialization)
- ✅ No cleanup needed (tracer is shared, not per-request)
- ✅ Legacy tracing untouched (journey event buffering unchanged)
- ✅ Zero overhead when disabled (boolean + endpoint check)
- ✅ Graceful degradation with warning log on failure
- ✅ Backward compatible (all parameters optional)

#### Tests (All Passing)

1. **`test_tracer_init_when_endpoint_set()`**
   - Uses mocking to avoid OTEL collector dependency
   - Verifies `scheduler.tracer` is set to mock tracer
   - Verifies `init_tracer()` called with correct parameters

2. **`test_tracer_none_when_endpoint_not_set()`**
   - Tests 3 negative cases: no endpoint, disabled flag, both
   - Verifies `scheduler.tracer` is None in all cases
   - Verifies no overhead when disabled

3. **`test_scheduler_init_succeeds_with_tracing_enabled()`**
   - Uses mocking for deterministic test
   - Smoke test ensuring tracing config doesn't break scheduler
   - Verifies tracer attribute exists and is set correctly

4. **`test_tracer_init_handles_failure_gracefully()`**
   - Mocks `init_tracer()` to raise exception
   - Verifies scheduler initializes successfully despite failure
   - Verifies tracer is None when init fails
   - Verifies defensive exception handling works

**Test Results**: ✅ All 85 tests in test_scheduler.py passing (4 new, 81 existing)

**Size**: ~25 lines production code, ~110 lines test code, 4 tests
**Review Time**: ~10 minutes
**Actual Implementation Time**: ~1 hour

---

### PR #2: Engine - Core Span Create AND Close ✅ COMPLETED

**Branch**: `pr2ofjourney`

**Status**: ✅ **MERGED** (commit fdbe492de)

**Completed**: 2026-01-26

**Goal**: Create core spans on `add_request`, guarantee closure on all termination paths.

**Why Safe**: Spans created AND cleaned in same PR. All termination paths covered.

**CRITICAL**: This PR includes both creation and cleanup. No "we'll add cleanup later".

#### Changes

```python
# vllm/v1/core/sched/scheduler.py

class Scheduler:
    def __init__(self, ...):
        # ... existing init including tracer from PR #1 ...

        # NEW: Track active core spans (request_id → Span)
        # Always initialize to avoid AttributeError when journey tracing disabled
        self._core_spans: dict[str, Any] = {}

    def _create_core_span(self, request: Request) -> Any | None:
        """Create child span for engine-core journey tracing.

        Extracts parent span context from request.trace_headers and creates
        child span in the same distributed trace.

        Args:
            request: The request for which to create a span

        Returns:
            Span object if tracer available and creation succeeds, None otherwise
        """
        if not self.tracer:
            return None

        try:
            from vllm.tracing import SpanAttributes, extract_trace_context
            from opentelemetry.trace import SpanKind
        except ImportError:
            return None

        # Extract parent context from trace_headers (injected by API layer)
        parent_context = None
        if request.trace_headers:
            parent_context = extract_trace_context(request.trace_headers)

        # Create child span
        core_span = self.tracer.start_span(
            name="llm_core",
            kind=SpanKind.INTERNAL,
            context=parent_context,
            start_time=time.time_ns(),
        )

        # Set span attributes
        core_span.set_attribute(SpanAttributes.GEN_AI_REQUEST_ID, request.request_id)

        return core_span

    def _end_core_span_and_cleanup(self, request: Request) -> None:
        """End core span and cleanup all journey tracing state.

        CRITICAL: This is the centralized cleanup method that must be called
        for ALL request termination paths to prevent memory leaks.

        This method handles TWO INDEPENDENT concerns:
        1. Core span cleanup (runs if span exists, regardless of flags)
        2. Journey state cleanup (runs if journey tracing enabled) - added in PR #3

        Safe to call multiple times for same request (idempotent).

        Args:
            request: The request being terminated
        """
        request_id = request.request_id

        # Cleanup #1: Core spans (independent of journey tracing flag)
        # CRITICAL: ALWAYS runs if span exists, never gate behind feature flags
        core_span = self._core_spans.pop(request_id, None)
        if core_span is not None:
            try:
                import time
                core_span.end(end_time=time.time_ns())
            except Exception as e:
                logger.warning(
                    "Failed to end core span for request %s: %s",
                    request_id,
                    e
                )

        # Cleanup #2: Journey state (only if enabled) - will be added in PR #3
        # (PR #3 will add: self._first_token_emitted.discard(request_id), etc.)

    def add_request(self, request: Request) -> None:
        """Add new request to waiting queue."""
        # ... existing code: add to self.requests, self.waiting ...

        # Existing: record event if log_stats enabled
        if self.log_stats:
            request.record_event(EngineCoreEventType.QUEUED)

        # NEW: Create child span
        if self._enable_journey_tracing:
            core_span = self._create_core_span(request)
            if core_span:
                self._core_spans[request.request_id] = core_span

        # NOTE: Journey event emission will be added in PR #4

    def finish_requests(
        self,
        request_ids: list[str],
        finished_status: RequestStatus,
    ) -> None:
        """Abort requests and cleanup all resources.

        Called when explicitly aborting requests (e.g., client disconnect).
        """
        for request_id in request_ids:
            request = self.requests.get(request_id)
            if request is None:
                continue

            # ... existing status update code ...
            request.status = finished_status

            # NOTE: Journey event emission will be added in PR #4

            # NEW: End core span and cleanup (CRITICAL: in same PR as span creation)
            self._end_core_span_and_cleanup(request)

    def _update_from_output(
        self,
        request: Request,
        scheduler_output: SchedulerOutput,
        req_output: RequestOutput,
        seq_output: SequenceOutput,
    ) -> None:
        """Update request state from engine output.

        Called on every output token from the engine.
        """
        # ... existing code: update request state, handle stopped ...

        if stopped:
            # NEW: Ensure cleanup always happens, even if routed_experts fails
            try:
                routed_experts = self._get_routed_experts(request)
            except Exception:
                pass  # routed_experts is optional; failures must not prevent cleanup
            finally:
                # CRITICAL: Always cleanup, even on exceptions
                self._end_core_span_and_cleanup(request)

            # ... existing code: _free_request (called AFTER cleanup) ...
            kv_transfer_params = self._free_request(request)
            # ... rest of existing code ...
```

#### Safety Checklist

- ✅ **Spans created → Spans closed on all paths**
  - ✅ `finish_requests()` → cleanup called
  - ✅ Natural completion (`stopped=True`) → cleanup called in finally block
  - ✅ Exception during `routed_experts` → cleanup called in finally block
- ✅ `_core_spans` dict cleaned on all termination paths
- ✅ Tests prove no leaks
- ✅ Legacy tracing untouched
- ✅ Zero overhead when disabled (early return)
- ✅ Graceful degradation when tracer is None

#### Termination Paths Covered

1. **Natural completion**: `_update_from_output()` when `stopped=True` → finally block ensures cleanup
2. **Explicit abort**: `finish_requests()` → cleanup called explicitly
3. **Exception during completion**: `_update_from_output()` finally block → cleanup called
4. **All paths** lead to `_end_core_span_and_cleanup()` before `_free_request()`

#### Tests

1. **`test_core_span_created_on_add_request()`**
   - Verify span exists in `_core_spans` dict after `add_request()`
   - Verify span has correct name and attributes

2. **`test_core_span_closed_on_finish_requests()`**
   - Call `finish_requests()`
   - Verify span.end() called
   - Verify span removed from `_core_spans` dict

3. **`test_core_span_closed_on_natural_completion()`**
   - Simulate natural completion in `_update_from_output()`
   - Verify cleanup called in finally block
   - Verify span ended and removed

4. **`test_core_span_closed_on_exception()`**
   - Mock `_get_routed_experts()` to raise exception
   - Verify cleanup still called in finally block
   - Verify span ended and removed

5. **`test_no_span_leak_when_tracer_none()`**
   - Set `tracer=None`
   - Add multiple requests
   - Verify `_core_spans` dict stays empty
   - Verify no memory growth

6. **`test_parent_context_extraction()`**
   - Create request with `trace_headers` containing traceparent
   - Verify `extract_trace_context()` called
   - Verify parent context passed to span creation

#### Implementation Summary

**What Was Built**:
- Added `_core_spans` dictionary to track active spans per request
- Created `_create_core_span()` helper with explicit OpenTelemetry parameters
  - Uses `SpanKind.INTERNAL`, `start_time=time.time_ns()`
  - Extracts parent context from `trace_headers`
  - Sets `GEN_AI_REQUEST_ID` attribute for correlation
- Created `_end_core_span_and_cleanup()` idempotent helper
  - Uses explicit `end_time=time.time_ns()` for consistency
  - Safe to call multiple times (idempotent via `.pop()`)
- Modified `add_request()` to create and store core spans
- Modified `update_from_output()` with try/finally (natural completion path)
- Modified `finish_requests()` with try/finally (explicit abort path)
- Added 6 comprehensive tests covering all safety properties

**Test Results**: ✅ All 91 tests passing (6 new, 85 existing)

**Safety Guarantees**:
- ✅ All termination paths properly cleanup spans (no leaks)
- ✅ Cleanup uses try/finally (runs even if teardown throws)
- ✅ Defensive error handling (tracing never crashes requests)
- ✅ Zero overhead when tracing disabled
- ✅ Idempotent cleanup (safe to call multiple times)

**Size**: ~125 lines production code, ~245 lines test code, 6 tests
**Review Time**: ~20 minutes
**Actual Implementation Time**: ~2 hours

---

### PR #3: Engine - Journey State WITH Cleanup ✅ COMPLETED

**Branch**: `pr3ofjourney`

**Status**: ✅ **COMPLETED** (commit f4cf7903c, PR #33126)

**Completed**: 2026-01-26

**Goal**: Add journey progress tracking state, integrate cleanup into existing `_end_core_span_and_cleanup()`.

**Why Safe**: State created AND cleaned in same function as PR #2. All termination paths already covered.

**Scope**: ONLY state initialization and cleanup. Progress snapshot semantics tested in PR #4 when actually used.

#### Changes

```python
# vllm/v1/core/sched/scheduler.py

class Scheduler:
    def __init__(self, ...):
        # ... existing init including _core_spans from PR #2 ...

        if self._enable_journey_tracing:
            # NEW: Track which requests have emitted FIRST_TOKEN (dedup)
            self._first_token_emitted: set[str] = set()
            # NEW: Prefill progress high-water marks (survives preemption)
            self._journey_prefill_hiwater: dict[str, int] = {}

    # NOTE: _compute_progress_snapshot() method will be added in PR #4
    # when event emission starts using it. This PR only adds the STATE,
    # not the logic that uses it.

    def _end_core_span_and_cleanup(self, request: Request) -> None:
        """End core span and cleanup all journey tracing state.

        CRITICAL: Extended from PR #2 to also clean journey state.

        This method handles TWO INDEPENDENT concerns:
        1. Core span cleanup (always runs if span exists)
        2. Journey state cleanup (only if journey tracing enabled)

        Args:
            request: The request being terminated
        """
        request_id = request.request_id

        # Cleanup #1: Core spans (from PR #2, independent of journey flag)
        core_span = self._core_spans.pop(request_id, None)
        if core_span is not None:
            try:
                import time
                core_span.end(end_time=time.time_ns())
            except Exception as e:
                logger.warning(
                    "Failed to end core span for request %s: %s",
                    request_id,
                    e
                )

        # Cleanup #2: Journey state (NEW in PR #3, only if enabled)
        if self._enable_journey_tracing:
            self._first_token_emitted.discard(request_id)
            self._journey_prefill_hiwater.pop(request_id, None)
```

#### Safety Checklist

- ✅ **State created → State cleaned in same function as PR #2**
- ✅ **Core span cleanup decoupled** from journey flag (always runs if span exists)
- ✅ **Journey state cleanup gated** behind `_enable_journey_tracing` flag
- ✅ All termination paths already call `_end_core_span_and_cleanup()` (from PR #2)
- ✅ Tests prove no set/dict growth
- ✅ Legacy tracing untouched
- ✅ Zero overhead when disabled (sets/dicts not created)

#### Termination Paths Covered

Same as PR #2 (already covered):
1. Natural completion → cleanup called
2. Explicit abort → cleanup called
3. Exception → cleanup called

#### Tests

1. **`test_journey_state_created()`**
   - Verify `_first_token_emitted` set initialized
   - Verify `_journey_prefill_hiwater` dict initialized

2. **`test_journey_state_cleaned_on_finish()`**
   - Add request, populate state
   - Call `finish_requests()`
   - Verify state cleaned (not in set/dict)

3. **`test_journey_state_cleaned_on_completion()`** ⭐ **CRITICAL**
   - Add request, populate state
   - Simulate natural completion
   - Verify state cleaned (catches memory leak bug)

4. **`test_no_state_leak()`**
   - Add and complete 100 requests
   - Verify `_first_token_emitted` set size = 0
   - Verify `_journey_prefill_hiwater` dict size = 0

**NOTE**: Progress snapshot logic (`_compute_progress_snapshot()`) will be added and tested in PR #4 when event emission starts using it. This PR focuses purely on state initialization and cleanup.

**Size**: ~40 lines production code, ~200 lines test code, 4 tests
**Review Time**: ~15 minutes

---

### PR #4: Engine - Emit Journey Events

**Status**: ✅ Completed (PR #12 merged)

**Branch**: `pr4ofjourney`

**Goal**: Add progress snapshot logic and emit events to core spans. No new resources, just additive event emission.

**Why Safe**: No new resources created, just emitting events to existing spans. Defensive error handling.

**What was implemented**:
- Extended `_emit_journey_event()` to accept optional span parameter
- Added span emission logic with defensive error handling (try/except around all OTEL calls)
- Updated all 6 call sites to pass span from `_core_spans` dict
- Added FINISHED emission in natural completion path (update_from_output)
- Extended `_compute_progress_snapshot()` to support WAITING phase for QUEUED events
- Changed QUEUED scheduler_step from None to counter (typically 0)
- Added 9 comprehensive tests covering all event types and edge cases
- Progress snapshot computed once and reused for both span and buffering (performance optimization)

**Tests added**: 9 tests (328 lines), all passing
- `test_events_emitted_to_span()` - Verify QUEUED, SCHEDULED emitted
- `test_event_attributes_complete()` - Verify all attributes present
- `test_defensive_error_handling()` - Verify request continues when add_event raises
- `test_no_events_when_span_none()` - Verify graceful handling when tracer=None
- `test_legacy_buffering_still_works()` - Verify parallel buffering unchanged
- `test_first_token_dedup_set()` - Verify FIRST_TOKEN deduplication
- `test_first_token_transition_emitted()` - Verify FIRST_TOKEN on 0→N transition
- `test_finished_emitted_to_span()` - Verify FINISHED emission on natural completion
- `test_preempted_event_emitted()` - Verify PREEMPTED event

**Size**: ~113 lines production code (net), 328 lines test code

#### Changes

```python
# vllm/v1/core/sched/scheduler.py

def _compute_progress_snapshot(self, request: Request) -> dict[str, Any]:
    """Compute progress snapshot for journey events.

    Handles preemption correctly by using high-water marks for prefill.

    Args:
        request: The request to compute progress for

    Returns:
        Dict with keys: phase, prefill_done_tokens, prefill_total_tokens,
        decode_done_tokens, decode_max_tokens
    """
    # Determine phase
    if request.status == RequestStatus.WAITING:
        phase = "waiting"
    elif request.num_computed_tokens < request.num_prompt_tokens:
        phase = "prefill"
    else:
        phase = "decode"

    # Prefill progress (use hiwater mark to handle preemption)
    prefill_total = request.num_prompt_tokens
    prefill_done = min(request.num_computed_tokens, prefill_total)

    # Track high-water mark for prefill
    if self._enable_journey_tracing:
        current_hiwater = self._journey_prefill_hiwater.get(request.request_id, 0)
        prefill_done = max(prefill_done, current_hiwater)
        self._journey_prefill_hiwater[request.request_id] = prefill_done

    # Decode progress
    decode_done = max(0, request.num_computed_tokens - prefill_total)
    decode_max = request.max_tokens if request.max_tokens else 0

    return {
        "phase": phase,
        "prefill_done_tokens": prefill_done,
        "prefill_total_tokens": prefill_total,
        "decode_done_tokens": decode_done,
        "decode_max_tokens": decode_max,
    }

def _emit_journey_event(
    self,
    request: Request,
    event_type: RequestJourneyEventType,
    scheduler_step: int | None,
    span: Any | None = None,  # NEW parameter
    schedule_kind: ScheduleKind | None = None,
    finish_status: str | None = None,
) -> None:
    """Emit journey event to span (new) and buffer (legacy, parallel).

    This is the central emission point for all journey events. Events are
    emitted to both spans (new) and buffers (legacy) in parallel until
    legacy buffering is removed in PR #9.

    DEFENSIVE: Must never break request processing if tracing fails.

    Args:
        request: The request this event is for
        event_type: Type of lifecycle event
        scheduler_step: Scheduler step counter (never None in dual-stream)
        span: OTEL span to emit to (required for event emission)
        schedule_kind: FIRST or RESUME (SCHEDULED events only)
        finish_status: Terminal status string (FINISHED events only)
    """
    if not self._enable_journey_tracing:
        return  # Near-zero overhead: single boolean check

    # NEW: Emit to span (parallel to legacy buffering)
    if span and span.is_recording() and SpanAttributes is not None:
        # Compute progress snapshot
        progress = self._compute_progress_snapshot(request)

        # Capture timestamps
        ts_monotonic = time.monotonic()
        ts_epoch_ns = time.time_ns()

        # Build event attributes
        attributes = {
            SpanAttributes.JOURNEY_EVENT_TYPE: event_type.name,
            SpanAttributes.JOURNEY_TS_MONOTONIC: ts_monotonic,
            SpanAttributes.JOURNEY_SCHEDULER_STEP: scheduler_step,
            SpanAttributes.JOURNEY_PHASE: progress["phase"],
            SpanAttributes.JOURNEY_PREFILL_DONE_TOKENS: progress["prefill_done_tokens"],
            SpanAttributes.JOURNEY_PREFILL_TOTAL_TOKENS: progress["prefill_total_tokens"],
            SpanAttributes.JOURNEY_DECODE_DONE_TOKENS: progress["decode_done_tokens"],
            SpanAttributes.JOURNEY_DECODE_MAX_TOKENS: progress["decode_max_tokens"],
            SpanAttributes.JOURNEY_NUM_PREEMPTIONS: request.num_preemptions,
        }

        # Add optional fields
        if schedule_kind is not None:
            attributes[SpanAttributes.JOURNEY_SCHEDULE_KIND] = schedule_kind.name
        if finish_status is not None:
            attributes[SpanAttributes.JOURNEY_FINISH_STATUS] = finish_status

        # DEFENSIVE: Ensure tracing failures never break request processing
        try:
            span.add_event(
                name=f"journey.{event_type.name}",
                attributes=attributes,
                timestamp=ts_epoch_ns,
            )
        except Exception:
            # Tracing must never break request processing
            logger.debug(
                "Failed to emit journey event %s for request %s",
                event_type.name,
                request.request_id,
            )

    # EXISTING: Current buffering (unchanged, parallel operation)
    # RequestJourneyEvent objects are still created and buffered
    # for use by OutputProcessor.do_tracing() OTEL export
    # This buffering will be removed in PR #9 (but do_tracing() stays)
    # ... existing buffering code ...

# Update all call sites to pass span:

def add_request(self, request: Request) -> None:
    # ... existing span creation from PR #2 ...

    # NEW: Emit QUEUED event
    if self._enable_journey_tracing:
        core_span = self._core_spans.get(request.request_id)
        self._emit_journey_event(
            request,
            RequestJourneyEventType.QUEUED,
            scheduler_step=self.scheduler_step_counter,
            span=core_span,
        )

def schedule(self, ...):
    # ... in scheduling loop, when request scheduled ...

    # NEW: Pass span to SCHEDULED event
    if self._enable_journey_tracing and schedule_kind is not None:
        core_span = self._core_spans.get(request.request_id)
        self._emit_journey_event(
            request,
            RequestJourneyEventType.SCHEDULED,
            scheduler_step=curr_step,
            span=core_span,
            schedule_kind=schedule_kind,
        )

def _preempt_request(self, request: Request, scheduler_step: int) -> None:
    # ... existing preemption logic ...

    # NEW: Emit PREEMPTED event
    if self._enable_journey_tracing:
        core_span = self._core_spans.get(request.request_id)
        self._emit_journey_event(
            request,
            RequestJourneyEventType.PREEMPTED,
            scheduler_step=scheduler_step,
            span=core_span,
        )

def _update_from_output(self, ...):
    # ... existing logic ...

    # NEW: Emit FIRST_TOKEN (deduped)
    if (
        request.status == RequestStatus.RUNNING
        and num_output_tokens == 1
        and request.request_id not in self._first_token_emitted
    ):
        self._first_token_emitted.add(request.request_id)
        if self._enable_journey_tracing:
            core_span = self._core_spans.get(request.request_id)
            self._emit_journey_event(
                request,
                RequestJourneyEventType.FIRST_TOKEN,
                scheduler_step=scheduler_output.scheduler_step,
                span=core_span,
            )

    # NEW: Emit FINISHED (before cleanup)
    if stopped:
        try:
            routed_experts = self._get_routed_experts(request)

            # NEW: Emit FINISHED event before cleanup
            if self._enable_journey_tracing:
                core_span = self._core_spans.get(request.request_id)
                try:
                    self._emit_journey_event(
                        request,
                        RequestJourneyEventType.FINISHED,
                        scheduler_step=scheduler_output.scheduler_step,
                        span=core_span,
                        finish_status=_map_finish_status(request.status),
                    )
                except Exception:
                    pass  # Defensive: tracing must never break completion
        except Exception:
            pass
        finally:
            # Cleanup from PR #2 (unchanged)
            self._end_core_span_and_cleanup(request)

def finish_requests(self, ...):
    # ... existing code ...

    # NEW: Emit FINISHED event before cleanup
    if self._enable_journey_tracing:
        core_span = self._core_spans.get(request.request_id)
        finish_status_str = _map_finish_status(finished_status)
        self._emit_journey_event(
            request,
            RequestJourneyEventType.FINISHED,
            scheduler_step=self.scheduler_step_counter,
            span=core_span,
            finish_status=finish_status_str,
        )

    # Cleanup from PR #2 (unchanged)
    self._end_core_span_and_cleanup(request)
```

#### Safety Checklist

- ✅ No new resources created (just event emission)
- ✅ Defensive error handling (try/except around span.add_event)
- ✅ Safe when span is None or not recording
- ✅ Legacy buffering still works (parallel operation)
- ✅ Legacy tracing untouched
- ✅ Zero overhead when disabled

#### Tests

1. **`test_events_emitted_to_span()`**
   - Add and complete request
   - Verify QUEUED, SCHEDULED, FIRST_TOKEN, FINISHED events on span

2. **`test_event_attributes_complete()`**
   - Emit event
   - Verify all required attributes present
   - Verify optional attributes when applicable

3. **`test_defensive_error_handling()`**
   - Mock span.add_event() to raise exception
   - Verify request processing continues
   - Verify exception logged but not raised

4. **`test_no_events_when_span_none()`**
   - Set tracer=None (no span created)
   - Add and complete request
   - Verify no exceptions
   - Verify legacy buffering still works

5. **`test_current_buffering_still_works()`**
   - Enable journey tracing
   - Verify events still buffered for do_tracing() (parallel operation)
   - Verify both span emission and buffering happen

6. **`test_first_token_deduped()`**
   - Emit multiple FIRST_TOKEN events for same request
   - Verify only one event on span
   - Verify dedup set works correctly

7. **`test_progress_snapshot_correct()`** (moved from PR #3)
   - Test prefill phase progress calculation
   - Test decode phase progress calculation
   - Test hiwater mark handling across preemption
   - Verify progress attributes in emitted events

**Size**: ~160 lines production code (includes progress snapshot), ~280 lines test code, 7 tests
**Review Time**: ~20 minutes

---

### PR #5: API - Add Span Tracking Dict

**Status**: ✅ Completed (PR #13 merged)

**Branch**: `pr5ofjourney`

**Goal**: Add separate dict for tracking API spans, avoiding Pydantic serialization risks.

**Why Safe**: No per-request state in serializable models. Spans tracked in separate dict.

**Key Decision**: Store spans in `_api_spans` dict instead of Pydantic model to avoid serialization issues.

**What was implemented**:
- Added `_api_spans` dict to track `(span, arrival_time, first_response_time)` tuples
- Added `_cached_is_tracing_enabled` flag with defensive error handling (defaults to False on failure)
- Added `_get_is_tracing_enabled()` async method with caching to avoid repeated engine calls
- Added `_store_api_span()` method to insert span tracking entries
- Added `_get_api_span_info()` method returning `(None, None, None)` for missing entries
- Added `_cleanup_api_span()` method with idempotent `.pop()` for safe cleanup
- All state is private (`_` prefix) and lives outside Pydantic models
- Added 8 comprehensive unit tests covering all dict operations and edge cases

**Tests added**: 8 tests (177 lines), all passing
- `test_api_span_dict_initialized()` - Verify dict and cache initialized correctly
- `test_store_and_retrieve_api_span()` - Store and retrieve span info
- `test_retrieve_missing_request_returns_none_tuple()` - Safe defaults for missing entries
- `test_cleanup_removes_api_span()` - Cleanup removes dict entry
- `test_cleanup_nonexistent_request_is_safe()` - Cleanup is idempotent
- `test_tracing_enabled_cache_works()` - Cache reduces async calls (call_count == 1)
- `test_tracing_enabled_check_handles_errors()` - Error handling defaults to False
- `test_multiple_requests_tracked_independently()` - Multi-request isolation

**Size**: ~67 lines production code, 177 lines test code

#### Changes

```python
# vllm/entrypoints/openai/engine/serving.py

class OpenAIServing:
    def __init__(self, ...):
        # ... existing init ...

        # NEW: Track API spans separately (not in Pydantic model to avoid serialization)
        # Maps request_id → (span, arrival_time, first_response_time)
        # Type hint 'Any' avoids OTEL import dependency (PR #6 will import OTEL)
        self._api_spans: dict[str, tuple[Any, float, float | None]] = {}

        # NEW: Cache for tracing enabled check (cached once, assumes config immutable)
        self._cached_is_tracing_enabled: bool | None = None

    async def _get_is_tracing_enabled(self) -> bool:
        """Check if journey tracing is enabled.

        Caches result to avoid repeated async calls to engine.
        Cached once at startup; assumes tracing config is immutable.
        Defaults to False on error (tracing must never break serving).

        Returns:
            True if journey tracing is enabled, False otherwise
        """
        if self._cached_is_tracing_enabled is None:
            try:
                self._cached_is_tracing_enabled = (
                    await self.engine_client.is_tracing_enabled()
                )
            except Exception:
                # Defensive: if check fails, assume tracing disabled
                # Prevents repeated exceptions from impacting serving performance
                self._cached_is_tracing_enabled = False
        return self._cached_is_tracing_enabled

    def _store_api_span(
        self,
        request_id: str,
        span: Any,
        arrival_time: float,
    ) -> None:
        """Store API span and timing info for a request.

        Args:
            request_id: The request ID
            span: The OTEL span object
            arrival_time: time.monotonic() when request arrived
        """
        self._api_spans[request_id] = (span, arrival_time, None)

    def _get_api_span_info(
        self,
        request_id: str,
    ) -> tuple[Any | None, float | None, float | None]:
        """Get API span and timing info for a request.

        Returns:
            Tuple of (span, arrival_time, first_response_time)
            Returns (None, None, None) if request not found
        """
        return self._api_spans.get(request_id, (None, None, None))

    def _cleanup_api_span(self, request_id: str) -> None:
        """Remove API span tracking for a request.

        Safe to call even if span was never created (e.g., tracing disabled).
        Called after span.end() in normal flow (PR #6 responsibility).

        Args:
            request_id: The request ID to cleanup
        """
        self._api_spans.pop(request_id, None)
```

#### Safety Checklist

- ✅ No per-request state in Pydantic models (avoids serialization issues)
- ✅ Spans tracked in separate dict (safe for any object type)
- ✅ No spans created yet (just tracking infrastructure)
- ✅ Cleanup method provided (will be used in PR #6)
- ✅ Legacy tracing untouched

#### Tests

(See "What was implemented" section above for full test list - 8 tests implemented)

---

### PR #6: API - Parent Span WITH Full Closure

**Branch**: `journey-tracing-06-api-spans-full-lifecycle`

**Goal**: Create API parent spans AND ensure they're closed on all exit paths in the same PR.

**Why Safe**: Spans created AND DEPARTED/ABORTED events included in same PR. All termination paths covered.

**CRITICAL**: This PR must include DEPARTED and ABORTED events. No "we'll add them later".

#### Changes

```python
# vllm/entrypoints/openai/chat_completion/serving.py

class OpenAIServingChat:
    async def _create_api_span(
        self, request_id: str, raw_request: Request | None
    ) -> Any | None:
        """Create parent span for API-level journey tracing.

        Extracts incoming trace context from request headers (if present) and
        creates parent span that will be linked to child span in engine-core.

        Args:
            request_id: The request ID (e.g., chatcmpl-xxx)
            raw_request: The FastAPI request object with headers

        Returns:
            Span object if tracer available, None otherwise
        """
        try:
            from vllm.tracing import SpanAttributes, extract_trace_context
            from opentelemetry import trace
            from opentelemetry.trace import SpanKind
        except ImportError:
            return None

        # Get tracer from global provider (set by engine during init)
        try:
            tracer_provider = trace.get_tracer_provider()
            tracer = tracer_provider.get_tracer("vllm.api")
        except Exception:
            return None

        # Extract incoming trace context (if client provided traceparent header)
        parent_context = None
        if raw_request:
            trace_headers = await self._get_trace_headers(raw_request.headers)
            if trace_headers:
                parent_context = extract_trace_context(trace_headers)

        # Create parent span (becomes child if parent_context exists, root otherwise)
        api_span = tracer.start_span(
            name="llm_request",
            kind=SpanKind.SERVER,
            context=parent_context,
            start_time=time.time_ns(),
        )

        # Set basic attributes
        api_span.set_attribute(SpanAttributes.GEN_AI_REQUEST_ID, request_id)

        return api_span

    def _safe_emit_departed_event(
        self,
        request_id: str,
    ) -> None:
        """Emit api.DEPARTED event, end span, and cleanup tracking.

        CRITICAL: Idempotent - safe to call even if span already ended.

        Args:
            request_id: The request ID to close span for
        """
        # Get span from tracking dict (from PR #5)
        api_span, arrival_time, first_response_time = self._get_api_span_info(request_id)

        if not api_span or not api_span.is_recording():
            # Still cleanup even if span not recording
            self._cleanup_api_span(request_id)
            return

        try:
            from vllm.tracing import SpanAttributes

            # Emit DEPARTED event (minimal version for this PR)
            # PR #8 will add full latency metrics
            api_span.add_event(
                name="api.DEPARTED",
                attributes={SpanAttributes.EVENT_TS_MONOTONIC: time.monotonic()},
                timestamp=time.time_ns(),
            )

            # End span
            api_span.end(end_time=time.time_ns())
        except Exception:
            pass  # Defensive: tracing must never break response
        finally:
            # CRITICAL: Always cleanup tracking dict (from PR #5)
            self._cleanup_api_span(request_id)

    def _safe_emit_aborted_event(
        self,
        request_id: str,
        error_message: str,
        reason: str | None = None,
    ) -> None:
        """Emit api.ABORTED event, set error status, end span, and cleanup tracking.

        CRITICAL: Idempotent - safe to call even if span already ended.

        Args:
            request_id: The request ID to close span for
            error_message: Error message to record
            reason: Optional reason code (e.g., "client_disconnect")
        """
        # Get span from tracking dict (from PR #5)
        api_span, _, _ = self._get_api_span_info(request_id)

        if not api_span or not api_span.is_recording():
            # Still cleanup even if span not recording
            self._cleanup_api_span(request_id)
            return

        try:
            from vllm.tracing import SpanAttributes
            from opentelemetry.trace import Status, StatusCode

            # Set error status
            api_span.set_status(Status(StatusCode.ERROR, error_message))

            # Emit ABORTED event
            attributes: dict[str, Any] = {
                SpanAttributes.EVENT_TS_MONOTONIC: time.monotonic(),
                "error": error_message,
            }
            if reason:
                attributes["reason"] = reason

            api_span.add_event(
                name="api.ABORTED",
                attributes=attributes,
                timestamp=time.time_ns(),
            )

            # End span
            api_span.end(end_time=time.time_ns())
        except Exception:
            pass  # Defensive
        finally:
            # CRITICAL: Always cleanup tracking dict (from PR #5)
            self._cleanup_api_span(request_id)

    async def create_chat_completion(
        self,
        request: ChatCompletionRequest,
        raw_request: Request | None,
    ) -> AsyncGenerator[str, None] | ChatCompletionResponse | ErrorResponse:
        # ... existing request_id creation ...
        request_id = (
            f"chatcmpl-{self._base_request_id(raw_request, request.request_id)}"
        )

        # NEW: Create API span and store in tracking dict (from PR #5)
        is_tracing_enabled = await self._get_is_tracing_enabled()
        if is_tracing_enabled:
            api_span = await self._create_api_span(request_id, raw_request)
            if api_span:
                arrival_time = time.monotonic()
                self._store_api_span(request_id, api_span, arrival_time)

        # Initialize metadata (no span fields needed)
        request_metadata = RequestResponseMetadata(
            request_id=request_id,
        )

        # ... existing code ...

    async def chat_completion_stream_generator(
        self,
        request: ChatCompletionRequest,
        result_generator: AsyncIterator[RequestOutput],
        request_id: str,
        conversation: list[ConversationMessage],
        tokenizer: TokenizerLike | None,
        request_metadata: RequestResponseMetadata,
    ) -> AsyncGenerator[str, None]:
        """Streaming generator with span closure on all paths."""
        try:
            # ... existing streaming logic ...

            # ... existing code ...
        except GenerationError as e:
            # NEW: Close span on generation error (uses request_id from metadata)
            self._safe_emit_aborted_event(
                request_metadata.request_id,
                f"Generation error: {e}",
                "generation_error"
            )
            yield f"data: {self._convert_generation_error_to_streaming_response(e)}\n\n"
            yield "data: [DONE]\n\n"
            return  # Span closed and cleaned, safe to return
        except Exception as e:
            # NEW: Close span on unexpected exception
            self._safe_emit_aborted_event(
                request_metadata.request_id, str(e), "exception"
            )
            logger.exception("Error in chat completion stream generator.")
            data = self.create_streaming_error_response(e)
            yield f"data: {data}\n\n"
            yield "data: [DONE]\n\n"
            return  # Span closed and cleaned, safe to return

        # Send final done message
        yield "data: [DONE]\n\n"

        # NEW: Close span on success (retrieves span from dict, ends it, cleans up)
        self._safe_emit_departed_event(request_metadata.request_id)

    async def chat_completion_full_generator(
        self,
        request: ChatCompletionRequest,
        result_generator: AsyncIterator[RequestOutput],
        request_id: str,
        model_name: str,
        conversation: list[ConversationMessage],
        tokenizer: TokenizerLike | None,
        request_metadata: RequestResponseMetadata,
    ) -> ErrorResponse | ChatCompletionResponse:
        created_time = int(time.time())
        final_res: RequestOutput | None = None
        span_closed = False

        # CRITICAL: Outer try/finally ensures span cleanup on ANY exception path
        try:
            # Generator iteration
            try:
                async for res in result_generator:
                    final_res = res
            except asyncio.CancelledError:
                self._safe_emit_aborted_event(
                    request_metadata.request_id,
                    "Client disconnected",
                    "client_disconnect"
                )
                span_closed = True
                return self.create_error_response("Client disconnected")
            except ValueError as e:
                self._safe_emit_aborted_event(
                    request_metadata.request_id, str(e), "validation_error"
                )
                span_closed = True
                return self.create_error_response(e)

            # Response building
            assert final_res is not None

            # ... existing response building code ...

            # NEW: Close span on success (retrieves span from dict, ends it, cleans up)
            self._safe_emit_departed_event(request_metadata.request_id)
            span_closed = True

            return response
        finally:
            # CRITICAL: Outer finally catches ALL other exceptions
            # Check if span exists in dict (not if metadata has span field)
            if not span_closed:
                span, _, _ = self._get_api_span_info(request_metadata.request_id)
                if span:
                    self._safe_emit_aborted_event(
                        request_metadata.request_id,
                        "Unexpected error during request processing",
                    )
```

#### Safety Checklist

- ✅ **Spans created → Spans closed AND tracking dict cleaned on all paths**
  - ✅ Streaming success → DEPARTED + span.end() + cleanup dict
  - ✅ Streaming GenerationError → ABORTED + span.end() + cleanup dict
  - ✅ Streaming exception → ABORTED + span.end() + cleanup dict
  - ✅ Non-streaming success → DEPARTED + span.end() + cleanup dict
  - ✅ Non-streaming CancelledError → ABORTED + span.end() + cleanup dict
  - ✅ Non-streaming ValueError → ABORTED + span.end() + cleanup dict
  - ✅ Non-streaming unexpected exception → ABORTED + span.end() + cleanup dict in finally
- ✅ **Tracking dict cleanup** in finally blocks (from PR #5)
- ✅ **No Pydantic serialization risk** (spans in separate dict, not model fields)
- ✅ Idempotent span.end() (checks is_recording())
- ✅ Tests prove all paths close span AND cleanup dict
- ✅ Legacy tracing untouched
- ✅ Defensive error handling

#### Termination Paths Covered

**Streaming Generator:**
1. Success → DEPARTED + span.end()
2. GenerationError → ABORTED + span.end()
3. Exception → ABORTED + span.end()

**Non-Streaming Generator:**
1. Success → DEPARTED + span.end()
2. CancelledError → ABORTED + span.end()
3. ValueError → ABORTED + span.end()
4. Unexpected exception → ABORTED + span.end() (outer finally)

#### Tests

1. **`test_api_span_created()`**
   - Verify span created when tracing enabled
   - Verify span has correct name and attributes

2. **`test_api_span_none_when_disabled()`**
   - Verify None when tracing disabled

3. **`test_span_closed_on_streaming_success()`**
   - Verify DEPARTED + end on success

4. **`test_span_closed_on_streaming_generation_error()`**
   - Verify ABORTED + end on GenerationError

5. **`test_span_closed_on_streaming_exception()`**
   - Verify ABORTED + end on exception

6. **`test_span_closed_on_full_success()`**
   - Verify DEPARTED + end on success

7. **`test_span_closed_on_full_cancelled()`**
   - Verify ABORTED + end on CancelledError

8. **`test_span_closed_on_full_exception()`**
   - Verify ABORTED + end in outer finally

9. **`test_initialization_order()`**
   - Verify engine init before API span creation
   - Verify global tracer provider set

**Size**: ~150 lines, 9 tests
**Review Time**: ~30 minutes

**CRITICAL**: This PR is larger because it includes full lifecycle. Cannot split into "create" and "close" PRs.

---

### PR #7: API↔Engine - Context Propagation

**Branch**: `journey-tracing-07-context-propagation`

**Goal**: Inject API span context into trace_headers for parent-child linkage.

**Why Safe**: No new resources created, just injects context into existing dict. Defensive error handling.

#### Changes

```python
# vllm/entrypoints/openai/chat_completion/serving.py

async def create_chat_completion(self, ...):
    # ... existing code up to trace_headers creation ...

    trace_headers = (
        None
        if raw_request is None
        else await self._get_trace_headers(raw_request.headers)
    )

    # NEW: Inject API span context into trace_headers for parent-child linkage
    if api_span:
        try:
            from opentelemetry import trace
            from opentelemetry.trace.propagation.tracecontext import (
                TraceContextTextMapPropagator,
            )

            # Set API span as current in context
            ctx = trace.set_span_in_context(api_span)

            # Inject into carrier
            carrier: dict[str, str] = {}
            propagator = TraceContextTextMapPropagator()
            propagator.inject(carrier, context=ctx)

            # Merge with existing trace_headers
            if trace_headers is None:
                trace_headers = carrier
            else:
                trace_headers = {**trace_headers, **carrier}
        except Exception as e:
            logger.debug("Failed to inject trace context: %s", e)

    # ... continue with engine.generate() call ...
```

#### Safety Checklist

- ✅ No new resources created
- ✅ Just injects context into existing trace_headers dict
- ✅ Defensive error handling (failures don't break request)
- ✅ Legacy tracing untouched
- ✅ Core span already extracts context (from PR #2)

#### Tests

1. **`test_context_injection()`**
   - Create API span
   - Verify traceparent header created
   - Verify traceparent format valid

2. **`test_context_extraction_in_scheduler()`**
   - Create request with injected context
   - Submit to scheduler
   - Verify scheduler extracts context correctly

3. **`test_parent_child_same_trace_id()`**
   - Create API span (parent)
   - Submit request to scheduler (creates child span)
   - Verify both spans share same trace_id

4. **`test_injection_failure_graceful()`**
   - Mock propagator.inject() to raise exception
   - Verify request still processes
   - Verify error logged but not raised

**Size**: ~25 lines, 4 tests
**Review Time**: ~15 minutes

---

### PR #8: API - Emit Additional Events

**Branch**: `journey-tracing-08-api-additional-events`

**Goal**: Add remaining API events. No new resources, just additive event emission.

**Why Safe**: No new resources created, span closure already handled (PR #6).

#### Changes

```python
# vllm/entrypoints/openai/chat_completion/serving.py

async def _create_api_span(self, ...):
    # ... existing span creation from PR #6 ...

    # NEW: Emit ARRIVED event immediately after creation
    try:
        from vllm.tracing import SpanAttributes

        api_span.add_event(
            name="api.ARRIVED",
            attributes={SpanAttributes.EVENT_TS_MONOTONIC: time.monotonic()},
            timestamp=time.time_ns(),
        )
    except Exception:
        pass  # Defensive

    return api_span

def _set_api_span_request_attributes(
    self,
    api_span: Any,
    model_name: str,
    prompt_token_ids: list[int],
    sampling_params: SamplingParams,
) -> None:
    """Set request metadata attributes on API span.

    Args:
        api_span: The OTEL span object
        model_name: Model name
        prompt_token_ids: Prompt tokens
        sampling_params: Sampling parameters
    """
    if not api_span or not api_span.is_recording():
        return

    try:
        from vllm.tracing import SpanAttributes

        api_span.set_attribute(SpanAttributes.GEN_AI_RESPONSE_MODEL, model_name)
        api_span.set_attribute(
            SpanAttributes.GEN_AI_USAGE_PROMPT_TOKENS, len(prompt_token_ids)
        )
        if sampling_params.temperature is not None:
            api_span.set_attribute(
                SpanAttributes.GEN_AI_REQUEST_TEMPERATURE,
                sampling_params.temperature
            )
        if sampling_params.top_p is not None:
            api_span.set_attribute(
                SpanAttributes.GEN_AI_REQUEST_TOP_P,
                sampling_params.top_p
            )
        if sampling_params.max_tokens is not None:
            api_span.set_attribute(
                SpanAttributes.GEN_AI_REQUEST_MAX_TOKENS,
                sampling_params.max_tokens
            )
        if sampling_params.n is not None:
            api_span.set_attribute(
                SpanAttributes.GEN_AI_REQUEST_N,
                sampling_params.n
            )
    except Exception:
        pass  # Defensive

async def create_chat_completion(self, ...):
    # ... after prompt processing ...

    # NEW: Set request attributes on span
    if api_span and isinstance(sampling_params, SamplingParams):
        self._set_api_span_request_attributes(
            api_span,
            model_name,
            engine_prompt["prompt_token_ids"],
            sampling_params,
        )

    # ... after trace_headers and context injection (PR #7) ...

    # ... generate() call ...

    # NEW: Emit HANDOFF_TO_CORE after submitting to engine
    if request_metadata.api_span:
        try:
            from vllm.tracing import SpanAttributes
            request_metadata.api_span.add_event(
                name="api.HANDOFF_TO_CORE",
                attributes={SpanAttributes.EVENT_TS_MONOTONIC: time.monotonic()},
                timestamp=time.time_ns(),
            )
        except Exception:
            pass

async def chat_completion_stream_generator(self, ...):
    # ... in main loop, on first iteration ...

    if first_iteration:
        # NEW: Track first response time
        if request_metadata.first_response_time is None:
            request_metadata.first_response_time = time.monotonic()

            # Emit FIRST_RESPONSE_FROM_CORE event
            if request_metadata.api_span:
                try:
                    from vllm.tracing import SpanAttributes
                    request_metadata.api_span.add_event(
                        name="api.FIRST_RESPONSE_FROM_CORE",
                        attributes={
                            SpanAttributes.EVENT_TS_MONOTONIC:
                                request_metadata.first_response_time
                        },
                        timestamp=time.time_ns(),
                    )
                except Exception:
                    pass

# Similar for chat_completion_full_generator
async def chat_completion_full_generator(self, ...):
    # ... in generator iteration loop ...

    async for res in result_generator:
        # NEW: Track first response time
        if request_metadata.first_response_time is None:
            request_metadata.first_response_time = time.monotonic()

            if request_metadata.api_span:
                try:
                    from vllm.tracing import SpanAttributes
                    request_metadata.api_span.add_event(
                        name="api.FIRST_RESPONSE_FROM_CORE",
                        attributes={
                            SpanAttributes.EVENT_TS_MONOTONIC:
                                request_metadata.first_response_time
                        },
                        timestamp=time.time_ns(),
                    )
                except Exception:
                    pass

        final_res = res
```

#### Safety Checklist

- ✅ No new resources created (just event emission)
- ✅ All events defensive (try/except)
- ✅ Span closure already handled (PR #6)
- ✅ Legacy tracing untouched

#### Tests

1. **`test_arrived_event_emitted()`**
   - Create API span
   - Verify ARRIVED event present

2. **`test_handoff_event_emitted()`**
   - Submit request to engine
   - Verify HANDOFF_TO_CORE event present

3. **`test_first_response_event_emitted()`**
   - Get first output from engine
   - Verify FIRST_RESPONSE_FROM_CORE event present

4. **`test_first_response_only_once()`**
   - Get multiple outputs
   - Verify FIRST_RESPONSE_FROM_CORE emitted only once

5. **`test_request_attributes_set()`**
   - Create span with sampling params
   - Verify all span attributes set correctly

**Size**: ~80 lines, 5 tests
**Review Time**: ~15 minutes

---

### PR #9: Cleanup - Remove Journey Event Buffering (Clean Break)

**Branch**: `journey-tracing-09-remove-buffering`

**Goal**: Remove journey event buffering and export now that OTEL spans are the sole tracing path.

**Why Safe**: Spans work end-to-end (PRs #2-8). Journey event buffering no longer needed for tracing.

**Clean Break Decision**: Journey tracing moves to OTEL spans exclusively. Prometheus metrics use direct timestamp capture.

**What Gets Removed**:
- ❌ Journey event buffer dictionaries in scheduler
- ❌ Journey event buffering logic in `_emit_journey_event()`
- ❌ Journey event flushing in `schedule()`
- ❌ Journey event export logic in `do_tracing()` (OTEL path only)

**What Stays**:
- ✅ `do_tracing()` method (for non-journey tracing, if any other uses exist)
- ✅ `RequestJourneyEvent` dataclass (kept in code, just not buffered/exported)
- ✅ Prometheus metrics (use direct timestamp capture in scheduler, not journey events)

**Key Change**: Replace journey event timestamp extraction with direct capture:

```python
# OLD (removed):
# Buffer journey events → Extract timestamps in metrics collector

# NEW (PR #9):
# Capture timestamps directly for Prometheus metrics
if self.log_stats:
    request.queued_ts = time.time()  # Direct capture
    request.scheduled_ts = time.time()  # When scheduled
```

#### Changes

```python
# vllm/v1/core/sched/scheduler.py

class Scheduler:
    def __init__(self, ...):
        # ... existing init ...

        if self._enable_journey_tracing:
            # REMOVED: Per-client event buffers
            # self._journey_events_buffer_by_client: dict[int, list[RequestJourneyEvent]] = defaultdict(list)

            # Keep: span tracking, dedup sets, hiwater marks (from PRs #2-3)
            self._first_token_emitted: set[str] = set()
            self._journey_prefill_hiwater: dict[str, int] = {}

def _emit_journey_event(self, ...):
    """Emit journey event to span only (buffering removed)."""
    if not self._enable_journey_tracing:
        return

    # Emit to span (from PR #4) - unchanged
    if span and span.is_recording() and SpanAttributes is not None:
        # ... span emission code (unchanged) ...

    # REMOVED: Legacy buffering code
    # No more:
    # - RequestJourneyEvent creation
    # - Buffer append
    # - Client index tracking

def schedule(self, ...):
    # ... existing scheduling logic ...

    # REMOVED: Event buffering flushing at end of schedule()
    # No more loop through _journey_events_buffer_by_client

# vllm/v1/engine/output_processor.py

class OutputProcessor:
    def process_outputs(
        self,
        engine_core_outputs: list[EngineCoreOutput],
        engine_core_timestamp: float,
        journey_events: list[RequestJourneyEvent] | None = None,  # Keep for API compat
    ) -> OutputProcessorOutput:
        """Process outputs from engine core.

        Args:
            engine_core_outputs: Outputs from engine core
            engine_core_timestamp: Timestamp when outputs produced
            journey_events: Deprecated, no longer used (kept for API compatibility)
        """
        # REMOVED: journey_events buffering and export logic
        # Journey tracing now uses OTEL spans exclusively (PRs #2-8)

        # KEEP: All existing output processing logic
        # KEEP: do_tracing() method (for other tracing, if any)
        # REMOVED from do_tracing(): Journey event export code

        request_outputs: list[RequestOutput | PoolingRequestOutput] = []
        reqs_to_abort: list[str] = []

        for engine_core_output in engine_core_outputs:
            # ... existing processing (unchanged) ...

        return OutputProcessorOutput(
            request_outputs=request_outputs,
            reqs_to_abort=reqs_to_abort,
        )

    def do_tracing(self, ...):
        """Export tracing data to OTEL.

        UPDATED: Journey event export removed.
        This method remains for other tracing purposes if needed.
        """
        # REMOVED: Journey event export logic
        # Journey events now emitted directly to spans in real-time

        # KEEP: Any other tracing export logic that may exist
        pass

# vllm/v1/core/sched/scheduler.py

# NEW: Direct timestamp capture for Prometheus metrics
def add_request(self, request: Request) -> None:
    # ... existing code ...

    # Capture timestamp directly for Prometheus (replaces journey event extraction)
    if self.log_stats:
        request.queued_ts = time.time()

def schedule(self, ...):
    # ... when scheduling request ...

    # Capture timestamp directly for Prometheus
    if self.log_stats and schedule_kind == ScheduleKind.FIRST:
        request.scheduled_ts = time.time()

# vllm/v1/core/sched/journey_events.py

# NO CHANGES to this file
# KEEP: RequestJourneyEvent dataclass (needed for Prometheus metrics extraction)
# KEEP: RequestJourneyEventType enum (used for span events)
# KEEP: ScheduleKind enum (still used)
# KEEP: _map_finish_status helper (still used)
```

#### Safety Checklist

- ✅ Spans work end-to-end (PRs #2-8) - OTEL tracing complete
- ✅ No functionality lost (spans are sole tracing path, buffering obsolete)
- ✅ **Clean break**: Journey event buffering and export removed completely
- ✅ **do_tracing() PRESERVED** (method stays, journey export logic removed)
- ✅ **RequestJourneyEvent PRESERVED** (dataclass kept in code, not buffered/exported)
- ✅ **Prometheus metrics still work** (use direct timestamp capture, not journey events)
- ✅ Tests verify spans still work after buffering removed
- ✅ Tests verify Prometheus metrics still work with direct capture
- ✅ Backward compatible (journey_events parameter kept for API compatibility)

#### Tests

1. **`test_no_buffering_when_tracing()`**
   - Enable journey tracing
   - Verify buffer dict doesn't exist
   - Verify no buffer-related code executed

2. **`test_spans_still_work()`**
   - Add and complete request
   - Verify span emission unchanged
   - Verify all events present on span

3. **`test_end_to_end_journey_tracing()`**
   - Create API span
   - Submit to engine (creates core span)
   - Complete request
   - Verify parent-child linkage
   - Verify all events on both spans

4. **`test_prometheus_metrics_with_direct_capture()`**
   - Add and complete request with log_stats=True
   - Verify `request.queued_ts` set directly (not from journey events)
   - Verify `request.scheduled_ts` set directly
   - Verify Prometheus metrics collector can access timestamps
   - Prove metrics work WITHOUT journey event buffering

**Size**: ~150 lines removed, ~100 lines added (direct timestamp capture), 4 tests
**Review Time**: ~20 minutes

---

## Resource Safety Checklist (For EVERY PR)

Add this to every PR description:

```markdown
### Resource Safety Checklist

- [ ] If this PR creates spans, it also ends them on all exits (success/error/cancel)
- [ ] If this PR introduces per-request state (dicts/sets), it also cleans them on all termination paths
- [ ] No buffering when tracer/exporter is absent
- [ ] Legacy tracing untouched
- [ ] Tests prove cleanup (no dict/set growth, spans ended)
- [ ] Defensive error handling (tracing never breaks requests)
- [ ] Zero overhead when disabled (early returns, no allocations)

### Termination Paths Covered

- [ ] Natural completion (stopped=True)
- [ ] Explicit abort (finish_requests)
- [ ] Exceptions during processing
- [ ] Client cancellation (if applicable)
- [ ] All paths call cleanup function
```

---


## Benefits of This Disciplined Approach

1. **Every PR is Safe**: No "fix it later" - resources created and cleaned in same PR
2. **Independent Merging**: Any PR can be merged without waiting for later PRs
3. **Easy Rollback**: Any PR can be reverted without breaking others
4. **Clear Verification**: Each PR has explicit checklist proving safety
5. **No Technical Debt**: No temporary hacks or incomplete implementations
6. **Reviewer Confidence**: Clear evidence that no leaks are introduced
7. **Better Testing**: Each piece tested thoroughly before building next
8. **Clear Progress**: 9 milestones with visible checkpoints
9. **Parallel Potential**: Some PRs (e.g., #5) can be developed in parallel with others

---

## Success Criteria (All PRs Combined)

### ✅ Functionality
- API spans created and emitted correctly
- Core spans created and emitted correctly
- Parent-child linkage works via trace context
- All events emitted to correct spans

### ✅ Initialization
- Engine initializes tracer before API server starts
- Global tracer provider set correctly
- API layer gets tracer from global provider
- No race conditions

### ✅ Performance
- Zero overhead when disabled
- Minimal overhead when enabled
- No measurable performance regression

### ✅ Memory
- No memory leaks in any path
- All spans properly closed
- All dicts properly cleaned

### ✅ Safety
- Graceful degradation when OTLP not configured
- Defensive error handling throughout
- Exception safety on all paths

### ✅ Quality
- All tests passing (45+ tests total)
- Legacy tracing removed cleanly (but preserved until ready)
- Code well-documented
- Clean, minimal diffs

---

## Final Notes

### For Implementer

When implementing each PR:
1. Read the detailed PR section above
2. Verify you understand all termination paths
3. Implement create + cleanup together
4. Write tests proving no leaks
5. Check the Resource Safety Checklist
6. Verify zero overhead when disabled

### For Reviewer

When reviewing each PR:
1. Verify Resource Safety Checklist completed
2. Check that all termination paths covered
3. Verify tests prove no leaks
4. Check defensive error handling
5. Verify legacy tracing untouched
6. Confirm PR is independently safe

---

## Implementation History

Condensed summary of completed PRs. See individual PR sections above for detailed implementation notes.

### ✅ PR #0: Remove EngineCoreEvent System
- **Completed**: 2026-01-25 (prior to journey tracing)
- **Commit**: 717f90eb5
- **Changes**: ~130 lines removed, 1 new test
- **Impact**: Removed legacy v0.0.1 metrics system, restored Prometheus metrics using RequestJourneyEvent timestamps

---

### ✅ PR #1: Scheduler Tracer Initialization
- **Completed**: 2026-01-26
- **Branch**: `pr1ofjourney` | **Commit**: 24f263656 | **PR**: #10
- **Changes**: +19 production, +110 test lines | 4 tests
- **Key**: Added tracer initialization to scheduler with defensive error handling, zero per-request state

---

### ✅ PR #2: Core Span Lifecycle Management
- **Completed**: 2026-01-26
- **Branch**: `pr2ofjourney` | **Commit**: d46cdf231 | **PR**: #33115
- **Changes**: +125 production, +245 test lines | 6 tests
- **Key**: Added `_core_spans` dict, span creation/cleanup on all termination paths with try/finally blocks

---

### ✅ PR #3: Journey State Cleanup
- **Completed**: 2026-01-26
- **Branch**: `pr3ofjourney` | **Commit**: f4cf7903c | **PR**: #11
- **Changes**: 26 production modified, +162 test lines | 4 tests
- **Key**: Extended cleanup to handle journey state, fixed memory leak, decoupled span vs state cleanup

---

### ✅ PR #4: Emit Journey Events to Core Spans
- **Completed**: 2026-01-26
- **Branch**: `pr4ofjourney` | **Commit**: 6a58608de | **PR**: #12
- **Changes**: +113 production, +328 test lines | 9 tests
- **Key**: Added event emission to spans (QUEUED, SCHEDULED, PREEMPTED, FIRST_TOKEN, FINISHED), defensive error handling, parallel buffering

---

### ✅ PR #5: Add API Span Tracking Dict
- **Completed**: 2026-01-27
- **Branch**: `pr5ofjourney` | **Commit**: 3d11f662d | **PR**: #13
- **Changes**: +67 production, +177 test lines | 8 tests
- **Key**: Added `_api_spans` dict to OpenAIServing, helper methods with error handling, avoids Pydantic serialization risks

---

**Summary**: 5 PRs completed, ~350 production lines added, ~1022 test lines added, 31 tests passing
