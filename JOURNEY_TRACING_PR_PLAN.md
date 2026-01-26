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

---

## PR Sequence Summary

| PR # | Branch | Goal | Size | Tests | Status |
|------|--------|------|------|-------|--------|
| #0 | `removelegacy` | Remove EngineCoreEvent | ~130 removed | 1 new + existing | ✅ **COMPLETED** |
| #1 | `pr1ofjourney` | Init tracer in scheduler | ~25 lines | 4 | ✅ **COMPLETED** |
| #2 | `journey-tracing-02-core-spans-lifecycle` | Create & cleanup core spans | ~100 lines | 6 | **Cleanup in same PR** ✅ |
| #3 | `journey-tracing-03-journey-state-cleanup` | Add journey state & cleanup | ~50 lines | 5 | Extends PR #2 cleanup |
| #4 | `journey-tracing-04-journey-events-emit` | Emit events to core spans | ~120 lines | 6 | No new resources, defensive |
| #5 | `journey-tracing-05-api-metadata` | Add API metadata fields | ~15 lines | 3 | No resources |
| #6 | `journey-tracing-06-api-spans-full-lifecycle` | Create & close API spans | ~150 lines | 9 | **All closure paths in same PR** ✅ |
| #7 | `journey-tracing-07-context-propagation` | Link parent-child spans | ~25 lines | 4 | No new resources |
| #8 | `journey-tracing-08-api-additional-events` | Emit API lifecycle events | ~80 lines | 5 | No new resources |
| #9 | `journey-tracing-09-remove-buffering` | Remove legacy buffering | ~150 removed | 4 | Keep do_tracing() |

**Total**: ~565 lines added, ~150 lines removed, 46 tests

---

## PR Dependencies

```
PR #0 (Remove EngineCoreEvent) ✅ COMPLETED
    ↓
PR #1 (Scheduler Tracer Init) ✅ COMPLETED
    ↓
PR #2 (Core Span + Cleanup) ← MUST include cleanup in same PR
    ↓
PR #3 (Journey State + Cleanup) ← extends PR #2 cleanup
    ↓
PR #4 (Core Event Emit) ← no new resources, safe
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

**Status**: ✅ **COMPLETED** - Ready to merge (pending final approval)

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

### PR #2: Engine - Core Span Create AND Close

**Branch**: `journey-tracing-02-core-spans-lifecycle`

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

        This PR only handles spans. PR #3 will extend this to clean
        other journey state (hiwater, dedup sets).

        Args:
            request: The request being terminated
        """
        if not self._enable_journey_tracing:
            return

        request_id = request.request_id

        # End and remove core span
        core_span = self._core_spans.pop(request_id, None)
        if core_span and core_span.is_recording():
            core_span.end(end_time=time.time_ns())

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

**Size**: ~100 lines, 6 tests
**Review Time**: ~25 minutes

---

### PR #3: Engine - Journey State WITH Cleanup

**Branch**: `journey-tracing-03-journey-state-cleanup`

**Goal**: Add journey progress tracking state, integrate cleanup into existing `_end_core_span_and_cleanup()`.

**Why Safe**: State created AND cleaned in same function as PR #2. All termination paths already covered.

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

    def _end_core_span_and_cleanup(self, request: Request) -> None:
        """End core span and cleanup all journey tracing state.

        CRITICAL: Extended from PR #2 to also clean journey state.

        Args:
            request: The request being terminated
        """
        if not self._enable_journey_tracing:
            return

        request_id = request.request_id

        # End and remove core span (from PR #2)
        core_span = self._core_spans.pop(request_id, None)
        if core_span and core_span.is_recording():
            core_span.end(end_time=time.time_ns())

        # NEW: Clean journey tracing state
        self._first_token_emitted.discard(request_id)
        self._journey_prefill_hiwater.pop(request_id, None)
```

#### Safety Checklist

- ✅ **State created → State cleaned in same function as PR #2**
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

3. **`test_journey_state_cleaned_on_completion()`**
   - Add request, populate state
   - Simulate natural completion
   - Verify state cleaned

4. **`test_no_state_leak()`**
   - Add and complete 100 requests
   - Verify `_first_token_emitted` set size = 0
   - Verify `_journey_prefill_hiwater` dict size = 0

5. **`test_progress_snapshot_correct()`**
   - Test prefill phase progress
   - Test decode phase progress
   - Test hiwater mark handling across preemption

**Size**: ~50 lines, 5 tests
**Review Time**: ~15 minutes

---

### PR #4: Engine - Emit Journey Events

**Branch**: `journey-tracing-04-journey-events-emit`

**Goal**: Emit events to core spans. No new resources, just additive event emission.

**Why Safe**: No new resources created, just emitting events to existing spans. Defensive error handling.

#### Changes

```python
# vllm/v1/core/sched/scheduler.py

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

**Size**: ~120 lines, 6 tests
**Review Time**: ~20 minutes

---

### PR #5: API - Add Request Metadata Fields

**Branch**: `journey-tracing-05-api-metadata`

**Goal**: Add fields to RequestResponseMetadata without creating any resources.

**Why Safe**: No per-request state introduced, no spans created, just field definitions.

#### Changes

```python
# vllm/entrypoints/openai/engine/protocol.py

from pydantic import BaseModel, ConfigDict

class RequestResponseMetadata(BaseModel):
    # Allow arbitrary types (OTEL Span) without validation
    model_config = ConfigDict(arbitrary_types_allowed=True)

    request_id: str
    final_usage_info: UsageInfo | None = None

    # NEW: API span tracking fields (no spans created yet)
    api_span: Any | None = None  # OTEL Span for API-level journey tracing
    # Timestamps for latency calculations (monotonic time)
    arrival_time: float | None = None  # time.monotonic() when span created
    first_response_time: float | None = None  # time.monotonic() when first output

# vllm/entrypoints/openai/engine/serving.py

class OpenAIServing:
    async def _get_is_tracing_enabled(self) -> bool:
        """Check if journey tracing is enabled.

        Caches result to avoid repeated async calls to engine.

        Returns:
            True if journey tracing is enabled, False otherwise
        """
        if not hasattr(self, '_cached_is_tracing_enabled'):
            self._cached_is_tracing_enabled = (
                await self.engine_client.is_tracing_enabled()
            )
        return self._cached_is_tracing_enabled
```

#### Safety Checklist

- ✅ No per-request state introduced
- ✅ No spans created
- ✅ No cleanup needed
- ✅ Legacy tracing untouched
- ✅ Just field definitions (pure metadata)

#### Tests

1. **`test_metadata_fields_exist()`**
   - Create RequestResponseMetadata instance
   - Verify `api_span`, `arrival_time`, `first_response_time` fields exist
   - Verify they accept None values

2. **`test_metadata_arbitrary_types_allowed()`**
   - Verify Pydantic config allows arbitrary types
   - Create metadata with mock Span object
   - Verify no validation errors

3. **`test_is_tracing_enabled_cached()`**
   - Call `_get_is_tracing_enabled()` multiple times
   - Verify engine_client.is_tracing_enabled() called only once
   - Verify cached value returned

**Size**: ~15 lines, 3 tests
**Review Time**: ~5 minutes

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
        api_span: Any,
        request_metadata: RequestResponseMetadata,
    ) -> None:
        """Emit api.DEPARTED event and end span.

        CRITICAL: Idempotent - safe to call even if span already ended.

        Args:
            api_span: The OTEL span object
            request_metadata: Request metadata with timing info
        """
        if not api_span or not api_span.is_recording():
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

    def _safe_emit_aborted_event(
        self,
        api_span: Any,
        error_message: str,
        reason: str | None = None,
    ) -> None:
        """Emit api.ABORTED event, set error status, and end span.

        CRITICAL: Idempotent - safe to call even if span already ended.

        Args:
            api_span: The OTEL span object
            error_message: Error message to record
            reason: Optional reason code (e.g., "client_disconnect")
        """
        if not api_span or not api_span.is_recording():
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

    async def create_chat_completion(
        self,
        request: ChatCompletionRequest,
        raw_request: Request | None,
    ) -> AsyncGenerator[str, None] | ChatCompletionResponse | ErrorResponse:
        # ... existing request_id creation ...
        request_id = (
            f"chatcmpl-{self._base_request_id(raw_request, request.request_id)}"
        )

        # NEW: Create API span
        api_span = None
        arrival_time = time.monotonic()
        is_tracing_enabled = await self._get_is_tracing_enabled()
        if is_tracing_enabled:
            api_span = await self._create_api_span(request_id, raw_request)

        # NEW: Initialize metadata with span
        request_metadata = RequestResponseMetadata(
            request_id=request_id,
            api_span=api_span,
            arrival_time=arrival_time,
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
            # NEW: Close span on generation error
            self._safe_emit_aborted_event(
                request_metadata.api_span,
                f"Generation error: {e}",
                "generation_error"
            )
            yield f"data: {self._convert_generation_error_to_streaming_response(e)}\n\n"
            yield "data: [DONE]\n\n"
            return  # Span closed, safe to return
        except Exception as e:
            # NEW: Close span on unexpected exception
            self._safe_emit_aborted_event(
                request_metadata.api_span, str(e), "exception"
            )
            logger.exception("Error in chat completion stream generator.")
            data = self.create_streaming_error_response(e)
            yield f"data: {data}\n\n"
            yield "data: [DONE]\n\n"
            return  # Span closed, safe to return

        # Send final done message
        yield "data: [DONE]\n\n"

        # NEW: Close span on success
        self._safe_emit_departed_event(request_metadata.api_span, request_metadata)

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
                    request_metadata.api_span,
                    "Client disconnected",
                    "client_disconnect"
                )
                span_closed = True
                return self.create_error_response("Client disconnected")
            except ValueError as e:
                self._safe_emit_aborted_event(
                    request_metadata.api_span, str(e), "validation_error"
                )
                span_closed = True
                return self.create_error_response(e)

            # Response building
            assert final_res is not None

            # ... existing response building code ...

            # NEW: Close span on success
            self._safe_emit_departed_event(request_metadata.api_span, request_metadata)
            span_closed = True

            return response
        finally:
            # CRITICAL: Outer finally catches ALL other exceptions
            if not span_closed and request_metadata.api_span:
                self._safe_emit_aborted_event(
                    request_metadata.api_span,
                    "Unexpected error during request processing",
                )
```

#### Safety Checklist

- ✅ **Spans created → Spans closed on all paths**
  - ✅ Streaming success → DEPARTED + span.end()
  - ✅ Streaming GenerationError → ABORTED + span.end()
  - ✅ Streaming exception → ABORTED + span.end()
  - ✅ Non-streaming success → DEPARTED + span.end()
  - ✅ Non-streaming CancelledError → ABORTED + span.end()
  - ✅ Non-streaming ValueError → ABORTED + span.end()
  - ✅ Non-streaming unexpected exception → ABORTED + span.end() in finally
- ✅ Idempotent span.end() (checks is_recording())
- ✅ Tests prove all paths close span
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

### PR #9: Cleanup - Remove Journey Buffering

**Branch**: `journey-tracing-09-remove-buffering`

**Goal**: Remove journey events buffering logic now that spans work end-to-end. Keep do_tracing() and RequestJourneyEvent dataclass.

**Why Safe**: Spans work end-to-end (PRs #2-8). Buffering no longer needed. do_tracing() and Prometheus metrics remain functional.

**IMPORTANT**: This PR removes buffering LOGIC only, not the underlying data structures:
- ✅ KEEP: `OutputProcessor.do_tracing()` (current OTEL export mechanism)
- ✅ KEEP: `RequestJourneyEvent` dataclass (needed for Prometheus timestamp extraction)
- ❌ REMOVE: Buffer dictionaries and buffering logic in scheduler
- ❌ REMOVE: Buffer flushing code in `schedule()`

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
        journey_events: list[RequestJourneyEvent] | None = None,  # Keep parameter for compatibility
    ) -> OutputProcessorOutput:
        """Process outputs from engine core.

        Args:
            engine_core_outputs: Outputs from engine core
            engine_core_timestamp: Timestamp when outputs produced
            journey_events: No longer buffered (kept for API compatibility)
        """
        # REMOVED: journey_events buffering in req_state.journey_events
        # Note: Journey events now emitted directly to spans in scheduler (PR #4)

        # KEEP: All existing output processing logic
        # KEEP: do_tracing() method (current OTEL export mechanism)
        # KEEP: Timestamp extraction from journey events (for Prometheus metrics)

        request_outputs: list[RequestOutput | PoolingRequestOutput] = []
        reqs_to_abort: list[str] = []

        for engine_core_output in engine_core_outputs:
            # ... existing processing (unchanged) ...

        return OutputProcessorOutput(
            request_outputs=request_outputs,
            reqs_to_abort=reqs_to_abort,
        )

# vllm/v1/core/sched/journey_events.py

# NO CHANGES to this file
# KEEP: RequestJourneyEvent dataclass (needed for Prometheus metrics extraction)
# KEEP: RequestJourneyEventType enum (used for span events)
# KEEP: ScheduleKind enum (still used)
# KEEP: _map_finish_status helper (still used)
```

#### Safety Checklist

- ✅ Spans work end-to-end (PRs #2-8)
- ✅ No functionality lost (spans replace buffering for OTEL export)
- ✅ **do_tracing() PRESERVED** (current OTEL export mechanism, NOT being removed)
- ✅ **RequestJourneyEvent PRESERVED** (needed for Prometheus metrics)
- ✅ **Prometheus metrics still work** (timestamp extraction from journey events preserved)
- ✅ Tests verify spans still work
- ✅ Backward compatible (journey_events parameter kept for compatibility)

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

4. **`test_current_tracing_preserved()`**
   - Verify do_tracing() method still exists
   - Verify do_tracing() still called (current OTEL export mechanism)
   - Verify Prometheus metrics still work (timestamp extraction functional)

**Size**: ~150 lines removed, 4 tests
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

## Timeline Estimate

| Phase | PRs | Time | Cumulative |
|-------|-----|------|------------|
| **Phase 0: Prerequisites** | #0 | ✅ DONE | 0 days |
| **Phase 1: Core** | #1-4 | 1 week | 1 week |
| **Phase 2: API** | #5-8 | 1 week | 2 weeks |
| **Phase 3: Cleanup** | #9 | 1 day | ~2 weeks |

**Breakdown**:
- PR #0: ✅ COMPLETED (Remove EngineCoreEvent)
- PR #1: ✅ COMPLETED (Scheduler tracer init - 0.5 day actual)
- PR #2: 2 days (critical, includes cleanup)
- PR #3: 1 day (extends cleanup)
- PR #4: 1.5 days (event emission)
- PR #5: 0.5 day (tiny, just fields)
- PR #6: 2 days (critical, includes all closure)
- PR #7: 1 day (context propagation)
- PR #8: 1 day (additional events)
- PR #9: 1 day (removal)

**Total**: ~2 weeks for complete implementation

---

## Review Time Estimates

| PR | Review Time | Reason |
|----|-------------|--------|
| #1 | 10 min | Tiny, just tracer init |
| #2 | 25 min | Critical, includes cleanup |
| #3 | 15 min | Extends cleanup |
| #4 | 20 min | Event emission |
| #5 | 5 min | Trivial, just fields |
| #6 | 30 min | Critical, all closure paths |
| #7 | 15 min | Context propagation |
| #8 | 15 min | Additional events |
| #9 | 20 min | Code removal |

**Total**: ~2.5 hours spread across 9 PRs
**Compare to**: Many hours for single 800-line PR

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

## Rollback Strategy

Each PR can be independently reverted without breaking others:

| Revert | Impact | System State |
|--------|--------|--------------|
| PR #1 | Tracer removed | No tracing, legacy works |
| PR #2 | Core spans removed | No core spans, legacy works |
| PR #3 | Journey state removed | Simplified core spans, legacy works |
| PR #4 | Event emission removed | Silent core spans, legacy works |
| PR #5 | Metadata fields removed | API spans can't be stored |
| PR #6 | API spans removed | No API spans, legacy works |
| PR #7 | Context propagation removed | Parent-child link broken but both work |
| PR #8 | Additional events removed | Fewer events but lifecycle complete |
| PR #9 | Legacy buffering restored | Both buffering and spans active |

**All rollbacks maintain system stability and correctness.**

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

## Key Differences from Initial Plan

| Initial Plan (Bad) | This Plan (Good) |
|-------------------|------------------|
| PR creates spans, separate PR cleans them | Same PR creates + cleans |
| PR creates API spans, later PR adds ABORTED | Same PR creates + adds DEPARTED/ABORTED |
| 11 PRs with dependencies | 9 PRs, each self-consistent |
| "We'll fix leaks later" | No leaks possible |
| Reviewer must trust future PRs | Reviewer can verify each PR independently |
| Technical debt accumulation | Zero technical debt |

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

### For Project Manager

Progress tracking:
- **After PR #0**: ✅ EngineCoreEvent removed, Prometheus metrics restored
- **After PR #1**: ✅ Tracer initialized (ready for span creation)
- **After PR #2**: Core spans working (create + cleanup)
- **After PR #3**: Journey state tracking working
- **After PR #4**: Core events flowing to spans
- **After PR #5**: API metadata ready
- **After PR #6**: API spans working (full lifecycle)
- **After PR #7**: Parent-child linkage working
- **After PR #8**: All API events flowing
- **After PR #9**: Journey event buffering removed, dual-stream feature complete
  - Note: do_tracing() and RequestJourneyEvent preserved (not removed)

Each milestone is independently safe and valuable.

---

## Implementation History

### ✅ PR #0: Remove EngineCoreEvent System

**Completed**: Prior to journey tracing implementation
**Branch**: `removelegacy`
**Commit**: 717f90eb5
**Changes**: ~130 lines removed, 1 new test + existing tests passing
**Impact**: Cleaned up legacy v0.0.1 metrics system, restored Prometheus metrics using RequestJourneyEvent timestamps

### ✅ PR #1: Scheduler Tracer Initialization

**Completed**: 2026-01-26
**Branch**: `pr1ofjourney`
**Status**: Ready to merge (pending final approval)

**Implementation Summary**:
- Added defensive `SpanAttributes` import with None fallback
- Added tracer initialization in `Scheduler.__init__()` with try/except
- Added `otlp_traces_endpoint` parameter to test utilities
- Implemented 4 comprehensive tests with proper mocking

**Changes**:
- Production code: 19 lines added
  - `vllm/v1/core/sched/scheduler.py`: 6 lines (import) + 13 lines (init)
- Test utilities: 2 lines modified
  - `tests/v1/core/utils.py`: Added parameter
- Test code: 110 lines added
  - `tests/v1/core/test_scheduler.py`: 4 new tests

**Test Results**: ✅ All 85 tests passing (81 existing + 4 new)

**Code Review**: Approved with fix applied
- Issue identified: Test 3 initially called real `init_tracer()`
- Fix applied: Added mock decorator for deterministic testing
- All tests now properly mocked and isolated

**Key Achievements**:
- ✅ Zero per-request state introduced
- ✅ Zero overhead when disabled
- ✅ Defensive error handling with warning logs
- ✅ Backward compatible (all parameters optional)
- ✅ No regressions in existing tests
- ✅ Foundation ready for PR #2 (core span creation)

**Next Steps**: PR #2 will use `self.tracer` to create core spans with complete cleanup in same PR
