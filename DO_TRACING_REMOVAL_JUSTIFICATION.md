# Justification for Disabling OutputProcessor.do_tracing()

## Executive Summary

Disabling `OutputProcessor.do_tracing()` was **necessary and safe** because:
1. It was creating a **duplicate** `llm_request` span under the wrong scope
2. All 7 attributes it set are **preserved** on the correct span (vllm.api)
3. Journey events were **never** emitted by `do_tracing()` (scheduler does this)
4. The change aligns with **documented behavior** (JOURNEY_TRACING.md)
5. Comprehensive tests verify no regression

## What do_tracing() Was Doing (Before Removal)

Location: `vllm/v1/engine/output_processor.py:622-669`

```python
def do_tracing(self, engine_core_output, req_state, iteration_stats):
    # Created a NEW span with:
    tracer = self.tracer  # vllm.llm_engine scope
    with tracer.start_as_current_span(
        "llm_request",  # ❌ DUPLICATE name (same as API span!)
        kind=SpanKind.SERVER,
        context=trace_context,
        start_time=arrival_time_nano_seconds,  # ❌ Wrong timing (after completion)
    ) as span:
        # Set 7 attributes:
        span.set_attribute(GEN_AI_USAGE_PROMPT_TOKENS, ...)
        span.set_attribute(GEN_AI_USAGE_COMPLETION_TOKENS, ...)
        span.set_attribute(GEN_AI_REQUEST_ID, ...)
        span.set_attribute(GEN_AI_REQUEST_TOP_P, ...)
        span.set_attribute(GEN_AI_REQUEST_MAX_TOKENS, ...)
        span.set_attribute(GEN_AI_REQUEST_TEMPERATURE, ...)
        span.set_attribute(GEN_AI_REQUEST_N, ...)
```

### The Problem

1. **Duplicate Span**: Created a third `llm_request` span (in addition to vllm.api and vllm.scheduler)
2. **Wrong Scope**: Used `vllm.llm_engine` scope (not documented, conflicts with providers)
3. **Wrong Timing**: Created AFTER request completion (start_time set to arrival, but span created at finish)
4. **Undocumented**: Not mentioned in JOURNEY_TRACING.md (only vllm.api and vllm.scheduler expected)

## Complete Attribute Mapping (All 7 Preserved)

| Attribute | Old Location (do_tracing) | New Location | File:Line |
|-----------|---------------------------|--------------|-----------|
| `GEN_AI_USAGE_PROMPT_TOKENS` | vllm.llm_engine span | vllm.api span | chat_completion/serving.py:713 |
| `GEN_AI_USAGE_COMPLETION_TOKENS` | vllm.llm_engine span | vllm.api span | chat_completion/serving.py:1571 + 2024 |
| `GEN_AI_REQUEST_ID` | vllm.llm_engine span | vllm.api span | engine/serving.py:464 |
| `GEN_AI_REQUEST_TOP_P` | vllm.llm_engine span | vllm.api span | chat_completion/serving.py:718 |
| `GEN_AI_REQUEST_MAX_TOKENS` | vllm.llm_engine span | vllm.api span | chat_completion/serving.py:721 |
| `GEN_AI_REQUEST_TEMPERATURE` | vllm.llm_engine span | vllm.api span | chat_completion/serving.py:717 |
| `GEN_AI_REQUEST_N` | vllm.llm_engine span | vllm.api span | chat_completion/serving.py:723 |

**Result**: All 7 attributes now set on the **correct** span (vllm.api) instead of a duplicate.

## Journey Events Were NEVER From do_tracing()

The comment in `do_tracing()` itself says:

```python
# Note: Journey events are now emitted directly to OTEL core spans in the scheduler (PR #9).
# This method only handles other request attributes for the API-level span.
```

Journey events (QUEUED, SCHEDULED, FIRST_TOKEN, FINISHED) are emitted by:
- **Location**: `vllm/v1/core/sched/scheduler.py`
- **Span**: `llm_core` (vllm.scheduler scope)
- **Method**: `_emit_journey_event()` called throughout scheduling lifecycle

`do_tracing()` was **never** responsible for journey events.

## Expected Span Structure (Per Documentation)

From `JOURNEY_TRACING.md`:

```
You'll see a timeline with two spans:
- **llm_request** (API layer) - parent span
- **llm_core** (Engine layer) - child span
```

Two services expected:
1. **vllm.api** - Parent span with API events
2. **vllm.scheduler** - Child span with journey events

**NOT documented**: vllm.llm_engine (this was the bug!)

## Why vllm.llm_engine Scope Was Wrong

1. **Initialization Conflict**: Each `init_tracer()` call was overwriting the global TracerProvider
   - API process: `init_tracer("vllm.api", endpoint)` → Sets provider
   - Engine process: `init_tracer("vllm.llm_engine", endpoint)` → **Overwrites provider**
   - Scheduler: `init_tracer("vllm.scheduler", endpoint)` → **Overwrites again**

2. **Result**: Only the LAST initialized tracer would export properly

3. **Fix**: Singleton pattern ensures ONE provider shared by all scopes

## Test Coverage

### Existing Tests (test_tracing_fixes.py)
- ✅ `test_no_llm_engine_in_async_llm`: Verifies vllm.llm_engine removed from async_llm.py
- ✅ `test_no_llm_engine_in_llm_engine`: Verifies vllm.llm_engine removed from llm_engine.py
- ✅ `test_output_processor_tracing_disabled`: Verifies do_tracing() call commented out

### New Tests (test_completion_tokens_attribute.py)
- ✅ `test_completion_tokens_set_on_full_generator`: Verifies attr set in non-streaming
- ✅ `test_completion_tokens_set_on_stream_generator`: Verifies attr set in streaming
- ✅ `test_all_do_tracing_attributes_accounted_for`: Maps all 7 attributes to new locations
- ✅ `test_only_two_services_expected`: Documents expected structure
- ✅ `test_journey_events_emitted_by_scheduler`: Confirms events NOT from do_tracing()

**Total**: 18 tests covering the removal (12 original + 6 new)

## Backward Compatibility

### What Changed
- vllm.llm_engine scope removed (was undocumented, causing bugs)
- OutputProcessor.do_tracing() disabled (was creating duplicate spans)

### What Stayed The Same
- All 7 OTEL attributes still exported (on correct span)
- Journey events still emitted (by scheduler, as before)
- Two-span structure maintained (vllm.api parent + vllm.scheduler child)
- Trace context propagation works (traceparent headers)
- API events still emitted (ARRIVED, HANDOFF_TO_CORE, FIRST_RESPONSE, DEPARTED)

### For Existing Users
If you were:
1. **Viewing traces in Jaeger**: You'll now see TWO services (vllm.api, vllm.scheduler) instead of three
2. **Querying OTEL attributes**: All attributes still present on vllm.api span
3. **Using journey events**: No change (always came from scheduler)
4. **Building dashboards**: May need to update service filter from "vllm.llm_engine" to "vllm.api"

## Evidence This Was The Right Fix

### From Git History
- PR #9 (commit 1d9b9f37c): Removed journey event buffering, but LEFT do_tracing() call
- Comment in do_tracing() added in PR #9: "This method only handles other request attributes"
- The method was already recognized as only handling attributes, not events

### From Documentation
- JOURNEY_TRACING.md: Only mentions vllm.api and vllm.scheduler
- No mention of vllm.llm_engine anywhere in docs
- Expected structure: TWO spans, not three

### From Code Inspection
- All 7 attributes mapped to new locations
- Journey events confirmed to come from scheduler
- TracerProvider singleton prevents overwrites

## Conclusion

**Disabling do_tracing() was necessary because**:
- It created a duplicate span under the wrong scope
- It caused TracerProvider conflicts
- It was undocumented and not expected by users

**No regression because**:
- ✅ All 7 attributes preserved on correct span
- ✅ Journey events still emitted (never from do_tracing)
- ✅ Two-span structure maintained as documented
- ✅ 18 tests verify correct behavior

**The change actually FIXES bugs**:
- Eliminates duplicate spans
- Fixes provider overwriting
- Aligns code with documentation
- Improves trace clarity (2 services instead of 3)
