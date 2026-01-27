# PR #7: API↔Engine Context Propagation - Implementation Summary

## Overview

This PR implements W3C Trace Context propagation from API spans to core spans, enabling parent-child linkage in distributed traces. This completes the handshake between PR #6 (API span lifecycle) and PR #2 (core span lifecycle).

## Changes Made

### 1. Core Implementation

#### `vllm/tracing.py`
- **Added `inject_trace_context()` function** (lines 98-127)
  - Mirrors existing `extract_trace_context()` for symmetric API
  - Injects span context into carrier dict using W3C Trace Context propagation
  - Handles None span and None carrier gracefully
  - Defensive error handling (returns carrier on failure, no exceptions propagate)
  - Early return when OTEL unavailable (zero overhead)

#### `vllm/entrypoints/openai/chat_completion/serving.py`
- **Added context injection after API span creation** (lines 434-449)
  - Injection occurs immediately after span creation succeeds
  - Before `engine.generate()` call (critical ordering preserved)
  - Wrapped in try-except (defensive, logs DEBUG on failure)
  - Modified `trace_headers` flows to both beam_search and engine.generate paths

### 2. Test Coverage

#### `tests/entrypoints/openai/test_context_propagation.py` (new file, 407 lines)
All 12 tests pass, covering behavioral properties:

**Unit Tests (inject_trace_context helper):**
1. ✅ `test_inject_trace_context_with_span` - Basic injection with None carrier
2. ✅ `test_inject_trace_context_with_existing_carrier` - Injection preserves existing headers
3. ✅ `test_inject_trace_context_when_span_is_none` - Early return when span is None
4. ✅ `test_inject_trace_context_when_otel_unavailable` - Early return when OTEL unavailable
5. ✅ `test_inject_trace_context_graceful_failure` - Exception handling (returns carrier)

**Integration Tests (W3C format and behavior):**
6. ✅ `test_traceparent_header_presence_and_format` - **Property 2/G2**: W3C format validity
7. ✅ `test_injection_preserves_existing_headers` - Headers not removed during injection
8. ✅ `test_injection_only_when_span_exists` - **Property 5/G6**: Conditional injection
9. ✅ `test_tracing_disabled_no_injection` - **Property 4/I1**: Backward compatibility

**End-to-End Tests:**
10. ✅ `test_injection_called_with_api_span` - Integration point verification
11. ✅ `test_injection_failure_returns_carrier` - **Property 3/G4 & G5**: Graceful failure
12. ✅ `test_trace_id_preserved_through_chain` - **Property 1/G1 & G3**: Trace continuity

### 3. Regression Testing

- ✅ All 17 existing API span lifecycle tests pass (`test_api_span_lifecycle.py`)
- ✅ No breaking changes to span creation/finalization logic

## Behavioral Guarantees Verified

### Unit-Testable Guarantees
- **G1: Trace ID Continuity** ✅ - API and core spans share same trace_id
- **G2: W3C Trace Context Injection** ✅ - traceparent header present with valid format
- **G3: Trace Continuation** ✅ - trace_id preserved through Client→API→Core chain
- **G4: Graceful Degradation** ✅ - Request continues on injection failure
- **G5: No Exception Propagation** ✅ - Injection failures never break requests
- **G6: Conditional Injection** ✅ - Only when API span exists

### Invariants
- **I1: Backward Compatibility** ✅ - Early return when tracing disabled
- **I2: Zero Overhead When Disabled** ✅ - No allocations/propagator access when disabled
- **I3: No Resource Leaks** ✅ - Only modifies existing trace_headers dict (code review)

### Integration-Only (Not Unit-Testable)
- **G7: Parent Span ID Relationship** - Requires real OTLP exporter (noted in plan)

## Ordering Constraints Preserved

The critical ordering is maintained:
```
1. Create API span (PR #6) ✅
2. Inject API span context into trace_headers (THIS PR) ✅
3. Pass modified trace_headers to engine.generate() ✅
4. Scheduler extracts context from trace_headers (PR #2) ✅
5. Scheduler creates core span with parent context (PR #2) ✅
```

## Edge Cases & Fallback Behavior

1. **Injection failure**: Request continues, core span created as root span (no linkage)
2. **Span is None**: Early return, no injection attempted
3. **OTEL unavailable**: Early return, original headers preserved
4. **Tracing disabled**: Early return before any propagator access (zero overhead)
5. **Existing client traceparent**: API span created as child (PR #6), outgoing headers contain API span context (trace_id preserved)

## Safety Properties

### No Lifecycle Risk
- ✅ Zero new resources (only modifies existing `trace_headers` dict)
- ✅ No cleanup obligations (dict managed by request lifecycle)
- ✅ Stateless transformation (span context → headers)

### Defensive Error Handling
- ✅ All OTEL operations wrapped in try-except
- ✅ Failures logged at DEBUG level only
- ✅ Request processing never interrupted

### Performance
- ✅ Early return when tracing disabled (before propagator instantiation)
- ✅ No allocations when disabled
- ✅ Single injection point (no redundant operations)

## Files Modified

1. `vllm/tracing.py` - Added `inject_trace_context()` helper (~30 lines)
2. `vllm/entrypoints/openai/chat_completion/serving.py` - Added injection call (~16 lines)
3. `tests/entrypoints/openai/test_context_propagation.py` - New test file (~407 lines)

**Total**: ~453 lines added (including tests and documentation)

## Verification Checklist

- [x] All 12 new tests pass
- [x] All 17 existing API span tests pass (no regressions)
- [x] Trace ID continuity verified (G1)
- [x] W3C format validity verified (G2)
- [x] Trace continuation semantics verified (G3)
- [x] Graceful failure verified (G4, G5)
- [x] Conditional injection verified (G6)
- [x] Backward compatibility verified (I1)
- [x] Zero overhead confirmed (I2 - code review)
- [x] No resource leaks confirmed (I3 - code review)
- [x] Ordering constraints preserved
- [x] Defensive error handling in place

## Next Steps (Future PRs)

- **PR #8**: Add remaining API events (HANDOFF_TO_CORE, FIRST_RESPONSE_FROM_CORE)
- **PR #9**: Remove journey event buffering (now obsolete with OTEL spans)

## Notable Design Decisions

1. **Symmetric API**: `inject_trace_context()` mirrors `extract_trace_context()` for consistency
2. **In-place modification**: Carrier dict modified in place (not creating new dict on each injection)
3. **Graceful degradation**: Injection failures return carrier unchanged (core span becomes root)
4. **DEBUG logging only**: Injection failures don't warrant ERROR logs (tracing is best-effort)
5. **Single injection point**: Context injected once, right after span creation

## Testing Strategy

Tests focus on **behavioral properties** rather than implementation details:
- Unit tests verify helper function behavior (span None, OTEL unavailable, graceful failure)
- Integration tests verify W3C format, header preservation, conditional logic
- End-to-end tests verify trace_id continuity and trace continuation semantics
- No tests require real OTLP export (mocks used for parent_span_id relationship)

## Compliance with PR #7 Plan

✅ All scope constraints met:
- Context injection added (no new resources)
- W3C Trace Context propagation used
- Defensive error handling implemented
- Zero overhead when disabled
- All unit-testable properties verified

✅ All hard constraints met:
- Ordering preserved (inject between span creation and engine call)
- Trace continuation semantics correct (trace_id preserved)
- Graceful degradation on failure
- No exceptions propagate

✅ Testing requirements satisfied:
- 12 tests covering properties A-E from plan
- Behavioral assertions (not implementation details)
- No runtime tests for parent_span_id (integration-only)
