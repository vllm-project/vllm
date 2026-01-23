# Request Journey Tracing in vLLM

**A comprehensive observability feature for tracking request lifecycles through the v1 scheduler**

---

## Table of Contents

- [What is Request Journey Tracing?](#what-is-request-journey-tracing)
- [Why Use Journey Tracing?](#why-use-journey-tracing)
- [Quick Start](#quick-start)
- [Semantics & Guarantees](#semantics--guarantees)
- [Event Types](#event-types)
- [Event Data Structure](#event-data-structure)
- [Common Use Cases](#common-use-cases)
- [Understanding Progress Tracking](#understanding-progress-tracking)
- [Performance Considerations](#performance-considerations)
- [Architecture Overview](#architecture-overview)
- [Troubleshooting](#troubleshooting)
- [FAQ](#faq)

---

## What is Request Journey Tracing?

Request journey tracing is a lightweight observability feature that tracks the lifecycle of every request as it moves through the vLLM v1 scheduler. It emits **sparse lifecycle events** at key state transitions, providing detailed visibility into:

- When requests enter the scheduler (added to waiting queue)
- When requests get scheduled for execution
- When the first token is generated (important latency metric)
- When requests get preempted (resource contention)
- When requests complete within the scheduler (terminal state)

**Scope**: Events track the request lifecycle **within the v1 scheduler**, from `add_request()` to terminal status. This does not include system-level ingress (API arrival) or egress (response departure).

Each event includes a **complete progress snapshot** with accurate token counts that survive preemption, making it perfect for debugging, monitoring, and performance analysis.

---

## Why Use Journey Tracing?

### 🔍 **Debugging**
- Trace the exact path a request takes through the scheduler
- Identify why requests are slow or stuck
- Debug preemption behavior and resource contention

### 📊 **Performance Analysis**
- Measure time-to-first-token (TTFT) accurately
- Track prefill vs decode phase duration
- Understand scheduling patterns and bottlenecks

### 🎯 **Monitoring & Alerting**
- Integrate events with your existing telemetry stack
- Set up alerts for abnormal patterns (excessive preemptions, long queuing)
- Track SLO compliance for request latencies

### 🔬 **Research & Optimization**
- Analyze scheduler behavior under different workloads
- Correlate journey events with model metrics
- Validate optimization hypotheses

---

## Quick Start

**Note**: Journey tracing is a **v1 engine-core feature**. It is configured at the scheduler/engine level, not via the high-level `LLM` API.

### Enable Journey Tracing via CLI (Recommended)

The easiest way to enable journey tracing is using the `--enable-journey-tracing` flag with `vllm serve`:

```bash
# Start vLLM server with journey tracing enabled
vllm serve meta-llama/Llama-3.2-1B-Instruct --enable-journey-tracing

# Enable journey tracing WITH OTEL export (recommended for production)
vllm serve meta-llama/Llama-3.2-1B-Instruct \
    --enable-journey-tracing \
    --otlp-traces-endpoint http://localhost:4317

# Combine with other observability flags
vllm serve meta-llama/Llama-3.2-1B-Instruct \
    --enable-journey-tracing \
    --otlp-traces-endpoint http://localhost:4317 \
    --enable-mfu-metrics \
    --enable-logging-iteration-details
```

**Note**: Journey tracing is **disabled by default**. Simply omit the `--enable-journey-tracing` flag to keep it off.

**🎯 OTEL Integration**: When `--enable-journey-tracing` is enabled and OTEL tracing is configured (e.g., via `--otlp-traces-endpoint` or other trace exporters), journey events are **automatically exported** as OTEL span events. This means you can view journey lifecycle events (QUEUED, SCHEDULED, FIRST_TOKEN, PREEMPTED, FINISHED) in any OTEL-compatible backend (Jaeger, Tempo, Zipkin, etc.) without writing custom export code.

Journey events are attached to request spans when:
- Journey tracing is enabled (`--enable-journey-tracing`)
- OTEL tracing is active (spans are being recorded)
- This typically requires configuring an OTEL exporter, with `--otlp-traces-endpoint` being the most common method

**Without OTEL**: If you only enable `--enable-journey-tracing` without configuring trace export, events are collected internally and available via `EngineCoreOutputs.journey_events` for custom integration (see [Custom Integration](#custom-integration) below).

### Enable Journey Tracing Programmatically

For advanced use cases or custom engine integrations, you can enable journey tracing programmatically:

```python
from vllm.config import ObservabilityConfig, VllmConfig

# Create VllmConfig with journey tracing enabled
observability_config = ObservabilityConfig(
    enable_journey_tracing=True
)

vllm_config = VllmConfig(
    # ... model_config, scheduler_config, etc.
    observability_config=observability_config
)

# Pass to Scheduler or EngineCore
scheduler = Scheduler(
    vllm_config=vllm_config,
    # ... other params
)
```

## OTEL Integration (Automatic Export)

When you enable both journey tracing and OTEL tracing, journey events are automatically exported as span events within your request spans.

### Setup

```bash
# Start vLLM with OTEL endpoint
vllm serve meta-llama/Llama-3.2-1B-Instruct \
    --enable-journey-tracing \
    --otlp-traces-endpoint http://localhost:4317
```

### What Gets Exported

Each journey event becomes an OTEL span event with:

- **Event name**: `journey.QUEUED`, `journey.SCHEDULED`, `journey.FIRST_TOKEN`, `journey.PREEMPTED`, `journey.FINISHED`
- **Timestamp**: Export time (OTEL SDK timestamp); original monotonic timestamp available as attribute `ts.monotonic`
- **Attributes**:
  - `event.type`: Event type name (QUEUED, SCHEDULED, etc.)
  - `ts.monotonic`: Original monotonic timestamp from when the event occurred
  - `scheduler.step`: Scheduler iteration counter (omitted if None)
  - `phase`: PREFILL or DECODE
  - `prefill.done_tokens`: Tokens processed in prefill
  - `prefill.total_tokens`: Total prefill tokens
  - `decode.done_tokens`: Output tokens generated
  - `decode.max_tokens`: Maximum decode tokens
  - `num_preemptions`: Preemption count
  - `schedule.kind`: FIRST or RESUME (SCHEDULED events only)
  - `finish.status`: stopped/length/aborted/error (FINISHED events only)

### Viewing in Jaeger/Tempo

When you view a request trace in Jaeger or Tempo, you'll see:

1. **Main span**: `llm_request` with aggregate metrics (TTFT, latency, token counts)
2. **Span events**: Timeline of journey events showing the request's lifecycle through the scheduler

This provides both high-level metrics AND detailed lifecycle visibility in a single trace.

### Custom Integration (Advanced)

**Note**: If you're using OTEL (see above), you don't need custom integration - events are automatically exported. This section is for advanced use cases where you want to process events directly.

```python
# In your engine/scheduler integration code (v1 only)
engine_outputs = scheduler.update_from_output(scheduler_output, model_output)

for client_idx, eco in engine_outputs.items():
    if eco.journey_events:
        for event in eco.journey_events:
            print(f"[{event.event_type.name}] Request {event.request_id}")
            print(f"  Scheduler Step: {event.scheduler_step}")
            print(f"  Progress: {event.prefill_done_tokens}/{event.prefill_total_tokens} prefill, "
                  f"{event.decode_done_tokens}/{event.decode_max_tokens} decode")
            print(f"  Phase: {event.phase}, Preemptions: {event.num_preemptions_so_far}")
```

### Simple Latency Tracking

```python
from collections import defaultdict
import time

# Track journey events per request
journey_tracker = defaultdict(list)

def process_events(engine_outputs):
    for client_idx, eco in engine_outputs.items():
        if eco.journey_events:
            for event in eco.journey_events:
                journey_tracker[event.request_id].append(event)

# Calculate TTFT (Time To First Token)
# Note: This measures scheduler-QUEUED → first decode token
# (includes queueing + prefill, but not API ingress time)
def calculate_ttft(request_id):
    events = journey_tracker[request_id]
    queued = next((e for e in events if e.event_type == RequestJourneyEventType.QUEUED), None)
    first_token = next((e for e in events if e.event_type == RequestJourneyEventType.FIRST_TOKEN), None)

    if queued and first_token:
        ttft = first_token.ts_monotonic - queued.ts_monotonic
        print(f"TTFT for {request_id}: {ttft:.3f}s")
```

---

## Semantics & Guarantees

Understanding the scope and guarantees of journey tracing:

### Scope
- **Events track lifecycle within the v1 scheduler**, not end-to-end system flow
- **Start point**: `scheduler.add_request()` (QUEUED event)
- **End point**: Terminal scheduler status (FINISHED event)
- **Not included**: API server arrival time, response departure from system

### Timestamps
- `ts_monotonic` uses `time.monotonic()` and is **process-local**
- Use for relative timing within the same process
- Not suitable for distributed tracing across processes without correlation

### Event Delivery
- Events are **buffered per-client** during scheduler iteration
- Events are **flushed via `update_from_output()`** into `EngineCoreOutputs.journey_events`
- Delivery is **best-effort within the scheduler loop**
- External aborts may have `scheduler_step=None` (called outside schedule context)

### Guarantees
- **FIRST_TOKEN** is emitted at the token append site and **exactly once per request**
- **Progress tracking survives preemption** via scheduler-side high-water mark
- **scheduler_step** is monotonically increasing (when not None)
- Events for a single request may be delivered across multiple `EngineCoreOutputs` batches

### Reserved Event Types
- **DEPARTED** (event type 6): Reserved for future use, not currently implemented

---

## Event Types

Journey tracing defines 6 event types; **5 are currently emitted** (DEPARTED reserved). Typical sequences:

- **Without preemption**: QUEUED → SCHEDULED(FIRST) → FIRST_TOKEN → FINISHED
- **With preemption**: QUEUED → SCHEDULED(FIRST) → PREEMPTED → SCHEDULED(RESUME) → ... → FIRST_TOKEN → FINISHED

Events are buffered and flushed in batches via `update_from_output()`, so delivery may not be strictly sequential.

### 1. QUEUED 🔵
**When**: Request is added to the scheduler's waiting queue

**Meaning**: Request has been accepted and is waiting to be scheduled for execution

**Key Fields**:
- `scheduler_step`: `None` (not yet scheduled)
- `phase`: Always `"PREFILL"`
- `prefill_done_tokens`: `0`
- `decode_done_tokens`: `0`

**Example**:
```python
RequestJourneyEvent(
    request_id="req-123",
    event_type=RequestJourneyEventType.QUEUED,
    scheduler_step=None,  # Before first schedule
    prefill_done_tokens=0,
    prefill_total_tokens=100,
    phase="PREFILL"
)
```

---

### 2. SCHEDULED 🟢
**When**: Request transitions from WAITING/PREEMPTED → RUNNING

**Meaning**: Scheduler has allocated resources (KV cache blocks) and the request will be processed in the current iteration

**Key Fields**:
- `schedule_kind`:
  - `ScheduleKind.FIRST` - First time scheduled (WAITING → RUNNING)
  - `ScheduleKind.RESUME` - Resumed after preemption (PREEMPTED → RUNNING)
- `scheduler_step`: Current scheduler iteration counter
- `prefill_done_tokens`: Accurate count (survives preemption via high-water mark)

**Example (First Schedule)**:
```python
RequestJourneyEvent(
    request_id="req-123",
    event_type=RequestJourneyEventType.SCHEDULED,
    schedule_kind=ScheduleKind.FIRST,
    scheduler_step=42,
    prefill_done_tokens=0,  # Just starting
    prefill_total_tokens=100,
    phase="PREFILL"
)
```

**Example (Resume After Preemption)**:
```python
RequestJourneyEvent(
    request_id="req-123",
    event_type=RequestJourneyEventType.SCHEDULED,
    schedule_kind=ScheduleKind.RESUME,
    scheduler_step=58,
    prefill_done_tokens=50,  # Preserved from before preemption!
    prefill_total_tokens=100,
    num_preemptions_so_far=1,
    phase="PREFILL"
)
```

---

### 3. FIRST_TOKEN 🎯
**When**: First decode token is generated (output token count transitions from 0 → 1+)

**Meaning**: Prefill phase is complete, decode phase has started. **Critical latency milestone.**

**Key Fields**:
- `phase`: Always `"DECODE"`
- `prefill_done_tokens`: Equal to `prefill_total_tokens` (100% complete)
- `decode_done_tokens`: `>= 1` (typically 1, but could be more)

**Example**:
```python
RequestJourneyEvent(
    request_id="req-123",
    event_type=RequestJourneyEventType.FIRST_TOKEN,
    scheduler_step=45,
    prefill_done_tokens=100,  # Prefill complete
    prefill_total_tokens=100,
    decode_done_tokens=1,     # First token generated
    decode_max_tokens=50,
    phase="DECODE"            # Now in decode phase
)
```

**Use for TTFT**:
```python
ttft = first_token_event.ts_monotonic - queued_event.ts_monotonic
```

---

### 4. PREEMPTED 🟡
**When**: Request is preempted (moved from RUNNING → PREEMPTED)

**Meaning**: Scheduler reclaimed resources (freed KV cache blocks) to make room for higher priority requests or due to resource constraints

**Key Fields**:
- `num_preemptions_so_far`: Incremented count of preemptions
- `prefill_done_tokens`: **Preserved accurately** (not reset)
- `decode_done_tokens`: Preserved (output tokens not lost)

**Example**:
```python
RequestJourneyEvent(
    request_id="req-123",
    event_type=RequestJourneyEventType.PREEMPTED,
    scheduler_step=48,
    prefill_done_tokens=50,   # Progress preserved!
    prefill_total_tokens=100,
    decode_done_tokens=0,
    num_preemptions_so_far=1,  # First preemption
    phase="PREFILL"
)
```

**Why Preemption Happens**:
- Resource contention (insufficient KV cache blocks)
- Higher priority requests arrived
- Scheduler policy decisions

---

### 5. FINISHED 🏁
**When**: Request completes (transitions to any FINISHED_* status)

**Meaning**: Request lifecycle is complete, resources freed

**Key Fields**:
- `finish_status`: Terminal reason
  - `"stopped"` - Stop token hit (normal completion)
  - `"length"` - Max tokens reached
  - `"aborted"` - Client aborted (disconnect, cancel)
  - `"ignored"` - Request was ignored (e.g., validation error)
  - `"error"` - Internal error during processing
- `prefill_done_tokens`: Final prefill progress (should equal total)
- `decode_done_tokens`: Final output token count

**Example (Normal Completion)**:
```python
RequestJourneyEvent(
    request_id="req-123",
    event_type=RequestJourneyEventType.FINISHED,
    finish_status="stopped",
    scheduler_step=95,
    prefill_done_tokens=100,
    prefill_total_tokens=100,
    decode_done_tokens=47,    # Generated 47 tokens
    decode_max_tokens=50,     # Could have generated up to 50
    phase="DECODE",
    num_preemptions_so_far=1
)
```

**Example (Length Capped)**:
```python
RequestJourneyEvent(
    request_id="req-123",
    event_type=RequestJourneyEventType.FINISHED,
    finish_status="length",
    decode_done_tokens=50,    # Hit max_tokens limit
    decode_max_tokens=50
)
```

---

## Event Data Structure

Every event captures a **complete snapshot** of the request state:

```python
class RequestJourneyEvent(msgspec.Struct, frozen=True):
    # ===== Identity =====
    request_id: str                          # Unique request identifier
    event_type: RequestJourneyEventType      # QUEUED, SCHEDULED, FIRST_TOKEN, etc.
    ts_monotonic: float                      # time.monotonic() timestamp

    # ===== Scheduler Context =====
    scheduler_step: int | None               # Scheduler iteration counter
                                             # (None only for QUEUED)

    # ===== Progress Snapshot =====
    # These values are ACCURATE even after preemption
    prefill_done_tokens: int                 # Prompt tokens processed
    prefill_total_tokens: int                # Total prompt tokens
    decode_done_tokens: int                  # Output tokens generated
    decode_max_tokens: int                   # Max generation tokens
    phase: Literal["PREFILL", "DECODE"]      # Current processing phase

    # ===== Lifecycle Tracking =====
    num_preemptions_so_far: int              # Preemption count

    # ===== Event-Specific Fields =====
    schedule_kind: ScheduleKind | None       # FIRST or RESUME (SCHEDULED only)
    finish_status: str | None                # stopped/length/aborted/etc (FINISHED only)
```

### Field Descriptions

| Field | Type | Description | When Populated |
|-------|------|-------------|----------------|
| `request_id` | str | Unique identifier for the request | Always |
| `event_type` | IntEnum | Type of lifecycle event (1-5) | Always |
| `ts_monotonic` | float | Monotonic timestamp (for relative timing) | Always |
| `scheduler_step` | int? | Scheduler iteration counter | All except QUEUED |
| `prefill_done_tokens` | int | Prompt tokens processed (high-water mark) | Always |
| `prefill_total_tokens` | int | Total prompt tokens | Always |
| `decode_done_tokens` | int | Output tokens generated | Always |
| `decode_max_tokens` | int | Max tokens to generate | Always |
| `phase` | str | "PREFILL" or "DECODE" | Always |
| `num_preemptions_so_far` | int | Count of preemptions | Always |
| `schedule_kind` | IntEnum? | FIRST or RESUME | SCHEDULED only |
| `finish_status` | str? | Terminal reason | FINISHED only |

---

## Common Use Cases

### Use Case 1: Latency Breakdown

Track where time is spent in request processing:

```python
class LatencyAnalyzer:
    def __init__(self):
        self.events = defaultdict(list)

    def record_event(self, event):
        self.events[event.request_id].append(event)

    def analyze(self, request_id):
        events = sorted(self.events[request_id], key=lambda e: e.ts_monotonic)

        queued = next(e for e in events if e.event_type == RequestJourneyEventType.QUEUED)
        scheduled = next(e for e in events if e.event_type == RequestJourneyEventType.SCHEDULED)
        first_token = next((e for e in events if e.event_type == RequestJourneyEventType.FIRST_TOKEN), None)
        finished = next(e for e in events if e.event_type == RequestJourneyEventType.FINISHED)

        print(f"Request {request_id} Latency Breakdown:")
        print(f"  Queuing Time: {scheduled.ts_monotonic - queued.ts_monotonic:.3f}s")

        if first_token:
            print(f"  Time to First Token (TTFT): {first_token.ts_monotonic - queued.ts_monotonic:.3f}s")
            print(f"    - Prefill Time: {first_token.ts_monotonic - scheduled.ts_monotonic:.3f}s")
            print(f"    - Decode Time: {finished.ts_monotonic - first_token.ts_monotonic:.3f}s")

        print(f"  Total Time: {finished.ts_monotonic - queued.ts_monotonic:.3f}s")
        print(f"  Preemptions: {finished.num_preemptions_so_far}")
```

---

### Use Case 2: Preemption Analysis

Identify requests that suffer from excessive preemption:

```python
def analyze_preemption_impact(events_by_request):
    for request_id, events in events_by_request.items():
        preempted_events = [e for e in events if e.event_type == RequestJourneyEventType.PREEMPTED]

        if len(preempted_events) > 2:
            print(f"⚠️  Request {request_id} preempted {len(preempted_events)} times")

            for i, event in enumerate(preempted_events):
                print(f"  Preemption {i+1}:")
                print(f"    - At step: {event.scheduler_step}")
                print(f"    - Progress: {event.prefill_done_tokens}/{event.prefill_total_tokens} prefill")
                print(f"    - Phase: {event.phase}")
```

---

### Use Case 3: Export to Monitoring System

Integrate events with your existing telemetry stack:

```python
import json
from datetime import datetime

class JourneyEventExporter:
    def __init__(self, export_callback):
        self.export = export_callback

    def process_events(self, engine_outputs):
        for client_idx, eco in engine_outputs.items():
            if eco.journey_events:
                for event in eco.journey_events:
                    # Convert to JSON-friendly format
                    event_data = {
                        "timestamp": datetime.now().isoformat(),
                        "request_id": event.request_id,
                        "event_type": event.event_type.name,
                        "scheduler_step": event.scheduler_step,
                        "progress": {
                            "prefill": f"{event.prefill_done_tokens}/{event.prefill_total_tokens}",
                            "decode": f"{event.decode_done_tokens}/{event.decode_max_tokens}",
                            "phase": event.phase
                        },
                        "preemptions": event.num_preemptions_so_far,
                        "schedule_kind": event.schedule_kind.name if event.schedule_kind else None,
                        "finish_status": event.finish_status
                    }

                    # Export to your telemetry system
                    self.export(json.dumps(event_data))

# Example: integrate with your existing telemetry infrastructure
# (send_to_prometheus, send_to_otel, etc. are your implementation)
exporter = JourneyEventExporter(lambda data: send_to_your_telemetry(data))
```

---

### Use Case 4: SLO Compliance Tracking

Monitor if requests meet SLO targets:

```python
class SLOTracker:
    def __init__(self, ttft_target_ms=1000, total_target_ms=5000):
        self.ttft_target = ttft_target_ms / 1000.0
        self.total_target = total_target_ms / 1000.0
        self.violations = []

    def check_request(self, request_id, events):
        queued = next(e for e in events if e.event_type == RequestJourneyEventType.QUEUED)
        first_token = next((e for e in events if e.event_type == RequestJourneyEventType.FIRST_TOKEN), None)
        finished = next(e for e in events if e.event_type == RequestJourneyEventType.FINISHED)

        ttft = (first_token.ts_monotonic - queued.ts_monotonic) if first_token else None
        total = finished.ts_monotonic - queued.ts_monotonic

        violation = {}
        if ttft and ttft > self.ttft_target:
            violation['ttft'] = ttft
        if total > self.total_target:
            violation['total'] = total

        if violation:
            self.violations.append({
                'request_id': request_id,
                'violations': violation,
                'preemptions': finished.num_preemptions_so_far
            })

    def report(self):
        if self.violations:
            print(f"⚠️  {len(self.violations)} SLO violations detected")
            for v in self.violations[:5]:  # Show top 5
                print(f"  Request {v['request_id']}: {v['violations']}, preemptions={v['preemptions']}")
```

---

## Understanding Progress Tracking

### Why Progress Tracking is Hard

When a request gets **preempted**, the scheduler resets internal counters to reclaim resources. Traditional approaches lose track of how much work was already done. Journey tracing solves this with a **high-water mark** approach.

### How We Track Progress Accurately

#### The Challenge

```
Request starts: prompt_len = 100 tokens

Scheduler iteration 1:
  - Allocated blocks, scheduled 50 tokens
  - num_computed_tokens = 50 ✓

PREEMPTION occurs:
  - Blocks freed
  - num_computed_tokens = 0 ❌ (RESET!)

Scheduler iteration 2:
  - Resume: How much progress was made?
  - num_computed_tokens = 0 (wrong!)
  - num_cached_tokens = ??? (this is cache-hit length, not progress)
```

#### Our Solution: High-Water Mark

```python
# Scheduler maintains a high-water mark dict (only when tracing enabled)
self._journey_prefill_hiwater: dict[str, int] = {}

# When request is RUNNING in prefill phase:
if request.num_output_tokens == 0:  # Still prefill
    prompt_len = len(request.prompt_token_ids)
    prefill_done = min(num_computed_tokens, prompt_len)

    # Update high-water mark (never decreases!)
    self._journey_prefill_hiwater[request.request_id] = max(
        self._journey_prefill_hiwater.get(request.request_id, 0),
        prefill_done
    )
```

**Result**: Progress is preserved across preemptions!

```
Iteration 1: prefill_done = 50 → hiwater[req_id] = 50
PREEMPTION: num_computed_tokens = 0, but hiwater[req_id] = 50 ✓
Iteration 2: prefill_done = 70 → hiwater[req_id] = max(50, 70) = 70 ✓
```

### Decode Progress

Decode progress is simpler because **output tokens are never lost**:

```python
decode_done_tokens = request.num_output_tokens  # len(request._output_token_ids)
decode_max_tokens = request.max_tokens
```

Even after preemption, `num_output_tokens` is preserved.

---

## Performance Considerations

### Overhead When Disabled (Default)

Journey tracing is **disabled by default** with near-zero overhead:

- **CPU**: Single boolean check per emission point (6 checks per request)
- **Memory**: Minimal (empty list per request in OutputProcessor: ~56 bytes per request)
- **Throughput impact**: Negligible

**How?** Single boolean check at each emission point:
```python
def _emit_journey_event(...):
    if not self._enable_journey_tracing:
        return  # Fast path, no work done
```

### Overhead When Enabled

- **Event creation**: O(1) per event
- **Progress snapshot**: O(1) - simple field access
- **Buffering**: O(1) append to list
- **Flush**: O(E) where E = events for that client
- **Complexity**: O(events emitted), **NOT** O(all requests)

**Typical Event Count**:
- Without preemption: 5 events (QUEUED, SCHEDULED, FIRST_TOKEN, FINISHED)
- With 1 preemption: 7 events (+PREEMPTED, +SCHEDULED-RESUME)
- With N preemptions: 5 + 2N events

**Memory Usage**:
- Order of magnitude: hundreds of bytes per event
- Events live briefly (between schedule and update_from_output)
- Overhead scales with event emission rate

**Expected Overhead**:
- Overhead is proportional to the number of events emitted
- For normal workloads with modest preemption rates, overhead is expected to be small
- Benchmark your specific workload if throughput is critical

### When to Enable Journey Tracing

✅ **Good use cases**:
- Development and debugging
- Performance profiling sessions
- Production monitoring (see sampling note below)
- Research and analysis

❌ **Avoid when**:
- Maximum throughput is critical AND you don't need observability
- Memory is extremely constrained

**Production Note**: Sampling (tracking a subset of requests) can be implemented in your event consumer/exporter if you want to reduce overhead. The feature itself currently traces all requests when enabled.

---

## Event Delivery Guarantees & Caveats

### Delivery Semantics

**When events are delivered:**
- Events are accumulated in memory as they occur during request processing
- With OTEL integration: Events are exported as OTEL span events when the request **finishes**
- Without OTEL: Events are available via `EngineCoreOutputs.journey_events` but are accumulated in `RequestState` until request completion

**Important caveats:**

1. **Events accumulate until request completion**: For long-running streaming requests, journey events (QUEUED, SCHEDULED, FIRST_TOKEN, etc.) are buffered in memory until the request finishes. This is bounded by O(5 + 2*preemptions) events per request.

2. **OTEL export happens at request finish**: Journey events are exported to OTEL spans only when a request completes. For streaming requests generating tokens over minutes, you won't see journey events in your traces until the final token.

3. **Event order is preserved**: Events for a single request are emitted and exported in chronological order.

4. **No event sampling**: When enabled, ALL requests are traced. There is no built-in sampling mechanism. If you need sampling in production, implement it in your OTEL exporter configuration.

### Observability Trade-offs

**What you get:**
- Complete lifecycle visibility for every request
- Accurate progress tracking that survives preemption
- Low overhead event emission

**What to be aware of:**
- Events buffered until request completion (memory grows with request duration and preemption count)
- No real-time event streaming (batch export at finish)
- All-or-nothing: no per-request or probabilistic sampling

**Recommendations:**
- For production monitoring at scale, use OTEL sampling at the exporter level
- For debugging specific requests, enable tracing for the duration of investigation
- Monitor memory usage for workloads with high preemption rates or very long requests

---

## Architecture Overview

### Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                         Scheduler                               │
│                                                                 │
│  ┌──────────────┐                                               │
│  │ add_request  │ ──> QUEUED event ──┐                         │
│  └──────────────┘                     │                         │
│                                       │                         │
│  ┌──────────────┐                     │                         │
│  │  schedule()  │ ──> SCHEDULED ──────┤                         │
│  └──────────────┘         │           │                         │
│                           ▼           │                         │
│                   _preempt_request    │                         │
│                           │           │                         │
│                           ▼           │                         │
│                    PREEMPTED ─────────┤                         │
│                                       │                         │
│  ┌──────────────────────┐             │                         │
│  │ update_from_output() │             │                         │
│  │                      │             │                         │
│  │  - Token append ──> FIRST_TOKEN ──┤                         │
│  │                      │             │                         │
│  │  - Flush events ────────────────>  │  Per-Client Buffers    │
│  └──────────────────────┘             │                         │
│                                       │  ┌─────────────────┐    │
│  ┌──────────────────┐                 │  │ Client 0: [...] │    │
│  │ finish_requests  │ ──> FINISHED ───┤  │ Client 1: [...] │    │
│  └──────────────────┘                 │  │ Client 2: [...] │    │
│                                       │  └─────────────────┘    │
└───────────────────────────────────────┼─────────────────────────┘
                                        │
                                        ▼
                              ┌──────────────────┐
                              │ EngineCoreOutputs │
                              │  .journey_events  │
                              └──────────────────┘
                                        │
                                        ▼
                              ┌──────────────────┐
                              │  Frontend/Engine │
                              │  (your code)     │
                              └──────────────────┘
```

### Key Components

1. **Event Emission Points** (6 locations in scheduler.py)
   - Emit events at state transitions
   - Pass scheduler_step explicitly (no mutable global)

2. **Per-Client Buffering**
   - `_journey_events_buffer_by_client: dict[int, list[Event]]`
   - Events accumulate during scheduler iteration
   - Isolated per client (no cross-contamination)

3. **Flush Mechanism** (in `update_from_output`)
   - Copy events from buffer to `EngineCoreOutputs`
   - Clear buffer after flush (no duplication)
   - Events are surfaced even in iterations that produce no token outputs for a client (EngineCoreOutputs may be created solely to carry events).

4. **Progress Tracking** (high-water mark dict)
   - `_journey_prefill_hiwater: dict[str, int]`
   - Only allocated when tracing enabled
   - Updated during RUNNING state
   - Survives preemption

---

## Troubleshooting

### Events Not Appearing

**Problem**: `EngineCoreOutputs.journey_events` is always `None`

**Solutions**:
1. Check if journey tracing is enabled:
   ```python
   assert observability_config.enable_journey_tracing == True
   ```

2. Verify scheduler received the config:
   ```python
   assert scheduler._enable_journey_tracing == True
   ```

3. Check if requests are actually being scheduled:
   ```python
   scheduler_output = scheduler.schedule()
   assert scheduler_output.num_scheduled_tokens  # Non-empty
   ```

---

### Missing FIRST_TOKEN Events

**Problem**: Requests complete but never emit FIRST_TOKEN

**Possible Causes**:
1. **Pooling requests**: These don't generate tokens, only embeddings
2. **Aborted before decode**: Request finished during prefill
3. **Zero max_tokens**: Request configured with `max_tokens=0`

**Debug**:
```python
if event.event_type == RequestJourneyEventType.FINISHED:
    if event.decode_done_tokens == 0:
        print(f"Request {event.request_id} finished without generating tokens")
        print(f"  Finish status: {event.finish_status}")
        print(f"  Phase: {event.phase}")
```

---

### Inaccurate Progress Counts

**Problem**: `prefill_done_tokens` doesn't match expectations

**Check**:
1. **After preemption**: Progress should be preserved
   ```python
   if event.event_type == RequestJourneyEventType.PREEMPTED:
       # Should NOT be 0 if work was done
       assert event.prefill_done_tokens > 0
   ```

2. **Phase transition**: Prefill complete when decode starts
   ```python
   if event.event_type == RequestJourneyEventType.FIRST_TOKEN:
       # Should be 100% complete
       assert event.prefill_done_tokens == event.prefill_total_tokens
   ```

---

## FAQ

### Q: How do I enable journey tracing?

**A**: Enable journey tracing via CLI flag:

```bash
# Enable collection only (for custom integration)
vllm serve meta-llama/Llama-3.2-1B-Instruct --enable-journey-tracing

# Enable with automatic OTEL export (recommended)
vllm serve meta-llama/Llama-3.2-1B-Instruct \
    --enable-journey-tracing \
    --otlp-traces-endpoint http://localhost:4317
```

For programmatic API:
```python
from vllm.config import ObservabilityConfig

observability_config = ObservabilityConfig(
    enable_journey_tracing=True,
    otlp_traces_endpoint="http://localhost:4317"  # Optional, for OTEL export
)
```

Journey tracing is **disabled by default** with near-zero overhead when off.

---

### Q: What's the difference between journey events and EngineCoreEvent?

**A**: They serve different purposes:

| Feature | Journey Events | EngineCoreEvent |
|---------|---------------|-----------------|
| **Purpose** | End-to-end request tracking | Internal scheduler logging |
| **Granularity** | 5 sparse lifecycle events | More detailed events |
| **Progress** | Survives preemption | May not survive |
| **Use Case** | Observability, monitoring | Debugging, profiling |
| **Enabled** | Opt-in via config | Always enabled with log_stats |

Journey events are designed for **production observability**, while EngineCoreEvent is for **development debugging**.

---

### Q: Does journey tracing impact throughput?

**A**: When **disabled** (default), impact is negligible due to single boolean checks. When **enabled**, overhead depends on:
- Event rate (function of request rate and preemption rate)
- Overhead scales with the number of events emitted
- Benchmark your specific workload if throughput is critical
- Consumer-side sampling can reduce processing overhead

---

### Q: Can I use journey tracing for billing/metering?

**A**: Journey events provide accurate token counts, but they are **observability events**, not transactional records. For billing:
- ✅ Use as a cross-check or audit trail
- ❌ Don't use as the sole source of truth
- Consider also tracking at the API layer for redundancy

---

### Q: How do I export events to Prometheus/OpenTelemetry?

**A**: Journey events are **automatically exported to OpenTelemetry** when you set both flags:

```bash
vllm serve MODEL \
    --enable-journey-tracing \
    --otlp-traces-endpoint http://localhost:4317
```

Events appear as **span events** within the `llm_request` span and are sent to any OTEL collector endpoint you configure. From there, they can be:
- Stored in Jaeger, Tempo, or other OTEL-compatible backends
- Processed by OTEL collectors and exported to other systems
- Visualized in tracing UIs alongside request metrics

**For Prometheus**: Journey events are lifecycle events, not metrics. For Prometheus metrics:
1. Use the existing `--enable-mfu-metrics` and other vLLM metric flags
2. Or, write a custom OTEL collector processor to transform journey span events into Prometheus metrics

---

### Q: What happens to events when a request is aborted?

**A**: A FINISHED event is emitted with `finish_status="aborted"`. All progress up to that point is captured accurately.

---

### Q: Can I trace requests across multiple engines (distributed)?

**A**: Currently, journey tracing is per-scheduler (single engine). For distributed tracing:
- Use `request_id` as correlation key
- Aggregate events across engines
- Consider adding trace IDs for distributed correlation (future work)

---

### Q: Why is scheduler_step None for QUEUED events?

**A**: QUEUED events occur **before** the first `schedule()` call, so there's no scheduler step yet. The first SCHEDULED event will have `scheduler_step=1` (or higher if scheduled later).

---

### Q: How long are events buffered?

**A**: Events are buffered for a single scheduler iteration (duration depends on workload):
1. Events emitted during `schedule()` and `update_from_output()`
2. Buffered in `_journey_events_buffer_by_client`
3. Flushed at end of `update_from_output()`
4. Delivered in `EngineCoreOutputs.journey_events`

---

## Additional Resources

- **CLI Flag**: Use `vllm serve --help` to see all observability options
- **Source Code**: `vllm/v1/core/sched/journey_events.py`
- **Implementation**: `vllm/v1/core/sched/scheduler.py` (search for `_emit_journey_event`)
- **Tests**: `tests/v1/core/test_journey_events.py`
- **Configuration**: `vllm/config/observability.py`
- **CLI Arguments**: `vllm/engine/arg_utils.py` (EngineArgs and CLI registration)

---

**Happy Tracing! 🔍📊**

For questions or issues, please file a GitHub issue with the `observability` label.
