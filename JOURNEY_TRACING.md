# Request Journey Tracing in vLLM

**Real-time request observability using OpenTelemetry distributed tracing**

---

## What is Request Journey Tracing?

Request journey tracing gives you complete visibility into how requests flow through vLLM, from the moment they arrive at your API server to when responses are sent back to clients.

It uses **OpenTelemetry (OTEL)** to create distributed traces with two linked spans:

- **API Span** (`llm_request`) - Tracks the request through your API server
- **Core Span** (`llm_core`) - Tracks processing in the inference engine

Events are emitted **in real-time** as requests progress through different states, giving you detailed timing and progress information at every step.

---

## Why Use This?

### 🔍 Debug Performance Issues
- See exactly where time is spent: queuing, prefill, decode
- Identify bottlenecks in your serving pipeline
- Understand why some requests are slow

### 📊 Monitor Production
- Track time-to-first-token (TTFT) for every request
- Detect preemption patterns and resource contention
- Measure end-to-end latency from API arrival to departure

### 🎯 Optimize Resource Usage
- Understand how your workload behaves
- See which requests get preempted and why
- Correlate performance with load patterns

---

## Quick Start

### Step 1: Start an OTEL Collector

The easiest way to view traces is using Jaeger with OTEL:

```bash
# Start Jaeger all-in-one (includes OTEL collector)
docker run -d --name jaeger \
  -p 4317:4317 \
  -p 4318:4318 \
  -p 16686:16686 \
  jaegertracing/jaeger:latest

# Open Jaeger UI in your browser
open http://localhost:16686
```

### Step 2: Start vLLM with Journey Tracing

```bash
# Enable journey tracing and point to OTEL collector
vllm serve meta-llama/Llama-3.2-1B-Instruct \
    --enable-journey-tracing \
    --otlp-traces-endpoint http://localhost:4317
```

That's it! Journey tracing is now enabled.

### Step 3: Send Some Requests

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.2-1B-Instruct",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### Step 4: View Traces in Jaeger

1. Open http://localhost:16686
2. Select service "vllm.api" or "vllm.scheduler"
3. Click "Find Traces"
4. Click on any trace to see the complete request journey

You'll see a timeline with two spans:
- **llm_request** (API layer) - parent span
- **llm_core** (Engine layer) - child span

Each span contains events showing the request lifecycle.

---

## What You'll See in Traces

### Two-Layer Span Architecture

Every request creates two linked spans that show the complete journey:

```
┌─────────────────────────────────────────────────────────────┐
│ llm_request (API Layer - Parent Span)                       │
│                                                              │
│  ARRIVED → HANDOFF_TO_CORE → FIRST_RESPONSE → DEPARTED     │
│               │                                              │
│               └──┐                                           │
│                  │                                           │
│  ┌───────────────▼──────────────────────────────────────┐  │
│  │ llm_core (Engine Layer - Child Span)                 │  │
│  │                                                       │  │
│  │  QUEUED → SCHEDULED → FIRST_TOKEN → FINISHED        │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### API Layer Events (Parent Span)

Events on the `llm_request` span:

| Event | When | What It Means |
|-------|------|---------------|
| **ARRIVED** | Request received by API server | Client request has been accepted |
| **HANDOFF_TO_CORE** | Request sent to inference engine | API handed off to scheduler |
| **FIRST_RESPONSE_FROM_CORE** | First token received from engine | Prefill complete, tokens flowing |
| **DEPARTED** | Response sent to client | Request completed successfully |
| **ABORTED** | Request terminated with error | Client disconnect, timeout, or error |

### Engine Core Events (Child Span)

Events on the `llm_core` span:

| Event | When | What It Means |
|-------|------|---------------|
| **QUEUED** | Added to scheduler waiting queue | Waiting for GPU resources |
| **SCHEDULED** | Allocated resources, executing | Actively processing on GPU |
| **FIRST_TOKEN** | First output token generated | Prefill done, decode started |
| **PREEMPTED** | Resources reclaimed, paused | Temporarily paused for other requests |
| **FINISHED** | Completed in scheduler | Done processing, resources freed |

**Note on Event Names:** In OTEL traces, events appear with prefixes: API events use `api.<EVENT>` (e.g., `api.ARRIVED`, `api.DEPARTED`), and core events use `journey.<EVENT>` (e.g., `journey.QUEUED`, `journey.FINISHED`). The tables above show the event type names without prefixes for readability.

### Event Attributes

Each event includes detailed attributes:

**Progress Tracking:**
- `phase` - Current phase: "waiting", "prefill", or "decode"
- `prefill_done_tokens` / `prefill_total_tokens` - Prompt processing progress
- `decode_done_tokens` / `decode_max_tokens` - Output generation progress

**Timing:**
- `ts.monotonic` - High-precision timestamp for latency calculations
- `scheduler.step` - Scheduler iteration number (for correlation)

**Lifecycle:**
- `num_preemptions` - How many times request was preempted
- `schedule.kind` - Whether this is first schedule or resume after preemption
- `finish.status` - Terminal status: stopped, length, aborted, error

---

## Understanding Request Flow

### Normal Request (No Preemption)

```
API:    ARRIVED → HANDOFF_TO_CORE → FIRST_RESPONSE_FROM_CORE → DEPARTED
                       │                        │
Core:                  └→ QUEUED → SCHEDULED → FIRST_TOKEN → FINISHED
```

**Timeline:**
1. Request arrives at API server (ARRIVED)
2. API validates and sends to engine (HANDOFF_TO_CORE)
3. Engine queues request for scheduling (QUEUED)
4. Scheduler allocates resources (SCHEDULED)
5. Prefill completes, first token generated (FIRST_TOKEN)
6. First token sent back to API (FIRST_RESPONSE_FROM_CORE)
7. Generation continues, request finishes (FINISHED)
8. Final response sent to client (DEPARTED)

### Request with Preemption

```
Core: QUEUED → SCHEDULED → PREEMPTED → SCHEDULED → FIRST_TOKEN → FINISHED
                 (first)      ↓         (resume)
                              └─ Resources reclaimed temporarily
```

When the scheduler needs to free resources, it may **preempt** running requests:
- Request is paused (PREEMPTED)
- Resources freed for higher-priority requests
- Later, request is resumed (SCHEDULED with kind=RESUME)
- Progress is preserved - prefill work is not lost

### Request with Error

```
API:    ARRIVED → HANDOFF_TO_CORE → ABORTED
                       │
Core:                  └→ QUEUED → SCHEDULED → FINISHED (status=error)
```

Errors can occur at various points:
- Client disconnect → ABORTED (core events may be truncated if disconnect happens before scheduling)
- Validation error → ABORTED (core may not reach FINISHED)
- Generation error → FINISHED with status=error, then ABORTED

**Note:** Depending on when the error occurs, core events may be incomplete. Early client disconnects or validation failures may result in QUEUED without SCHEDULED/FINISHED. The core span ends when resources are freed, regardless of completion status.

---

## Common Use Cases

### 1. Measuring Time-to-First-Token (TTFT)

TTFT is critical for user experience. Journey tracing gives you precise measurements:

**What to look for:**
- Time from ARRIVED to FIRST_RESPONSE_FROM_CORE = End-to-end TTFT
- Time from QUEUED to FIRST_TOKEN = Engine-only TTFT (excludes API overhead)
- Time from SCHEDULED to FIRST_TOKEN = Prefill duration

**In Jaeger:**
1. Find your trace
2. Look at time between events on the timeline
3. Expand span events to see exact timestamps

**Typical TTFT breakdown:**
```
ARRIVED (t=0ms)
  ↓ API validation + request parsing (~2-5ms)
HANDOFF_TO_CORE (t=3ms)
  ↓ Queue waiting time (varies with load)
SCHEDULED (t=150ms)
  ↓ Prefill compute time (depends on prompt length)
FIRST_TOKEN (t=350ms)
  ↓ Network + API overhead (~1-3ms)
FIRST_RESPONSE_FROM_CORE (t=352ms)
```

### 2. Debugging Slow Requests

When a request is slow, traces show you exactly why:

**High queue time?**
- Long gap between QUEUED and SCHEDULED
- Solution: Scale up, optimize scheduling, reduce load

**High prefill time?**
- Long gap between SCHEDULED and FIRST_TOKEN
- Check: prompt length, model size, batch size

**Frequent preemptions?**
- Multiple PREEMPTED events
- Solution: Adjust scheduling policy, increase KV cache

**In Jaeger:**
- Look for the longest gaps in the timeline
- Check event attributes for clues (preemption count, phase, progress)
- Compare slow traces to fast ones to find patterns

### 3. Understanding Preemption Behavior

Preemption can impact request latency. Traces help you understand it:

**What to look for:**
- `num_preemptions` attribute on events (how many times preempted)
- Multiple SCHEDULED events with `schedule.kind=RESUME`
- Progress preserved: `prefill_done_tokens` doesn't reset

**Example trace with preemption:**
```
SCHEDULED (step=10, kind=FIRST, prefill_done=0/100)
  ↓ Processed 40 tokens
PREEMPTED (step=12, prefill_done=40/100)  ← Paused
  ↓ Other requests processed...
SCHEDULED (step=25, kind=RESUME, prefill_done=40/100)  ← Resumed from 40!
  ↓ Completed remaining 60 tokens
FIRST_TOKEN (step=28, prefill_done=100/100)
```

**High preemption impact?**
- Check system load and request patterns
- Consider increasing KV cache size
- Evaluate scheduling policy (FCFS vs priority-based)

### 4. Monitoring Production Workloads

Journey tracing helps you understand system behavior at scale:

**Key metrics to track:**
- P50/P95/P99 TTFT (from ARRIVED to FIRST_RESPONSE_FROM_CORE)
- Queue time distribution (QUEUED to SCHEDULED)
- Preemption rate (% of requests with preemptions)
- Error rate (% of requests ending in ABORTED)

**Sampling for production:**
vLLM traces all requests when enabled. For high-volume production:
- Use OTEL collector sampling (configure in collector config)
- Or use head-based sampling at the tracer level
- Start with 1-10% sampling, adjust based on overhead

**Example OTEL collector config with sampling:**
```yaml
processors:
  probabilistic_sampler:
    sampling_percentage: 10  # Sample 10% of traces

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [probabilistic_sampler]
      exporters: [jaeger]
```

---

## Configuration Options

### Required Flags

```bash
--enable-journey-tracing          # Enable the feature
--otlp-traces-endpoint URL        # OTEL collector endpoint
```

### Common Configurations

**Local development (Jaeger):**
```bash
vllm serve MODEL \
    --enable-journey-tracing \
    --otlp-traces-endpoint http://localhost:4317
```

**Production (external collector):**
```bash
vllm serve MODEL \
    --enable-journey-tracing \
    --otlp-traces-endpoint http://otel-collector.internal:4317
```

**With other observability features:**
```bash
vllm serve MODEL \
    --enable-journey-tracing \
    --otlp-traces-endpoint http://localhost:4317 \
    --enable-metrics \
    --enable-mfu-metrics
```

### OTEL Exporter Options

vLLM supports standard OTEL environment variables:

```bash
# Alternative: Use environment variables
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
export OTEL_SERVICE_NAME=vllm-production

vllm serve MODEL --enable-journey-tracing
```

---

## Trace Backends

Journey tracing works with any OTEL-compatible backend:

### Jaeger (Recommended for Development)

```bash
# All-in-one: collector + UI
docker run -d -p 4317:4317 -p 16686:16686 \
  jaegertracing/jaeger:latest
```

View traces at http://localhost:16686

### Grafana Tempo (Recommended for Production)

Tempo is designed for high-volume tracing:

```bash
# Example docker-compose.yml
services:
  tempo:
    image: grafana/tempo:latest
    command: ["-config.file=/etc/tempo.yaml"]
    ports:
      - "4317:4317"  # OTLP gRPC
      - "3200:3200"  # Tempo API
    volumes:
      - ./tempo.yaml:/etc/tempo.yaml

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
```

Configure Grafana to query Tempo for trace visualization.

### Other Backends

- **Zipkin** - Compatible via OTEL collector
- **Datadog APM** - Use Datadog OTEL exporter
- **New Relic** - Use New Relic OTEL exporter
- **Honeycomb** - Native OTEL support

---

## Performance Impact

### When Disabled (Default)

Journey tracing is **disabled by default** with negligible overhead:
- No spans created
- No events emitted
- Single boolean check per potential emission point

### When Enabled

Overhead depends on your workload:

**Typical overhead (ballpark estimates):**
- **CPU:** ~1-3% additional CPU for span creation and event emission
- **Memory:** ~1-2KB per active request for span state
- **Network:** ~5-10KB per trace exported to OTEL collector

*These are rough estimates based on typical workloads. Actual overhead varies by request rate, preemption frequency, and OTEL exporter configuration. Measure in your specific environment for accurate numbers.*

**Factors affecting overhead:**
- Request rate (more requests = more spans)
- Preemption rate (more preemptions = more events per request)
- OTEL exporter config (batching reduces overhead)

**Recommendations:**
- ✅ Safe to enable in production with moderate traffic
- ✅ Use sampling for high-volume production (>1000 RPS)
- ✅ Monitor CPU/memory before and after enabling
- ⚠️ Overhead scales with request rate, not model size

---

## Troubleshooting

### Traces Not Appearing

**Problem:** No traces showing up in Jaeger/Tempo

**Check:**

1. **Is journey tracing enabled?**
   ```bash
   # Check your vllm serve command includes:
   --enable-journey-tracing
   ```

2. **Is OTLP endpoint correct?**
   ```bash
   # Verify endpoint is reachable
   curl http://localhost:4317

   # Check vllm logs for connection errors
   grep -i "otlp" /path/to/vllm.log
   ```

3. **Is OTEL collector running?**
   ```bash
   # Check Jaeger is running
   docker ps | grep jaeger

   # Check collector logs
   docker logs jaeger
   ```

4. **Send a test request:**
   ```bash
   curl http://localhost:8000/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{
       "model": "your-model",
       "messages": [{"role": "user", "content": "test"}]
     }'

   # Wait a few seconds, then check Jaeger UI
   ```

### Missing Events

**Problem:** Some events missing from traces

**Common causes:**

1. **Request aborted early**
   - DEPARTED/ABORTED events may be missing if process crashed
   - Check vllm server logs for errors

2. **FIRST_TOKEN never emitted**
   - Request may have finished during prefill (max_tokens=0?)
   - Check FINISHED event for decode_done_tokens=0

3. **OTEL collector dropping events**
   - Check collector is not overloaded
   - Increase collector resource limits

### High Overhead

**Problem:** Journey tracing causing performance issues

**Solutions:**

1. **Enable sampling:**
   ```yaml
   # OTEL collector config
   processors:
     probabilistic_sampler:
       sampling_percentage: 10
   ```

2. **Use batch export:**
   ```bash
   # OTEL exporter batches traces
   export OTEL_BSP_SCHEDULE_DELAY=5000  # Batch every 5s
   export OTEL_BSP_MAX_EXPORT_BATCH_SIZE=512
   ```

3. **Check request rate:**
   - Very high request rates (>1000 RPS) may need sampling
   - Monitor CPU/memory after enabling

### Trace Context Issues

**Problem:** Parent-child spans not linked

**Check:**

1. **W3C Trace Context propagation**
   - vLLM automatically propagates context from API to engine
   - No manual configuration needed

2. **Trace IDs match?**
   - In Jaeger, both spans should have same trace ID
   - If different, context propagation failed (file a bug)

---

## FAQ

### Q: Do I need to change my client code?

**A:** No. Journey tracing is transparent to clients. Just enable it on the server.

### Q: Does this work with all models?

**A:** Yes. Journey tracing works with any model served by vLLM.

### Q: Does this work with distributed inference?

**A:** Currently, journey tracing tracks per-instance. For multi-node tensor parallelism, each node emits its own traces. Correlation across nodes is future work.

### Q: Can I use this with Prometheus?

**A:** Journey tracing uses OTEL spans, not Prometheus metrics. vLLM has separate Prometheus metrics (use `--enable-metrics`). Some OTEL collectors can convert span data to metrics.

### Q: What's the data retention?

**A:** Depends on your backend:
- Jaeger default: 24 hours
- Tempo: Configurable (days to months)
- Configure based on your storage capacity

### Q: Can I export to multiple backends?

**A:** Yes, use an OTEL collector with multiple exporters:

```yaml
exporters:
  jaeger:
    endpoint: localhost:14250
  otlp/tempo:
    endpoint: tempo:4317

service:
  pipelines:
    traces:
      receivers: [otlp]
      exporters: [jaeger, otlp/tempo]
```

### Q: What if my OTEL collector is down?

**A:** vLLM will log warnings but continue serving requests normally. Tracing is best-effort and never blocks request processing.

### Q: Can I customize span names or attributes?

**A:** Not currently. Span names (`llm_request`, `llm_core`) and event types are fixed for consistency.

---

## What's Next?

**For more help:**
- 💬 Ask in [vLLM Discord](https://discord.gg/vllm) #observability channel
- 🐛 File issues at [github.com/vllm-project/vllm/issues](https://github.com/vllm-project/vllm/issues)
- 📖 See [OTEL documentation](https://opentelemetry.io/docs/) for collector setup

**Related observability features:**
- `--enable-metrics` - Prometheus metrics
- `--enable-mfu-metrics` - Model FLOPs utilization
- `--enable-logging-iteration-details` - Detailed scheduler logs

**Advanced topics:**
- Custom OTEL collector pipelines
- Sampling strategies for production
- Correlating traces with logs and metrics

---

**Happy tracing! 🔍✨**
