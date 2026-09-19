# OpenTelemetry JSON stdout profile

The JSON log formatter emits one JSON object per line. This is an application
stdout profile of the OpenTelemetry Logs Data Model; it is not OTLP/JSON. A log
collector must apply the mapping below before exporting OTLP.

## Record contract

| JSON field | Type | OpenTelemetry field |
| --- | --- | --- |
| `timestamp` | RFC 3339 UTC string | `Timestamp` |
| `severity_text` | string | `SeverityText` |
| `severity_number` | integer | `SeverityNumber` |
| `body` | string | `Body` |
| `service.name` | string | `Resource.Attributes["service.name"]` |
| `logger` | string | `InstrumentationScope.Name` |
| `trace_id` | optional 32-character lowercase hex string | `TraceId` |
| `span_id` | optional 16-character lowercase hex string | `SpanId` |
| all other fields | any JSON value | `Attributes` |

Collectors must parse `timestamp`, move `service.name` to the resource, move
`logger` to the instrumentation scope, decode valid trace and span identifiers,
and retain every other application field as a log attribute.

Severity numbers use the OpenTelemetry bands: `TRACE=1`, `DEBUG=5`, `INFO=9`,
`WARN=13`, `ERROR=17`, and `FATAL=21`. Zap `DPANIC` and `PANIC` retain their
defined intermediate values, 18 and 19. `severity_text` uses these uppercase
names; `DPANIC` and `PANIC` are retained where the source runtime exposes them.

Example:

```json
{"timestamp":"2026-09-11T11:34:56.123Z","severity_text":"INFO","severity_number":9,"body":"request complete","logger":"vllm.engine","service.name":"vllm","trace_id":"4bf92f3577b34da6a3ce929d0e0e4736","span_id":"00f067aa0ba902b7"}
```

Trace fields are omitted when no valid span is active.
