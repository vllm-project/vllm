# Anonymous Usage Telemetry (`lmcache/usage_telemetry/`)

Phone-home usage statistics: what LMCache deployments look like and how much
caching they do. This is **not** operator observability — Prometheus/OTel
metrics for operators live in `lmcache/v1/mp_observability/` (MP mode) and
`lmcache/observability.py` (single-process mode). The two systems share event
sources in places but have different consumers (LMCache maintainers vs.
deployment operators), different transports, and independent opt-outs.

## Package layout

| Module | Contents |
|---|---|
| `messages.py` | **Wire schema — single source of truth.** Every message dataclass, `USAGE_SCHEMA_VERSION`, per-message `ENDPOINT` |
| `guard.py` | `swallow_telemetry_errors` decorator — the no-throw guarantee |
| `identity.py` | Opt-out gate (`is_usage_tracking_enabled`), `UsageIdentity` (session/machine ids) |
| `transport.py` | `UsageMessageSender` (HTTP), `build_usage_payload` (identity + schema stamping) |
| `env_probe.py` | Hardware/platform/cloud detection feeding `EnvMessage` |
| `context.py` | `UsageContextBase`, single-process `UsageContext`, `InitializeUsageContext` — the `/context` snapshot reporters |
| `continuous.py` | `ContinuousUsageContext` interval counters and lifespan histogram |
| `metric_specs.py` | `MetricSpec` (map-reduce metric contract) and the default metric registry (not re-exported from the package root) |
| `mp_continuous.py` | `MPContinuousUsageReporter` — buffers/reduces/sends the `MetricSpec` metrics for the MP server (not re-exported from the package root) |
| `mp.py` | MP server `MPUsageContext`, `InitializeMPUsageContext` |

`lmcache/usage_context.py` remains as a backward-compatibility shim
re-exporting the pre-package public names.

## Message schema contract

`messages.py` is the protobuf-style registry: every message LMCache can
phone home is a dataclass there, and nowhere else. The contract:

- The **class name** is the wire `message_type` discriminator —
  `build_usage_payload` derives it, so senders cannot disagree with the
  schema.
- Each message declares the stats-server **`ENDPOINT`** it POSTs to as a
  class attribute; senders route by it.
- Fields are **flat and JSON-serializable** (the backend stores flat
  key-value pairs; lists are joined into delimited strings).
- `session_id`, `machine_id`, `schema_version`, and `deployment_mode` are
  stamped onto every payload by `build_usage_payload`, not declared per
  message.

## Reporting paths

| Path | Trigger | Messages |
|---|---|---|
| Single-process one-shot | `LMCacheEngine.__init__` → `InitializeUsageContext` | `EnvMessage`, `EngineMessage`, `MetadataMessage` |
| MP server one-shot | `run_cache_server` → `InitializeMPUsageContext` | `EnvMessage`, `MPServerMessage` |
| Continuous (single-process) | `LMCacheStatsLogger.log_worker`, every `LMCACHE_USAGE_TRACK_INTERVAL` s (default 600) | `ContinuousContextMessage`, `CacheLifespanMessage` |
| Continuous (MP server) | EventBus (`MP_RETRIEVE_END`, `MP_STORE_END`; lmcache-driven transfers only), flushed every `LMCACHE_USAGE_TRACK_INTERVAL` s | `ContinuousContextMessage`; empty intervals double as heartbeats |

Model and KV-layout information only become visible to the MP server when
a serving engine registers its KV caches; a registration-time report
(`MPInstanceMessage`) is planned for a follow-up PR. MP continuous
callbacks run on the EventBus drain thread and only increment counters;
sends happen on the reporter's own flush thread, with a final flush when
the bus stops. See `docs/design/usage_telemetry/continuous_metrics.md`
for the metric roadmap.

## Failure isolation (no-throw guarantee)

A failure anywhere in telemetry must never affect caching or serving. The
mechanisms, outermost first:

- Every entry point called from serving code — `InitializeUsageContext`,
  `InitializeMPUsageContext`, `InitializeMPContinuousUsage`,
  `ContinuousUsageContext.incr_or_send_stats`, `report_once`, the MP
  event callbacks, and `MPContinuousUsageReporter.flush` — is decorated
  with
  `guard.swallow_telemetry_errors`: exceptions are logged at debug level
  and swallowed.
- One-shot sends run on a daemon thread; startup never blocks on the
  stats server.
- `UsageMessageSender.send` additionally swallows all transport errors
  with a 5 s timeout.
- Constructors reachable from serving code never raise on bad inputs:
  nonstandard kv shapes degrade (kv-bytes-per-token = 0) and malformed
  `LMCACHE_USAGE_TRACK_INTERVAL` values fall back to the default —
  necessary because `ContinuousUsageContext.GetOrCreate` is called
  unguarded inside `LMCacheStatsLogger.__init__` on the engine startup
  path.

Tests for this contract live in `TestFailureIsolation`
(`tests/test_usage_telemetry.py`): a transport that raises must not break
the startup report or the stats-logger flush.

## Identity and association contract

Every payload — one-shot or continuous — is stamped with:

- `session_id`: random UUID minted once per process. The join key: the
  `/context` rows keyed by a `session_id` describe the deployment that
  produced all other rows with that `session_id`.
- `machine_id`: random UUID persisted at `~/.config/lmcache/machine_id`.
  Groups sessions from the same machine across restarts (deployment-level
  dedup, restart cadence). Empty string when the file cannot be created.
- `schema_version`: bump `USAGE_SCHEMA_VERSION` whenever a field changes
  meaning so backend analysis can partition by schema.
- `deployment_mode`: `single_process` or `mp_server` — which deployment
  mode produced the payload. Needed because `EnvMessage` and the continuous
  messages are shared between modes and are otherwise ambiguous.

Continuous messages additionally carry `sequence_number` (monotonic per
session). Send failures drop the interval's data rather than retrying —
telemetry must never accumulate unbounded state — so sequence gaps are the
backend's signal that intervals were lost rather than idle.

Privacy rules:

- Identifiers are random UUIDs, never derived from hardware (no MAC address,
  hostname, or `/etc/machine-id`).
- The MP server's `instance_id` is deliberately excluded from payloads: it is
  operator-settable (`--instance-id`) and may contain identifying strings.
  It stays in the operator-facing OTel `service.instance.id` only.
- No prompts, keys, or KV-cache contents are ever sent. Model names are
  reported (they identify public model architectures, not deployments).

## Opt-out semantics

`is_usage_tracking_enabled()` is the single gate, checked by every path:

1. `LMCACHE_TRACK_USAGE=false` (LMCache-specific),
2. `DO_NOT_TRACK` in `1`/`true`/`yes` (cross-tool convention).

When disabled, the factory functions return `None`, the continuous reporter
no-ops, and no state files (`machine_id`) are created.

## Transport and testing

`UsageMessageSender` is the injectable transport boundary. Tests inject a
recording stub into the usage contexts (see `tests/test_usage_telemetry.py`);
no test may hit the network. The test suite globally disables tracking via an
autouse fixture in `tests/conftest.py` so no other test phones home.

## Extension guide

To add a new message type:

1. Define a dataclass in `messages.py` extending `UsageMessage`, with flat
   JSON-serializable fields and an `ENDPOINT` (the class name becomes the
   wire `message_type`).
2. Send it through `build_usage_payload` so identity and schema stamping
   stay uniform; wrap any new serving-code entry point with
   `swallow_telemetry_errors`.
3. Coordinate with the stats-server owner: unknown `message_type` values are
   dropped silently on the backend.
4. Bump `USAGE_SCHEMA_VERSION` only when an *existing* field changes meaning;
   adding a message type or field is not a schema bump.
