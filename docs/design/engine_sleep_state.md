# Engine sleep-state telemetry

Sleep is a combination of scheduler and memory resource states, rather than a
single mutually exclusive sleep level. The engine publishes confirmed snapshots
of these dimensions to every frontend, including when no generation is running.
Python and Rust expose the same new metric families with `model_name` and `engine`
labels. Each engine is recorded independently, including with data parallelism.

## Metrics

`vllm:engine_sleep_resource_state{resource, state}` is one for the current state
of a resource and zero for its other states:

| Resource | States |
| --- | --- |
| `scheduler` | `running`, `paused` |
| `weights` | `resident`, `offloaded`, `discarded`, `unknown` |
| `kv_cache` | `resident`, `released`, `unknown` |

`vllm:engine_fully_awake` is one when scheduling is running and both tracked
memory resources are confirmed resident. A successful full wake resumes
scheduling and sets this metric to one, consistent with the wake API result.
The metric remains zero for scheduler-only pauses, partial wakes, or an unknown
resource state. It describes confirmed operation state; it does not assert that
level-two discarded tensor contents have been restored by an external weight
reload. Residency alone does not certify model contents.

A failed collective memory RPC may leave workers in different states. Its
potentially affected resources become `unknown`, rather than claiming a
successful offload, release, or wake. State changes only after executor operations;
invalid or redundant wake requests do not invent residency changes. Scheduler
state is read from the actual scheduler pause state.

| Operation | Scheduler | Weights | KV cache |
| --- | --- | --- | --- |
| `sleep(level=0)` | paused | unchanged | unchanged |
| `sleep(level=1)` | paused | offloaded | released |
| `sleep(level=2)` | paused | discarded | released |
| `release_kv_cache_memory` after pause | paused | resident | released |
| partial wake of weights | paused | resident | released |
| wake of remaining KV cache | running | resident | resident |

## Transport and compatibility

Optional fields are appended to `SchedulerStats`: `sleep_state` contains the
snapshot and `sleep_state_only` marks a telemetry-only update. These updates
bypass normal request, cache, throughput, and routing-load recording. Normal
scheduler stats retain their existing meanings and positional field ordering.
No sleep or wake utility response format changes are needed.

New gauge children are created only once a snapshot arrives. Rust connected to
an older engine without snapshots does not claim that an unobserved engine is
asleep. The same applies before Python receives its first snapshot.

The existing Python `vllm:engine_sleep_state{sleep_state}` series remains for
migration. `awake` follows the new fully-awake predicate; `weights_offloaded`
follows the confirmed weight disposition; `discard_all` is set only while
weights are discarded and KV cache is released. All three may be zero for a
scheduler pause, partial wake, KV-only release, or unknown resources, so these
legacy flags are not a complete state model. Partial weight wake clears the
stale `weights_offloaded` flag immediately. New dashboards should use the
resource dimensions and `engine_fully_awake`. Rust does not add the deprecated
three-state metric.

The legacy stat-logger `record_sleep_state` method remains available for custom
logger compatibility. Engine-originated snapshots use the optional
`record_sleep_snapshot` callback instead; custom loggers can implement that
callback to observe per-engine resource state.
