# Forward-Pass Metrics

Forward-pass metrics (FPM) export one completed engine iteration at a time for
external performance modeling and orchestration. The feature is optional and
disabled by default.

## Enable and subscribe

```bash
vllm serve MODEL --forward-pass-metrics-port 20380

python examples/features/forward_pass_metrics/forward_pass_metrics_subscriber.py \
    --port 20380
```

The port flag enables iteration GPU timing and workload collection. It does not
enable verbose iteration logging and works with `--disable-log-stats`.
Use `--forward-pass-metrics-worker-id` to override the default vLLM instance ID.

Data-parallel rank `N` publishes on `BASE_PORT + N` on its **EngineCore host**.
Each engine owns one publisher, independently of the number or implementation
of API frontends. Python serving, offline clients, and native Rust frontends
(`vllm-rs serve`) use the same publication path. Ports bind during engine
initialization, including when elastic scaling starts a new engine.

Subscribers can join and leave independently of the engine. By default, the
example displays the latest record once per second while validating every
received message. Use `--interval 0` to display the full stream, or
`--interval 5` to display the latest record every five seconds. Each record
occupies at most two 80-character lines.

## Record semantics

The engine matches the scheduling snapshot and GPU timing by iteration ID
before handing a complete record to the exporter. The exporter does not read
scheduler state, calculate CUDA timing, or reconstruct associations.

The FPM v1 schema is defined in
[`vllm/v1/metrics/forward_pass_metrics.py`](../../vllm/v1/metrics/forward_pass_metrics.py).
Its main fields are:

| Field | Meaning |
| --- | --- |
| `worker_id`, `dp_rank` | Engine identity and data-parallel rank. |
| `counter_id` | Per-rank publication sequence; distinct from the internal iteration ID. |
| `timing_scope` | `model_step_cuda`: model execution, sampling, and speculative drafting on the recorded CUDA stream, including host-submission gaps. |
| `wall_time` | CUDA interval in seconds; zero is reserved for engine-confirmed idle. |
| `scheduled_requests` | Prefill/decode request counts, token/context-length sums, and population variances. |
| `queued_requests` | Waiting prefill/decode work captured by the iteration-metrics collector. |

With pipeline parallelism, timing comes from the output worker of the last
stage. It may include waiting for input from earlier stages, but does not
measure the whole pipeline or independent work on other CUDA streams.

Prefill length variance uses each request's
`kv_read + scheduled_tokens / 2`. Decode KV statistics describe logical context
lengths, not the number of input tokens scheduled in that iteration.

A data-parallel dummy iteration has measured timing and empty scheduled
metrics; queued metrics may be nonempty. Only the engine's explicitly confirmed
idle state produces a zero-duration heartbeat. Silence from a publisher does
not imply idle.

## Transport and overhead

The publisher uses the same three-frame ZeroMQ PUB envelope as KV events:
an empty topic, an eight-byte big-endian sequence number, and a MessagePack
payload. Unlike the KV-event publisher, it does not provide a replay socket.

EngineCore enqueues complete records without waiting for subscribers. A
background thread performs conversion, encoding, and socket operations,
including after the last request output. No frontend output consumption is
required. Adding subscribers does not add CUDA events or scheduler collection
work.

Delivery is best effort: the bounded queue, ZeroMQ high-water marks, startup
subscription handshakes, disconnects, and shutdown can drop records. Local
queue drops and bind failures are logged; sequence gaps reveal some losses.
There is no historical replay or acknowledgement from subscribers. Port
conflicts fail engine startup instead of silently disabling FPM.

Publication adds CPU overhead in addition to iteration collection.
Benchmark the complete enabled path with representative traffic; background
publication is not a zero-overhead guarantee.

FPM inherits the iteration timer's current restrictions: generative CUDA
models, `uni` or `mp` executors, and a multiprocessing EngineCore client.
Ordinary single-GPU serving is supported;
in-process EngineCore mode (`VLLM_ENABLE_V1_MULTIPROCESSING=0`) is not.

!!! warning
    The publisher binds all network interfaces and has no authentication.
    Expose its ports only on a trusted network.
