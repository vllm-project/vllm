# KV connector startup notification

Override `KVConnectorBase_V1.on_kv_cache_ready()` to finish deferred registration
before the first request. During normal startup, it runs after warmup, CUDA Graph
capture (when enabled), final KV cache sizing, and preceding device work. Eager
mode uses the same hook. The default is a no-op; `MultiConnector` forwards it to
each child.

With `VLLM_ELASTIC_EP_SCALE_UP_LAUNCH`, the hook runs before the deferred warmup
and capture in `warm_and_capture()`. It is not repeated afterward. Connectors
must not treat this notification as a post-capture boundary on that path.

`register_kv_caches()` receives tensors earlier. To defer registration, retain
those tensors and their layout until `on_kv_cache_ready()` runs.

Startup waits for every worker's hook to return. **The hook must return in bounded
time:** vLLM provides no hook timeout or cancellation, so an unbounded external
wait blocks initialization indefinitely. Set deadlines on external operations;
exceptions fail initialization. Background tasks started by the hook are not
awaited.

The hook runs only during startup. Idle recovery and sleep/wake continue to use
the connector's existing lifecycle.
