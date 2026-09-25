# Expert-load statistics

Expert-load statistics observe **logical expert routing assignments** without enabling expert parallel load balancing (EPLB). They do not change routing weights, expert placement, replication, or scheduling. This is an experimental opt-in diagnostic feature; measure its overhead on your workload before leaving it enabled.

## Enable interval summaries

```bash
vllm serve MODEL --expert-load-stats-config '{"enabled":true,"log_interval":1000,"output_dir":"./expert-load"}'
```

JSONL summaries contain a per-layer expert count vector, assignment total, maximum, mean, maximum-to-mean ratio, unused-expert count, and worker rank labels. `detail="summary"` omits the count vector. Counts reset after each reported interval by default; `reset_after_log=false` reports cumulative counts since startup. Console logging is one compact line per reporting worker and interval, without expert vectors. Set `output_dir` to retain per-layer statistics in rank-qualified JSONL files; without it, only the compact console overview is emitted.

## Use the same collection settings with or without EPLB

Expert-load statistics and expert balancing are independent controls. Use the same `--expert-load-stats-config` in both cases; enabling EPLB does not change the histogram's logical expert IDs, output schema, sampling interval, or file format.

```bash
# Observe routing without rearranging experts.
vllm serve MODEL --data-parallel-size 2 --enable-expert-parallel \
    --expert-load-stats-config '{"enabled":true,"log_interval":1000,"output_dir":"./expert-load"}'

# Observe the same logical distribution while EPLB rearranges physical experts.
vllm serve MODEL --data-parallel-size 2 --enable-expert-parallel --enable-eplb \
    --expert-load-stats-config '{"enabled":true,"log_interval":1000,"output_dir":"./expert-load"}'
```

Both configurations also accept dotted options such as `--expert-load-stats-config.enabled true`. Python callers can pass the same dictionary as `expert_load_stats_config` to `LLM` or `EngineArgs`.

`--eplb-config` continues to control balancing policy and its existing `log_balancedness` diagnostics. It is not required for histogram collection. EPLB's recording window and rebalancing interval do not set the histogram reporting interval. Logical routing skew can remain unchanged after successful balancing: replication and placement redistribute the physical work, not the model's logical expert selections. Use EPLB's physical-load diagnostics to assess that redistribution.

## Enable per-layer, per-iteration histograms

```bash
vllm serve MODEL --expert-load-stats-config '{"enabled":true,"log_interval":1000,"trace":true,"output_dir":"./expert-load-traces","trace_max_iterations":1024,"flush_interval":64}'
```

Trace records contain `iteration`, `forward_index`, `layer`, and `counts`, plus the same rank and model-role labels. Iterations are one-based **local target-model forward calls**, not global scheduler steps. Warmup, graph capture, empty calls, and DP dummy calls do not advance this counter. Different DP ranks can have different iteration numbers; do not align independent ranks solely by this field.

`trace_interval` samples every Nth target iteration, starting at iteration one. `trace_max_iterations` limits the initial trace window; summaries continue afterward. `layers` optionally selects global target layer IDs. Each pipeline stage observes only its selected local MoE layers; stages with no selected layers allocate no tracker or export buffers. Out-of-range IDs and selected dense layers are rejected. Per-iteration records go only to JSONL, not to the console or Prometheus.

Each top-k selection of a valid routed expert contributes one assignment. Padding, invalid expert IDs, and shared experts are excluded. With speculative decoding, **target verification tokens are included even if later rejected**. Draft-model forwards are not instrumented in this version; every record is explicitly labeled `model_role="target"` and `forward_index=0`.

## Distributed meaning

Only `scope="local"` is supported. Collection adds no all-reduce or all-gather.

- A gathered routing batch is sliced to this DP rank's source tokens.
- Routing replicated across TP ranks is counted on TP rank zero only; other replicas do not instrument those layers or emit all-zero files.
- For sequence-parallel routing before dispatch, each TP rank records its own token shard. Add the shards to reconstruct that DP rank's histogram.
- Counts describe the originating tokens' **logical selections**, not the physical EP rank that executes an expert. They are not expert kernel time, communication time, or a complete utilization metric.

With EPLB enabled, logical histogram collection is fused into the existing logical-to-physical mapping kernel. Physical EPLB counters retain their existing sampling and replication semantics; monitoring has separate logical counter storage because summing those physical counters would otherwise include duplicated tokens and depend on placement epochs. There is one routing-ID traversal, not a second routing pass. Existing `eplb_config.log_balancedness` behavior is unchanged; its physical-rank balancedness is not the logical-expert maximum-to-mean ratio reported here.

## Cost and export behavior

Disabled monitoring creates no tracker, device buffers, streams, transfers, reporting thread, or extra GPU kernels. The disabled compilation hash is unchanged.

Enabled monitoring uses persistent int64 counters. Without EPLB there is one histogram kernel per selected MoE layer; with EPLB it adds histogram work to the mapping kernel. One end-of-iteration kernel accumulates summaries, optionally saves a trace row, and clears the pass counters. A device scalar controls valid token count across CUDA-graph replay. This instrumentation has a cost even if the reporting interval is large.

Two preallocated export slots bound GPU and pinned-CPU memory. For L selected layers, E experts and trace chunk capacity C, counter and export storage is approximately `8 * L * E * (2 + 2 * (C + 1))` GPU bytes and `8 * L * E * 2 * (C + 1)` pinned-CPU bytes, plus scalar/event overhead. Without tracing, C is zero. CPU serialization also needs bounded per-chunk workspace.

The inference stream snapshots counters; a separate copy stream waits for that snapshot and transfers to pinned memory. Only the reporting thread waits for D2H completion and writes JSONL. A slot cannot be reused until its write finishes. If both slots remain busy, trace samples are dropped and counted in `dropped_trace_iterations`; summary counts continue accumulating and their interval may extend. Slow export never waits on the inference thread. A filesystem error logs an exception once, increments `export_errors`, and disables further JSONL writes until restart; serving and compact summary logging continue. A failed write can leave the last JSONL record incomplete. CUDA failures are not suppressed. Normal shutdown drains outstanding chunks and emits a partial summary; abrupt process termination can lose buffered records.

## Initial coverage and validation

This version supports CUDA target models using non-monolithic `BaseRouter` MoE paths, including MiniMax-M3's standard `FusedMoEFactory` path. It coexists with returned routed-expert capture without replacing its callback. Eager, piecewise, and full CUDA-graph paths use the same persistent recording buffers.

Monolithic fused router/expert kernels, custom routing implementations (including specialized MegaMoE paths), LoRA, ubatching/DBO, context parallelism, elastic EP, encoder disaggregation, KV-sharing fast prefill, and adaptive speculative verification are rejected. Selected layers owned by a worker must be MoE layers and have the same logical expert count. This is not a claim that every MoE backend or distributed combination has been end-to-end validated.

Correctness tests:

```bash
.venv/bin/python -m pytest tests/v1/worker/test_expert_load_stats.py -v
```

GPU tests exercise exact histograms, padding/invalid IDs, live valid-token counts across graph replay, export correctness, and EPLB mapping/sampling preservation. CPU tests exercise configuration, token ownership, summary resets, trace records, and buffer/writer lifecycle.

Kernel-only instrumentation benchmark (CUDA required):

```bash
.venv/bin/python benchmarks/kernels/benchmark_expert_load_stats.py --tokens 1 8 64 512 4096
```

This measures standalone histogram cost and EPLB mapping with/without monitoring, with uniform and skewed routing. It does not measure JSON serialization, asynchronous export, or end-to-end serving overhead.

Before production use, run a same-commit A/B benchmark with monitoring disabled, summaries enabled, and traces enabled. Compare output equivalence, throughput, TPOT, host overhead, and device memory. Profile small decode batches and skewed routing, and test both EPLB settings on the intended DP/TP/backend configuration. GPU tests and end-to-end M3 performance measurements require a CUDA environment; CPU test results do not establish their correctness or overhead.
