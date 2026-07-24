# Production Metrics

vLLM exposes a number of metrics that can be used to monitor the health of the
system. These metrics are exposed via the `/metrics` endpoint on the vLLM
OpenAI compatible API server.

You can start the server using Python, or using [Docker](../deployment/docker.md):

```bash
vllm serve unsloth/Llama-3.2-1B-Instruct
```

Then query the endpoint to get the latest metrics from the server:

??? console "Output"

    ```console
    $ curl http://0.0.0.0:8000/metrics

    # HELP vllm:iteration_tokens_total Histogram of number of tokens per engine_step.
    # TYPE vllm:iteration_tokens_total histogram
    vllm:iteration_tokens_total_sum{model_name="unsloth/Llama-3.2-1B-Instruct"} 0.0
    vllm:iteration_tokens_total_bucket{le="1.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
    vllm:iteration_tokens_total_bucket{le="8.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
    vllm:iteration_tokens_total_bucket{le="16.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
    vllm:iteration_tokens_total_bucket{le="32.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
    vllm:iteration_tokens_total_bucket{le="64.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
    vllm:iteration_tokens_total_bucket{le="128.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
    vllm:iteration_tokens_total_bucket{le="256.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
    vllm:iteration_tokens_total_bucket{le="512.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
    ...
    ```

The following metrics are exposed:

## General Metrics

--8<-- "docs/generated/metrics/general.inc.md"

## Interpreting prefix caching metrics

`vllm:prefix_cache_queries` and `vllm:prefix_cache_hits` are token-level
counters. They count the number of tokens queried in the local prefix cache and
the number found there, rather than the number of requests. A local
prefix-cache token hit rate can be calculated from counter increases over the
same time window:

```promql
increase(vllm:prefix_cache_hits_total[5m])
/
increase(vllm:prefix_cache_queries_total[5m])
```

Prometheus client libraries expose counters with the `_total` suffix in the
time series name.
The ratio above is not the fraction of requests with a cache hit because a
request can reuse only part of its prefix.

Related metrics answer different questions:

- `vllm:prompt_tokens_cached` counts prompt tokens skipped during prefill
  through local prefix-cache reuse and external KV transfer.
- `vllm:request_prefill_kv_computed_tokens` is a per-request histogram of newly
  computed prefill KV tokens, excluding cached tokens.
- `vllm:kv_cache_usage_perc` reports KV-cache usage from 0 to 1; it is not a
  prefix-cache hit rate.
- The sampled `vllm:kv_block_lifetime_seconds`,
  `vllm:kv_block_idle_before_evict_seconds`, and
  `vllm:kv_block_reuse_gap_seconds` histograms describe block-level residency
  and reuse. They do not report per-request cache hit rates.

See [Automatic Prefix Caching](../features/automatic_prefix_caching.md) for a
workflow to verify prefix reuse.

## Speculative Decoding Metrics

--8<-- "docs/generated/metrics/spec_decode.inc.md"

## NIXL KV Connector Metrics

--8<-- "docs/generated/metrics/nixl_connector.inc.md"

## Model Flops Utilization (MFU) Performance Metrics

These metrics are available via `--enable-mfu-metrics`:

--8<-- "docs/generated/metrics/perf.inc.md"

## Deprecation Policy

Note: when metrics are deprecated in version `X.Y`, they are hidden in version `X.Y+1`
but can be re-enabled using the `--show-hidden-metrics-for-version=X.Y` escape hatch,
and are then removed in version `X.Y+2`.
