# Bug Report: Negative Prompt Token Stats in P/D Disaggregation

## Summary

In P/D (Prefill/Decode) disaggregated deployments, the `local_cache_hit` metric can become negative, causing Prometheus counter increment failures with `ValueError: Counters can only be incremented by non-negative amounts.`

## Environment

- vLLM version: Latest main (commit `4403e3ed4` and later)
- Deployment: P/D disaggregation with NIXL connector
- Hardware: GB200 (8x GPU gang scheduling)

## Error Message

```
(ApiServer_0 pid=308) ERROR 02-08 03:01:13 [v1/engine/async_llm.py:698] AsyncLLM output_handler failed.
(ApiServer_0 pid=308) ERROR 02-08 03:01:13 [v1/engine/async_llm.py:698] Traceback (most recent call last):
  File "/usr/local/lib/python3.12/dist-packages/vllm/v1/metrics/loggers.py", line 1309, in record
    logger.record(
  File "/usr/local/lib/python3.12/dist-packages/vllm/v1/metrics/loggers.py", line 1113, in record
    self.counter_prompt_tokens_by_source[source][engine_idx].inc(
  File "/usr/local/lib/python3.12/dist-packages/prometheus_client/metrics.py", line 339, in inc
    raise ValueError('Counters can only be incremented by non-negative amounts.')
ValueError: Counters can only be incremented by non-negative amounts.
```

## Root Cause

The bug is in `vllm/v1/metrics/stats.py` in the `PromptTokenStats.update_from_output()` method (lines 278-280):

```python
self.local_cache_hit += (
    num_cached_tokens + recomputed - num_external_computed_tokens
)
```

### Why it goes negative:

In P/D disaggregation:
1. The **prefill service** computes tokens and sends KV cache to the decode service
2. The **decode service** receives these tokens via `num_external_computed_tokens`
3. The calculation assumes: `local_cache_hit = num_cached_tokens + recomputed - num_external_computed_tokens`

However, when:
- `num_external_computed_tokens > num_cached_tokens + recomputed`

This results in a **negative `local_cache_hit`**, which then causes the Prometheus counter to fail.

### Scenario where this happens:

1. Prefill service processes a long prompt (e.g., 7000 tokens)
2. Prefill sends all KV cache to decode via NIXL
3. Decode receives `num_external_computed_tokens = 7000`
4. Decode's scheduler reports `num_cached_tokens = 0` (no local cache)
5. `local_cache_hit = 0 + 0 - 7000 = -7000` ← **NEGATIVE!**

## Proposed Fix

Clamp `local_cache_hit` to non-negative values:

```python
def update_from_output(
    self,
    num_cached_tokens: int,
    num_external_computed_tokens: int,
    prompt_len: int,
) -> None:
    """Update stats from a prefill output."""
    recomputed = 1 if (num_cached_tokens + 1 == prompt_len) else 0

    self.computed += prompt_len - num_cached_tokens
    self.external_kv_transfer += num_external_computed_tokens
    # Clamp to non-negative to handle P/D disagg edge cases
    local_hit = max(0, num_cached_tokens + recomputed - num_external_computed_tokens)
    self.local_cache_hit += local_hit
    self.cached_tokens += num_cached_tokens
    self.recomputed_tokens += recomputed
    self.total += prompt_len
```

## Impact

- **Severity**: High - causes request failures (400 Bad Request)
- **Affected deployments**: All P/D disaggregated setups using NIXL or similar KV transfer
- **Introduced in**: Commit `4403e3ed4` - "[Metrics] Add labeled prompt token metrics for P/D disaggregation (#33290)"

## Steps to Reproduce

1. Deploy vLLM in P/D disaggregated mode with NIXL connector
2. Configure prefill and decode services with `--kv-transfer-config`
3. Send a request that triggers KV transfer from prefill to decode
4. Observe the error in decode service logs

## Related

- PR #33290: [Metrics] Add labeled prompt token metrics for P/D disaggregation
