# Adaptive token budgets (scheduler example)

The [adaptive budget example](../../examples/features/adaptive_budget/scheduler.py)
uses `scheduler_cls` to vary the total number of tokens scheduled per iteration.
It is opt-in: the standard scheduler and its defaults are unchanged. This is a
reference extension for experimenting with the latency/throughput trade-off,
not a universally faster scheduling policy.

## Run the example

From a vLLM source checkout, make the example importable in the engine process:

```bash
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" vllm serve Qwen/Qwen3-1.7B \
    --scheduler-cls examples.features.adaptive_budget.scheduler.AdaptiveBudgetScheduler \
    --no-async-scheduling --enable-chunked-prefill \
    --max-num-batched-tokens 8192 \
    --additional-config '{"adaptive_budget":{"high_budget":8192,"low_budget":1024,"guard_budget":4096}}'
```

Synchronous scheduling and chunked prefill are required. Use a single GPU and
a text-only decoder model. Speculative decoding, LoRA,
pooling, encoder-decoder models, multimodal models, KV/encoder transfer and
tensor/pipeline/data parallelism are rejected by this example. Prefix caching
is allowed; a waiting request is conservatively counted as prefill until the
native scheduler resolves its cache hit. Remove `--scheduler-cls` and the
`adaptive_budget` entry to restore the standard scheduler.

## Configuration and behavior

All options below belong to `additional_config.adaptive_budget`.
Unknown keys, nonintegral token budgets, and nonfinite timeouts are rejected.

| Option | Default | Meaning |
| --- | ---: | --- |
| `high_budget` | 8192 | Open-state total token budget; must fit the configured scheduling capacity. |
| `low_budget` | 1024 | Total token budget while mixed prefill/decode work is under pressure. |
| `enter_prefills` | 2 | Enter the limited state at this number of runnable prefills with active decode. |
| `exit_prefills` | 0 | Leave the limited state at or below this count; must be below `enter_prefills`. |
| `min_dwell_steps` | 4 | Minimum steps between pressure transitions. |
| `guard_enabled` | true | Allow a periodic budget boost after a prefill stops advancing. |
| `max_stall_ms` | 250 | Monotonic elapsed time since the last observed computed-token increase. |
| `guard_budget` | 4096 | Boosted budget; between `low_budget` and `high_budget`. |
| `guard_interval_steps` | 8 | Minimum number of steps between boosts. |

When either prefill or decode work disappears, the controller immediately
returns to `high_budget`, bypassing minimum residence. With the default
`exit_prefills=0`, this immediate-release rule handles exit; a positive exit
threshold can also exit while mixed work remains. Recomputed generated tokens
after preemption count as prefill work. Grammar/KV-blocked requests are excluded.

The example delegates allocation, preemption, cancellation and request ordering
to the native scheduler. It temporarily narrows `max_num_scheduled_tokens` and
restores it in `finally`; it never resizes the model runner's compilation or
buffer capacity. This relies on the concrete Scheduler implementation, so it
must be checked when upgrading vLLM. It does not add GPU synchronization.

A boost creates an opportunity for progress. It cannot guarantee admission or
freedom from starvation: KV pressure and the native request order still apply.
The observer scans the request map once per step, so CPU cost grows with queue
depth. A smaller total budget may also delay decode requests, especially if it
is smaller than the number of active sequences. Start with a low budget at least
as large as `max_num_seqs` and measure the actual workload.

## Evaluate the trade-off

Compare against the standard scheduler, several fixed runtime budgets, and the
existing `long_prefill_token_threshold` controls. Keep runner capacity, model,
arrival traces, output lengths and warmup constant. Report TTFT, TPOT, end-to-end
latency, throughput, failures and long/short request slices; record run-to-run
variation. Smaller chunks can improve token interarrival latency while worsening
TTFT and overall latency. Do not infer production gains from a single synthetic
trace.

The pressure predicate is inspired by [P-PAS RFC #52906](https://github.com/vllm-project/vllm/issues/52906).
P-PAS caps aggregate **prefill** work while retaining the global token budget.
This example caps **all** scheduled tokens and adds residence, hysteresis and a
rate-limited progress boost. It is not an implementation or reproduction of
P-PAS, and its results should not be labeled as P-PAS results.
