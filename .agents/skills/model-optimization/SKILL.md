---
name: model-optimization
description: Turn a vague "optimize this model" request into a concrete vLLM performance goal — which parallelism, which serving regime, which request shape, which benchmark harness, and which metric.
---

# Model Optimization

"Optimize <model>" is not actionable on its own. It does not say which
parallelism to tune, which batch size and KV length to tune for, which harness
defines the number, or whether the target is throughput or perf/W. A change that
wins one cell of the matrix below routinely loses another.

Before touching code, pin down all five. If the request does not supply them,
pick a cell, say which one, and state it in the PR description.

1. **Parallelism** — TP, TP+EP, DP+EP, DCP, or PCP+EP.
2. **Regime** — decode, prefill, or decode-prefill mixed.
3. **Shape** — concurrency and KV/prompt length (the matrix columns).
4. **Harness** — which benchmark produces the reported number.
5. **Metric** — tokens/s (per GPU or per request), latency percentile, or
   tokens/s/W.

## Performance goal matrix

Each cell is a separate goal. Name the one you are targeting.

| Parallelism | Decode: 1 req, 50K KV | Decode: 128 reqs, 50-70K KV | Prefill: 1 req, 8K | Prefill: 4 reqs, 2K | Decode-prefill mixed |
| --- | --- | --- | --- | --- | --- |
| TP | | | | | |
| TP + EP | | | | | |
| DP + EP | | | | | |
| DCP | | | | | |
| PCP + EP | | | | | |

DCP shards KV across ranks for decode and PCP shards context during prefill, so
each is normally paired with the regime it serves. See
[context parallel deployment](../../../docs/serving/context_parallel_deployment.md)
for how both are configured.

The two decode columns pull in opposite directions: 1 request at 50K KV is
latency- and memory-bandwidth-bound, while 128 requests at 50-70K KV is
throughput-bound and usually KV-capacity-limited. Say which one you are
optimizing before claiming a decode win. The same split applies to prefill:
1x8K is a single large compute-bound pass, 4x2K stresses batching and launch
overhead.

## Choosing the harness

Report which harness produced the number, because they are not comparable:

- `vllm bench serve` / `vllm bench latency` — the in-repo default. Reproducible
  from a single command, so prefer it unless the goal is tied to an external
  leaderboard.
- External leaderboards (for example SemiAnalysis InferenceX or Artificial
  Analysis) — use when the goal is to move a published number. Their request
  mix and SLA definitions differ from the in-repo defaults, so a win there does
  not transfer to `vllm bench serve` or vice versa.

A matrix cell maps directly onto `vllm bench serve` flags, e.g. the
128-request decode column:

```bash
vllm bench serve --model <model> --dataset-name random \
    --random-input-len 60000 --random-output-len 1000 \
    --max-concurrency 128 --num-prompts 256
```

## Choosing the metric

Default to throughput at a fixed latency SLA, and state the SLA. Use perf/W
(tokens/s/W, measured with `nvidia-smi` power draw averaged over the steady-state
window) only when the goal is explicitly energy efficiency — it ranks
configurations differently from raw throughput, and mixing the two in one
comparison is the most common way these results become unfalsifiable.

## Reporting

A performance claim is not reviewable without: the matrix cell, the harness and
full command, model and hardware, branch/commit, baseline number, new number,
and an accuracy check (`tests/evals/`) when the change can affect output.
