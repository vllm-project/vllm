# EPLB Expert Metrics Tuning Guide

This guide explains how to configure and use vLLM's per-expert EPLB metrics for
debugging load imbalance and correlating expert load with SLOs such as TTFT and
TPOT tail latency.

The feature is opt-in:

```json
{"enable_expert_metrics": true}
```

## What the metrics represent

EPLB exposes two views of expert state:

- **Topology:** which logical expert occupies each physical expert slot.
- **Token load:** how many tokens each physical expert processed.

Token load is split into three batch phases:

- `phase="prefill"`: the batch contains only prefill work.
- `phase="decode"`: the batch contains only decode work.
- `phase="mixed"`: the batch contains both prefill and decode work.

A mixed batch is not split per token. Doing so would require a per-token phase
mask in every MoE router backend, including CUDA-graph and MegaMoE paths.

## Metrics

| Metric | Type | Purpose |
| --- | --- | --- |
| `vllm:eplb_expert_topology` | Gauge | Logical expert currently held by each physical expert |
| `vllm:eplb_expert_tokens_total` | Counter | Tokens routed to each physical expert over the last sample |
| `vllm:eplb_expert_load_timestamp_seconds` | Gauge | When the last load sample was taken |
| `vllm:eplb_expert_load_window_steps` | Gauge | Number of engine steps covered by the last sample |
| `vllm:eplb_rearrangements_total` | Counter | Completed EPLB rearrangements |
| `vllm:eplb_rearrangement_timestamp_seconds` | Gauge | When the last rearrangement occurred |
| `vllm:eplb_expert_topology_timestamp_seconds` | Gauge | When the current topology was committed |

`vllm:eplb_expert_tokens_total` includes the labels `logical_expert` and
`phase`. It is a counter because each sample adds the tokens observed since the
previous sample. Use `rate()` or `increase()` to query the token rate over time.

## Why the sampling interval matters

`expert_metrics_interval` is expressed in **engine steps**, not seconds.

The effective wall-clock resolution is:

```text
sample_period_seconds =
    expert_metrics_interval * median_step_seconds
```

The same step interval therefore gives different resolutions across models,
hardware, batch shapes, and prefill/decode mixes.

A large step interval hides transient hotspots. For example:

```text
step 1: expert 1 = 100 tokens, expert 2 = 0
step 2: expert 1 = 0,   expert 2 = 100 tokens
```

An aggregate over both steps reports:

```text
expert 1 = 100
expert 2 = 100
```

That does not show that the overload moved between steps.

## Choosing `expert_metrics_interval`

There is no universal value. Choose it from the temporal resolution needed for
SLO analysis, then convert that to steps.

```text
expert_metrics_interval ≈
    target_sample_seconds / median_step_seconds
```

Examples:

| Median step time | 1s sample | 5s sample | 15s sample |
| ---: | ---: | ---: | ---: |
| 20 ms | 50 | 250 | 750 |
| 50 ms | 20 | 100 | 300 |
| 100 ms | 10 | 50 | 150 |
| 250 ms | 4 | 20 | 60 |

The default is `100`.

### Suggested starting points

| Goal | Suggested setting |
| --- | --- |
| Production Prometheus monitoring | 10–15 second equivalent |
| SLO incident correlation | 1–5 second equivalent |
| Exact step-by-step diagnosis | `expert_metrics_interval=1` plus JSONL tracing |
| Very large wide-EP deployment | Start with 10–30 second equivalent and measure overhead |

A useful rule is to sample at least 5–10 times finer than the SLO evaluation
window:

| SLO analysis window | Useful sample period |
| --- | --- |
| 30 seconds | 1–5 seconds |
| 1 minute | 5–10 seconds |
| 5 minutes | 15–30 seconds |
| 15 minutes | 30–60 seconds |

## Prometheus vs exact per-step tracing

Prometheus is scrape-based. If EPLB samples every second but Prometheus scrapes
every 15 seconds, you will not see every sample. The counter still contains the
correct cumulative value, and `rate()` over a longer window will reflect the
aggregate, but individual one-second spikes are lost.

Use Prometheus for:

- near-real-time monitoring,
- alerts,
- comparing load trends over minutes,
- correlation with aggregated TTFT/TPOT metrics.

Use JSONL tracing for:

- exact step-by-step hotspot movement,
- offline analysis,
- debugging why one expert or rank is hot,
- reconstructing global per-step load across DP/EP ranks.

## Exact per-step JSONL tracing

Enable tracing:

```json
{
  "enable_expert_metrics": true,
  "expert_metrics_interval": 1,
  "expert_metrics_trace": true,
  "expert_metrics_trace_interval": 1,
  "expert_metrics_trace_path": "/tmp/eplb_expert_trace.jsonl"
}
```

Each DP/EP rank writes its own file, for example:

```text
/tmp/eplb_expert_trace.jsonl.dp0.ep1.model.jsonl
```

Each line contains:

```json
{
  "timestamp": 123.0,
  "step": 42,
  "model": "my-model",
  "dp_rank": 0,
  "ep_rank": 1,
  "num_ep_ranks": 8,
  "num_local_experts": 36,
  "phase": "decode",
  "tokens": [[0, 12, 3, 0]]
}
```

`tokens[layer][physical_expert]` is the load contributed by that source EP rank
during that step. Sum across rank files to reconstruct global per-step load.

Tracing performs synchronous device-to-host copies and writes files. Use it for
diagnostic runs, not as a permanent production setting.

## Recommended workflow

1. Measure median and p95 engine step time under a representative workload.
2. Pick a target sample period based on the SLO or latency window.
3. Convert the period to steps.
4. Run with JSONL tracing for a short diagnostic window.
5. Check how quickly the hottest expert/rank changes.
6. Choose the shortest Prometheus interval that preserves the changes you care
   about.
7. Measure the overhead of the sampling all-reduce at that interval.
8. Compare the resulting expert load timeline with p95/p99 TTFT and TPOT.

## Is the interval model/workload/topology dependent?

Yes, indirectly:

- **Model:** number of MoE layers, expert count, hidden size, and step latency
  change both signal and sampling cost.
- **Workload:** burstiness, prompt length, chunked prefill, and routing skew
  change how fast hotspots move.
- **Topology:** EP/DP size changes all-reduce cost, placement, and metric label
  cardinality.
- **Hardware:** faster GPUs shorten step time, so a fixed step interval covers
  less wall-clock time.

The primary design parameter should be wall-clock resolution. The model,
workload, and topology determine the step time and the acceptable sampling
overhead.

## Example PromQL

```promql
# Global tokens per logical expert and phase.
sum by (model_name, layer, logical_expert, phase) (
  rate(vllm:eplb_expert_tokens_total[1m])
)

# Current placement on EP rank 2, layer 0.
vllm:eplb_expert_topology{layer="0", ep_rank="2"}

# Age of the last expert-load sample.
time() - max by (model_name, engine, model, phase) (
  vllm:eplb_expert_load_timestamp_seconds
)
```

## Limitations

- Prometheus cannot reliably retain every short sample.
- Mixed prefill/decode batches are labeled `mixed`; they are not split per token.
- Token load is per source EP rank. Aggregate across engines for global load.
- Per-expert topology has high label cardinality; keep the feature opt-in.
- Sampling with `expert_metrics_interval=1` adds one EP-group all-reduce per
  step.
