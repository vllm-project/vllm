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
| `vllm:eplb_rearrangement_pause_seconds` | Histogram | Blocking time spent rearranging experts |
| `vllm:eplb_rearrangement_blocked_request_seconds_total` | Counter | Rearrangement pause weighted by in-flight requests |
| `vllm:eplb_rearrangement_blocked_token_seconds_total` | Counter | Rearrangement pause weighted by scheduled tokens |
| `vllm:eplb_load_rank_skew` | Histogram | Max/mean token load across EP ranks per layer |
| `vllm:eplb_critical_rank_excess_tokens` | Histogram | Tokens above the rank mean on the busiest rank per layer |
| `vllm:eplb_load_expert_skew` | Histogram | Max/mean token load across physical experts per layer |
| `vllm:eplb_critical_expert_excess_tokens` | Histogram | Tokens above the expert mean on the busiest expert per layer |
| `vllm:eplb_expert_token_gini` | Histogram | Gini coefficient of per-expert token load per layer |
| `vllm:eplb_expert_topk_token_share` | Histogram | Share of layer tokens held by the top-k hottest experts |
| `vllm:eplb_active_experts_per_rank` | Gauge | Experts with nonzero load on each EP rank per layer |
| `vllm:eplb_expert_topology_timestamp_seconds` | Gauge | When the current topology was committed |

`vllm:eplb_expert_tokens_total` includes the labels `logical_expert` and
`phase`. It is a counter because each sample adds the tokens observed since the
previous sample. Use `rate()` or `increase()` to query the token rate over time.

The skew histograms carry a `layer` label. They are computed at the source from
the complete per-layer load vector, so experts and ranks with zero load are
included in the max/mean calculation. This differs from aggregating
`vllm:eplb_expert_tokens_total` in PromQL, where zero-token experts are absent.

`vllm:eplb_expert_token_gini` reports inequality across all physical experts in
a layer: `0` is perfectly balanced and approaches `1` as load concentrates in
fewer experts. `vllm:eplb_expert_topk_token_share` reports the fraction of layer
tokens held by the top `expert_metrics_top_k` experts (default `8`); its `top_k`
label records the configured value. Both include zero-load experts.
`vllm:eplb_active_experts_per_rank` reports how many expert slots each rank
actually touched, which distinguishes sparse routing from a rank that is simply
idle. Windows with no tokens for a phase or layer are not exported at all, so
these metrics always describe a window with real load.

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
| Exact step-by-step diagnosis | `expert_metrics_interval=1` |
| Very large wide-EP deployment | Start with 10–30 second equivalent and measure overhead |

A useful rule is to sample at least 5–10 times finer than the SLO evaluation
window:

| SLO analysis window | Useful sample period |
| --- | --- |
| 30 seconds | 1–5 seconds |
| 1 minute | 5–10 seconds |
| 5 minutes | 15–30 seconds |
| 15 minutes | 30–60 seconds |

## Tail-latency metrics

Load imbalance predicts mean throughput, not tail latency. A bulk-synchronous
decode step waits for its slowest EP rank, but the rank skew measured over a
sampling window is usually small and averages away the transients that produce
SLO breaches. Rearrangement stalls are the tail-dominant EPLB event: the
serving loop is blocked while experts move, so every in-flight request gains
the pause duration.

The tail metrics are event-based, which makes them robust to Prometheus
scrape aliasing:

- `vllm:eplb_rearrangement_pause_seconds` observes the blocking duration of
  each rearrangement. Use `histogram_quantile` for the tail.
- `vllm:eplb_rearrangement_blocked_request_seconds_total` adds
  `pause_seconds * num_requests` per rearrangement.
- `vllm:eplb_rearrangement_blocked_token_seconds_total` adds
  `pause_seconds * num_scheduled_tokens` per rearrangement.
- `vllm:eplb_load_rank_skew`, `vllm:eplb_critical_rank_excess_tokens`,
  `vllm:eplb_load_expert_skew`, and
  `vllm:eplb_critical_expert_excess_tokens` are guardrails for sustained
  per-layer imbalance. They are not tail predictors on their own.

The **EPLB tail exposure ratio (ETER)** is the fraction of served
request-time lost to EPLB stalls:

```promql
sum(increase(vllm:eplb_rearrangement_blocked_request_seconds_total[5m]))
/
(
  sum(avg_over_time(vllm:num_requests_running[5m]))
  * 300
)
```

Compare `histogram_quantile(0.99, rate(vllm:eplb_rearrangement_pause_seconds_bucket[5m]))`
with the SLO headroom (p99 SLO latency minus median latency). A pause longer
than the headroom is a direct tail-breach candidate. Label results by `phase`
and `async` so synchronous and asynchronous EPLB are not mixed.

Avoid using `vllm:eplb_rearrangement_timestamp_seconds` to reconstruct when
events happened: it is a last-value gauge and aliases whenever the scrape
period is longer than the rearrangement period.

## Recommended workflow

1. Measure median and p95 engine step time under a representative workload.
2. Pick a target sample period based on the SLO or latency window.
3. Convert the period to steps.
4. Run with `expert_metrics_interval=1` for a short diagnostic window.
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

# p95 per-layer rank skew.
histogram_quantile(
  0.95,
  sum by (le, model_name, layer, phase) (
    rate(vllm:eplb_load_rank_skew_bucket[5m])
  )
)

# p95 per-layer expert skew.
histogram_quantile(
  0.95,
  sum by (le, model_name, layer, phase) (
    rate(vllm:eplb_load_expert_skew_bucket[5m])
  )
)

# p95 per-layer expert Gini.
histogram_quantile(
  0.95,
  sum by (le, model_name, layer, phase) (
    rate(vllm:eplb_expert_token_gini_bucket[5m])
  )
)

# Experts touched per rank and layer.
vllm:eplb_active_experts_per_rank{model_name="my-model"}
```

## Limitations

- Prometheus cannot reliably retain every short gauge value. Event histograms
  and counters are not affected by scrape loss.
- Mixed prefill/decode batches are labeled `mixed`; they are not split per token.
- Token load is per source EP rank. Aggregate across engines for global load.
- Per-expert topology has high label cardinality; keep the feature opt-in.
- Sampling with `expert_metrics_interval=1` adds one EP-group all-reduce per
  step.
