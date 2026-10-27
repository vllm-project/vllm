# Experimental dynamic pipeline parallelism

Dynamic pipeline parallelism moves a layer boundary between two pipeline
stages for independent, complete prefills. Each stage loads a few overlapping
layers. A bounded activation buffer lets a producer submit another batch while
an earlier send is pending. Completed device timings and consumer credits then
guide the boundary toward a balanced split.

The feature is disabled by default and requires Model Runner V2. The initial
model adapter is NVIDIA `DeepseekV4ForCausalLM`. Other models must implement
`SupportsDynamicPP` before enabling boundary movement. Activation buffering
without boundary movement can be enabled independently for MRv2 PP models.

## Enable

```bash
VLLM_USE_V2_MODEL_RUNNER=1 vllm serve deepseek-ai/DeepSeek-V4-Flash \
    --pipeline-parallel-size 2 --tensor-parallel-size 4 \
    --enforce-eager --no-enable-prefix-caching --no-enable-chunked-prefill \
    --max-model-len 20992 --max-num-batched-tokens 20480 \
    --additional-config '{
      "pp_send_buffer_size": 2,
      "dynamic_pp_config": {
        "enabled": true,
        "redundant_layers": 3,
        "initial_offset": -3,
        "high_watermark": 2,
        "cooldown_steps": 8,
        "log_interval_seconds": 10
      }
    }'
```

Send requests with `max_tokens=1`. Every prompt must fit in a single scheduled
batch. An initial offset of -3 makes PP0 execute three fewer layers than its
base partition. Later batches may move this boundary within the preloaded
range. The active boundary is carried with each batch, so work already in
flight continues to use its original split.

To use activation buffering alone, set
`--additional-config '{"pp_send_buffer_size": 2}'` without enabling
`dynamic_pp_config`. This mode retains the model's fixed layer partition.

## Configuration

The following fields belong to `dynamic_pp_config`:

| Field | Default | Meaning |
| --- | --- | --- |
| `enabled` | `false` | Enable boundary movement. |
| `redundant_layers` | `2` | Load up to this many extra layers around the base boundary. |
| `initial_offset` | `0` | Initial PP0 layer-count offset, within the redundant range. |
| `adaptive` | `true` | Set `false` to hold the initial boundary for comparisons. |
| `low_watermark` | `0` | Low completed-activation queue depth. |
| `high_watermark` | `2` | High completed-activation queue depth; must exceed the low watermark. |
| `cooldown_steps` | `8` | Matching completed timing samples required before a move. |
| `max_step` | `1` | Maximum layers moved in one decision. |
| `balance_tolerance_percent` | `3` | Hold when stage times differ by no more than this percentage. |
| `feedback_timeout_seconds` | `30` | Hold the boundary when consumer feedback becomes stale. |
| `log_interval_seconds` | `10` | Periodic strategy logging; `0` disables it. |

`pp_send_buffer_size` defaults to `0`. A positive value reserves that many
independent activation snapshots before memory profiling determines the KV
budget. Dynamic mode requires a capacity at least as large as the high
watermark. More slots and overlapping weights consume additional device memory.
Snapshots also add a device copy per inter-stage output.

## Feedback and observation

The queue depth is completed producer work minus credited consumer starts.
Posted communication handles are reported separately and never count as ready
activations. TP0 polls device events and exchanges bounded snapshots through
the existing PP CPU store; the hot path does not read a device tensor on the
host or synchronize the device for timing. TP0 broadcasts each decision within
its tensor-parallel group.

The controller matches timings by batch, boundary version, token count, and
request count. It uses a median window, ignores stale versions, and holds
within a tolerance. Watermarks shorten the confirmation window when they agree
with the timing imbalance. When credits lag, sustained timing evidence can
still trigger a move. Previously measured neighboring cuts prevent repeated
oscillation between integer layer counts; changes in device load invalidate
incomparable costs.

`Dynamic PP strategy` logs show both execution ranges, boundary version,
completed queue depth, stage times, decision reason, and the splits used during
the reporting interval. A stable boundary with `balanced_hold`, or an adjacent
cut that is no faster, identifies a candidate steady-state split for that
workload. It is not a universal partition for other prompt or batch sizes.

## Current limits

Boundary movement supports eager PP=2, TP, complete fresh prefills, and one
output token per request. Prefix caching, chunked prefill, decode, KV
connectors, speculation, LoRA, DP, EP, context parallelism, sequence parallel
MoE, microbatching, and cross-layer index reuse are rejected. PP buffering
does not support sleep mode or the external launcher.

Movable-layer KV replicas are not implemented. Fresh prefills compute each
logical layer exactly once and finish before historical KV can be needed.
Supporting cached prefixes or subsequent chunks requires consistent cache
ownership and updates on every rank that might execute those layers. Disabling
prefix caching alone does not make chunked prefill or decode safe.

Balancing can reduce bubbles caused by unequal stage times. Startup, drain,
communication, and the indivisibility of layers still impose limits. Measure
end-to-end throughput and latency with the intended workload before drawing a
performance conclusion.
