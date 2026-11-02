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
| `allow_mtp_prefill` | `false` | Opt into the restricted eager DeepSeek-V4 MTP1 fresh-prefill path described below. |
| `allow_mtp_decode` | `false` | Opt into eager DeepSeek-V4 MTP3 with isolated cache state and acknowledged ownership handoff. |
| `allow_prefix_caching` | `false` | Enable local prefix reuse with dedicated movable-layer cache pages and ownership handoff. |
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

The default boundary path supports eager PP=2, TP, complete fresh prefills, and one
output token per request. Prefix caching, chunked prefill, decode, KV
connectors, speculative verification, LoRA, DP, EP, context parallelism, sequence parallel
MoE, microbatching, and cross-layer index reuse are rejected. PP buffering
does not support sleep mode or the external launcher.

Speculative configuration is rejected by default. The experimental
`allow_mtp_prefill=true` opt-in permits only DeepSeek-V4 with
`--speculative-config '{"method":"mtp","num_speculative_tokens":1,"enforce_eager":true}'`.
It retains the complete fresh-prefill and `max_tokens=1` checks, and rejects
scheduled verification tokens. The platform adapter must isolate the PP=1
draft's configuration and MoE execution context from the target pipeline.
This is a prefill/TTFT probe: draft work can add last-stage cost even for OUT1,
but it does not establish verification, acceptance-rate, or decode support.

The default prefill path does not allocate movable-layer KV replicas. Fresh prefills compute each
logical layer exactly once and finish before historical KV can be needed.
Supporting cached prefixes or subsequent chunks requires consistent cache
ownership and updates on every rank that might execute those layers. Disabling
prefix caching alone does not make chunked prefill or decode safe.

### Experimental MTP3 state handoff

`allow_mtp_decode=true` is a separate opt-in for DeepSeek-V4 with eager MTP and
exactly three speculative tokens. It cannot be combined with `allow_mtp_prefill`.
The model adapter must allocate independent complete backing for every movable
layer's attention, compressed, indexer, and compressor state. Copies include
physical page padding and indexer scales. Shared cache pools cannot be copied
as if they were independent layer state.

Before either stage executes a batch, PP0 acquires its activation-send slot
and publishes its boundary to the PP CPU-group Store. PP1 consumes and deletes
the header before receiving that batch. Completed metadata sends bound unconsumed
headers by the ring capacity. PP0 does not wait for PP1 on an unchanged cut.
On an ownership change both stages finish prior device work,
transfer every changed layer's complete state from its previous owner, and
wait for completion. The activation header must match this control epoch.
Unchanged boundaries copy no KV and add no device synchronization or two-way
PP control rendezvous. Each batch still publishes and consumes one small header. Stage feedback includes the
last stage's MTP proposals; ownership-transfer time is reported separately and
remains part of end-to-end latency.

This path assumes the central KV planner's common group/block mapping and
checks matching component shapes, types, byte counts, and independent storage.
It does not add pooled KV, resumed requests, chunking, or graphs. Local prefix
reuse requires the additional opt-in below.
Independent state requires memory beyond the ordinary KV budget; the adapter
must check capacity before allocating it. This complete-state handoff is an
experimental implementation, not an incremental background KV replication path.

### Local prefix caching

Enable `--enable-prefix-caching` together with
`"allow_prefix_caching": true` in `dynamic_pp_config`. Keep eager execution,
PP=2, and chunked prefill disabled. This opt-in is supported by the NVIDIA
DeepSeek-V4 adapter, for OUT1 prefills or together with `allow_mtp_decode` for
MTP3 decode. The restricted MTP1 prefill mode can also reuse local prefixes;
its one-output-token limit remains in force. KV connectors and HiSparse are
not supported. A platform adapter must implement both cache-state and
cache-layout accessors before using this opt-in.

The scheduler continues to own prefix hashes, reference counts, eviction,
and physical block IDs. A cache hit executes the **entire uncached suffix**
in one batch, including the last prompt token on a full-prefix match.
Partial suffix scheduling and resumed requests are rejected.

Each movable component gets independent, padded cache pages before binding.
This includes SWA, compressed KV, indexer bytes/scales, and FP32 compressor
state. Scheduler block IDs and kernel view shapes stay the same. The normal
copy-on-write and block-zeroing paths operate on these views, including raw
page padding. Nonmovable layers keep their ordinary pooled allocation.

Both PP stages check the movable components' scheduler group IDs, cache
specifications, block count, kernel block sizes, view strides, hit alignment,
hash block size, and retention policy before the first real batch. A mismatch
aborts instead of reading another group's blocks. When a layer changes owner,
the handoff copies **every physical block** of that layer, including reusable
prefixes belonging to completed requests. No copy is needed on an unchanged
cut. Prefix-cache reset and eviction keep their scheduler semantics; the
runtime does not create a second hash index or invalidate hits on each move.

Dedicated pages consume memory beyond the ordinary KV pool. Allocation checks
their complete size plus 1 GiB device headroom. Use a conservative explicit
`--kv-cache-memory-bytes` budget for this prototype; the extra storage is not
yet integrated into the scheduler's automatic KV capacity calculation.
Ownership changes quiesce both stages and transfer complete state, so a large
cache can make frequent moves expensive. This is local prefix reuse with
ownership handoff, not background incremental replication or pooled KV import.

For MTP workloads, start with a **positive** `initial_offset` (for example `2`)
to give PP0 more target layers: PP1 also runs the draft steps. The feedback
measures that last-stage draft work and can adjust the split further.

Balancing can reduce bubbles caused by unequal stage times. Startup, drain,
communication, and the indivisibility of layers still impose limits. Measure
end-to-end throughput and latency with the intended workload before drawing a
performance conclusion.
