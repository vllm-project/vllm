# Mooncake layerwise transfer

`MooncakeLayerwiseConnector` sends KV data from prefill (P) to decode (D) by
WRITE, overlapping ready attention layers with computation. Both peers must
use this connector from the same revision. The implementation described
here has **static validation only**: GPU/NPU payload correctness, distributed
execution, accuracy, and performance have not been validated for these changes.

## Configuration and implemented boundaries

Use `"kv_connector": "MooncakeLayerwiseConnector"` with `kv_producer` on P and
`kv_consumer` on D in the existing [Mooncake deployment](mooncake_connector_usage.md).
P requires eager or PIECEWISE execution with Python save hooks. Producer PD
requests should use `max_tokens=1`; the layerwise-specific enforcement was
withdrawn. Further P decoding could mutate the final recurrent/ring state while
a destination is still waiting. KV writers must use
the compute stream fenced by the layer hook or final forward event.

| Capability | Implementation boundary |
| --- | --- |
| FA / MLA | Incremental complete blocks, followed by the final partial block |
| Multiple transfer groups | Independent logical block units, component readiness, and group-preserving address plans |
| PP | Different P/D stage counts and layer partitions; full cache-name intersection and complete D coverage |
| SWA / SWA MLA | Final requested window, including retained-token and bounded-replay semantics; no cross-chunk window streaming |
| Mamba / GDN | Final resident states after the final forward; align null slots excluded; equal TP required |
| CircularBuffer | One final physical ring page; no token-block expansion or ring reordering; equal TP required |
| PCP | Complete replicas only, explicitly declared on both peers; P replica zero sends to all required D replicas |
| DCP | Unsupported; configuration must use DCP=1 |

For PCP, set the following **only when every PCP rank already holds complete
transferable cache/state**, independently of any TP partition:

```json
{
  "kv_connector": "MooncakeLayerwiseConnector",
  "kv_role": "kv_producer",
  "kv_connector_extra_config": {
    "pcp_cache_mode": "replicated"
  }
}
```

Use the corresponding consumer role on D. This option does not gather or
reconstruct PCP shards. Known sharded specs are rejected. There is no automatic
replica failover: selecting another source requires a new deployment.
Engine identity is synchronized across PCP; only source replica zero registers
its PP/TP endpoints with bootstrap. Other P replicas still participate in
model execution and worker completion aggregation.

P/D component semantics and kernel block sizes must match. Byte strides remain
side-specific; this does not implement dtype conversion, cache transposition,
or incompatible block-internal layouts. Individually addressable views sharing
storage are retained as separate components. Opaque packed rows or registration
that loses writer identity are rejected. Chunked-local attention, storage
replacement, sleep mode, and role switching are not supported.

A windowed `FullAttentionSpec` still uses the full-attention cache manager.
Its full requested suffix is sent at final readiness so D cannot publish prefix
hashes for uninitialized earlier pages; only `SlidingWindowSpec`-managed groups
use tail clipping.

Mamba support refers to the cache-spec path, not a validated model matrix.
Mamba2, GDN variants, checkpoint configurations, and speculative decoding require
separate runtime validation. The connector preserves the base scheduler's
handoff-token adjustment; it does not independently advance or rewind state.

### Receive deadline and send margin

D includes `recv_timeout_s` in every ready message, including messages that reuse
a cached layout. Its value is D's `VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT + 60`
seconds, matching the base receiver's socket timeout. On entering the ready
handler, P records `send_deadline = time.monotonic() + recv_timeout_s` in the
destination. Layout processing and waiting in the model-thread queue consume this
same budget. No timestamps are compared across hosts and no extra handshake is
required. A layout retry starts a new receive window; intermediate ACKs do not
extend P's deadline.

P reads `layerwise_send_timeout_margin_s` from `kv_connector_extra_config`, with
a default of `10` seconds. For example, on the producer:

```json
"kv_connector_extra_config": {"layerwise_send_timeout_margin_s": 10}
```

P rejects a request when its remaining receive window is at most this margin.
The executor checks again after address planning and the device event, immediately
before WRITE. Requests too close to expiry fail without issuing their WRITEs or
advancing their send progress. All layers of those requests are removed from the
batch; healthy requests in the same batch still transfer. An absent timeout is
treated as expired; both peers must use the same connector revision.

P/D wall-clock synchronization is unnecessary. This policy assumes ready's
network and listener delay fit within the configured margin (10 seconds by
default); P starts its clock later than D, so this is a bounded-delay assumption,
not an exact remote deadline. Allow room for WRITE duration as well. The check
reduces late submissions but does not cancel an in-flight WRITE or guarantee
its completion before D releases the destination.

## Why the control flow changes

### Per-step send completion

Layer hooks submit ready data in the background. `wait_for_save()` is the step's
send barrier: it collects all existing sends, submits the remaining currently
available layers, and waits for those sends too. It returns with no pending WRITE
tasks. `start_load_kv()` therefore replaces metadata without waiting for an old
send, and `get_finished()` only reports completion; it does not submit new work.
If a speculative runner polls before finalization, pending sends defer completion
reporting until after the barrier.

Ready messages arriving after the barrier's queue drain are handled in the next
step, including empty scheduler steps that retain completed prefill requests.
The barrier does not wait for destinations that have not sent ready. A slow
destination can still delay the current step; there is no cross-step send backlog.

An executor exception fails only requests present in that executor batch.
Successful independent batches retain their progress, and all in-flight batches
are collected before cleanup. ACK scheduling is tracked separately from actual
delivery: `ack_scheduled_workers` is not proof that D received the ACK.
ACK Futures log failures in a completion callback without polling or accessing
model-thread request state.
Nonoverlapping PP destinations can receive an empty ACK without waiting for final
layer readiness. Non-sending PCP replicas do not retain late-ready tombstones.

Registration caches each group's logical-to-kernel block ratio. Each peer layout
caches its expected TP/PP/PCP worker set, and each ready destination caches the
layers required by that request's nonempty remote block groups. These caches
avoid repeated spec traversal and completion-list construction; block ranges and
addresses still follow the current request snapshot. PCP size participates in
the layout cache key, and queued destinations retain their layouts after LRU eviction.

### Group and stage identity

A group chooses the block table and semantic mapping. A cache name chooses the
writer/readiness unit. A transfer region chooses physical addresses and bytes.
These identities cannot be substituted for one another. Remote group numbers
are mapped by cache name because PP-local group numbering can differ.

D queries all relevant P stages, including when PP counts are equal. A P stage
sends only its intersection and reports the names covered. D requires the union
of successful stage results to cover all required cache components before
publishing receive completion. Destination identity includes engine, TP, PP,
and PCP coordinates, so equal TP ranks on different stages cannot alias.

### Control rollback for review

The additional lifecycle/scheduling safety mechanisms have been withdrawn at
the author's request. The shared scheduler and connector base interface now
match the rebased original layerwise baseline (`ed227fdbc0`). This is a feature
development branch, not a claim of safe cancellation or failure recovery.

The original model-thread/executor handoff, device events, per-step send-result
collection, preemption notifications, and ordinary Mooncake receive handling
remain. PP/PCP worker identities, component readiness, final-state sends, and
coverage aggregation remain because they implement the retained cache features.

Removed mechanisms include the transfer-step allocation fence, explicit
in-flight/data-complete fields, allocation attempts and result replay, query/cancel
messages, drain certificates, producer generations, bounded terminal records,
and worker-fatal handling of unconfirmed writes. ACK contexts are again removed
before socket submission; WRITE/event errors become ordinary batch failures;
executor result waits have no layerwise deadline.

The existing scheduler still holds D blocks when a request is cancelled while
waiting for remote KV. That hold ends when receive completion is reported,
including ordinary failure completion. Neither a receive timeout nor a failed
WRITE establishes that remote DMA has stopped. The original consumer-only
deferred-free behavior for overlapping compute batches is unchanged.

See [the unresolved lifecycle cases](mooncake_layerwise_validation.md#control-rollback-and-unresolved-cases)
before choosing a replacement design. The send margin above limits new WRITE
submissions; it does not restore cancellation or confirm that in-flight DMA stopped.

## Validation status and reproduction

### Deferred: D-ready after P becomes fully idle

If P times out and releases a request's KV, and has no other work keeping
scheduling active, a later D-ready is received by the listener but remains queued.
No `start_load_kv` call consumes it, so D waits for its receive timeout instead
of receiving an immediate rejection. The released KV must not be sent again.
Empty-step transfer progression handles requests still retained by P; it does
not wake a fully idle P. In the Ascend validation with a 20-second P timeout,
this late D request failed after approximately 80 seconds. This limitation is
recorded and intentionally deferred at the author's request; no wakeup or
background rejection mechanism is added.

See [the static delivery report](mooncake_layerwise_validation.md) for the fixed
baseline, commit rationale, completed checks, and the runtime acceptance matrix.
A successful static check or pure block-mapping test does not establish correct
registration, device-event ordering, actual payload delivery, or model accuracy.
