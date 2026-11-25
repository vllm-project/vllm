# Decode instance cache lending

`PeerL1Connector` with `role: "decode"` lets prefill instances reuse cached KV
from a decode instance over the configured RDMA transport. It publishes local
cache blocks and adjusts eviction order; it never acquires KV for a request.
PD remains responsible for all decode request loads. No background backfill or
separate lending pool is created.

Use the [peer L1 requirements and security settings](peer_l1_share.md), including
MoE, prefix caching, compatible weights/geometry and the required ROCm copy
priority environment variables. Configure both PD legs for MoRIIO READ and
[PD replay-tail retention](moriio_connector_usage.md#retaining-replay-tails).
`DFlashSlidingWindowManager` groups and WRITE mode are unsupported for replay-tail
retention. DSpark drafters using ordinary sliding-window managers are supported.

## Enable a decode owner

On decode, combine the PD consumer with an owner-only peer child:

```json
{
  "kv_connector": "MultiConnector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "connectors": [
      {
        "kv_connector": "MoRIIOConnector",
        "kv_role": "kv_consumer",
        "kv_connector_extra_config": {
          "read_mode": true,
          "pd_replay_retention": true
        }
      },
      {
        "kv_connector": "PeerL1Connector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
          "role": "decode",
          "instance_id": "decode-b",
          "kv_compatibility_id": "weights-and-config-v1",
          "control_host": "10.0.0.2",
          "control_port": 28600,
          "peer_seeds": ["tcp://10.0.0.1:28500"],
          "shared_secret_env": "PEER_L1_SECRET"
        }
      }
    ]
  }
}
```

Pass this object to `--kv-transfer-config`. Prefill keeps the default peer
`full` role and adds the decode endpoint to its seeds. Use the same compatibility
ID and secret on both instances. The secret named by `shared_secret_env` must be
inherited by every worker; plain-text `shared_secret` configuration is rejected.
Restrict control and transport endpoints to the trusted deployment.

Decode subscribes only to prefill placement hints. Prefill reads decode owners
as ordinary cross-instance sources. Equal-length same-instance IPC hits remain
preferred, and the remote minimum-gain rule still applies.

## Limits and capacity

Set `role="decode"` in the peer child's `kv_connector_extra_config`.
`protected_cap_blocks` defaults to twice the longest model-context prefix row
bound, capped at 65536; zero disables new grants. It limits unique externally
protected decode rows plus admission reserve. The [complete options
table](peer_l1_share.md#options-and-limits) lists all accepted keys.

Maintenance queries at most 256 redundant candidates per step. The gate closes
when admissible free blocks fall below `max(1, min(256, num_blocks // 100))`,
when running requests exceed scheduler `max_num_seqs`, or when any compute-waiting
request is present. PD loads, grammar and streaming waits are excluded from the
waiting count. These thresholds are derived or fixed, rather than extra options.

The budget conservatively charges every unique externally protected row,
including locally used rows and references awaiting pool-side release, plus
the admission reserve. A new immediate window uses at most half the remaining
budget. Deferred grants can use the remaining budget at absorb and retain the
normal free-headroom limit. Existing grants are never revoked to satisfy a cap
or gate. The fixed 16384-row window and 65536-reference pin cap remain upper
bounds. A gated decode owner is skipped immediately rather than treated as a
busy window.

The default budget sums full-attention rows at `max_model_len` and up to two
replay-boundary tails per sliding-window group. Doubling leaves room for both
one such read and its reserve. This can reserve substantially more capacity
than a percentage default on models with a long context; set an explicit cap
to choose a smaller admission-loss bound. Readers shorten a hit at coordinator
alignment boundaries to the decode owner's advertised remaining budget and pin capacity rather
than repeatedly requesting an oversized grant.

Relative to ordinary PD, admission loses at most this protected cap plus the
separately planned temporary replay-tail storage. This does not guarantee zero
ITL impact: remote reads also consume HBM and network bandwidth. Increase the
cap only after choosing an acceptable admission-loss budget.

Decode eviction order is a best-effort preference: under memory pressure,
newly freed rows with another copy on prefill are preferred for reuse. Spare
empty capacity preserves normal LRU placement, and bounded maintenance lowers
that eviction preference when a prefill copy disappears. Hints may lag; losing
the last cached copy costs a later miss. See the [ordering contract](
../design/peer_l1_decode_lending.md#free-row-ordering) for the pressure gate and
its bounds. This is separate from the admission gate for new read grants.

## Metrics

All names use the `vllm:peer_l1_` prefix and per-engine labels. See the
[complete metric table](peer_l1_share.md#metrics).

On a prefill reader, query
`vllm:peer_l1_reader_tokens_total{transport="rdma",owner_role="decode"}` for the
number of peer prefix tokens selected from decode instances. This counts selected
prefixes, including receives that later fail, rather than completed restores.

- `owner_protected_blocks` and `owner_protected_high_water` report protected rows
  plus admission reserve, for checking the configured admission-loss bound.
- `decode_gate_open` reports whether new acquires are allowed.
  `owner_rejections_total{reason="gate",stage="immediate"}` and the `deferred`
  stage count gate rejections; `reason="protected_cap",stage="immediate"` and
  `reason="budget",stage="deferred"` expose budget rejections.
- `decode_order_rows_total{action}` counts rows placed at the front (`front`),
  formerly redundant rows moved to the tail (`retired`), and redundant candidates
  queried by maintenance (`scanned`), including unknown-hint attempts.
- `reader_truncations_total{reason="protected_cap"}` counts shortened candidates.
- `reader_tokens_total{transport="rdma",owner_role="decode"}`,
  `worker_copy_bytes_total{transport="rdma",owner_role="decode"}` and
  `worker_rdma_failures_total{reason,owner_role="decode"}` describe decode sources
  on prefill readers. Tokens count selected prefixes, not completed restores.
