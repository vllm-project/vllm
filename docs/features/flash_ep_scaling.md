# Flash EP Scaling

Flash EP scaling is a fast path for resizing the **active** Expert Parallel
(EP) group of a running vLLM serving instance. It targets MoE deployments
where the **maximum** DP/EP size is known at startup: the server pays the
cost of weight loading, kernel compilation, and NCCL topology construction
**once**, and later scaling operations only resize the active subset of
that prepared topology.

In our 8-GPU single-node tests this brings end-to-end EP scale-up/down
from tens of seconds (with the existing
[Elastic EP](https://vllm.ai/blog/2026-05-14-elastic-expert-parallelism)
rebuild flow) down to **~1–2 seconds**.

!!! note
    Flash EP scaling is exposed via a development endpoint and requires
    `VLLM_SERVER_DEV_MODE=1`. It is intended for operators, not for
    end-user requests.

## How it works

The server starts at the maximum DP/EP size and stays there for the
lifetime of the process. `/flash_epscale` then runs one of three paths:

```text
                   ┌─────────────────────────────────────────────────┐
                   │  /flash_epscale  ep_size=N                      │
                   └─────────────────────────────────────────────────┘
                                       │
                       ┌───────────────┼───────────────┐
                       ▼               ▼               ▼
                     noop          scale_down       scale_up
                       │               │               │
                       │               │               │
                       │  drain target ranks (active   │  pause
                       │  ranks keep serving)          │
                       │               │               │  wake offloaded
                       │               │  pause        │  ranks
                       │               │               │
                       │               │  wake / EPLB  │  resize NCCL
                       │               │  remap / NCCL │  group
                       │               │  split        │
                       │               │               │  resume
                       │               │  offload to   │
                       │               │  CPU          │  open routing
                       │               │               │  to new ranks
                       │               │  resume       │
                       └───────────────┴───────────────┘
```

- **Scale down** rearranges experts off the to-be-inactive EP ranks,
  offloads their weights and KV cache to CPU (sleep/wake-style), splits
  the NCCL communicator with `ncclCommSplit`, and masks those ranks in
  NIXL all2all. Active ranks keep serving while the soon-to-sleep ranks
  drain in flight.
- **Scale up** wakes the previously offloaded ranks, restores the
  logical expert mapping, resizes the NCCL group, and only then opens
  routing back to the larger active set, so requests never reach a rank
  whose communicator is still being resized.

## Requirements

To use this endpoint, the server must be started with:

- `--enable-sleep-mode` — the offload path reuses sleep/wake
  infrastructure.
- `--enable-expert-parallel` and `--enable-eplb` — the EPLB state
  manages the logical-sleep mapping during resize.
- `--enable-elastic-ep` — required for routing requests away from
  to-be-inactive DP ranks.
- `--all2all-backend nixl_ep` — peer masking during scale_down uses the
  NIXL EP backend.
- `--data-parallel-backend ray` and a `--data-parallel-size` equal to
  the maximum EP world size.
- `VLLM_SERVER_DEV_MODE=1` in the environment to expose
  `/flash_epscale`.

## HTTP endpoint

### `POST /flash_epscale`

Resize the active EP group to `ep_size`.

#### Request body

```json
{
  "ep_size": 4,
  "tags": ["expert_weights", "kv_cache"],
  "drain_timeout": 300
}
```

| Field | Type | Required | Default | Description |
| --- | --- | --- | --- | --- |
| `ep_size` | int | yes | — | New active EP size; must be in `[1, ep_world_size]`. |
| `tags` | list&nbsp;of&nbsp;str | no | `["expert_weights", "kv_cache"]` | CuMem allocator tags to offload (scale-down) or wake (scale-up). |
| `drain_timeout` | number&nbsp;(seconds) | no | `300` | Time to wait for in-flight requests on to-be-inactive ranks to finish. Only used by `scale_down`. |

#### Response body

```json
{
  "ok": true,
  "ep_world_size": 8,
  "active_ep_size": 4,
  "sleeping_ep_ranks": [4, 5, 6, 7],
  "changed": true,
  "action": "scale_down",
  "tags": ["expert_weights", "kv_cache"],
  "timing_ms": {
    "query_state": 0.0,
    "route_shrink": 0.0,
    "drain": 0.0,
    "pause": 0.0,
    "wake": 0.0,
    "resize": 0.0,
    "sleep": 0.0,
    "resume": 0.0,
    "final_state": 0.0,
    "total": 0.0
  }
}
```

| Field | Description |
| --- | --- |
| `action` | One of `noop`, `scale_up`, `scale_down`. |
| `changed` | `true` if the active group was resized. |
| `sleeping_ep_ranks` | Suffix of EP ranks currently offloaded. |
| `timing_ms` | Per-step timing for diagnosing latency. Only steps that actually ran are present. |

`timing_ms` keys correspond to the orchestration steps:

- `query_state` / `final_state` — pre/post EP-state queries.
- `route_shrink` / `route_grow` — front-end routing updates.
- `drain` — wait for to-be-inactive DP ranks to finish in flight (scale-down only).
- `pause` / `resume` — engine stop-the-world enter/exit.
- `wake` — `wake_up_ep_ranks` collective RPC.
- `resize` — `resize_sleep_ep_ranks` collective RPC (EPLB remap +
  NIXL mask + `ncclCommSplit`).
- `sleep` — `sleep_ep_ranks_by_tags` collective RPC.
- `total` — wall time of the whole call.

#### Error responses

- `400` — invalid `ep_size`, `tags`, or `drain_timeout`.
- `500` — workers reported inconsistent EP state, or a step inside
  the pause window failed. The engine is always resumed before
  returning the error.

## Helper endpoints

These are reused internally by `/flash_epscale` but can also be called
directly when scripting recovery flows.

### `POST /sleep_ep_ranks_tags`

Offload selected ranks for the given tags. Body:

```json
{
  "sleeping_ep_ranks": [4, 5, 6, 7],
  "tags": ["expert_weights", "kv_cache"]
}
```

### `POST /wake_up_ep_ranks_tags`

Reverse of the above; restores the listed tags on the listed ranks.

## Example session

Start the server at the maximum size:

```bash
VLLM_SERVER_DEV_MODE=1 vllm serve "$MODEL" \
    --trust-remote-code \
    --enable-sleep-mode \
    --enable-expert-parallel \
    --enable-eplb \
    --enable-elastic-ep \
    --all2all-backend nixl_ep \
    --eplb-config.num_redundant_experts 64 \
    --data-parallel-backend ray \
    --distributed-executor-backend ray \
    --data-parallel-size 8 \
    --data-parallel-size-local 8 \
    --port 8005 \
    --enforce-eager
```

Scale down from EP=8 to EP=4 while serving:

```bash
curl -X POST http://localhost:8005/flash_epscale \
  -H 'Content-Type: application/json' \
  -d '{"ep_size": 4}'
```

Scale back up:

```bash
curl -X POST http://localhost:8005/flash_epscale \
  -H 'Content-Type: application/json' \
  -d '{"ep_size": 8}'
```

## Limitations

- **Bounded by the startup size.** This path only resizes the active
  subset within the EP world that was constructed at startup. Going
  beyond the startup maximum requires the existing ElasticEP flow or
  NCCL `grow`/`shrink` primitives.
- **Wait policy during scaling.** In-flight requests on the
  to-be-inactive ranks are drained before the transition runs. Active
  ranks keep serving during the drain.
- **First scaling call is slower.** Sleep/wake (level 1) allocates a
  pinned CPU buffer the first time it is invoked (~20 s). Subsequent
  scaling calls run at ~1 s per offload/restore.
- **MoE only.** The endpoint only makes sense for EPLB-managed MoE
  deployments.

## Related

- [Sleep mode](sleep_mode.md) — the offload primitive flash EP scaling
  builds on.
- [Elastic EP](https://vllm.ai/blog/2026-05-14-elastic-expert-parallelism) —
  the existing path for changes that exceed the startup topology.
