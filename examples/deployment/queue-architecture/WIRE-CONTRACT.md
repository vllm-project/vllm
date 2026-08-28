# NATS wire contract (Redis replacement)

This is the locked contract for swapping Valkey/Redis Streams out of this
example. Implement against **roles**, not Redis commands. Do not keep a Redis
backend path.

Semantics: the **client** retries application failures. The queue retries
**only** when a sidecar dies mid-job (at most once). Tokens and HTTP results
are transient.

```
Client ──HTTP──► Proxy
                   │  JetStream Publish (job + reply inbox)
                   ▼
              Stream (memory, work-queue)
                   │  one durable queue consumer; all sidecars pull it
                   ▼
              Sidecar ──HTTP──► vLLM
                   │  core NATS Publish to inbox
                   ▼
              Proxy (still holding the HTTP request)
```

JetStream holds **work**. Core NATS holds the **reply** (one result or a token
stream). There is no `result:` key, no Pub/Sub channel, no `XAUTOCLAIM`.

---

## Broker

Local compose: single `nats-server -js`. Production: 3-node NATS with JetStream
and stream `Replicas: 3`. Develop against R=1; do not require a cluster to
merge the swap.

Server `max_payload` **must** be ≥ proxy `MAX_BODY_BYTES` (default 10 MiB).
NATS default `max_payload` is 1 MiB; leaving it at default will 413/publish-fail
legitimate jobs.

| Env | Who | Notes |
|---|---|---|
| `NATS_URL` | proxy, sidecar | e.g. `nats://nats:4222`. Replaces `REDIS_ADDR`. |
| `STREAM_NAME` | proxy, sidecar | JetStream **stream** name. Default `vllm_requests`. |
| `STREAM_SUBJECT` | proxy, sidecar | Publish/filter subject. Default `vllm.requests`. |
| `CONSUMER_NAME` | sidecar | Durable consumer name, **shared** by every sidecar. Default `vllm-sidecars`. Not a per-replica Redis consumer id. |

Drop: `REDIS_ADDR`, `IDLE_THRESHOLD`, `RECLAIM_INTERVAL`. There is no reclaim
loop.

---

## Stream

Create-if-missing on sidecar (and optionally proxy) startup.

| Field | Value | Why |
|---|---|---|
| Name | `STREAM_NAME` | Stable handle for KEDA / `nats` CLI. |
| Subjects | `STREAM_SUBJECT` | One subject is enough. |
| Storage | `Memory` | HTTP is ephemeral; Raft/R=3 is for node HA later. |
| Retention | `WorkQueue` | Delete on ACK. One consumer **definition**. |
| Discard | `New` | Full stream → publish error → proxy 503, not silent drop-oldest. |
| MaxMsgSize | `MAX_BODY_BYTES` (10 MiB) | Reject monsters at ingest. |
| MaxBytes | compose: 256 MiB | RAM cap. Tune in prod. |
| MaxAge | `1h` | Match default proxy `REQUEST_TIMEOUT` / `STREAM_TIMEOUT`. |
| Replicas | 1 local / 3 prod | |

WorkQueue means **one durable consumer**, many sidecar processes pulling it
(competing consumers). Do not create one durable per pod.

---

## Consumer

| Field | Value | Why |
|---|---|---|
| Durable | `CONSUMER_NAME` (`vllm-sidecars`) | Shared. |
| FilterSubject | `STREAM_SUBJECT` | |
| AckPolicy | Explicit | ACK only after reply is sent (or Term). |
| DeliverPolicy | All | Do not start at `$` / new-only (Redis group `$` skipped backlog). |
| MaxDeliver | **2** | Delivery 1 + one pickup after sidecar death. Never a poison loop. |
| AckWait | `30s` | Short; keep alive with `InProgress`. |
| MaxAckPending | `MAX_CONCURRENT_REQUESTS` (default 2) | Do not hide lag behind a huge PEL. |

`InProgress` at least every `AckWait / 2` while vLLM is generating so a long
job is not redelivered to a second sidecar.

---

## Job message (JetStream payload)

JSON body, same shape as today, plus reply routing:

```json
{
  "job_id": "<ulid>",
  "method": "POST",
  "path": "/v1/chat/completions",
  "headers": { "Content-Type": "application/json" },
  "body": "<raw HTTP body bytes, JSON-encoded>",
  "stream": false,
  "reply_to": "_INBOX.xxx"
}
```

- `job_id`: ULID on **both** stream and non-stream paths (today streaming uses
  `job-%d`; stop that).
- `reply_to`: proxy inbox. Also set NATS header `Nats-Reply-To` to the same
  value. Sidecar prefers the header, then the JSON field.
- Subscribe on the inbox **before** JetStream publish (same race as Redis
  SUBSCRIBE-before-XADD).

Proxy publish failure (MaxBytes, MaxMsgSize, disconnected): HTTP 503 or 413.
Do not invent a local retry onto the stream.

---

## Reply messages (core NATS, inbox)

Not JetStream. If the proxy is gone, publishes fail; sidecar **ACK**s the work
message anyway so a ghost client does not occupy the GPU via redelivery.

### Non-stream (exactly one payload, then ACK work)

```json
{
  "status": 200,
  "headers": { "Content-Type": "application/json" },
  "body": "<response body>"
}
```

On vLLM/forward failure, still one payload (`status` ≥ 400 or `status` 502 and
an error `body`), then **ACK**. Do not Nak. Client sees the error and retries.

### Stream (many payloads, then done, then ACK work)

Each token: the JSON object that would have been an SSE `data:` line (same as
today’s Redis Pub/Sub payload).

Terminal:

```json
{"__done": true}
```

Upstream HTTP error (non-2xx):

```json
{"error": true, "status": 400, "body": "..."}
```

then `__done`, then ACK.

---

## ACK / Term / Nak

| Event | Work message | Reply to inbox |
|---|---|---|
| Success | ACK after replies | result or tokens + `__done` |
| vLLM / forward error | ACK (or Term; same outcome with MaxDeliver) | error payload |
| Proxy/inbox gone | ACK | none |
| Sidecar process death | no ACK → redelivery if `MaxDeliver` allows | none; new sidecar runs the job |
| Second delivery also fails or MaxDeliver exceeded | Term | if still connected, error payload |

Do not Nak on application failure. Nak is only for “vLLM not ready yet” if the
capacity gate already waited and you still want a delayed retry **within**
MaxDeliver; prefer waiting in `waitForCapacity` instead of Nak.

There is **no** `EXISTS result:` idempotency. Double-reply after a stolen job
is prevented by `InProgress`, not by a result key. First successful HTTP write
on the proxy wins; ignore extra inbox messages after the handler returns.

---

## Sidecar behavior that does not change

- `WaitForHealthy` before first pull.
- `waitForCapacity` against vLLM `/metrics` before the next pull.
- `shutdownCtx` vs `workCtx`: stop pulling on SIGTERM; finish in-flight;
  `InProgress` until ACK; `MAX_DRAIN_TIMEOUT` still applies.
- `ForwardNonStreaming` / HTTP to `VLLM_TARGET` stay as they are.

Delete `ReclaimLoop`, `Claim`, `ResultExists`.

---

## Proxy behavior that does not change

- Catch-all HTTP, `MaxBytesReader`, peek `"stream": true`.
- OpenAI path is whatever the client sent (`job.Path`).
- `REQUEST_TIMEOUT` / `STREAM_TIMEOUT` bound how long the proxy waits on the
  inbox (defaults 1h). Timeout → HTTP 504 / SSE error; the sidecar may still
  complete and ACK (orphan reply is dropped).

---

## Out of scope for this contract

- 3-node compose cluster
- Durable token replay
- Fast vs large job lanes
- NATS KV
- Running Redis beside NATS
