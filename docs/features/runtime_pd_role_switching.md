# Runtime prefill/decode role switching

Runtime P/D switching changes the serving role of an entire engine group while
keeping its workers, model weights, KV registrations, expert placement, process
groups, and CUDA graphs alive. The group stops admitting requests while it drains.
A router can keep the service available by sending new work to other ready groups.

This experimental implementation keeps both KV transfer capabilities enabled at
startup. It changes admission state; it does not resize EP or change the all-to-all
backend. In particular, DeepEP v2 can serve both roles through its existing
graph-compatible execution path. Switching to separately optimized communication
paths is outside this feature.

## Configuration

Set the initial role with `pd_role` in the KV transfer configuration:

```bash
vllm serve Qwen/Qwen3-30B-A3B \
  --data-parallel-size 2 --api-server-count 1 \
  --enable-expert-parallel --all2all-backend deepep_v2 \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both","pd_role":"prefill"}'
```

The first implementation requires NIXL pull transfer, one Python API process with
internal DP load balancing, PP=PCP=DCP=1, and a text generation model. Speculative
decoding, bidirectional KV reuse, sleep mode, and concurrent elastic EP resizing
are unsupported. The model, precision, topology, and memory allocation remain
fixed. Size the initial configuration for both roles and enable the CUDA graphs
needed by decode at startup.

With switching enabled, inference uses `/v1/completions` or
`/v1/chat/completions`. Each request must carry the current
`X-vLLM-PD-Role` and `X-vLLM-PD-Epoch` headers. The NIXL request parameters must
select exactly one direction: `do_remote_decode` on prefill, or
`do_remote_prefill` on decode. Other mutating endpoints are unavailable in this
mode. The ordinary health, metrics, model information, and tokenization endpoints
remain available.

## Transition protocol

Read the group's current role, epoch, and phase:

```bash
curl http://localhost:8000/v1/pd_role
```

Before requesting a transition, the router must stop assigning new work to this
group. It must also account for already assigned handoffs: a decode reservation
may still be waiting for its prefill response. Deliver these reserved requests
before closing admission. Existing accepted streams continue normally.

```bash
curl -X POST http://localhost:8000/v1/pd_role \
  -H 'Content-Type: application/json' \
  -d '{"role":"decode","expected_epoch":0,"drain_timeout":120}'
```

The endpoint returns HTTP 202. Poll `GET /v1/pd_role` until the requested role and
incremented epoch are reported with `phase="ready"`, then add the group to the
new routing pool. The operation continues if its HTTP caller disconnects. Use
the configured API key when authentication is enabled.

Admission closes before any asynchronous work starts. Accepted HTTP requests,
including preprocessing, queued work, and streaming responses, complete first.
Every engine rank then prepares the new epoch while scheduling continues to
release outstanding NIXL transfers and retained KV blocks. The frontend publishes
the new role only after every engine has acknowledged commit. No cache reset,
worker restart, communicator reconstruction, or graph capture is performed by the
transition.

Requests with stale role/epoch headers, and new requests arriving during a
transition, receive HTTP 409 before inference. A router may retry those requests
on another ready group. It must not retry requests whose execution has started.
The router remains responsible for reserved handoffs and for retaining enough
prefill and decode capacity throughout the transition.

A drain timeout restores admission only after every prepared rank has cancelled
the old transaction. The group stays fenced with `phase="rolling_back"` until
cancellation is acknowledged. A failed cancellation or uncertain commit leaves
the group fenced with `phase="failed"`; it must not be returned to routing
automatically. A successful
transition preserves request execution, but draining and temporarily reduced
fleet capacity can affect latency. Continuous availability requires another ready
group for each role.
