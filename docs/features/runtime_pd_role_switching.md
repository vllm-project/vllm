# Runtime prefill/decode role switching

Runtime P/D switching moves an entire engine group between prefill and decode.
Workers, model weights, KV registrations, expert placement, communicators, and
CUDA graphs stay resident. The group drains before switching; the router sends
new work to other ready groups.

This experimental feature keeps both KV transfer capabilities enabled at startup.
EP membership and the all-to-all backend stay fixed. DeepEP v2 serves both roles
through its existing graph-compatible path.

## Configuration

Set the initial role with `pd_role` in the KV transfer configuration:

```bash
vllm serve Qwen/Qwen3-30B-A3B \
  --data-parallel-size 2 --api-server-count 1 \
  --enable-expert-parallel --all2all-backend deepep_v2 \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both","pd_role":"prefill"}'
```

Switching requires NIXL pull transfer, one Python API process with
internal DP load balancing, PP=PCP=DCP=1, and a text generation model. Speculative
decoding, bidirectional KV reuse, sleep mode, and concurrent elastic EP resizing
are unsupported. Size the initial configuration for both roles and enable decode
CUDA graphs at startup.

With switching enabled, inference uses `/v1/completions` or
`/v1/chat/completions`. Each request must carry the current
`X-vLLM-PD-Role` and `X-vLLM-PD-Epoch` headers. The NIXL request parameters must
select exactly one direction: `do_remote_decode` on prefill, or
`do_remote_prefill` on decode. Other mutating endpoints are unavailable in this
mode; health, metrics, model information, and tokenization remain available.

## Transition protocol

Read the group's current role, epoch, and phase:

```bash
curl http://localhost:8000/v1/pd_role
```

Before requesting a transition, the router must stop assigning new work to the
group and deliver reserved handoffs, including decode requests still waiting for
their prefill response. Accepted streams continue normally.

```bash
curl -X POST http://localhost:8000/v1/pd_role \
  -H 'Content-Type: application/json' \
  -d '{"role":"decode","expected_epoch":0,"drain_timeout":120}'
```

The endpoint returns HTTP 202. Poll `GET /v1/pd_role` until the requested role and
incremented epoch are reported with `phase="ready"`, then add the group to the
new routing pool. The operation continues if its HTTP caller disconnects. Use
the configured API key when authentication is enabled.

Admission closes before asynchronous work starts. Accepted HTTP requests finish,
including preprocessing and streaming responses. Every engine core then prepares
the new epoch while scheduling drains queued work, NIXL transfers, and retained
KV. The frontend publishes readiness after every core acknowledges commit.

Requests with stale role/epoch headers, and new requests arriving during a
transition, receive HTTP 409 before inference. A router may retry those requests
on another ready group. It must not retry requests whose execution has started.
The router must retain enough prefill and decode capacity during the transition.

A drain timeout keeps admission closed with `phase="rolling_back"` until every
core acknowledges cancellation, then restores the old role. Failed cancellation
or an uncertain commit leaves `phase="failed"` and requires operator recovery.
Draining and reduced capacity can affect latency; continuous availability requires
another ready group for each role.
