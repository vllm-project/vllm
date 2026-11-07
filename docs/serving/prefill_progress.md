# Prefill Progress over gRPC

The experimental Rust frontend can report the remaining prefill work for each
request. Progress is disabled by default and is available through the
`vllm.Inference/GenerateStream` gRPC API.

## Start the server

Start the Rust frontend with a gRPC port:

```bash
VLLM_USE_RUST_FRONTEND=1 vllm serve Qwen/Qwen3-0.6B \
  --host 127.0.0.1 --port 8000 --grpc-port 50051
```

## Request progress

From the vLLM repository root, use [grpcurl](https://github.com/fullstorydev/grpcurl)
to check whether the server supports progress:

```bash
grpcurl -plaintext -import-path rust/proto -proto control.proto \
  -d '{}' localhost:50051 vllm.Control/GetServerInfo
```

If `supportsPrefillProgress` is true, set `include_prefill_progress` on a
streaming request:

```bash
grpcurl -plaintext -import-path rust/proto -proto inference.proto \
  -d '{"request_id":"example","text":"Explain KV caching.","include_prefill_progress":true,"stopping":{"max_new_tokens":16}}' \
  localhost:50051 vllm.Inference/GenerateStream
```

If the capability is unavailable, omit the flag and use the normal stream.
Unary `Generate` does not accept the flag.

## Consume updates

- Read `prefillProgress` responses alongside the normal `outputs` responses.
  For each request, replace the previous estimate with the newest `sequence`
  and `remainingComputeTokens`; do not add snapshots together.
- Remaining work can increase after preemption or recomputation. When
  `cacheAccountingFinal` is false, cache-hit counts are not yet confirmed.
- Track update freshness and release request reservations on completion, error,
  or cancellation, even if no final progress update arrives.

See the [gRPC protocol](https://github.com/vllm-project/vllm/tree/main/rust/proto)
for the schema and client bindings.
