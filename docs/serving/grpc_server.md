# gRPC Server

vLLM includes a standalone gRPC server entrypoint:

```bash
python -m vllm.entrypoints.grpc_server --model <model>
```

This server exposes the `VllmEngine` protobuf service defined in
`vllm/grpc/vllm_engine.proto`.

## KV Event Streaming

The gRPC server supports server-streaming KV cache events via:

- `SubscribeKvEvents(SubscribeKvEventsRequest) returns (stream KvEventBatch)`

KV event streaming is disabled by default for backward compatibility.  
When disabled, `SubscribeKvEvents` returns `UNIMPLEMENTED`.

To enable KV event streaming, set `--kv-events-config` with ZMQ publishing:

```bash
python -m vllm.entrypoints.grpc_server \
  --model <model> \
  --kv-events-config '{"enable_kv_cache_events": true, "publisher": "zmq", "endpoint": "tcp://*:5557", "replay_endpoint": "tcp://*:5558", "topic": "kv-events"}'
```

### Resume and Replay Semantics

- `start_sequence_number=0` starts from live stream state.
- `start_sequence_number>0` requests replay from that sequence number before
  live streaming, if `replay_endpoint` is configured.
- Replay completion is indicated by the publisher sentinel `(-1, empty payload)`.

### Data Parallel Endpoints

For data parallel configurations, the gRPC bridge follows the same endpoint
port offset logic as the internal ZMQ publisher. It also normalizes bind-style
`tcp://*:<port>` endpoints to connectable loopback addresses.

## Notes

- This is a dedicated gRPC entrypoint and is separate from the OpenAI-compatible
  HTTP server (`vllm serve`).
- Keep generated protobuf stubs in sync when changing `vllm/grpc/vllm_engine.proto`.
