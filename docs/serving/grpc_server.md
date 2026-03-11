# gRPC Server

vLLM provides a gRPC entrypoint at `vllm.entrypoints.grpc_server`.

Install gRPC dependencies:

```bash
pip install "vllm[grpc]"
```

Start the gRPC server:

```bash
python -m vllm.entrypoints.grpc_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --host 0.0.0.0 \
  --port 50051
```

## KV Event Streaming RPC

`VllmEngine.SubscribeKvEvents` streams KV cache events as `KvEventBatch`.

To enable it, start the server with KV events enabled and a ZMQ publisher:

```bash
python -m vllm.entrypoints.grpc_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --kv-events-config '{"enable_kv_cache_events": true, "publisher": "zmq", "topic": "kv-events"}'
```

Notes:

- The RPC request accepts `start_sequence_number` for replay/catch-up.
- If KV events are disabled, the RPC returns `UNIMPLEMENTED`.
- For replay support, set `replay_endpoint` in `--kv-events-config`.
- For security, `SubscribeKvEvents` only allows local peers (`127.0.0.1`,
  `::1`, or `unix:`) by default. This protects sensitive `token_ids`.
- To allow remote subscribers, explicitly set
  `allow_remote_subscribe: true` in `--kv-events-config` and use trusted
  network controls.
- The local-peer check is transport-level. If you deploy behind a proxy or
  sidecar, enforce authentication and authorization at that boundary.

For a KV events example, see [examples/online_serving/disaggregated_serving/kv_events.sh](../../examples/online_serving/disaggregated_serving/kv_events.sh).
