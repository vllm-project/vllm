This is a simple example of [Decoupled Mode](https://github.com/triton-inference-server/python_backend/tree/main/examples/decoupled) based on the [Triton Python Backend](https://github.com/triton-inference-server/python_backend), which achieves throughput and latency close to [API Server](https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/api_server.py). It uses [Streaming gRPC](https://grpc.io/docs/what-is-grpc/core-concepts/#bidirectional-streaming-rpc).

Only the `prompt` and `max_tokens` parameters have been configured, users can configure custom configurations according to their own usage scenarios.

The example is designed to show the integration with Triton and in no way should be used in production.
