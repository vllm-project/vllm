# Native Rust gRPC request retirement

The native Rust frontend's `vllm.Inference` and `vllm.Control` services can
acknowledge that EngineCore has processed scheduler cancellation. This is
useful when an orchestrator needs more than a locally generated terminal chunk
or a successfully sent Abort message.

Start the native Rust frontend with a gRPC listener:

```bash
VLLM_USE_RUST_FRONTEND=1 vllm serve Qwen/Qwen3-0.6B \
  --host 127.0.0.1 --port 8000 --grpc-port 50051
```

See the [Rust frontend instructions](https://github.com/vllm-project/vllm/tree/main/rust)
for installation. This contract is separate from the Python `--grpc` service.

Set `wait_for_retirement: true` on a `GenerateRequest` or `AbortRequest`:

```json
{"request_ids": ["my-active-request"], "wait_for_retirement": true}
```

For generation, the terminal `GenerateResponse` carries
`scheduler_retired: true` only after the engine replies. This includes a
stop-string match in the frontend. For explicit cancellation, wait for the
`AbortResponse.scheduler_retired` value; the generation stream can end first.
The default false value, including a response from an older server, is not an
acknowledgement.

Generation errors are returned before waiting for retirement confirmation,
preserving the original gRPC status. Such errors do not acknowledge retirement.

The server captures randomized internal IDs before local cancellation removes
their public-ID mappings, and reuses EngineCore's existing `abort_requests`
utility on its ordered input path. It waits for all connected engines to
reply. An unknown public ID, or one whose active mapping has been removed,
returns `FAILED_PRECONDITION` when waiting is requested. Generation also requires
a unique active public ID.
The ordinary Abort API's idempotent no-op behavior is unchanged when waiting
is disabled. An empty Abort returns no retirement acknowledgement.

The wait has a 30-second server limit; clients should also set their RPC
deadline. An error, timeout, or disconnection leaves completion unknown and
does not roll back cancellation. Dropping the generation stream early still
triggers the existing automatic cancellation without providing a receipt.

!!! warning "Scheduler retirement is not a resource-release fence"
    The acknowledgement does not wait for GPU steps, deferred KV block frees,
    remote KV/encoder transfers, or readers of external media buffers. Keep
    those resources until their own lifetime contract permits release. Cached
    prefix blocks may legitimately remain cached.

These fields are available in `vllm-proto` 0.8. Generate bindings from the
canonical `rust/proto` schemas for clients in other languages.
