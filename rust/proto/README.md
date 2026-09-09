# gRPC protocol

This directory is the canonical source for vLLM's gRPC schema.

`Control.GetServerInfo` exposes per-group cache geometry through
`kv_cache_metadata.groups`. The same descriptors are available in Python through
`await engine_client.get_kv_cache_group_metadata()` and in the engine startup
response's `kv_cache_group_metadata` field. Use Python and Rust built from the
same vLLM revision when consuming this metadata.

Each descriptor contains `group_id` (matching KV events' `group_idx`), `kind`,
`block_size` (physical tokens per block), and `logical_block_size` (the initialized
manager's effective full-block size). Partial events carry their own actual size;
consumers must not replace it with the full-block size. Unknown cache kinds use
`"unknown"`.

An absent `kv_cache_metadata` means metadata is unavailable, including with older
engine handshakes. Python returns `None` for unsupported schedulers. A present
empty group list means there are no cache groups. Discovery fails with `FAILED_PRECONDITION`
if data-parallel ranks report different descriptors or availability.

The schema is published to `buf.build/vllm-project/vllm`:

- A daily workflow publishes the latest Git `main` schema to the `nightly` label.
- The workflow can be run manually to retry nightly publication.
- Tags matching `v*` update the Buf `main` label and publish the corresponding release label.
- Buf commits and generated SDK versions are immutable and can be pinned by consumers.

Repository setup requires a `BUF_TOKEN` GitHub Actions secret with permission to create and push the public Buf module. Register the Prost and Tonic generated SDKs for the `main` and `nightly` labels once so subsequent pushes generate them automatically.
