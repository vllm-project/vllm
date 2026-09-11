# gRPC protocol

This directory is the canonical source for vLLM's gRPC schema.

`Control.GetServerInfo.effective_attention_block_size` reports the initialized
full-attention block size in tokens, including DCP scaling. `kv_block_size` keeps
its physical-size meaning. The optional field is absent when unavailable or
when engines disagree; clients can continue using the existing fields.
See [context parallel deployment](../../docs/serving/context_parallel_deployment.md)
for Python access and details.

The schema is published to `buf.build/vllm-project/vllm`:

- A daily workflow publishes the latest Git `main` schema to the `nightly` label.
- The workflow can be run manually to retry nightly publication.
- Tags matching `v*` update the Buf `main` label and publish the corresponding release label.
- Buf commits and generated SDK versions are immutable and can be pinned by consumers.

Repository setup requires a `BUF_TOKEN` GitHub Actions secret with permission to create and push the public Buf module. Register the Prost and Tonic generated SDKs for the `main` and `nightly` labels once so subsequent pushes generate them automatically.
