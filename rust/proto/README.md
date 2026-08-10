# gRPC protocol

This directory is the canonical source for vLLM's gRPC schema.

The schema is published to `buf.build/vllm-project/vllm`:

- A daily workflow publishes the latest Git `main` schema to the `nightly` label.
- The workflow can be run manually to retry nightly publication.
- Tags matching `v*` update the Buf `main` label and publish the corresponding release label.
- Buf commits and generated SDK versions are immutable and can be pinned by consumers.

Repository setup requires a `BUF_TOKEN` GitHub Actions secret with permission to create and push the public Buf module. Register the Prost and Tonic generated SDKs for the `main` and `nightly` labels once so subsequent pushes generate them automatically.
