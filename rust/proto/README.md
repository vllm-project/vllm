# gRPC protocol

This directory is the canonical source for vLLM's gRPC schema.

See [context parallel deployment](../../docs/serving/context_parallel_deployment.md)
for effective attention block-size metadata in Python and `Control.GetServerInfo`.

Schema updates are no longer published to the Buf Schema Registry. Rust consumers
should use `vllm-proto` from crates.io; consumers in other languages can generate
bindings from the `.proto` files in this directory. Buf still builds and lints
the schemas on pull requests, including on forks.

## Rust crate

`vllm-proto` exposes the Prost message types and Tonic client and server modules for all APIs:

```toml
[dependencies]
vllm-proto = "0.7"
```

For example, import `vllm_proto::inference_client::InferenceClient`,
`vllm_proto::control_client::ControlClient`,
`vllm_proto::kv_events_client::KvEventsClient`, or
`vllm_proto::render_client::RenderClient`. The vLLM frontend uses this same crate.
The package includes the canonical `.proto` files and generates bindings during
compilation using `protox` and `tonic-prost-build`. Consumers do not need Buf
credentials, the vLLM checkout, or a separately installed `protoc`.

### Versioning and releases

The crate has its own version in `Cargo.toml`, independent of the vLLM release
number and the other Rust workspace crates. Release it when changes to the
schema, generated Rust API, or dependencies require a new version. Review both
Rust API and protobuf wire compatibility when choosing the version bump;
a wire-compatible schema addition can still break Rust callers.

On pull requests and releases, `cargo-semver-checks` compares the crate with its
latest published version. Include any required version bump in the protocol
change PR. This check becomes available after the first manual publication.

1. Update the crate version in `rust/proto/Cargo.toml` and update `rust/Cargo.lock`.
2. Run `cargo publish --manifest-path rust/proto/Cargo.toml --locked --dry-run`
   and the frontend gRPC tests. Record the tested vLLM releases or revisions in
   the release notes; matching crate versions alone do not establish runtime compatibility.
3. After the change merges, create a `proto-v<version>` tag on that commit.
   The `proto-crate.yml` workflow checks that the tag matches the crate version,
   verifies the package, and publishes it to crates.io.

There is no scheduled crate publication. Pull requests and manual workflow runs only
verify packaging, including on forks. Retry a failed release by rerunning its
tag workflow; crates.io versions cannot be overwritten.

### Publisher setup

A vLLM maintainer must publish the first version and establish the crate's owners.
For subsequent releases, configure [crates.io Trusted Publishing](https://crates.io/docs/trusted-publishing)
for repository `vllm-project/vllm`, workflow `proto-crate.yml`, and environment
`crates-io`. Configure that GitHub environment to allow `proto-v*` release tags.
Only tag pushes in the upstream repository can publish; forks do not obtain
publishing credentials.

## KV cache events

`KvEvents.SubscribeKvEvents` streams the KV cache events of an engine started
with `--kv-events-config '{"enable_kv_cache_events": true}'`, one stream per
data-parallel rank, with the same batches and sequence numbers as the engine's
ZMQ publisher. Nothing else needs configuring: at startup the frontend gives
each engine a private events address and replay address, which the publisher
binds in addition to its configured endpoints: IPC when every engine it connects
to runs on its node, and TCP on each engine's own node otherwise (under
`VLLM_USE_RUST_FRONTEND=1 vllm serve`, only for the engines on other nodes).
`Control.GetKvEventSources` lists the stream as a source with transport `grpc`.

A subscriber without `resume` receives live batches. With `resume`, the
frontend first replays the batches the engine still buffers (`buffer_steps`)
from the cursor, then continues live; it keeps no history of its own, and fills
batches its own subscription misses from the same buffer. A cursor older than
the buffer fails with `OUT_OF_RANGE`; batches that can no longer be recovered
end the stream with `DATA_LOSS`.

Each batch is one gRPC message. Requests that set `kv_cache_report_mode=full`
can produce batches above the 4 MiB that gRPC clients accept by default, so
subscribers should raise their receive limit.
