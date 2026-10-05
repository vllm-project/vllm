# gRPC protocol

This directory is the canonical source for vLLM's gRPC schema.

See [context parallel deployment](../../docs/serving/context_parallel_deployment.md)
for effective attention block-size metadata in Python and `Control.GetServerInfo`.

Schema updates are no longer published to the Buf Schema Registry. Rust consumers
should use `vllm-proto` from crates.io; consumers in other languages can generate
bindings from the `.proto` files in this directory. Buf still builds and lints
the schemas on pull requests, including on forks.

## Rust crate

`vllm-proto` exposes the Prost message types and Tonic client and server modules for both APIs:

```toml
[dependencies]
vllm-proto = "0.4"
```

For example, import `vllm_proto::inference_client::InferenceClient` or
`vllm_proto::control_client::ControlClient`. The vLLM frontend uses this same crate.
The package includes the canonical `.proto` files and generates bindings during
compilation using `protox` and `tonic-prost-build`. Consumers do not need Buf
credentials, the vLLM checkout, or a separately installed `protoc`.

### Protocol history

These are protocol crate versions, not vLLM Python package versions. Dates are
the release-tag creation dates in America/Los_Angeles time. The linked tags
identify the schemas included in each release.

| Tag date (2026) | Crate release | Schema additions since the previous release |
| --- | --- | --- |
| Sep 11 | [0.1.0](https://github.com/vllm-project/vllm/tree/proto-v0.1.0/rust/proto) | First published crate for the existing `Inference` and `Control` services ([#56365](https://github.com/vllm-project/vllm/pull/56365)). |
| Sep 16 | [0.2.0](https://github.com/vllm-project/vllm/tree/proto-v0.2.0/rust/proto) | Preprocessed multimodal features ([#55047](https://github.com/vllm-project/vllm/pull/55047)), optional request watermarking ([#56338](https://github.com/vllm-project/vllm/pull/56338)), and optional effective attention block-size metadata ([#56538](https://github.com/vllm-project/vllm/pull/56538)). |
| Sep 16 | [0.3.0](https://github.com/vllm-project/vllm/tree/proto-v0.3.0/rust/proto) | `ParallelismInfo.data_parallel_size_local = 7` for local rank ownership in hybrid DP ([#57116](https://github.com/vllm-project/vllm/pull/57116), [#57233](https://github.com/vllm-project/vllm/pull/57233)). |
| Sep 30 | [0.4.0](https://github.com/vllm-project/vllm/tree/proto-v0.4.0/rust/proto) | Response sampling masks ([#56777](https://github.com/vllm-project/vllm/pull/56777)), the request KV-hints envelope ([#53423](https://github.com/vllm-project/vllm/pull/53423)), and `Control.Shutdown` ([#59316](https://github.com/vllm-project/vllm/pull/59316)). |

### Compatibility across these releases

The schema changes from 0.1.0 through 0.4.0 are additive: existing field numbers,
field types, and RPC signatures are unchanged. A client generated from 0.4.0 can
encode requests and decode responses for the RPCs shared with a 0.2.0 server.
The reverse direction also retains the shared wire format. This is
[protobuf binary wire compatibility](https://protobuf.dev/programming-guides/proto3/#binary),
not a guarantee that an older server implements newer features or that every
vLLM runtime combination has been tested.

| Newer client talking to a 0.2.0 server | Behavior |
| --- | --- |
| Existing generation and discovery RPCs | Shared fields retain their wire representation; use only features implemented by the server. |
| Local-DP ownership | The absent `data_parallel_size_local` decodes as zero, meaning unknown. Do not infer a hybrid frontend's rank range from global DP size and starting rank alone. Require a server implementing the 0.3.0 metadata for hybrid ownership. |
| KV hints | The 0.2.0 schema has no `GenerateRequest.kv_hints`; an older implementation does not interpret that field. Successful generation does not establish that the requested hint was applied. |
| Sampling masks | A 0.2.0 response has no `SequenceOutput.sampling_mask`; a newer decoder sees an empty repeated field. |
| Shutdown | A server without `Control.Shutdown` returns gRPC `UNIMPLEMENTED` for that RPC. Other Control RPCs remain independent. |

Adding a field can still break Rust source that constructs a generated message
with a complete struct literal. The minor crate bumps therefore do not by
themselves mean that existing RPC traffic became incompatible. Consumers must
handle absent fields and unsupported RPCs explicitly; the crate version is not
negotiated on each gRPC connection.

Launcher support is separate from schema support. For example, Python
`vllm serve --grpc-port` requires [#59659](https://github.com/vllm-project/vllm/pull/59659),
while standalone `vllm-rs serve` already exposes a gRPC port. A protocol tag alone
does not establish that a packaged Python launcher accepts that flag. Keep the
Python engine and its Rust frontend from the same vLLM build; their internal
MessagePack interface is a separate compatibility boundary.

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
