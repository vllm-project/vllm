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
vllm-proto = "0.8"
```

For example, import `vllm_proto::inference_client::InferenceClient`,
`vllm_proto::control_client::ControlClient`, or
`vllm_proto::render_client::RenderClient`. The vLLM frontend uses this same crate.
The package includes the canonical `.proto` files and generates bindings during
compilation using `protox` and `tonic-prost-build`. Consumers do not need Buf
credentials, the vLLM checkout, or a separately installed `protoc`.

### Scheduler retirement acknowledgements

`Generate`, `GenerateStream`, and `Control.Abort` accept
`wait_for_retirement=true`. Generation delays its terminal response until the
engine replies to an ordered scheduler abort utility, including when a stop
string ends generation in the frontend. `Abort` can still terminate the
generation stream before its own response arrives; wait for its response.

Only `scheduler_retired=true` is an acknowledgement. Older servers ignore the
new request field and return the default false value. Empty Abort requests do
not produce an acknowledgement. Opted-in Abort requires every supplied ID to
still have an active mapping, and generation requires a unique active external
ID. Request-ID randomization must be enabled: the captured internal IDs must
never refer to a later generation reusing the same external ID.

The wait is bounded to 30 seconds, or the RPC deadline if shorter. Failure,
disconnect, and timeout mean retirement was not confirmed; they do not undo
the cancellation. Dropping a generation stream before its terminal response
retains the existing auto-abort behavior and provides no acknowledgement.

This acknowledges **scheduler retirement**, not completion of in-flight GPU
steps, KV/encoder transfers, deferred resource release, or external shared-memory
reads. It must not authorize recycling external media buffers. It neither
flushes prefix cache nor pauses other requests.

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
