# gRPC protocol

This directory is the canonical source for vLLM's gRPC schema.

See [context parallel deployment](../../docs/serving/context_parallel_deployment.md)
for effective attention block-size metadata in Python and `Control.GetServerInfo`.

The schema is published to `buf.build/vllm-project/vllm`:

- A daily workflow publishes the latest Git `main` schema to the `nightly` label.
- The workflow can be run manually to retry nightly publication.
- Tags matching `v*` update the Buf `main` label and publish the corresponding release label.
- Buf commits and generated SDK versions are immutable and can be pinned by consumers.

Repository setup requires a `BUF_TOKEN` GitHub Actions secret with permission to create and push the public Buf module. Register the Prost and Tonic generated SDKs for the `main` and `nightly` labels once so subsequent pushes generate them automatically.

## Rust crate

`vllm-proto` exposes the Prost message types and Tonic client and server modules for both APIs:

```toml
[dependencies]
vllm-proto = "0.2"
```

For example, import `vllm_proto::inference_client::InferenceClient` or
`vllm_proto::control_client::ControlClient`. The vLLM frontend uses this same crate.
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

1. Update the crate version and the `vllm-proto` workspace dependency together,
   and update `rust/Cargo.lock`.
2. Run `cargo publish --manifest-path rust/proto/Cargo.toml --locked --dry-run`
   and the frontend gRPC tests. Record the tested vLLM releases or revisions in
   the release notes; matching crate versions alone do not establish runtime compatibility.
3. After the change merges, create a `proto-v<version>` tag on that commit.
   The `proto-crate.yml` workflow checks that the tag matches the crate version,
   verifies the package, and publishes it to crates.io.

Use `proto-v*` tags so they do not trigger Buf's `v*` release workflow. There is
no scheduled crate publication. Pull requests and manual workflow runs only
verify packaging, including on forks. Retry a failed release by rerunning its
tag workflow; crates.io versions cannot be overwritten.

### Publisher setup

A vLLM maintainer must publish the first version and establish the crate's owners.
For subsequent releases, configure [crates.io Trusted Publishing](https://crates.io/docs/trusted-publishing)
for repository `vllm-project/vllm`, workflow `proto-crate.yml`, and environment
`crates-io`. Configure that GitHub environment to allow `proto-v*` release tags.
Only tag pushes in the upstream repository can publish; forks do not obtain
publishing credentials.
